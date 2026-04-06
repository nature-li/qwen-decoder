#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "gpu_decoder.h"
#include "pd_comm.h"

struct SharedState {
  std::mutex mu;
  std::condition_variable cv;
  std::queue<Request*> ready_queue;
  std::atomic<bool> shutdown{false};
  std::atomic<int> total_output_tokens{0};
};

struct EncodeQueue {
  std::mutex mu;
  std::condition_variable cv;
  std::queue<Request*> q;
  bool closed = false;

  void push(Request* req) {
    std::lock_guard<std::mutex> lock(mu);
    q.push(std::move(req));
    cv.notify_one();
  }

  bool pop(Request** req) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return !q.empty() || closed; });
    if (q.empty()) return false;
    *req = q.front();
    q.pop();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu);
    closed = true;
    cv.notify_all();
  }
};

void d_send_thread_func(EncodeQueue& encode_queue, DNode& dnode) {
  Request* req = nullptr;
  while (encode_queue.pop(&req)) {
    pd::PrefillRequest grpc_req;
    grpc_req.set_d_node_id(dnode.d_node_id);
    grpc_req.set_req_id(req->id);
    for (int tok : req->prompt_tokens) {
      grpc_req.add_token_ids(tok);
    }
    if (!dnode.send_request(grpc_req)) {
      fprintf(stderr, "[D.send] 发送失败 req_id=%d\n", req->id);
      break;
    }
    // fprintf(stderr, "[D.send] req_id=%d n_tokens=%d\n", req->id, (int)req->prompt_tokens.size());
  }
  fprintf(stderr, "[D.send] 线程退出\n");
}

void d_recv_thread_func(std::atomic<int>& total_requests, std::atomic<bool>& encode_done,
                        GPUDecoder* decoder, DNode& dnode, BlockPool* pool, int n_layers,
                        int kv_dim, SharedState& state, std::vector<Request*>& all_reqs,
                        std::mutex& reqs_mu) {
  int received = 0;
  while (true) {
    if (encode_done && received >= total_requests) break;

    pd::PrefillResponse grpc_resp;
    fprintf(stderr, "[D.recv] 等待 response...\n");
    if (!dnode.recv_response(grpc_resp)) {
      fprintf(stderr, "[D.recv] recv_response 失败，退出\n");
      break;
    }
    fprintf(stderr, "[D.recv] 收到 response req_id=%d\n", grpc_resp.req_id());

    Request* req = nullptr;
    {
      std::lock_guard<std::mutex> lock(reqs_mu);
      req = all_reqs[received];
    }

    auto* r = new Request();
    r->id = grpc_resp.req_id();
    r->pos = req->prompt_tokens.size();
    r->prefill_done = true;
    r->finished = false;
    r->cur_token = grpc_resp.first_token();
    r->input = req->input;

    int need_blocks = (req->prompt_tokens.size() - 1) / BLOCK_SIZE + 1;
    bool oom = false;
    for (int i = 0; i < need_blocks; i++) {
      int block_id = pool->allocate();
      if (block_id < 0) {
        oom = true;
        break;
      }
      r->block_table.add_block(block_id);
    }

    if (oom) {
      delete r;
      break;
    }

    std::vector<int> slots(req->prompt_tokens.size());
    for (int j = 0; j < (int)req->prompt_tokens.size(); j++) {
      slots[j] = r->block_table.physical_idx(j) * BLOCK_SIZE + r->block_table.block_offset(j);
    }

    auto t_kv_start = std::chrono::steady_clock::now();
    printf("before nccl_recv_kv\n");
    nccl_recv_kv(dnode.nccl, pool->get_k_cache(), pool->get_v_cache(), slots,
                 req->prompt_tokens.size(), n_layers, kv_dim);
    printf("after nccl_recv_kv\n");
    auto t_kv_end = std::chrono::steady_clock::now();
    double kv_ms = std::chrono::duration<double, std::milli>(t_kv_end - t_kv_start).count();
    double kv_mb =
        (double)req->prompt_tokens.size() * n_layers * kv_dim * 2 * sizeof(__half) / 1024 / 1024;
    fprintf(stderr, "[D.recv] req_id=%d KV: %.1fMB in %.1fms\n", grpc_resp.req_id(), kv_mb, kv_ms);

    {
      std::lock_guard<std::mutex> lock(state.mu);
      state.ready_queue.push(r);
    }
    state.cv.notify_one();
    received++;
  }
  fprintf(stderr, "[D.recv] 线程退出，共收 %d 个\n", received);
}

void decode_thread_func(GPUDecoder* decoder, BlockPool* pool, int max_batch, int max_new_toks,
                        float temperature, int top_k, int total_requests, SharedState& state) {
  const Config& cfg = decoder->get_config();
  int max_blk = decoder->get_max_blocks_per_seq();
  std::mt19937 rng(42);

  std::vector<Request*> running(max_batch, nullptr);
  int completed = 0;
  int steps = 0;
  int total_batch_size = 0;
  double total_fwd_s = 0.0;
  double total_wait_s = 0.0;

  auto t_start = std::chrono::steady_clock::now();
  auto last_step_time = std::chrono::steady_clock::now();

  while (completed < total_requests) {
    std::vector<int> changed;
    int n_running = 0;

    {
      std::unique_lock<std::mutex> lock(state.mu);
      for (int i = 0; i < max_batch; i++) {
        if (running[i] != nullptr) n_running++;
      }

      if (n_running == 0 && state.ready_queue.empty()) {
        auto t_wait_start = std::chrono::steady_clock::now();
        state.cv.wait(lock, [&] { return !state.ready_queue.empty() || state.shutdown; });
        total_wait_s +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wait_start).count();
      }

      while (!state.ready_queue.empty()) {
        int slot = -1;
        for (int i = 0; i < max_batch; i++) {
          if (running[i] == nullptr) {
            slot = i;
            break;
          }
        }
        if (slot == -1) break;

        Request* r = state.ready_queue.front();
        state.ready_queue.pop();
        running[slot] = r;
        changed.push_back(slot);
        n_running++;
      }
    }

    if (n_running == 0) continue;

    std::vector<FlatRequest> flat_reqs;
    std::vector<int> flat_tokens, flat_positions, token_to_req;
    std::vector<int> token_slot, last_tok_idx;
    std::vector<int> dec_flat, dec_pos;

    for (int i = 0; i < (int)running.size(); i++) {
      Request* r = running[i];
      if (r == nullptr || r->finished) continue;

      int need = r->pos / BLOCK_SIZE + 1;
      int old_blocks = r->block_table.num_blocks();
      while (r->block_table.num_blocks() < need) {
        int block_id = pool->allocate();
        if (block_id < 0) {
          r->finished = true;
          break;
        }
        r->block_table.add_block(block_id);
      }
      if (r->finished) continue;
      if (r->block_table.num_blocks() != old_blocks) changed.push_back(i);

      FlatRequest fr;
      fr.req_idx = i;
      fr.flat_offset = (int)flat_tokens.size();
      fr.n_tokens = 1;
      fr.start_pos = r->pos;
      fr.is_prefill = false;

      int phy = r->block_table.physical_idx(r->pos);
      int off = r->block_table.block_offset(r->pos);
      flat_tokens.push_back(r->cur_token);
      flat_positions.push_back(r->pos);
      token_to_req.push_back(i);
      token_slot.push_back(phy * BLOCK_SIZE + off);
      last_tok_idx.push_back(fr.flat_offset);
      dec_flat.push_back(fr.flat_offset);
      dec_pos.push_back(r->pos);
      flat_reqs.push_back(fr);
    }

    if (flat_reqs.empty()) continue;

    std::sort(changed.begin(), changed.end());
    changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
    if (!changed.empty()) decoder->update_block_table_partial(running, changed, max_blk);

    auto now = std::chrono::steady_clock::now();
    double between = std::chrono::duration<double, std::milli>(now - last_step_time).count();
    last_step_time = now;
    fprintf(stdout, "between: %.2f, step %d: total_tokens=%d n_decode=%d\n", between, steps,
            (int)flat_tokens.size(), (int)dec_pos.size());

    auto t_fwd_start = std::chrono::steady_clock::now();
    decoder->forward_flat(flat_reqs, flat_tokens, flat_positions, token_to_req, token_slot,
                          last_tok_idx, dec_flat, (int)flat_tokens.size());
    total_fwd_s +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_fwd_start).count();
    steps++;
    total_batch_size += (int)flat_reqs.size();

    for (int fi = 0; fi < (int)flat_reqs.size(); fi++) {
      auto& fr = flat_reqs[fi];
      Request* r = running[fr.req_idx];
      if (r->finished) continue;

      float* logits = decoder->get_logits_batch(fi);
      int next_token = sample_topk(logits, cfg.vocab_size, top_k, temperature, rng);

      const char* piece = decode(decoder->get_tokenizer(), r->cur_token);
      if (piece) r->output += piece;
      r->pos++;
      r->cur_token = next_token;
      r->n_generated++;
      state.total_output_tokens++;

      if (next_token == decoder->get_tokenizer().eos_token_id ||
          next_token == decoder->get_tokenizer().bos_token_id ||
          decoder->get_tokenizer().vocab[next_token] == "<|im_end|>" ||
          r->n_generated >= max_new_toks || r->pos >= cfg.seq_len) {
        r->finished = true;
      }
    }

    for (int i = 0; i < (int)running.size(); i++) {
      Request* r = running[i];
      if (r != nullptr && r->finished) {
        printf("request id: %d\nprompt: %s\noutput: %s\n--------------------\n", r->id,
               r->input.c_str(), r->output.c_str());
        r->block_table.free_blocks([pool](int id) { pool->free(id); });
        running[i] = nullptr;
        completed++;
      }
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();
  elapsed -= total_wait_s;

  fprintf(stderr, "=== decode 统计 ===\n");
  fprintf(stderr, "总输出 token:  %d\n", (int)state.total_output_tokens);
  fprintf(stderr, "总时间:        %.2fs\n", elapsed);
  fprintf(stderr, "等待时间:      %.2fs\n", total_wait_s);
  fprintf(stderr, "forward时间:   %.2fs\n", total_fwd_s);
  fprintf(stderr, "decode 吞吐:   %.1f tokens/s\n", state.total_output_tokens / elapsed);
  fprintf(stderr, "GPU 利用率:    %.1f%%\n", total_fwd_s / elapsed * 100);
  fprintf(stderr, "总步数:        %d\n", steps);
  fprintf(stderr, "平均 batch:    %.1f\n", (float)total_batch_size / steps);

  state.shutdown = true;
  state.cv.notify_all();
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <model_file> <prefill_node_ip> [max_batch]\n", argv[0]);
    return 1;
  }

  const char* model_file = argv[1];
  const char* prefill_node_ip = argv[2];
  int max_batch = (argc >= 4) ? atoi(argv[3]) : 64;

  auto* decoder = new GPUDecoder(model_file, max_batch, 4096, 512, 4096);
  fprintf(stderr, "D节点模型加载完成\n");

  DNode dnode;
  // 1. 连接 P节点
  if (!dnode.init(prefill_node_ip)) return 1;

  // 2. 注册并初始化 NCCL
  if (!dnode.register_and_init_nccl()) return 1;
  fprintf(stderr, "D节点[%d] 初始化完成\n", dnode.d_node_id);

  // 3. 建立 Prefill 双向流
  if (!dnode.open_prefill_stream()) return 1;

  const Config& cfg = decoder->get_config();
  int n_layers = cfg.n_layers;
  int head_dim = cfg.dim / cfg.n_heads;
  int kv_dim = cfg.n_kv_heads * head_dim;
  BlockPool* pool = decoder->get_block_pool();

  // 读用户输入
  std::vector<std::string> user_inputs;
  std::string line;
  int idx = 0;
  while (true) {
    printf("User[%d] (empty to start): ", idx++);
    if (!std::getline(std::cin, line) || line.empty()) break;
    user_inputs.push_back(line);
  }

  if (user_inputs.empty()) {
    fprintf(stderr, "no input\n");
    delete decoder;
    return 1;
  }

  std::vector<Request*> all_requests;
  for (int i = 0; i < (int)user_inputs.size(); i++) {
    auto* req = new Request();
    req->id = i;
    req->input = user_inputs[i];
    std::string prompt = apply_chat_template(user_inputs[i]);
    decoder->encode_pub(prompt, req->prompt_tokens);
    req->prompt_tokens.insert(req->prompt_tokens.begin(), decoder->get_tokenizer().bos_token_id);
    all_requests.push_back(req);
  }

  fprintf(stderr, "%d requests\n", (int)user_inputs.size());

  SharedState state;
  EncodeQueue encode_queue;
  std::mutex reqs_mu;
  std::atomic<int> total_requests{0};
  std::atomic<bool> encode_done{false};

  float temperature = 0.0f;
  int top_k = 30;
  int max_new_toks = 256;

  std::thread send_th(d_send_thread_func, std::ref(encode_queue), std::ref(dnode));
  std::thread recv_th(d_recv_thread_func, std::ref(total_requests), std::ref(encode_done), decoder,
                      std::ref(dnode), pool, n_layers, kv_dim, std::ref(state),
                      std::ref(all_requests), std::ref(reqs_mu));
  std::thread decode_th(decode_thread_func, decoder, pool, max_batch, max_new_toks, temperature,
                        top_k, (int)user_inputs.size(), std::ref(state));

  for (int i = 0; i < (int)all_requests.size(); i++) {
    Request* req = all_requests[i];
    req->id = i;
    total_requests++;
    encode_queue.push(req);
  }

  encode_queue.close();
  encode_done = true;
  fprintf(stderr, "[main] encode 完成，共 %d 个请求\n", (int)user_inputs.size());

  send_th.join();
  recv_th.join();
  decode_th.join();

  // 正常退出，关闭 prefill 流
  dnode.send_shutdown();

  for (auto* req : all_requests) delete req;
  delete decoder;
  return 0;
}
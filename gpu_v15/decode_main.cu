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

// ============================================================================
// 共享状态：通信线程 → decode 线程
// ============================================================================
struct SharedState {
  std::mutex mu;
  std::condition_variable cv;
  std::queue<Request*> ready_queue;
  std::atomic<bool> shutdown{false};
  std::atomic<int> total_output_tokens{0};
};

// ============================================================================
// ReqInfo：tokenize 后的请求信息
// ============================================================================
struct ReqInfo {
  int req_id;
  int n_tokens;
  std::vector<int> prompt_tokens;
  std::string input;
};

// encode 后的请求队列
struct EncodeQueue {
  std::mutex mu;
  std::condition_variable cv;
  std::queue<ReqInfo> q;
  bool closed = false;

  void push(ReqInfo info) {
    std::lock_guard<std::mutex> lock(mu);
    q.push(std::move(info));
    cv.notify_one();
  }

  bool pop(ReqInfo& info) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return !q.empty() || closed; });
    if (q.empty()) return false;
    info = std::move(q.front());
    q.pop();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu);
    closed = true;
    cv.notify_all();
  }
};

// ============================================================================
// D节点发送线程
// ============================================================================
void d_send_thread_func(EncodeQueue& encode_queue, DNode& dnode) {
  ReqInfo info;
  while (encode_queue.pop(info)) {
    PrefillRequest req_msg;
    req_msg.req_id = info.req_id;
    req_msg.n_tokens = info.n_tokens;
    dnode.send_request(req_msg, info.prompt_tokens);
    fprintf(stderr, "[D.send] req_id=%d n_tokens=%d\n", info.req_id, info.n_tokens);
  }
  fprintf(stderr, "[D.send] 线程退出\n");
}

// ============================================================================
// D节点接收线程
// ============================================================================
void d_recv_thread_func(std::atomic<int>& total_requests, std::atomic<bool>& encode_done,
                        GPUDecoder* decoder, NcclComm& nccl, DNode& dnode, BlockPool* pool,
                        int n_layers, int kv_dim, SharedState& state,
                        std::vector<ReqInfo>& all_reqs, std::mutex& reqs_mu) {
  int received = 0;
  while (true) {
    // 等到 encode 完成或者还有请求没收
    if (encode_done && received >= total_requests) break;

    PrefillResponse resp;
    if (!dnode.recv_response(resp)) break;

    // 从 all_reqs 里找对应的 req
    ReqInfo info;
    {
      std::lock_guard<std::mutex> lock(reqs_mu);
      // 按顺序收，resp.req_id 就是 received
      info = all_reqs[received];
    }

    auto* r = new Request();
    r->id = resp.req_id;
    r->pos = info.n_tokens;
    r->prefill_done = true;
    r->finished = false;
    r->cur_token = resp.first_token;
    r->input = info.input;

    int need_blocks = (info.n_tokens - 1) / BLOCK_SIZE + 1;
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

    std::vector<int> slots(info.n_tokens);
    for (int j = 0; j < info.n_tokens; j++) {
      slots[j] = r->block_table.physical_idx(j) * BLOCK_SIZE + r->block_table.block_offset(j);
    }

    auto t_kv_start = std::chrono::steady_clock::now();
    nccl_recv_kv(nccl, pool->get_k_cache(), pool->get_v_cache(), slots, info.n_tokens, n_layers,
                 kv_dim);
    auto t_kv_end = std::chrono::steady_clock::now();
    double kv_ms = std::chrono::duration<double, std::milli>(t_kv_end - t_kv_start).count();
    double kv_mb = (double)info.n_tokens * n_layers * kv_dim * 2 * sizeof(__half) / 1024 / 1024;
    fprintf(stderr, "[D.recv] req_id=%d KV: %.1fMB in %.1fms (%.1f MB/s)\n", resp.req_id, kv_mb,
            kv_ms, kv_mb / kv_ms * 1000);

    {
      std::lock_guard<std::mutex> lock(state.mu);
      state.ready_queue.push(r);
    }
    state.cv.notify_one();
    received++;
  }
  fprintf(stderr, "[D.recv] 线程退出，共收 %d 个\n", received);
}

// ============================================================================
// decode 线程：batch decode
// ============================================================================
void decode_thread_func(GPUDecoder* decoder, BlockPool* pool, int max_batch, int max_new_toks,
                        float temperature, int top_k, int total_requests, SharedState& state) {
  const Config& cfg = decoder->get_config();
  int max_blk = decoder->get_max_blocks_per_seq();
  std::mt19937 rng(42);

  std::vector<Request*> running(max_batch, nullptr);
  int completed = 0;
  int steps = 0;
  int total_batch_size = 0;
  double total_wait_s = 0.0;
  double total_fwd_s = 0.0;

  auto t_start = std::chrono::steady_clock::now();
  while (completed < total_requests) {
    std::vector<int> changed;

    // 统计当前 running 里有多少非空槽
    int n_running = 0;
    {
      std::unique_lock<std::mutex> lock(state.mu);
      for (int i = 0; i < max_batch; i++) {
        if (running[i] != nullptr) {
          n_running++;
        }
      }

      // running 全空且 ready_queue 也空时才等待
      if (n_running == 0 && state.ready_queue.empty()) {
        auto t_wait_start = std::chrono::steady_clock::now();
        state.cv.wait(lock, [&] { return !state.ready_queue.empty() || state.shutdown; });
        total_wait_s +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wait_start).count();
      }

      // 找空槽，填入就绪请求
      while (!state.ready_queue.empty()) {
        // 找空槽
        int slot = -1;
        for (int i = 0; i < max_batch; i++) {
          if (running[i] == nullptr) {
            slot = i;
            break;
          }
        }
        if (slot == -1) {
          break;  // 没有空槽
        }

        Request* r = state.ready_queue.front();
        state.ready_queue.pop();
        running[slot] = r;
        changed.push_back(slot);
        n_running++;
        fprintf(stderr, "[decode] 加入请求 id=%d slot=%d\n", r->id, slot);
      }
    }

    if (n_running == 0) continue;

    // 组装 flat batch（纯 decode）
    std::vector<FlatRequest> flat_reqs;
    std::vector<int> flat_tokens, flat_positions, token_to_seq;
    std::vector<int> slot_mapping, last_tok_idx;
    std::vector<int> dec_flat, dec_pos, dec_seq;

    for (int i = 0; i < (int)running.size(); i++) {
      Request* r = running[i];
      if (r == nullptr || r->finished) {
        continue;
      }

      // 确保有足够的 block
      int need = r->pos / BLOCK_SIZE + 1;
      int old_blocks = r->block_table.num_blocks();
      while (r->block_table.num_blocks() < need) {
        int block_id = pool->allocate();
        if (block_id < 0) {
          fprintf(stderr, "[decode] OOM\n");
          r->finished = true;
          break;
        }
        r->block_table.add_block(block_id);
      }
      if (r->finished) {
        continue;
      }
      if (r->block_table.num_blocks() != old_blocks) {
        changed.push_back(i);
      }

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
      token_to_seq.push_back(i);
      slot_mapping.push_back(phy * BLOCK_SIZE + off);
      last_tok_idx.push_back(fr.flat_offset);
      dec_flat.push_back(fr.flat_offset);
      dec_pos.push_back(r->pos);
      dec_seq.push_back(i);

      flat_reqs.push_back(fr);
    }

    if (flat_reqs.empty()) {
      continue;
    }

    // 去重 changed
    std::sort(changed.begin(), changed.end());
    changed.erase(std::unique(changed.begin(), changed.end()), changed.end());

    // 更新 block_table
    if (!changed.empty()) decoder->update_block_table_partial(running, changed, max_blk);

    // generate_continuous 里 forward_flat 调用前加
    fprintf(stdout, "step %d: total_tokens=%d n_decode=%d n_prefill=%d prefill_tokens=%d\n", steps,
            (int)flat_tokens.size(), (int)dec_pos.size(),
            (int)flat_reqs.size() - (int)dec_pos.size(),
            (int)flat_tokens.size() - (int)dec_pos.size());

    // forward
    auto t_fwd_start = std::chrono::steady_clock::now();
    decoder->forward_flat(flat_reqs, flat_tokens, flat_positions, token_to_seq, slot_mapping,
                          last_tok_idx, dec_flat, dec_pos, dec_seq, (int)flat_tokens.size());
    total_fwd_s +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_fwd_start).count();

    steps++;
    total_batch_size += (int)flat_reqs.size();

    // 采样、更新状态
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

    // 移出完成的请求
    for (int i = 0; i < running.size(); i++) {
      Request* r = running[i];
      if (r != nullptr && r->finished) {
        printf("request id: %d\nprompt: %s\noutput: %s\n--------------------\n", r->id,
               r->input.c_str(), r->output.c_str());
        r->block_table.free_blocks([pool](int id) { pool->free(id); });
        delete r;
        running[i] = nullptr;
        completed++;
        fprintf(stderr, "[decode] 完成 %d/%d\n", completed, total_requests);
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

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <model_file> <prefill_node_ip> [max_batch]\n", argv[0]);
    return 1;
  }

  const char* model_file = argv[1];
  const char* prefill_node_ip = argv[2];
  int max_batch = (argc >= 4) ? atoi(argv[3]) : 64;

  int max_pps = 512;
  int max_total_tokens = 4096;
  auto* decoder = new GPUDecoder(model_file, max_batch, 4096, max_pps, max_total_tokens);
  fprintf(stderr, "D节点模型加载完成\n");

  // ZMQ 初始化
  DNode dnode;
  if (!dnode.init(prefill_node_ip)) return 1;

  // NCCL 初始化
  NcclComm nccl;
  if (!nccl_init_decode(nccl, dnode)) return 1;

  const Config& cfg = decoder->get_config();
  int n_layers = cfg.n_layers;
  int head_dim = cfg.dim / cfg.n_heads;
  int kv_dim = cfg.n_kv_heads * head_dim;
  BlockPool* pool = decoder->get_block_pool();

  fprintf(stderr, "D节点就绪，开始接收请求...\n");

  // 收集用户输入
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

  fprintf(stderr, "%d requests\n", (int)user_inputs.size());

  // 共享状态
  SharedState state;
  EncodeQueue encode_queue;
  std::vector<ReqInfo> all_reqs;
  std::mutex reqs_mu;
  std::atomic<int> total_requests{0};
  std::atomic<bool> encode_done{false};

  float temperature = 0.0f;
  int top_k = 30;
  int max_new_toks = 256;

  // 先启动三个线程
  std::thread send_th(d_send_thread_func, std::ref(encode_queue), std::ref(dnode));

  std::thread recv_th(d_recv_thread_func, std::ref(total_requests), std::ref(encode_done), decoder,
                      std::ref(nccl), std::ref(dnode), pool, n_layers, kv_dim, std::ref(state),
                      std::ref(all_reqs), std::ref(reqs_mu));

  std::thread decode_th(decode_thread_func, decoder, pool, max_batch, max_new_toks, temperature,
                        top_k, (int)user_inputs.size(), std::ref(state));

  // main 线程：encode 一条放一条，立刻送给 send 线程
  auto encode_start = std::chrono::steady_clock::now();
  for (int i = 0; i < (int)user_inputs.size(); i++) {
    ReqInfo info;
    info.req_id = i;
    info.input = user_inputs[i];
    std::string prompt = decoder->apply_chat_template_pub(user_inputs[i]);
    decoder->encode_pub(prompt, info.prompt_tokens);
    info.prompt_tokens.insert(info.prompt_tokens.begin(), decoder->get_tokenizer().bos_token_id);
    info.n_tokens = (int)info.prompt_tokens.size();

    // 放入 all_reqs（recv 线程查询 n_tokens 用）
    {
      std::lock_guard<std::mutex> lock(reqs_mu);
      all_reqs.push_back(info);
    }
    total_requests++;

    // 放入 encode_queue（send 线程消费）
    encode_queue.push(info);

    fprintf(stderr, "[main] encode req_id=%d n_tokens=%d\n", info.req_id, info.n_tokens);
  }
  auto encode_end = std::chrono::steady_clock::now();
  double encode_elapsed = std::chrono::duration<double>(encode_end - encode_start).count();
  std::cout << "encode elapsed: " << encode_elapsed << " seconds" << std::endl;

  // encode 完成，关闭队列，send 线程退出
  encode_queue.close();
  encode_done = true;
  fprintf(stderr, "[main] encode 完成，共 %d 个请求\n", (int)user_inputs.size());

  send_th.join();
  recv_th.join();
  decode_th.join();

  // 通知 P节点退出
  PrefillRequest shutdown_msg;
  shutdown_msg.req_id = -1;
  shutdown_msg.n_tokens = 0;
  dnode.send_request(shutdown_msg, {});

  delete decoder;
  return 0;
}
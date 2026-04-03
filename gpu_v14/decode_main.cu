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
// 通信线程：一次性发所有请求，逐个收 response + KV cache
// ============================================================================
void comm_thread_func(const std::vector<std::string>& user_inputs, GPUDecoder* decoder,
                      NcclComm& nccl, DNode& dnode, BlockPool* pool, int n_layers, int kv_dim,
                      SharedState& state) {
  const Config& cfg = decoder->get_config();

  // 预先 tokenize 所有请求
  struct ReqInfo {
    int req_id;
    int n_tokens;
    std::vector<int> prompt_tokens;
    std::string input;
  };
  std::vector<ReqInfo> reqs;

  for (int i = 0; i < (int)user_inputs.size(); i++) {
    ReqInfo info;
    info.req_id = i;
    info.input = user_inputs[i];
    std::string prompt = decoder->apply_chat_template_pub(user_inputs[i]);
    decoder->encode_pub(prompt, info.prompt_tokens);
    info.prompt_tokens.insert(info.prompt_tokens.begin(), decoder->get_tokenizer().bos_token_id);
    info.n_tokens = (int)info.prompt_tokens.size();
    reqs.push_back(std::move(info));
  }

  // 1. 一次性发送所有请求给 P节点
  fprintf(stderr, "[comm] 发送所有 %d 个请求\n", (int)reqs.size());
  for (auto& info : reqs) {
    PrefillRequest req_msg;
    req_msg.req_id = info.req_id;
    req_msg.n_tokens = info.n_tokens;
    dnode.send_request(req_msg, info.prompt_tokens);
    fprintf(stderr, "[comm] 已发送 req_id=%d n_tokens=%d\n", info.req_id, info.n_tokens);
  }
  fprintf(stderr, "[comm] 所有请求已发送\n");

  // 2. 逐个接收 PrefillResponse + KV cache
  for (auto& info : reqs) {
    if (state.shutdown) break;

    auto t_ttft_start = std::chrono::steady_clock::now();

    // 等待 PrefillResponse
    PrefillResponse resp;
    if (!dnode.recv_response(resp)) break;

    auto t_ttft_end = std::chrono::steady_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(t_ttft_end - t_ttft_start).count();
    fprintf(stderr, "[comm] TTFT req_id=%d: %.1fms first_token=%d\n", resp.req_id, ttft_ms,
            resp.first_token);

    // 分配 D节点 BlockPool slots
    auto* r = new Request();
    r->id = info.req_id;
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
        fprintf(stderr, "[comm] OOM\n");
        oom = true;
        break;
      }
      r->block_table.add_block(block_id);
    }
    if (oom) {
      delete r;
      break;
    }

    // 收集 slots
    std::vector<int> slots(info.n_tokens);
    for (int j = 0; j < info.n_tokens; j++) {
      int phy = r->block_table.physical_idx(j);
      int off = r->block_table.block_offset(j);
      slots[j] = phy * BLOCK_SIZE + off;
    }

    // 接收 KV cache
    auto t_kv_start = std::chrono::steady_clock::now();
    nccl_recv_kv(nccl, pool->get_k_cache(), pool->get_v_cache(), slots, info.n_tokens, n_layers,
                 kv_dim);
    auto t_kv_end = std::chrono::steady_clock::now();
    double kv_ms = std::chrono::duration<double, std::milli>(t_kv_end - t_kv_start).count();
    double kv_mb = (double)info.n_tokens * n_layers * kv_dim * 2 * sizeof(__half) / 1024 / 1024;
    fprintf(stderr, "[comm] KV cache req_id=%d: %.1fMB in %.1fms (%.1f MB/s)\n", info.req_id, kv_mb,
            kv_ms, kv_mb / kv_ms * 1000);

    // 放入 ready_queue，通知 decode 线程
    {
      std::lock_guard<std::mutex> lock(state.mu);
      state.ready_queue.push(r);
    }
    state.cv.notify_one();
  }

  fprintf(stderr, "[comm] 所有请求处理完毕\n");
}

// ============================================================================
// decode 线程：batch decode
// ============================================================================
void decode_thread_func(GPUDecoder* decoder, BlockPool* pool, int max_batch, int max_new_toks,
                        float temperature, int top_k, int total_requests, SharedState& state) {
  const Config& cfg = decoder->get_config();
  int max_blk = decoder->get_max_blocks_per_seq();
  std::mt19937 rng(42);

  std::vector<Request*> running;
  int completed = 0;
  int steps = 0;
  int total_batch_size = 0;
  double total_wait_s = 0.0;

  auto t_start = std::chrono::steady_clock::now();

  int step = 0;
  while (completed < total_requests) {
    step++;
    // 从 ready_queue 取就绪请求
    std::vector<int> changed;
    {
      std::unique_lock<std::mutex> lock(state.mu);
      auto t_wait_start = std::chrono::steady_clock::now();
      bool waited = false;

      if (running.empty() && state.ready_queue.empty()) {
        // running 为空时阻塞等待
        waited = true;
        state.cv.wait(lock, [&] { return !state.ready_queue.empty() || state.shutdown; });
      }

      if (waited) {
        double wait_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wait_start).count();
        std::cout << "----------------step: " << step << ", waited seconds:" << wait_s << std::endl;
        total_wait_s += wait_s;
      }

      // 非阻塞取出所有就绪请求
      while (!state.ready_queue.empty() && (int)running.size() < max_batch) {
        int idx = (int)running.size();
        running.push_back(state.ready_queue.front());
        state.ready_queue.pop();
        changed.push_back(idx);  // 新请求必须更新 block_table
        fprintf(stderr, "[decode] 加入请求 id=%d batch=%d\n", running.back()->id,
                (int)running.size());
      }
    }

    if (running.empty()) continue;

    // 组装 flat batch（纯 decode）
    std::vector<FlatRequest> flat_reqs;
    std::vector<int> flat_tokens, flat_positions, token_to_seq;
    std::vector<int> slot_mapping, last_tok_idx;
    std::vector<int> dec_flat, dec_pos, dec_seq;

    for (int i = 0; i < (int)running.size(); i++) {
      Request* r = running[i];
      if (r->finished) continue;

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
      if (r->block_table.num_blocks() != old_blocks) changed.push_back(i);
      if (r->finished) continue;

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

    if (flat_reqs.empty()) continue;

    // 去重 changed
    std::sort(changed.begin(), changed.end());
    changed.erase(std::unique(changed.begin(), changed.end()), changed.end());

    // 更新 block_table
    if (!changed.empty()) decoder->update_block_table_partial(running, changed, max_blk);

    std::cout << "step: " << step << ", requests: " << flat_reqs.size()
              << ", total_tokens:" << flat_tokens.size() << std::endl;

    // forward
    decoder->forward_flat(flat_reqs, flat_tokens, flat_positions, token_to_seq, slot_mapping,
                          last_tok_idx, dec_flat, dec_pos, dec_seq, (int)flat_tokens.size());

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
    for (auto it = running.begin(); it != running.end();) {
      Request* r = *it;
      if (r->finished) {
        printf("request id: %d\nprompt: %s\noutput: %s\n--------------------\n", r->id,
               r->input.c_str(), r->output.c_str());
        r->block_table.free_blocks([pool](int id) { pool->free(id); });
        delete r;
        it = running.erase(it);
        completed++;
        fprintf(stderr, "[decode] 完成 %d/%d\n", completed, total_requests);
      } else {
        ++it;
      }
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();
  double decode_elapsed = elapsed - total_wait_s;

  fprintf(stderr, "=== decode 统计 ===\n");
  fprintf(stderr, "总输出 token:  %d\n", (int)state.total_output_tokens);
  fprintf(stderr, "总时间:        %.2fs\n", elapsed);
  fprintf(stderr, "等待时间:      %.2fs\n", total_wait_s);
  fprintf(stderr, "实际decode:    %.2fs\n", decode_elapsed);
  fprintf(stderr, "decode 吞吐:   %.1f tokens/s\n", state.total_output_tokens / decode_elapsed);
  fprintf(stderr, "GPU 利用率:    %.1f%%\n", decode_elapsed / elapsed * 100);
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
  int max_total_tokens = max_batch + max_pps;
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
    // 发 shutdown 信号给 P节点
    PrefillRequest shutdown_msg;
    shutdown_msg.req_id = -1;
    shutdown_msg.n_tokens = 0;
    dnode.send_request(shutdown_msg, {});
    delete decoder;
    return 1;
  }

  fprintf(stderr, "%d requests\n", (int)user_inputs.size());

  SharedState state;
  float temperature = 0.0f;
  int top_k = 30;
  int max_new_toks = 256;

  // 启动 decode 线程
  std::thread decode_th(decode_thread_func, decoder, pool, max_batch, max_new_toks, temperature,
                        top_k, (int)user_inputs.size(), std::ref(state));

  // 主线程跑通信逻辑
  comm_thread_func(user_inputs, decoder, nccl, dnode, pool, n_layers, kv_dim, state);

  // 等待 decode 线程完成
  decode_th.join();

  // 发 shutdown 信号给 P节点
  PrefillRequest shutdown_msg;
  shutdown_msg.req_id = -1;
  shutdown_msg.n_tokens = 0;
  dnode.send_request(shutdown_msg, {});

  delete decoder;
  return 0;
}
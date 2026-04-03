#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

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
  std::queue<Request*> ready_queue;  // KV cache 已就绪，等待加入 batch
  std::atomic<bool> shutdown{false};
  std::atomic<int> total_output_tokens{0};
};

// ============================================================================
// TCP 连接
// ============================================================================
static int tcp_connect(const char* ip, int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    fprintf(stderr, "socket failed: %s\n", strerror(errno));
    return -1;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, ip, &addr.sin_addr) <= 0) {
    fprintf(stderr, "inet_pton failed\n");
    close(fd);
    return -1;
  }

  fprintf(stderr, "连接 P节点 %s:%d ...\n", ip, port);
  if (connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
    fprintf(stderr, "connect failed: %s\n", strerror(errno));
    close(fd);
    return -1;
  }
  fprintf(stderr, "已连接 P节点\n");
  return fd;
}

// ============================================================================
// 通信线程：负责和 P节点交互，把就绪请求放入 ready_queue
// ============================================================================
void comm_thread_func(const std::vector<std::string>& user_inputs, GPUDecoder* decoder,
                      NcclComm& nccl, int tcp_fd, BlockPool* pool, int n_layers, int kv_dim,
                      SharedState& state) {
  const Config& cfg = decoder->get_config();
  int req_id = 0;

  for (auto& input : user_inputs) {
    if (state.shutdown) break;

    // 1. tokenize
    std::string prompt = decoder->apply_chat_template_pub(input);
    std::vector<int> prompt_tokens;
    decoder->encode_pub(prompt, prompt_tokens);
    prompt_tokens.insert(prompt_tokens.begin(), decoder->get_tokenizer().bos_token_id);
    int n_tokens = (int)prompt_tokens.size();

    fprintf(stderr, "[comm] req_id=%d n_tokens=%d\n", req_id, n_tokens);

    // 2. 发送 PrefillRequest
    auto t_ttft_start = std::chrono::steady_clock::now();

    int body_size = sizeof(PrefillRequest) + n_tokens * sizeof(int);
    std::vector<char> body(body_size);
    PrefillRequest* req_msg = (PrefillRequest*)body.data();
    req_msg->req_id = req_id;
    req_msg->n_tokens = n_tokens;
    memcpy(body.data() + sizeof(PrefillRequest), prompt_tokens.data(), n_tokens * sizeof(int));
    tcp_send_msg(tcp_fd, MsgType::PREFILL_REQUEST, body.data(), body_size);

    // 3. 等待 PrefillResponse
    MsgHeader resp_header;
    if (!tcp_recv_header(tcp_fd, resp_header)) break;
    if (resp_header.type != MsgType::PREFILL_RESPONSE) break;
    PrefillResponse resp;
    if (!tcp_recv(tcp_fd, &resp, sizeof(resp))) break;

    auto t_ttft_end = std::chrono::steady_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(t_ttft_end - t_ttft_start).count();
    fprintf(stderr, "[comm] TTFT: %.1fms req_id=%d first_token=%d\n", ttft_ms, resp.req_id,
            resp.first_token);

    // 4. 分配 D节点 BlockPool slots
    auto* r = new Request();
    r->id = req_id;
    r->pos = n_tokens;
    r->prefill_done = true;
    r->finished = false;
    r->cur_token = resp.first_token;
    r->input = input;

    fprintf(stderr, "[comm] resp.first_token=%d r->cur_token=%d\n", resp.first_token, r->cur_token);

    int need_blocks = (n_tokens - 1) / BLOCK_SIZE + 1;
    bool oom = false;
    for (int i = 0; i < need_blocks; i++) {
      int block_id = pool->allocate();
      if (block_id < 0) {
        fprintf(stderr, "[comm] OOM: no free blocks\n");
        oom = true;
        break;
      }
      r->block_table.add_block(block_id);
    }
    if (oom) {
      delete r;
      break;
    }

    // 5. 收集 slots
    std::vector<int> slots(n_tokens);
    for (int j = 0; j < n_tokens; j++) {
      int phy = r->block_table.physical_idx(j);
      int off = r->block_table.block_offset(j);
      slots[j] = phy * BLOCK_SIZE + off;
    }

    // 6. 通过 NCCL 接收 KV cache
    auto t_kv_start = std::chrono::steady_clock::now();
    nccl_recv_kv(nccl, pool->get_k_cache(), pool->get_v_cache(), slots, n_tokens, n_layers, kv_dim);
    auto t_kv_end = std::chrono::steady_clock::now();
    double kv_ms = std::chrono::duration<double, std::milli>(t_kv_end - t_kv_start).count();
    double kv_mb = (double)n_tokens * n_layers * kv_dim * 2 * sizeof(__half) / 1024 / 1024;
    fprintf(stderr, "[comm] KV cache: %.1fMB in %.1fms (%.1f MB/s)\n", kv_mb, kv_ms,
            kv_mb / kv_ms * 1000);

    // 7. 放入 ready_queue，通知 decode 线程
    {
      std::lock_guard<std::mutex> lock(state.mu);
      state.ready_queue.push(r);
    }
    state.cv.notify_one();

    req_id++;
  }

  // 所有请求处理完，通知 decode 线程准备退出
  fprintf(stderr, "[comm] 所有请求已发送，等待 decode 完成\n");
}

// ============================================================================
// decode 线程：batch decode
// ============================================================================
void decode_thread_func(GPUDecoder* decoder, BlockPool* pool, int max_batch, int max_new_toks,
                        float temperature, int top_k, int total_requests, SharedState& state) {
  const Config& cfg = decoder->get_config();
  int max_blk = decoder->get_max_blocks_per_seq();
  std::mt19937 rng(42);

  // running batch：正在 decode 的请求
  std::vector<Request*> running;
  int completed = 0;

  auto t_decode_start = std::chrono::steady_clock::now();
  double total_wait_seconds = 0;

  int step = 0;
  while (completed < total_requests) {
    step++;
    // 从 ready_queue 取就绪的请求加入 running batch
    std::vector<int> changed;
    {
      std::unique_lock<std::mutex> lock(state.mu);
      // 如果 running 为空，等待新请求；否则非阻塞取
      auto from = std::chrono::steady_clock::now();
      bool waited = false;
      if (running.empty() && state.ready_queue.empty()) {
        waited = true;
        state.cv.wait(lock, [&] { return !state.ready_queue.empty() || state.shutdown; });
      }

      if (waited) {
        auto to = std::chrono::steady_clock::now();
        double waited_seconds = std::chrono::duration<double>(to - from).count();
        std::cout << "----------------step: " << step << ", waited seconds:" << waited_seconds << std::endl;
        total_wait_seconds += waited_seconds;
      }

      // 取出所有就绪请求
      while (!state.ready_queue.empty() && (int)running.size() < max_batch) {
        Request* req = state.ready_queue.front();
        state.ready_queue.pop();
        int idx = (int)running.size();
        running.push_back(req);
        changed.push_back(idx);
        fprintf(stderr, "[decode] 加入请求 id=%d idx=%d\n", req->id, idx);
      }
    }

    if (running.empty()) continue;

    // 组装 flat batch（纯 decode）
    std::vector<FlatRequest> flat_reqs;
    std::vector<int> flat_tokens, flat_positions, token_to_seq;
    std::vector<int> slot_mapping, last_tok_idx;
    std::vector<int> dec_flat, dec_pos, dec_seq;

    // 更新 block_table，确保每个请求有足够的 block
    for (int i = 0; i < (int)running.size(); i++) {
      Request* r = running[i];

      // 确保有足够的 block
      int need = r->pos / BLOCK_SIZE + 1;
      int old_blocks = r->block_table.num_blocks();
      while (r->block_table.num_blocks() < need) {
        int block_id = pool->allocate();
        if (block_id < 0) {
          fprintf(stderr, "[decode] OOM during decode\n");
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

    // 更新 block_table
    // 把 running 转成 vector<Request*>，changed 是 running 里的下标
    // 去重
    std::sort(changed.begin(), changed.end());
    changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
    if (!changed.empty()) {
      // 临时构造 running_tmp，和 running 一一对应
      decoder->update_block_table_partial(running, changed, max_blk);
    }

    std::cout << "step: " << step << ", requests: " << flat_reqs.size() << ", total_tokens:" << flat_tokens.size() << std::endl;
    // forward
    decoder->forward_flat(flat_reqs, flat_tokens, flat_positions, token_to_seq, slot_mapping,
                          last_tok_idx, dec_flat, dec_pos, dec_seq, (int)flat_tokens.size());

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

  auto t_decode_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t_decode_end - t_decode_start).count();
  elapsed -= total_wait_seconds;
  fprintf(stderr, "decode: %d tokens in %.2fs (%.1f tokens/s)\n", (int)state.total_output_tokens,
          elapsed, state.total_output_tokens / elapsed);

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

  int tcp_fd = tcp_connect(prefill_node_ip, PD_TCP_PORT);
  if (tcp_fd < 0) return 1;

  NcclComm nccl;
  if (!nccl_init_decode(nccl, tcp_fd)) return 1;

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
    tcp_send_msg(tcp_fd, MsgType::SHUTDOWN, nullptr, 0);
    delete decoder;
    close(tcp_fd);
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
  comm_thread_func(user_inputs, decoder, nccl, tcp_fd, pool, n_layers, kv_dim, state);

  // 等待 decode 线程完成
  decode_th.join();

  // 通知 P节点退出
  tcp_send_msg(tcp_fd, MsgType::SHUTDOWN, nullptr, 0);
  close(tcp_fd);
  delete decoder;
  return 0;
}
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "gpu_decoder.h"
#include "pd_comm.h"

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

  fprintf(stdout, "已连接 P节点\n");
  return fd;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <model_file> <prefill_node_ip> [max_batch]\n", argv[0]);
    return 1;
  }

  const char* model_file = argv[1];
  const char* prefill_node_ip = argv[2];
  int max_batch = (argc >= 4) ? atoi(argv[3]) : 8;

  // D节点只做 decode
  int max_pps = 512;
  int max_total_tokens = max_batch + max_pps;
  Decoder* decoder = new GPUDecoder(model_file, max_batch, 4096, max_pps, max_total_tokens);
  fprintf(stderr, "D节点模型加载完成\n");

  int tcp_fd = tcp_connect(prefill_node_ip, PD_TCP_PORT);
  if (tcp_fd < 0) {
    return 1;
  }

  NcclComm nccl;
  if (!nccl_init_decode(nccl, tcp_fd)) {
    return 1;
  }

  const Config& cfg = decoder->get_config();
  int n_layers = cfg.n_layers;
  int head_dim = cfg.dim / cfg.n_heads;
  int kv_dim = cfg.n_kv_heads * head_dim;
  BlockPool* pool = decoder->get_block_pool();

  std::mt19937 rng(42);
  float temperature = 0.0f;
  int top_k = 30;
  int max_new_toks = 256;
  int max_blk = decoder->get_max_blocks_per_seq();

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
    tcp_send_msg(tcp_fd, MsgType::SHUTDOWN, nullptr, 0);
    delete decoder;
    close(tcp_fd);
    return 1;
  }

  auto t_start = std::chrono::steady_clock::now();
  int total_output_tokens = 0;
  int req_id = 0;

  for (auto& input : user_inputs) {
    // ----------------------------------------------------------------
    // 1. tokenize
    // ----------------------------------------------------------------
    std::string prompt = decoder->apply_chat_template_pub(input);
    std::vector<int> prompt_tokens;
    decoder->encode_pub(prompt, prompt_tokens);
    prompt_tokens.insert(prompt_tokens.begin(), decoder->get_tokenizer().bos_token_id);
    int n_tokens = (int)prompt_tokens.size();

    fprintf(stderr, "req_id=%d n_tokens=%d\n", req_id, n_tokens);

    // ----------------------------------------------------------------
    // 2. 发送 PrefillRequest 给 P节点
    // ----------------------------------------------------------------
    int body_size = sizeof(PrefillRequest) + n_tokens * sizeof(int);
    std::vector<char> body(body_size);
    PrefillRequest* req_msg = (PrefillRequest*)body.data();
    req_msg->req_id = req_id;
    req_msg->n_tokens = n_tokens;
    memcpy(body.data() + sizeof(PrefillRequest), prompt_tokens.data(), n_tokens * sizeof(int));
    tcp_send_msg(tcp_fd, MsgType::PREFILL_REQUEST, body.data(), body_size);
    fprintf(stderr, "已发送 PrefillRequest\n");

    // ----------------------------------------------------------------
    // 3. 等待 PrefillResponse
    // ----------------------------------------------------------------
    MsgHeader resp_header;
    if (!tcp_recv_header(tcp_fd, resp_header)) break;
    if (resp_header.type != MsgType::PREFILL_RESPONSE) {
      fprintf(stderr, "期望 PREFILL_RESPONSE，收到 %d\n", (int)resp_header.type);
      break;
    }
    PrefillResponse resp;
    if (!tcp_recv(tcp_fd, &resp, sizeof(resp))) break;
    fprintf(stderr, "收到 PrefillResponse req_id=%d\n", resp.req_id);

    // ----------------------------------------------------------------
    // 4. 在 D节点 BlockPool 里分配 slots
    // ----------------------------------------------------------------
    auto* r = new Request();
    r->id = req_id;
    r->pos = n_tokens;  // decode 从 n_tokens 开始
    r->prefill_done = true;
    r->finished = false;
    r->cur_token = resp.first_token;

    int need_blocks = (n_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < need_blocks; i++) {
      int block_id = pool->allocate();
      if (block_id < 0) {
        fprintf(stderr, "OOM: no free blocks\n");
        delete r;
        goto done;
      }
      r->block_table.add_block(block_id);
    }

    {
      // 收集 slots
      std::vector<int> slots(n_tokens);
      for (int j = 0; j < n_tokens; j++) {
        int phy = r->block_table.physical_idx(j);
        int off = r->block_table.block_offset(j);
        slots[j] = phy * BLOCK_SIZE + off;
      }

      // ----------------------------------------------------------------
      // 5. 通过 NCCL 接收 KV cache，写入本地 BlockPool
      // ----------------------------------------------------------------
      fprintf(stderr, "开始接收 KV cache...\n");
      auto t0 = std::chrono::steady_clock::now();

      nccl_recv_kv(nccl, pool->get_k_cache(), pool->get_v_cache(), slots, n_tokens, n_layers,
                   kv_dim);

      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      double mb = (double)n_tokens * n_layers * kv_dim * 2 * sizeof(__half) / 1024 / 1024;
      fprintf(stderr, "KV cache 接收完成：%.1fMB in %.1fms (%.1f MB/s)\n", mb, ms, mb / ms * 1000);
    }

    // 6.省略，不再重新采样

    // ----------------------------------------------------------------
    // 7. decode 循环
    // ----------------------------------------------------------------
    {
      std::string output;

      while (!r->finished) {
        // 确保有足够的 block
        {
          int need = (r->pos + BLOCK_SIZE - 1) / BLOCK_SIZE;
          while (r->block_table.num_blocks() < need) {
            int block_id = pool->allocate();
            if (block_id < 0) {
              fprintf(stderr, "OOM during decode\n");
              r->finished = true;
              break;
            }
            r->block_table.add_block(block_id);
          }
          if (r->finished) break;
        }

        // 组装 flat batch（纯 decode，1个 token）
        FlatRequest fr;
        fr.req_idx = 0;
        fr.flat_offset = 0;
        fr.n_tokens = 1;
        fr.start_pos = r->pos;
        fr.is_prefill = false;

        int phy = r->block_table.physical_idx(r->pos);
        int off = r->block_table.block_offset(r->pos);

        std::vector<FlatRequest> flat_reqs = {fr};
        std::vector<int> flat_tokens = {r->cur_token};
        std::vector<int> flat_positions = {r->pos};
        std::vector<int> token_to_seq = {0};
        std::vector<int> slot_mapping = {phy * BLOCK_SIZE + off};
        std::vector<int> last_tok_idx = {0};
        std::vector<int> dec_flat = {0};
        std::vector<int> dec_pos = {r->pos};
        std::vector<int> dec_seq = {0};

        std::vector<Request*> running_tmp(1, r);
        std::vector<int> changed = {0};
        decoder->update_block_table_partial(running_tmp, changed, max_blk);

        decoder->forward_flat(flat_reqs, flat_tokens, flat_positions, token_to_seq, slot_mapping,
                              last_tok_idx, dec_flat, dec_pos, dec_seq, 1);

        // 采样
        float* logits = decoder->get_logits_batch(0);
        int next_token = sample_topk(logits, cfg.vocab_size, top_k, temperature, rng);

        // 输出当前 token
        const char* piece = decode(decoder->get_tokenizer(), r->cur_token);
        if (piece) output += piece;

        r->pos++;
        r->cur_token = next_token;
        r->n_generated++;
        total_output_tokens++;

        if (next_token == decoder->get_tokenizer().eos_token_id ||
            next_token == decoder->get_tokenizer().bos_token_id ||
            decoder->get_tokenizer().vocab[next_token] == "<|im_end|>" ||
            r->n_generated >= max_new_toks || r->pos >= cfg.seq_len) {
          r->finished = true;
        }
      }

      printf("request id: %d\nprompt: %s\noutput: %s\n--------------------\n", req_id,
             input.c_str(), output.c_str());
    }

  cleanup:
    r->block_table.free_blocks([pool](int id) { pool->free(id); });
    delete r;
    req_id++;
  }

done:
  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();
  fprintf(stderr, "total: %d tokens in %.2fs (%.1f tokens/s)\n", total_output_tokens, elapsed,
          total_output_tokens / elapsed);

  tcp_send_msg(tcp_fd, MsgType::SHUTDOWN, nullptr, 0);
  close(tcp_fd);
  delete decoder;
  return 0;
}
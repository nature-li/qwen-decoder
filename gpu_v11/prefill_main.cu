#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gpu_decoder.h"
#include "pd_comm.h"

static int tcp_listen_and_accept() {
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) {
    fprintf(stderr, "socket failed: %s\n", strerror(errno));
    return -1;
  }

  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(PD_TCP_PORT);

  if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
    fprintf(stderr, "bind failed: %s\n", strerror(errno));
    close(server_fd);
    return -1;
  }

  if (listen(server_fd, 1) < 0) {
    fprintf(stderr, "listen failed: %s\n", strerror(errno));
    close(server_fd);
    return -1;
  }

  fprintf(stdout, "P节点监听 port %d，等待 D节点连接...\n", PD_TCP_PORT);

  sockaddr_in client_addr{};
  socklen_t client_len = sizeof(client_addr);
  int client_fd = accept(server_fd, (sockaddr*)&client_addr, &client_len);
  if (client_fd < 0) {
    fprintf(stderr, "accept failed: %s\n", strerror(errno));
    close(server_fd);
    return -1;
  }

  char ip[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &client_addr.sin_addr, ip, sizeof(ip));
  fprintf(stderr, "D节点已连接：%s\n", ip);

  close(server_fd);
  return client_fd;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model_file>\n", argv[0]);
    return 1;
  }

  // P节点只做 prefill，max_batch=1
  Decoder* decoder = new GPUDecoder(argv[1], 1, 4096, 4096, 4096);
  fprintf(stderr, "P节点模型加载完成\n");

  int tcp_fd = tcp_listen_and_accept();
  if (tcp_fd < 0) {
    return 1;
  }

  NcclComm nccl;
  if (!nccl_init_prefill(nccl, tcp_fd)) {
    return 1;
  }

  const Config& cfg = decoder->get_config();
  int n_layers = cfg.n_layers;
  int head_dim = cfg.dim / cfg.n_heads;
  int kv_dim = cfg.n_kv_heads * head_dim;
  BlockPool* pool = decoder->get_block_pool();

  fprintf(stderr, "P节点就绪，等待 prefill 请求...\n");

  while (true) {
    // 1. 接收消息头
    MsgHeader header;
    if (!tcp_recv_header(tcp_fd, header)) {
      fprintf(stderr, "D节点断开连接\n");
      break;
    }

    if (header.type == MsgType::SHUTDOWN) {
      fprintf(stderr, "收到 SHUTDOWN，退出\n");
      break;
    }

    if (header.type != MsgType::PREFILL_REQUEST) {
      fprintf(stderr, "未知消息类型: %d\n", (int)header.type);
      break;
    }

    // 2. 接收 PrefillRequest + token ids
    std::vector<char> body(header.body_size);
    if (!tcp_recv(tcp_fd, body.data(), header.body_size)) {
      break;
    }

    PrefillRequest* req_msg = (PrefillRequest*)body.data();
    int req_id = req_msg->req_id;
    int n_tokens = req_msg->n_tokens;
    int* token_ids = (int*)(body.data() + sizeof(PrefillRequest));

    fprintf(stdout, "收到 prefill 请求 req_id=%d n_tokens=%d\n", req_id, n_tokens);

    // 3. 构造 Request，分配物理块
    auto* r = new Request();
    r->id = req_id;
    r->pos = 0;
    r->prefill_done = false;
    r->finished = false;
    r->prompt_tokens.assign(token_ids, token_ids + n_tokens);

    int need_blocks = (n_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < need_blocks; i++) {
      int block_id = pool->allocate();
      if (block_id < 0) {
        fprintf(stderr, "OOM: no free blocks\n");
        delete r;
        goto shutdown;
      }
      r->block_table.add_block(block_id);
    }

    {
      // 4. 组装 flat batch（纯 prefill，1个请求）
      FlatRequest fr;
      fr.req_idx = 0;
      fr.flat_offset = 0;
      fr.n_tokens = n_tokens;
      fr.start_pos = 0;
      fr.is_prefill = true;

      std::vector<FlatRequest> flat_reqs = {fr};
      std::vector<int> flat_tokens(token_ids, token_ids + n_tokens);
      std::vector<int> flat_positions(n_tokens);
      std::vector<int> token_to_seq(n_tokens, 0);
      std::vector<int> slot_mapping(n_tokens);
      std::vector<int> last_tok_idx = {n_tokens - 1};
      std::vector<int> dec_flat, dec_pos, dec_seq;  // 空，没有 decode token

      for (int j = 0; j < n_tokens; j++) {
        flat_positions[j] = j;
        int phy = r->block_table.physical_idx(j);
        int off = r->block_table.block_offset(j);
        slot_mapping[j] = phy * BLOCK_SIZE + off;
      }

      // 5. 更新 block_table，forward
      std::vector<Request*> running_tmp(1, r);
      std::vector<int> changed = {0};
      decoder->update_block_table_partial(running_tmp, changed, decoder->get_max_blocks_per_seq());

      decoder->forward_flat(flat_reqs, flat_tokens, flat_positions, token_to_seq, slot_mapping,
                            last_tok_idx, dec_flat, dec_pos, dec_seq, n_tokens);

      fprintf(stderr, "prefill 完成 req_id=%d\n", req_id);

      // 6. 收集 slots
      std::vector<int> slots(n_tokens);
      for (int j = 0; j < n_tokens; j++) {
        int phy = r->block_table.physical_idx(j);
        int off = r->block_table.block_offset(j);
        slots[j] = phy * BLOCK_SIZE + off;
      }

      // 7. 通知 D节点 prefill 完成
      PrefillResponse resp;
      resp.req_id = req_id;
      resp.n_tokens = n_tokens;
      resp.n_layers = n_layers;
      resp.kv_dim = kv_dim;
      tcp_send_msg(tcp_fd, MsgType::PREFILL_RESPONSE, &resp, sizeof(resp));

      // 8. 通过 NCCL 传输 KV cache
      fprintf(stderr, "开始传输 KV cache n_tokens=%d n_layers=%d kv_dim=%d\n", n_tokens, n_layers,
              kv_dim);
      auto t0 = std::chrono::steady_clock::now();

      nccl_send_kv(nccl, pool->get_k_cache(), pool->get_v_cache(), slots, n_tokens, n_layers,
                   kv_dim);

      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      double mb = (double)n_tokens * n_layers * kv_dim * 2 * sizeof(__half) / 1024 / 1024;
      fprintf(stderr, "KV cache 传输完成：%.1fMB in %.1fms (%.1f MB/s)\n", mb, ms, mb / ms * 1000);
    }

    // 9. 释放物理块，等待下一个请求
    r->block_table.free_blocks([pool](int id) { pool->free(id); });
    delete r;
    continue;

  shutdown:
    break;
  }

  close(tcp_fd);
  delete decoder;
  return 0;
}
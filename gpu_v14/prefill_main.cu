#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "gpu_decoder.h"
#include "pd_comm.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model_file>\n", argv[0]);
    return 1;
  }

  // P节点只做 prefill，max_batch=1
  auto* decoder = new GPUDecoder(argv[1], 1, 4096, 4096, 4096);
  fprintf(stderr, "P节点模型加载完成\n");

  // ZMQ 初始化
  PNode pnode;
  if (!pnode.init()) return 1;

  // NCCL 初始化
  NcclComm nccl;
  if (!nccl_init_prefill(nccl, pnode)) return 1;

  const Config& cfg = decoder->get_config();
  int n_layers = cfg.n_layers;
  int head_dim = cfg.dim / cfg.n_heads;
  int kv_dim   = cfg.n_kv_heads * head_dim;
  BlockPool* pool = decoder->get_block_pool();

  std::mt19937 rng(42);
  fprintf(stderr, "P节点就绪，等待 prefill 请求...\n");

  // ----------------------------------------------------------------
  // 主循环：收到请求立刻 prefill，完成后立刻发回
  // ----------------------------------------------------------------
  while (true) {
    // 1. 接收 PrefillRequest
    PrefillRequest req_msg;
    std::vector<int> token_ids;
    if (!pnode.recv_request(req_msg, token_ids)) {
      fprintf(stderr, "recv_request 失败，退出\n");
      break;
    }

    int req_id   = req_msg.req_id;
    int n_tokens = req_msg.n_tokens;

    // shutdown 信号：req_id = -1
    if (req_id == -1) {
      fprintf(stderr, "收到 SHUTDOWN，退出\n");
      break;
    }

    fprintf(stderr, "[P] prefill req_id=%d n_tokens=%d\n", req_id, n_tokens);

    // 2. 构造 Request，分配物理块
    auto* r = new Request();
    r->id           = req_id;
    r->pos          = 0;
    r->prefill_done = false;
    r->finished     = false;
    r->prompt_tokens.assign(token_ids.begin(), token_ids.end());

    int need_blocks = (n_tokens - 1) / BLOCK_SIZE + 1;
    bool oom = false;
    for (int i = 0; i < need_blocks; i++) {
      int block_id = pool->allocate();
      if (block_id < 0) {
        fprintf(stderr, "OOM: no free blocks\n");
        oom = true;
        break;
      }
      r->block_table.add_block(block_id);
    }
    if (oom) { delete r; break; }

    // 3. 组装 flat batch
    FlatRequest fr;
    fr.req_idx     = 0;
    fr.flat_offset = 0;
    fr.n_tokens    = n_tokens;
    fr.start_pos   = 0;
    fr.is_prefill  = true;

    std::vector<FlatRequest> flat_reqs   = {fr};
    std::vector<int> flat_tokens(token_ids.begin(), token_ids.end());
    std::vector<int> flat_positions(n_tokens);
    std::vector<int> token_to_seq(n_tokens, 0);
    std::vector<int> slot_mapping(n_tokens);
    std::vector<int> last_tok_idx = {n_tokens - 1};
    std::vector<int> dec_flat, dec_pos, dec_seq;

    for (int j = 0; j < n_tokens; j++) {
      flat_positions[j] = j;
      int phy          = r->block_table.physical_idx(j);
      int off          = r->block_table.block_offset(j);
      slot_mapping[j]  = phy * BLOCK_SIZE + off;
    }

    std::vector<Request*> running_tmp(1, r);
    std::vector<int> changed = {0};
    decoder->update_block_table_partial(running_tmp, changed,
                                         decoder->get_max_blocks_per_seq());

    // 4. prefill
    auto t0 = std::chrono::steady_clock::now();
    decoder->forward_flat(flat_reqs, flat_tokens, flat_positions,
                           token_to_seq, slot_mapping, last_tok_idx,
                           dec_flat, dec_pos, dec_seq, n_tokens);
    auto t1 = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "[P] prefill 完成 req_id=%d %.1fms (%.1f tokens/s)\n",
            req_id, prefill_ms, n_tokens / prefill_ms * 1000);

    // 5. 采样 first_token
    float* logits   = decoder->get_logits_batch(0);
    int first_token = sample_topk(logits, cfg.vocab_size, 30, 0.0f, rng);

    // 6. 收集 slots
    std::vector<int> slots(n_tokens);
    for (int j = 0; j < n_tokens; j++) {
      int phy  = r->block_table.physical_idx(j);
      int off  = r->block_table.block_offset(j);
      slots[j] = phy * BLOCK_SIZE + off;
    }

    // 7. 立刻发回 PrefillResponse
    PrefillResponse resp;
    resp.req_id      = req_id;
    resp.n_tokens    = n_tokens;
    resp.n_layers    = n_layers;
    resp.kv_dim      = kv_dim;
    resp.first_token = first_token;
    pnode.send_response(resp);

    // 8. 立刻发 KV cache
    auto t2 = std::chrono::steady_clock::now();
    nccl_send_kv(nccl, pool->get_k_cache(), pool->get_v_cache(),
                  slots, n_tokens, n_layers, kv_dim);
    auto t3 = std::chrono::steady_clock::now();
    double kv_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double kv_mb = (double)n_tokens * n_layers * kv_dim * 2 * sizeof(__half) / 1024 / 1024;
    fprintf(stderr, "[P] KV cache 传输完成：%.1fMB in %.1fms (%.1f MB/s)\n",
            kv_mb, kv_ms, kv_mb / kv_ms * 1000);

    // 9. 释放物理块
    r->block_table.free_blocks([pool](int id) { pool->free(id); });
    delete r;
  }

  delete decoder;
  return 0;
}
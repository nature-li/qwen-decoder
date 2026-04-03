#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "gpu_decoder.h"
#include "pd_comm.h"

// 线程安全队列
template <typename T>
struct SafeQueue {
  std::mutex mu;
  std::condition_variable cv;
  std::queue<T> q;
  bool closed = false;

  void push(T val) {
    std::lock_guard<std::mutex> lock(mu);
    q.push(std::move(val));
    cv.notify_one();
  }

  // 取出一个，返回 false 表示队列已关闭且为空
  bool pop(T& val) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return !q.empty() || closed; });
    if (q.empty()) return false;
    val = std::move(q.front());
    q.pop();
    return true;
  }

  bool try_pop(T& val) {
    std::lock_guard<std::mutex> lock(mu);
    if (q.empty()) return false;
    val = std::move(q.front());
    q.pop();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu);
    closed = true;
    cv.notify_all();
  }
};

// P节点请求
struct PrefillTask {
  int req_id;
  int n_tokens;
  std::vector<int> token_ids;
};

// P节点结果
struct PrefillResult {
  int req_id;
  int n_tokens;
  int first_token;
  std::vector<int> slots;  // KV cache 在 BlockPool 里的位置
  Request* r;              // 用完后释放物理块
};

// 线程1：接收 D节点请求
void p_recv_thread(PNode& pnode, SafeQueue<PrefillTask>& task_queue) {
  for (;;) {
    PrefillRequest req_msg;
    std::vector<int> token_ids;
    if (!pnode.recv_request(req_msg, token_ids)) break;

    // shutdown 信号
    if (req_msg.req_id == -1) {
      fprintf(stderr, "[P.recv] 收到 SHUTDOWN\n");
      break;
    }

    PrefillTask task;
    task.req_id = req_msg.req_id;
    task.n_tokens = req_msg.n_tokens;
    task.token_ids = std::move(token_ids);
    task_queue.push(std::move(task));
    fprintf(stderr, "[P.recv] req_id=%d n_tokens=%d\n", task.req_id, task.n_tokens);
  }
  task_queue.close();
  fprintf(stderr, "[P.recv] 线程退出\n");
}

// 线程2：prefill
void p_prefill_thread(GPUDecoder* decoder, BlockPool* pool, SafeQueue<PrefillTask>& task_queue,
                      SafeQueue<PrefillResult>& result_queue, const Config& cfg,
                      std::mt19937& rng) {
  int max_batch = decoder->get_max_batch();
  int max_pps = decoder->get_max_prefill_tokens_per_step();
  int max_blk = decoder->get_max_blocks_per_seq();

  // running：固定大小，nullptr 表示空槽
  std::vector<Request*> running(max_batch, nullptr);
  std::vector<PrefillTask> running_tasks(max_batch);
  std::vector<int> prefill_offset(max_batch, 0);

  int completed = 0;

  while (true) {
    std::vector<int> changed;

    // ----------------------------------------------------------------
    // 1. 填空槽
    // ----------------------------------------------------------------
    // 先非阻塞填所有空槽
    for (int i = 0; i < max_batch; i++) {
      if (running[i] != nullptr) continue;
      PrefillTask task;
      if (!task_queue.try_pop(task)) continue;
      // 分配物理块
      auto* r = new Request();
      r->id = task.req_id;
      r->pos = 0;
      int need_blocks = (task.n_tokens - 1) / BLOCK_SIZE + 1;
      bool oom = false;
      for (int b = 0; b < need_blocks; b++) {
        int block_id = pool->allocate();
        if (block_id < 0) {
          oom = true;
          break;
        }
        r->block_table.add_block(block_id);
      }
      if (oom) {
        delete r;
        goto done;
      }
      running[i] = r;
      running_tasks[i] = std::move(task);
      prefill_offset[i] = 0;
      changed.push_back(i);
      // fprintf(stderr, "[P.prefill] 加入 req_id=%d slot=%d n_tokens=%d\n", r->id, i,
      //         running_tasks[i].n_tokens);
    }

    // running 全空时阻塞等一个
    {
      bool any = false;
      for (int i = 0; i < max_batch; i++)
        if (running[i]) {
          any = true;
          break;
        }

      if (!any) {
        // 找第一个空槽
        int slot = -1;
        for (int i = 0; i < max_batch; i++) {
          if (running[i] == nullptr) {
            slot = i;
            break;
          }
        }
        if (slot == -1) {
          continue;  // 没有空槽，不应该发生
        }

        PrefillTask task;
        if (!task_queue.pop(task)) goto done;  // 队列关闭，退出

        auto* r = new Request();
        r->id = task.req_id;
        r->pos = 0;
        int need_blocks = (task.n_tokens - 1) / BLOCK_SIZE + 1;
        bool oom = false;
        for (int b = 0; b < need_blocks; b++) {
          int block_id = pool->allocate();
          if (block_id < 0) {
            oom = true;
            break;
          }
          r->block_table.add_block(block_id);
        }
        if (oom) {
          delete r;
          goto done;
        }
        running[slot] = r;
        running_tasks[slot] = std::move(task);
        prefill_offset[slot] = 0;
        changed.push_back(slot);
        fprintf(stderr, "[P.prefill] 阻塞等到 req_id=%d slot=%d\n", r->id, slot);
      }
    }

    // ----------------------------------------------------------------
    // 2. 组装 flat batch（chunked prefill）
    // ----------------------------------------------------------------
    {
      std::vector<FlatRequest> flat_reqs;
      std::vector<int> flat_tokens, flat_positions, token_to_seq, slot_mapping;
      std::vector<int> last_tok_idx;
      std::vector<int> dec_flat, dec_pos, dec_seq;

      int flat_offset = 0;
      int prefill_budget = max_pps;

      for (int i = 0; i < max_batch; i++) {
        Request* r = running[i];
        if (r == nullptr) continue;

        auto& tk = running_tasks[i];
        int& poff = prefill_offset[i];
        int remaining = tk.n_tokens - poff;
        int n_tok = std::min(remaining, prefill_budget);
        if (n_tok <= 0) continue;

        // ensure_pages
        int old_blocks = r->block_table.num_blocks();
        int need = (r->pos + n_tok - 1) / BLOCK_SIZE + 1;
        while (r->block_table.num_blocks() < need) {
          int block_id = pool->allocate();
          if (block_id < 0) goto done;
          r->block_table.add_block(block_id);
        }
        if (r->block_table.num_blocks() != old_blocks) changed.push_back(i);

        FlatRequest fr;
        fr.req_idx = i;
        fr.flat_offset = flat_offset;
        fr.n_tokens = n_tok;
        fr.start_pos = r->pos;
        fr.is_prefill = true;
        flat_reqs.push_back(fr);

        for (int j = 0; j < n_tok; j++) {
          int tok = tk.token_ids[poff + j];
          int pos = r->pos + j;
          int phy = r->block_table.physical_idx(pos);
          int off = r->block_table.block_offset(pos);
          flat_tokens.push_back(tok);
          flat_positions.push_back(pos);
          token_to_seq.push_back(i);
          slot_mapping.push_back(phy * BLOCK_SIZE + off);
        }
        last_tok_idx.push_back(flat_offset + n_tok - 1);
        flat_offset += n_tok;
        prefill_budget -= n_tok;
      }

      if (flat_reqs.empty()) continue;

      // 去重 changed，更新 block_table
      std::sort(changed.begin(), changed.end());
      changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
      if (!changed.empty()) decoder->update_block_table_partial(running, changed, max_blk);

      // forward
      auto t0 = std::chrono::steady_clock::now();
      decoder->forward_flat(flat_reqs, flat_tokens, flat_positions, token_to_seq, slot_mapping,
                            last_tok_idx, dec_flat, dec_pos, dec_seq, flat_offset);
      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      fprintf(stderr, "[P.prefill] forward batch=%d tokens=%d %.1fms\n", (int)flat_reqs.size(),
              flat_offset, ms);

      // 更新状态，完成的放入 result_queue
      for (int fi = 0; fi < (int)flat_reqs.size(); fi++) {
        auto& fr = flat_reqs[fi];
        int i = fr.req_idx;
        Request* r = running[i];
        if (r == nullptr) continue;

        auto& tk = running_tasks[i];
        int& poff = prefill_offset[i];

        r->pos += fr.n_tokens;
        poff += fr.n_tokens;

        if (poff >= tk.n_tokens) {
          // prefill 完成，采样 first_token
          float* logits = decoder->get_logits_batch(fi);
          int first_token = sample_topk(logits, cfg.vocab_size, 30, 0.0f, rng);

          // 收集 slots
          std::vector<int> slots(tk.n_tokens);
          for (int j = 0; j < tk.n_tokens; j++) {
            slots[j] = r->block_table.physical_idx(j) * BLOCK_SIZE + r->block_table.block_offset(j);
          }

          PrefillResult result;
          result.req_id = tk.req_id;
          result.n_tokens = tk.n_tokens;
          result.first_token = first_token;
          result.slots = std::move(slots);
          result.r = r;
          result_queue.push(std::move(result));

          // 清空槽位
          running[i] = nullptr;
          prefill_offset[i] = 0;
          completed++;
          // fprintf(stderr, "[P.prefill] 完成 req_id=%d slot=%d completed=%d\n", tk.req_id, i,
          //         completed);
        }
      }
    }
  }

done:
  result_queue.close();
  fprintf(stderr, "[P.prefill] 线程退出，共完成 %d 个\n", completed);
}

// 线程3：发送 response + KV cache
void p_send_thread(PNode& pnode, NcclComm& nccl, BlockPool* pool,
                   SafeQueue<PrefillResult>& result_queue, const Config& cfg) {
  int n_layers = cfg.n_layers;
  int kv_dim = cfg.n_kv_heads * (cfg.dim / cfg.n_heads);

  PrefillResult result;
  while (result_queue.pop(result)) {
    // 发 response
    PrefillResponse resp;
    resp.req_id = result.req_id;
    resp.n_tokens = result.n_tokens;
    resp.n_layers = n_layers;
    resp.kv_dim = kv_dim;
    resp.first_token = result.first_token;
    pnode.send_response(resp);

    // 发 KV cache
    auto t0 = std::chrono::steady_clock::now();
    nccl_send_kv(nccl, pool->get_k_cache(), pool->get_v_cache(), result.slots, result.n_tokens,
                 n_layers, kv_dim);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double mb = (double)result.n_tokens * n_layers * kv_dim * 2 * sizeof(__half) / 1024 / 1024;
    fprintf(stderr, "[P.send] req_id=%d KV: %.1fMB in %.1fms (%.1f MB/s)\n", result.req_id, mb, ms,
            mb / ms * 1000);

    // 释放物理块
    result.r->block_table.free_blocks([pool](int id) { pool->free(id); });
    delete result.r;
  }

  fprintf(stderr, "[P.send] 线程退出\n");
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model_file>\n", argv[0]);
    return 1;
  }

  int max_batch = (argc >= 4) ? atoi(argv[3]) : 64;
  auto* decoder = new GPUDecoder(argv[1], max_batch, 4096, 4096, 4096);
  fprintf(stderr, "P节点模型加载完成\n");

  PNode pnode;
  if (!pnode.init()) return 1;

  NcclComm nccl;
  if (!nccl_init_prefill(nccl, pnode)) return 1;

  const Config& cfg = decoder->get_config();
  BlockPool* pool = decoder->get_block_pool();
  std::mt19937 rng(42);

  SafeQueue<PrefillTask> task_queue;
  SafeQueue<PrefillResult> result_queue;

  // 启动三个线程
  std::thread recv_th(p_recv_thread, std::ref(pnode), std::ref(task_queue));
  std::thread prefill_th(p_prefill_thread, decoder, pool, std::ref(task_queue),
                         std::ref(result_queue), std::ref(cfg), std::ref(rng));
  std::thread send_th(p_send_thread, std::ref(pnode), std::ref(nccl), pool, std::ref(result_queue),
                      std::ref(cfg));

  recv_th.join();
  prefill_th.join();
  send_th.join();

  fprintf(stderr, "P节点完成\n");
  delete decoder;
  return 0;
}
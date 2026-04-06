#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <future>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include "gpu_decoder.h"
#include "pd_comm.h"

template <typename T>
struct SafeQueue {
  std::mutex mu;
  std::condition_variable cv_not_full;
  std::condition_variable cv_not_empty;
  std::queue<T> q;
  bool closed = false;
  int max_size;

  explicit SafeQueue(int max_size = 0) : max_size(max_size) {}

  void push(T val) {
    std::unique_lock<std::mutex> lock(mu);
    if (max_size > 0) {
      cv_not_full.wait(lock, [&] { return (int)q.size() < max_size || closed; });
    }
    if (closed) return;
    q.push(std::move(val));
    cv_not_empty.notify_one();
  }

  bool pop(T& val) {
    std::unique_lock<std::mutex> lock(mu);
    cv_not_empty.wait(lock, [&] { return !q.empty() || closed; });
    if (q.empty()) return false;
    val = std::move(q.front());
    q.pop();
    cv_not_full.notify_one();
    return true;
  }

  enum class PopResult { OK, EMPTY, CLOSED };

  PopResult try_pop(T& val) {
    std::lock_guard<std::mutex> lock(mu);
    if (!q.empty()) {
      val = std::move(q.front());
      q.pop();
      cv_not_full.notify_one();
      return PopResult::OK;
    }
    return closed ? PopResult::CLOSED : PopResult::EMPTY;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu);
    closed = true;
    cv_not_empty.notify_all();
    cv_not_full.notify_all();
  }
};

struct PrefillTask {
  int d_node_id;
  int req_id;
  int n_tokens;
  std::vector<int> token_ids;
};

struct PrefillResult {
  int d_node_id;
  int req_id;
  int n_tokens;
  int first_token;
  std::vector<int> slots;
  Request* r;
};

/**
 * prefill 计算线程
 */
void p_prefill_thread(GPUDecoder* decoder, BlockPool* pool, PNode& pnode,
                      SafeQueue<PrefillTask>& task_queue, SafeQueue<PrefillResult>& result_queue,
                      const Config& cfg, std::mt19937& rng, PrefixCache& prefix_cache,
                      bool enable_prefix_cache) {
  int max_batch = decoder->get_max_batch();
  int max_pps = decoder->get_max_prefill_tokens_per_step();
  int max_blk = decoder->get_max_blocks_per_seq();

  std::vector<Request*> running(max_batch, nullptr);
  std::vector<PrefillTask> running_tasks(max_batch);
  std::vector<int> prefill_offset(max_batch, 0);
  int completed = 0;

  while (true) {
    std::vector<int> changed;

    auto try_add_request = [&](int slot, PrefillTask task) -> bool {
      // D 节点已断开，丢弃任务
      if (!pnode.is_alive(task.d_node_id)) {
        fprintf(stderr, "[P.prefill] D节点[%d] 已断开，丢弃 req_id=%d\n", task.d_node_id,
                task.req_id);
        return true;
      }

      auto* r = new Request();
      r->id = task.req_id;
      r->pos = 0;
      prefill_offset[slot] = 0;

      if (enable_prefix_cache) {
        int hit_blocks = prefix_cache.match(task.token_ids, r->block_table);
        if (hit_blocks > 0) {
          r->pos = hit_blocks * BLOCK_SIZE;
          prefill_offset[slot] = hit_blocks * BLOCK_SIZE;
          fprintf(stderr, "[P.prefill] prefix cache hit: req_id=%d hit_blocks=%d hit_tokens=%d\n",
                  task.req_id, hit_blocks, hit_blocks * BLOCK_SIZE);
        }
      }

      int need_blocks = (task.n_tokens - 1) / BLOCK_SIZE + 1;
      for (int b = r->block_table.num_blocks(); b < need_blocks; b++) {
        int block_id = pool->allocate();
        if (block_id < 0) {
          delete r;
          return false;
        }
        r->block_table.add_block(block_id);
      }

      running[slot] = r;
      running_tasks[slot] = std::move(task);
      changed.push_back(slot);
      return true;
    };

    // 尝试填充空槽
    for (int i = 0; i < max_batch; i++) {
      if (running[i] != nullptr) continue;
      PrefillTask task;
      auto ret = task_queue.try_pop(task);
      if (ret == SafeQueue<PrefillTask>::PopResult::CLOSED) goto done;
      if (ret == SafeQueue<PrefillTask>::PopResult::EMPTY) break;
      if (!try_add_request(i, std::move(task))) goto done;
    }

    // 没有任何请求在跑，阻塞等一个
    {
      bool any = false;
      for (int i = 0; i < max_batch; i++) {
        if (running[i]) {
          any = true;
          break;
        }
      }

      if (!any) {
        int slot = -1;
        for (int i = 0; i < max_batch; i++) {
          if (running[i] == nullptr) {
            slot = i;
            break;
          }
        }
        if (slot == -1) continue;

        PrefillTask task;
        if (!task_queue.pop(task)) goto done;
        fprintf(stderr, "[P.prefill] 阻塞等到 req_id=%d slot=%d\n", task.req_id, slot);
        if (!try_add_request(slot, std::move(task))) goto done;
      }
    }

    // 构建 flat batch
    {
      std::vector<FlatRequest> flat_reqs;
      std::vector<int> flat_tokens, flat_positions, token_to_seq, slot_mapping;
      std::vector<int> last_tok_idx;
      std::vector<int> dec_flat;
      int flat_offset = 0;
      int prefill_budget = max_pps;

      for (int i = 0; i < max_batch; i++) {
        Request* r = running[i];
        if (r == nullptr) continue;

        auto& tk = running_tasks[i];
        int& poff = prefill_offset[i];

        // D 节点已断开，丢弃
        if (!pnode.is_alive(tk.d_node_id)) {
          fprintf(stderr, "[P.prefill] D节点[%d] 已断开，丢弃 req_id=%d\n", tk.d_node_id,
                  tk.req_id);
          r->block_table.free_blocks([pool](int id) { pool->dec_ref(id); });
          delete r;
          running[i] = nullptr;
          prefill_offset[i] = 0;
          completed++;
          continue;
        }

        int remaining = tk.n_tokens - poff;
        int n_tok = std::min(remaining, prefill_budget);
        if (n_tok <= 0) continue;

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

      std::sort(changed.begin(), changed.end());
      changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
      if (!changed.empty()) decoder->update_block_table_partial(running, changed, max_blk);

      auto t0 = std::chrono::steady_clock::now();
      decoder->forward_flat(flat_reqs, flat_tokens, flat_positions, token_to_seq, slot_mapping,
                            last_tok_idx, dec_flat, flat_offset);
      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      fprintf(stderr, "[P.prefill] forward batch=%d tokens=%d %.1fms\n", (int)flat_reqs.size(),
              flat_offset, ms);

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
          // D 节点已断开，丢弃结果
          if (!pnode.is_alive(tk.d_node_id)) {
            fprintf(stderr, "[P.prefill] D节点[%d] 已断开，丢弃 prefill 结果 req_id=%d\n",
                    tk.d_node_id, tk.req_id);
            r->block_table.free_blocks([pool](int id) { pool->dec_ref(id); });
            delete r;
            running[i] = nullptr;
            prefill_offset[i] = 0;
            completed++;
            continue;
          }

          float* logits = decoder->get_logits_batch(fi);
          int first_token = sample_topk(logits, cfg.vocab_size, 30, 0.0f, rng);

          if (enable_prefix_cache) {
            prefix_cache.insert(tk.token_ids, r->block_table);
          }

          std::vector<int> slots(tk.n_tokens);
          for (int j = 0; j < tk.n_tokens; j++) {
            slots[j] = r->block_table.physical_idx(j) * BLOCK_SIZE + r->block_table.block_offset(j);
          }

          PrefillResult result;
          result.d_node_id = tk.d_node_id;
          result.req_id = tk.req_id;
          result.n_tokens = tk.n_tokens;
          result.first_token = first_token;
          result.slots = std::move(slots);
          result.r = r;
          fprintf(stderr, "[P.prefill] push result req_id=%d d_node_id=%d\n", tk.req_id,
                  tk.d_node_id);
          result_queue.push(std::move(result));

          running[i] = nullptr;
          prefill_offset[i] = 0;
          completed++;
        }
      }
    }
  }

done:
  result_queue.close();
  fprintf(stderr, "[P.prefill] 线程退出，共完成 %d 个\n", completed);
}

/**
 * 发送 KV cache 给 D节点
 */
void p_send_thread(PNode& pnode, BlockPool* pool, SafeQueue<PrefillResult>& result_queue,
                   const Config& cfg) {
  int n_layers = cfg.n_layers;
  int kv_dim = cfg.n_kv_heads * (cfg.dim / cfg.n_heads);

  PrefillResult result;
  while (result_queue.pop(result)) {
    // D 节点已断开，丢弃
    if (!pnode.is_alive(result.d_node_id)) {
      fprintf(stderr, "[P.send] D节点[%d] 已断开，丢弃 req_id=%d\n", result.d_node_id,
              result.req_id);
      result.r->block_table.free_blocks([pool](int id) { pool->dec_ref(id); });
      delete result.r;
      continue;
    }

    // 发 response
    pd::PrefillResponse resp;
    resp.set_req_id(result.req_id);
    resp.set_n_tokens(result.n_tokens);
    resp.set_n_layers(n_layers);
    resp.set_kv_dim(kv_dim);
    resp.set_first_token(result.first_token);
    pnode.send_response(result.d_node_id, resp);

    // 发 KV
    auto t0 = std::chrono::steady_clock::now();
    pnode.send_kv(result.d_node_id, pool->get_k_cache(), pool->get_v_cache(), result.slots,
                  result.n_tokens, n_layers, kv_dim);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double mb = (double)result.n_tokens * n_layers * kv_dim * 2 * sizeof(__half) / 1024 / 1024;
    fprintf(stderr, "[P.send] d_node_id=%d req_id=%d KV: %.1fMB in %.1fms (%.1f MB/s)\n",
            result.d_node_id, result.req_id, mb, ms, mb / ms * 1000);

    result.r->block_table.free_blocks([pool](int id) { pool->dec_ref(id); });
    delete result.r;
  }
  fprintf(stderr, "[P.send] 线程退出\n");
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model_file> [max_batch] [enable_prefix_cache]\n", argv[0]);
    return 1;
  }

  int max_batch = (argc >= 3) ? atoi(argv[2]) : 64;
  bool enable_prefix_cache = (argc >= 4) ? atoi(argv[3]) != 0 : true;

  fprintf(stderr, "max_batch=%d enable_prefix_cache=%s\n", max_batch,
          enable_prefix_cache ? "true" : "false");

  auto* decoder = new GPUDecoder(argv[1], max_batch, 4096, 4096, 4096);
  fprintf(stderr, "P节点模型加载完成\n");

  const Config& cfg = decoder->get_config();
  BlockPool* pool = decoder->get_block_pool();
  std::mt19937 rng(42);
  PrefixCache prefix_cache(pool);

  SafeQueue<PrefillTask> task_queue(max_batch * 2);
  SafeQueue<PrefillResult> result_queue(max_batch * 2);

  PNode pnode;

  // 设置回调：收到 prefill 请求时推入 task_queue
  pnode.on_request = [&task_queue](pd::PrefillRequest req) {
    PrefillTask task;
    task.d_node_id = req.d_node_id();
    task.req_id = req.req_id();
    task.token_ids = {req.token_ids().begin(), req.token_ids().end()};
    task.n_tokens = (int)task.token_ids.size();
    task_queue.push(std::move(task));
  };

  // 设置回调：D节点断开时打印日志
  pnode.on_disconnect = [](int d_node_id) {
    fprintf(stderr, "[P.main] D节点[%d] 已断开\n", d_node_id);
  };

  // 启动 gRPC server
  if (!pnode.init()) return 1;
  fprintf(stderr, "P节点 gRPC server 启动完成\n");

  // prefill 线程
  std::thread prefill_th(p_prefill_thread, decoder, pool, std::ref(pnode), std::ref(task_queue),
                         std::ref(result_queue), std::ref(cfg), std::ref(rng),
                         std::ref(prefix_cache), enable_prefix_cache);

  // 发送线程
  std::thread send_th(p_send_thread, std::ref(pnode), pool, std::ref(result_queue), std::ref(cfg));

  // P节点常驻，等待 gRPC server 退出
  pnode.server->Wait();

  task_queue.close();
  result_queue.close();

  prefill_th.join();
  send_th.join();

  delete decoder;
  return 0;
}
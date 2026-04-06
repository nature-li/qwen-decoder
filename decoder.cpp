#include "decoder.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

void Decoder::generate_continuous(std::vector<std::string>& user_inputs, int max_batch,
                                  int max_new_tokens, float temperature, int top_k,
                                  std::mt19937& rng, bool enable_prefix_cache) {
  Scheduler scheduler(max_batch, BLOCK_SIZE, get_block_pool(), enable_prefix_cache);

  // 初始化所有请求，插入 BOS token，加入 waiting 队列
  std::vector<Request*> all_requests;
  for (int i = 0; i < (int)user_inputs.size(); i++) {
    auto* req = new Request();
    req->id = i;
    req->input = user_inputs[i];
    std::string prompt = apply_chat_template(user_inputs[i]);
    encode(tokenizer, prompt, req->prompt_tokens);
    req->prompt_tokens.insert(req->prompt_tokens.begin(), tokenizer.bos_token_id);
    scheduler.add_request(req);
    all_requests.push_back(req);
  }

  auto t_start = std::chrono::steady_clock::now();
  int output_tokens = 0;
  int max_blk = get_max_blocks_per_seq();
  // 每步 prefill token 预算
  int max_pps = get_max_prefill_tokens_per_step();

  int step = 0;
  auto last_step_time = std::chrono::steady_clock::now();
  while (!scheduler.all_done()) {
    // 把 waiting 队列里的请求填入空槽
    scheduler.fill_slots();
    scheduler.last_changed.clear();

    /**
     * 组装 flat batch
     */
    // 每个请求在 flat batch 里的元信息（offset, n_tokens, is_prefill 等）
    std::vector<FlatRequest> flat_requests;
    // [total_tokens] 所有 token 的 id，prefill token + decode token 按请求顺序打包
    std::vector<int> flat_tokens;
    // [total_tokens] 每个 token 的绝对位置（pos），用于 RoPE 和 KV cache 写入
    std::vector<int> flat_positions;
    // [total_tokens] 每个 token 属于哪个请求（scheduler.running 的下标），用于查 block_table
    std::vector<int> token_to_req;
    // [total_tokens] 每个 token 对应的绝对物理槽位，用于 KV cache 写入
    std::vector<int> token_slot;
    // [batch_size] 每个请求最后一个 token 在 flat batch 里的位置
    std::vector<int> last_token_indices;

    // decode token 单独收集，用于并行 attention
    // [n_decode] decode token 在 flat batch 里的位置，用于 gather/scatter q 和 xb
    std::vector<int> decode_flat_indices;
    // [n_decode] decode token 的绝对位置，用于 decode attention 计算 kv_end
    std::vector<int> decode_positions;
    // [n_decode] decode token 属于哪个请求，用于查 block_table 做 decode attention
    std::vector<int> decode_req_indices;

    int flat_offset = 0;
    // 本步剩余 prefill token 预算（所有请求共享）
    int prefill_budget = max_pps;

    for (int i = 0; i < max_batch; i++) {
      Request* req = scheduler.get_running_req(i);
      if (req == nullptr || req->finished) {
        continue;
      }

      FlatRequest fr;
      fr.req_idx = i;
      fr.flat_offset = flat_offset;
      fr.start_pos = req->pos;

      if (!req->prefill_done) {
        // prefill 阶段
        fr.is_prefill = true;

        // 新请求第一次 prefill，block_table 为空，需要更新 GPU 上的 block_table
        if (req->prefill_offset == 0) {
          scheduler.last_changed.push_back(i);
        }

        // chunked prefill：本步最多处理 prefill_budget 个 token
        int remaining = (int)req->prompt_tokens.size() - req->prefill_offset;
        fr.n_tokens = std::min(remaining, prefill_budget);

        // 本步 prefill 预算已耗尽，跳过
        if (fr.n_tokens <= 0) {
          continue;
        }

        if (req->pos + fr.n_tokens >= config.seq_len) {
          fprintf(stderr, "request %d: prompt too long\n", req->id);
          req->finished = true;
          continue;
        }

        // 为本步 prefill 的 token 分配物理块
        int old_blocks = (int)req->block_table.num_blocks();
        if (!scheduler.ensure_blocks(req, fr.n_tokens)) {
          req->finished = true;
          continue;
        }

        // 分配了新物理块，需要更新 GPU 上的 block_table
        if ((int)req->block_table.num_blocks() != old_blocks) {
          scheduler.last_changed.push_back(i);
        }

        // 把本步 prefill 的 token 打包进 flat batch
        for (int j = 0; j < fr.n_tokens; j++) {
          int tok = req->prompt_tokens[req->prefill_offset + j];
          int pos = req->pos + j;
          int phy = req->block_table.physical_idx(pos);
          flat_tokens.push_back(tok);
          flat_positions.push_back(pos);
          token_to_req.push_back(i);
          token_slot.push_back(phy * BLOCK_SIZE + pos % BLOCK_SIZE);
        }

        // 消耗本步 prefill 预算
        prefill_budget -= fr.n_tokens;
      } else {
        // decode 阶段
        fr.is_prefill = false;
        fr.n_tokens = 1;

        // 为下一个 decode token 分配物理块（如果当前块已满）
        int old_blocks = (int)req->block_table.num_blocks();
        if (!scheduler.ensure_blocks(req, 1)) {
          req->finished = true;
          continue;
        }

        // 分配了新物理块，需要更新 GPU 上的 block_table
        if ((int)req->block_table.num_blocks() != old_blocks) {
          scheduler.last_changed.push_back(i);
        }

        int phy = req->block_table.physical_idx(req->pos);
        flat_tokens.push_back(req->cur_token);
        flat_positions.push_back(req->pos);
        token_to_req.push_back(i);
        token_slot.push_back(phy * BLOCK_SIZE + req->pos % BLOCK_SIZE);

        // decode token 单独记录，forward_flat 里用于并行 attention
        decode_flat_indices.push_back(flat_offset);
        decode_positions.push_back(req->pos);
        decode_req_indices.push_back(i);
      }

      // 该请求最后一个 token 在 flat batch 里的位置，用于提取 logits
      last_token_indices.push_back(flat_offset + fr.n_tokens - 1);
      flat_offset += fr.n_tokens;
      flat_requests.push_back(fr);
    }

    if (flat_requests.empty()) continue;

    /**
     * 增量更新 GPU 上的 block_table，只更新有变化的槽位
     * 先去重再更新
     */
    std::sort(scheduler.last_changed.begin(), scheduler.last_changed.end());
    scheduler.last_changed.erase(
        std::unique(scheduler.last_changed.begin(), scheduler.last_changed.end()),
        scheduler.last_changed.end());
    update_block_table_partial(scheduler.running(), scheduler.last_changed, max_blk);

    // generate_continuous 里 forward_flat 调用前加
    auto now = std::chrono::steady_clock::now();
    double between = std::chrono::duration<double, std::milli>(now - last_step_time).count();
    last_step_time = now;
    fprintf(stdout,
            "between %.02f, step %d: total_tokens=%d n_decode=%d n_prefill=%d prefill_tokens=%d\n",
            between, step, (int)flat_tokens.size(), (int)decode_positions.size(),
            (int)flat_requests.size() - (int)decode_positions.size(),
            (int)flat_tokens.size() - (int)decode_positions.size());

    // forward: 一次处理所有 token
    forward_flat(flat_requests, flat_tokens, flat_positions, token_to_req, token_slot,
                 last_token_indices, decode_flat_indices, (int)flat_tokens.size());

    // 采样、更新状态
    for (int fi = 0; fi < (int)flat_requests.size(); fi++) {
      auto& fr = flat_requests[fi];
      Request* req = scheduler.get_running_req(fr.req_idx);
      if (req == nullptr || req->finished) {
        continue;
      }

      // logits 按 flat_requests 顺序存放，fi 对应第 fi 个请求
      float* logits = get_logits_batch(fi);
      int next_token = sample_topk(logits, config.vocab_size, top_k, temperature, rng);

      if (!req->prefill_done) {
        req->pos += fr.n_tokens;
        req->prefill_offset += fr.n_tokens;
        // prefill 全部完成，采样第一个 decode token
        if (req->prefill_offset >= (int)req->prompt_tokens.size()) {
          req->prefill_done = true;
          req->cur_token = next_token;
          scheduler.on_prefill_done(req);
        }
        // prefill 未完成时 next_token 丢弃，下一步继续 prefill
      } else {
        // decode：输出当前 token，更新状态
        const char* piece = decode(tokenizer, req->cur_token);
        if (piece) {
          req->output += piece;
        }

        req->pos++;
        req->cur_token = next_token;
        req->n_generated++;
        output_tokens++;

        // 检查结束条件
        if (next_token == tokenizer.eos_token_id || next_token == tokenizer.bos_token_id ||
            tokenizer.vocab[next_token] == "<|im_end|>" || req->n_generated >= max_new_tokens ||
            req->pos >= config.seq_len) {
          req->finished = true;
        }
      }
    }

    // 释放完成请求的槽位和物理块，waiting 队列里的请求下一步可以填入
    scheduler.release_finished();

    step++;
  }

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();

  for (auto* req : all_requests) {
    printf("request id: %d\nprompt: %s\noutput: %s\n--------------------\n", req->id,
           req->input.c_str(), req->output.c_str());
    delete req;
  }

  fprintf(stderr, "total: %d tokens in %.2fs (%.1f tokens/s)\n", output_tokens, elapsed,
          output_tokens / elapsed);
}

#include "decoder.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

void Decoder::generate_continuous(std::vector<std::string>& user_inputs,
                                  int max_batch, int max_new_tokens,
                                  float temperature, int top_k,
                                  std::mt19937& rng) {
  Scheduler scheduler(max_batch, BLOCK_SIZE, get_page_pool());

  // 把所有请求加入等待队列
  std::vector<Request*> all_requests;
  for (int i = 0; i < (int)user_inputs.size(); i++) {
    auto* req = new Request();
    req->id = i;
    req->input = user_inputs[i];
    std::string prompt = apply_chat_template(user_inputs[i]);
    encode(tokenizer, prompt, req->prompt_tokens);
    scheduler.add_request(req);
    all_requests.push_back(req);
  }

  auto t_start = std::chrono::steady_clock::now();
  int total_tokens = 0;

  while (!scheduler.all_done()) {
    // 1. 把等待队列里的请求填入空槽
    scheduler.fill_slots();

    // 2. 对每个空槽里刚进来的请求做 prefill
    for (int i = 0; i < max_batch; i++) {
      Request* req = scheduler.running[i];
      if (req == nullptr || req->prefill_done) continue;

      // prefill BOS
      int bos = tokenizer.bos_token_id;
      if (!scheduler.ensure_pages(req, 1)) {
        req->finished = true;
        continue;
      }
      forward_prefill(&bos, 1, req->pos, i, req->block_table.table);
      req->pos++;

      // prefill prompt
      if (req->pos + (int)req->prompt_tokens.size() >= config.seq_len) {
        fprintf(stderr, "request %d: prompt too long\n", req->id);
        req->finished = true;
        continue;
      }

      if (!scheduler.ensure_pages(req, (int)req->prompt_tokens.size())) {
        req->finished = true;
        continue;
      }
      forward_prefill(req->prompt_tokens.data(), (int)req->prompt_tokens.size(),
                      req->pos, i, req->block_table.table);
      req->pos += (int)req->prompt_tokens.size();
      req->prefill_done = true;

      // prefill 完成后采样第一个 token
      float* logits = get_logits_batch(i);
      req->cur_token =
          sample_topk(logits, config.vocab_size, top_k, temperature, rng);
    }

    // 3. 收集所有 prefill 完成的请求做 batch decode
    std::vector<int> batch_tokens(max_batch, 0);
    std::vector<int> batch_positions(max_batch, 0);
    bool has_decode = false;
    for (int i = 0; i < max_batch; i++) {
      Request* req = scheduler.running[i];
      if (req == nullptr || !req->prefill_done || req->finished) continue;
      batch_tokens[i] = req->cur_token;
      batch_positions[i] = req->pos;
      has_decode = true;
    }

    if (!has_decode) continue;

    int max_blk = get_max_blocks_per_seq();
    update_block_table(scheduler.running, max_blk);
    forward_batch(batch_tokens.data(), batch_positions.data(), max_batch);

    // 4. 采样、收集输出、检查结束条件
    for (int i = 0; i < max_batch; i++) {
      Request* req = scheduler.running[i];
      if (req == nullptr || !req->prefill_done || req->finished) continue;
      if (!scheduler.ensure_pages(req, 1)) {
        req->finished = true;
        continue;
      }

      float* logits = get_logits_batch(i);
      int next_token =
          sample_topk(logits, config.vocab_size, top_k, temperature, rng);

      const char* piece = decode(tokenizer, req->cur_token);
      if (piece) req->output += piece;

      req->pos++;
      req->cur_token = next_token;
      req->n_generated++;
      total_tokens++;

      if (next_token == tokenizer.eos_token_id ||
          next_token == tokenizer.bos_token_id ||
          tokenizer.vocab[next_token] == "<|im_end|>" ||
          req->n_generated >= max_new_tokens || req->pos >= config.seq_len) {
        req->finished = true;
      }
    }

    // 5. 释放完成的槽位，下一轮 fill_slots 会填入新请求
    scheduler.release_finished();
  }

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();

  // 打印所有请求的输出
  for (auto* req : all_requests) {
    printf("=== Request %d ===\nContext: %s\nAssistant: %s\n\n", req->id,
           req->input.c_str(), req->output.c_str());
    delete req;
  }

  fprintf(stderr, "total: %d tokens in %.2fs (%.1f tokens/s)\n", total_tokens,
          elapsed, total_tokens / elapsed);
}
#include "decoder.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

void Decoder::generate_batch(const std::vector<std::string>& user_inputs,
                             int max_new_tokens, float temperature, int top_k,
                             std::mt19937& rng) {
  int batch_size = (int)user_inputs.size();

  std::vector<std::vector<int>> prompt_tokens(batch_size);
  std::vector<int> pos(batch_size, 0);
  std::vector<int> cur_token(batch_size, 0);
  std::vector<bool> finished(batch_size, false);
  std::vector<int> n_generated(batch_size, 0);
  std::vector<std::string> outputs(batch_size);  // 缓存每个请求的输出

  // 1. encode + prefill 所有请求
  for (int i = 0; i < batch_size; i++) {
    std::string prompt = apply_chat_template(user_inputs[i]);
    encode(tokenizer, prompt, prompt_tokens[i]);

    // prefill BOS
    int bos = tokenizer.bos_token_id;
    forward_prefill(&bos, 1, pos[i], i);
    pos[i]++;

    // prefill prompt
    if (pos[i] + (int)prompt_tokens[i].size() >= config.seq_len) {
      fprintf(stderr, "request %d: prompt too long\n", i);
      finished[i] = true;
      continue;
    }
    auto t_prefill_start = std::chrono::steady_clock::now();
    forward_prefill(prompt_tokens[i].data(), (int)prompt_tokens[i].size(),
                    pos[i], i);
    pos[i] += (int)prompt_tokens[i].size();
    auto t_prefill_end = std::chrono::steady_clock::now();
    double prefill_time =
        std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();
    fprintf(stderr, "req%d prefill: %.2f tokens/s (%d tokens in %.2fs)\n", i,
            prompt_tokens[i].size() / prefill_time,
            (int)prompt_tokens[i].size(), prefill_time);

    // prefill 完成后采样第一个 token
    float* logits = get_logits_batch(i);
    cur_token[i] =
        sample_topk(logits, config.vocab_size, top_k, temperature, rng);
  }

  // 2. batch decode 循环
  auto t_start = std::chrono::steady_clock::now();

  while (true) {
    bool all_done = true;
    for (int i = 0; i < batch_size; i++)
      if (!finished[i]) {
        all_done = false;
        break;
      }
    if (all_done) break;

    std::vector<int> batch_tokens(batch_size);
    std::vector<int> batch_positions(batch_size);
    for (int i = 0; i < batch_size; i++) {
      batch_tokens[i] = finished[i] ? 0 : cur_token[i];
      batch_positions[i] = pos[i];
    }

    forward_batch(batch_tokens.data(), batch_positions.data(), batch_size);

    for (int i = 0; i < batch_size; i++) {
      if (finished[i]) continue;

      float* logits = get_logits_batch(i);
      int next_token =
          sample_topk(logits, config.vocab_size, top_k, temperature, rng);

      // 缓存输出，不实时打印
      const char* piece = decode(tokenizer, cur_token[i]);
      if (piece) outputs[i] += piece;

      pos[i]++;
      cur_token[i] = next_token;
      n_generated[i]++;

      if (next_token == tokenizer.eos_token_id ||
          next_token == tokenizer.bos_token_id ||
          tokenizer.vocab[next_token] == "<|im_end|>" ||
          n_generated[i] >= max_new_tokens || pos[i] >= config.seq_len) {
        finished[i] = true;
      }
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();
  int total_tokens = 0;
  for (int i = 0; i < batch_size; i++) total_tokens += n_generated[i];

  // 所有请求完成后统一打印
  for (int i = 0; i < batch_size; i++) {
    std::cout << "---------------------------------" << std::endl;
    std::cout << user_inputs[i] << std::endl;
    std::cout << outputs[i] << std::endl;
    std::cout << std::endl;
  }

  fprintf(stderr, "generate: %.2f tokens/s (%d tokens in %.2fs)\n",
          total_tokens / elapsed, total_tokens, elapsed);
}

void Decoder::generate_continuous(std::vector<std::string>& user_inputs,
                                  int max_batch, int max_new_tokens,
                                  float temperature, int top_k,
                                  std::mt19937& rng) {
  Scheduler scheduler(max_batch);

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
      forward_prefill(&bos, 1, req->pos, i);
      req->pos++;

      // prefill prompt
      if (req->pos + (int)req->prompt_tokens.size() >= config.seq_len) {
        fprintf(stderr, "request %d: prompt too long\n", req->id);
        req->finished = true;
        continue;
      }
      forward_prefill(req->prompt_tokens.data(), (int)req->prompt_tokens.size(),
                      req->pos, i);
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

    forward_batch(batch_tokens.data(), batch_positions.data(), max_batch);

    // 4. 采样、收集输出、检查结束条件
    for (int i = 0; i < max_batch; i++) {
      Request* req = scheduler.running[i];
      if (req == nullptr || !req->prefill_done || req->finished) continue;

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
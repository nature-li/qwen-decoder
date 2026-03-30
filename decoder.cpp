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
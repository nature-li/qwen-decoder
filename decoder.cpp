#include "decoder.h"

#include <chrono>
#include <cstdio>
#include <vector>

void Decoder::generate(const std::string& user_input, int max_new_tokens,
                       float temperature, int top_k, std::mt19937& rng) {
  // 套 chat template
  std::string prompt = apply_chat_template(user_input);

  // encode
  std::vector<int> prompt_tokens;
  encode(tokenizer, prompt, prompt_tokens);

  // prefill BOS
  int pos = 0;
  forward(tokenizer.bos_token_id, pos++);

  // prefill prompt
  auto t_prefill_start = std::chrono::steady_clock::now();
  if (pos + (int)prompt_tokens.size() >= config.seq_len) {
    fprintf(stderr, "prompt too long\n");
    return;
  }
  forward_prefill(prompt_tokens.data(), (int)prompt_tokens.size(), pos);
  pos += (int)prompt_tokens.size();

  auto t_prefill_end = std::chrono::steady_clock::now();
  double prefill_time =
      std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();
  fprintf(stderr, "prefill: %d tokens in %.3fs (%.1f tokens/s)\n",
          (int)prompt_tokens.size(), prefill_time,
          prompt_tokens.size() / prefill_time);

  // 生成阶段开始计时
  int n_generated = 0;
  auto t_start = std::chrono::steady_clock::now();

  // 生成
  printf("Assistant: ");
  while (pos < config.seq_len &&
         pos < max_new_tokens + (int)prompt_tokens.size() + 1) {
    float* logits = get_logits();
    int next_token =
        sample_topk(logits, config.vocab_size, top_k, temperature, rng);

    if (next_token == tokenizer.eos_token_id) break;
    if (next_token == tokenizer.bos_token_id) break;
    if (tokenizer.vocab[next_token] == "<|im_end|>") break;

    printf("%s", decode(tokenizer, next_token));
    fflush(stdout);

    forward(next_token, pos++);
    n_generated++;
  }
  printf("\n");

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();
  fprintf(stderr, "\n%.2f tokens/s (%d tokens in %.2fs)\n",
          n_generated / elapsed, n_generated, elapsed);
}
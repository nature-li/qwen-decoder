// gpu_v11/main.cu
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "gpu_decoder.h"

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model> [max_batch] [max_prefill_len] [max_prefill_step]\n",
            argv[0]);
    return 1;
  }

  // 最大并发请求数
  int max_batch = 64;
  // 单个请求 prompt 的最大长度
  int max_prefill_len = 4096;
  // 每步允许处理的最大 prefill token 数(所有请求共享的预算)
  int max_prefill_tokens_per_step = 512;
  // 每步最多处理多少 token
  int max_total_tokens = 1024;

  if (argc >= 3) max_batch = atoi(argv[2]);
  if (argc >= 4) max_prefill_len = atoi(argv[3]);
  if (argc >= 5) max_prefill_tokens_per_step = atoi(argv[4]);
  if (argc >= 6) max_total_tokens = atoi(argv[5]);

  fprintf(stderr,
          "max_batch=%d max_total_tokens=%d "
          "max_prefill_len=%d max_prefill_step=%d\n",
          max_batch, max_total_tokens, max_prefill_len, max_prefill_tokens_per_step);

  Decoder* decoder = new GPUDecoder(argv[1], max_batch, max_prefill_len,
                                    max_prefill_tokens_per_step, max_total_tokens);

  std::mt19937 rng(42);
  float temperature = 0.0f;
  int top_k = 30;
  int max_new_toks = 256;

  std::vector<std::string> inputs;
  std::string line;
  int idx = 0;
  while (true) {
    printf("User[%d] (empty to start): ", idx++);
    if (!std::getline(std::cin, line) || line.empty()) break;
    inputs.push_back(line);
  }

  if (inputs.empty()) {
    fprintf(stderr, "no input\n");
    delete decoder;
    return 1;
  }
  fprintf(stderr, "%d requests\n", (int)inputs.size());

  decoder->generate_continuous(inputs, max_batch, max_new_toks, temperature, top_k, rng);

  delete decoder;
  return 0;
}

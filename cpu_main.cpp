#include <ctime>
#include <iostream>
#include <random>

#include "cpu_decoder.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model_file>\n", argv[0]);
    return 1;
  }

  CPUDecoder decoder(argv[1]);

  std::mt19937 rng(time(nullptr));
  float temperature = 0.7f;
  int top_k = 40;
  int max_new_tokens = 256;

  std::string user_input;
  printf("User: ");
  std::getline(std::cin, user_input);
  fprintf(stderr, "%s\n", user_input.c_str());

  decoder.generate(user_input, max_new_tokens, temperature, top_k, rng);

  return 0;
}
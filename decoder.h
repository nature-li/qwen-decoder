#pragma once
#include <random>
#include <string>
#include <vector>

#include "common.h"

class Decoder {
 public:
  virtual ~Decoder() = default;
  virtual void forward(int token, int pos) = 0;
  virtual float* get_logits() = 0;
  void generate(const std::string& user_input, int max_new_tokens,
                float temperature, int top_k, std::mt19937& rng);

 protected:
  Config config;
  Tokenizer tokenizer;
};
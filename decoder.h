#pragma once
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "scheduler.h"

class Decoder {
 public:
  virtual ~Decoder() = default;
  // prefill: 单请求多 token
  virtual void forward_prefill(const int* tokens, int n_tokens, int start_pos,
                               int batch_idx, const std::vector<int>& block_table_cpu) = 0;
  virtual void forward_batch(const int* tokens, const int* positions,
                             int batch_size) = 0;
  // 获取 batch logits, batch idx 指定第几个请求
  virtual float* get_logits_batch(int batch_idx) = 0;
  void generate_batch(const std::vector<std::string>& user_inputs,
                      int max_new_tokens, float temperature, int top_k,
                      std::mt19937& rng);
  void generate_continuous(std::vector<std::string>& user_inputs, int max_batch,
                           int max_new_tokens, float temperature, int top_k,
                           std::mt19937& rng);
  virtual PagePool* get_page_pool() { return nullptr; }
  // 上传 block_table
  virtual void update_block_table(const std::vector<Request*>& running,
                                  int max_blocks_per_seq) {}
  // decoder.h
  virtual int get_max_blocks_per_seq() const { return 0; }

 protected:
  Config config;
  Tokenizer tokenizer;
};
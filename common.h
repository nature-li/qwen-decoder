#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf_loader.h"

// ============================================================================
// Config - 从 GGUF metadata 里读出来
// ============================================================================
struct Config {
  int dim;               // embedding_length = 896
  int hidden_dim;        // feed_forward_length = 4864
  int n_layers;          // block_count = 24
  int n_heads;           // attention.head_count = 14
  int n_kv_heads;        // attention.head_count_kv = 2
  int vocab_size;        // 151936
  int seq_len;           // context_length = 8192
  float rope_freq_base;  // rope.freq_base = 1000000.0
};

// ============================================================================
// Weights - 指向 mmap 内存的指针
// fp16 权重用 uint16_t 存储
// ============================================================================
struct Weights {
  uint16_t* token_embedding;  // [vocab_size, dim] fp16
  float** rms_att;            // [n_layers][dim] fp32
  uint16_t** wq;              // [n_layers][dim, dim] fp16
  uint16_t** wk;              // [n_layers][kv_dim, dim] fp16
  uint16_t** wv;              // [n_layers][kv_dim, dim] fp16
  uint16_t** wo;              // [n_layers][dim, dim] fp16
  float** bq;                 // [n_layers][dim] fp32
  float** bk;                 // [n_layers][kv_dim] fp32
  float** bv;                 // [n_layers][kv_dim] fp32
  float** rms_ffn;            // [n_layers][dim] fp32
  uint16_t** w1;              // [n_layers][hidden_dim, dim] fp16
  uint16_t** w2;              // [n_layers][dim, hidden_dim] fp16
  uint16_t** w3;              // [n_layers][hidden_dim, dim] fp16
  float* rms_final;           // [dim] fp32
  uint16_t* wcls;             // [vocab_size, dim] fp16
};

// ============================================================================
// ModelFile - mmap 文件
// ============================================================================
struct ModelFile {
  int fd;
  void* data;
  size_t size;
};

// ============================================================================
// Tokenizer
// ============================================================================
struct Tokenizer {
  int vocab_size;
  std::vector<std::string> vocab;  // token 字符串
  std::vector<float> scores;       // BPE merge 分数
  std::vector<int> token_type;     // token 类型
  int bos_token_id;                // 151643
  int eos_token_id;                // 151645
};

// fp16 转 float
float fp16_to_float(uint16_t h);
int open_model(const std::string& path, ModelFile& mf);

void close_model(ModelFile& mf);

// 从 GGUF 元数据里读 Config
int load_config(Config& config, const GGUFFile& gguf);

// 根据张量名找到数据指针
static void* find_tensor(const GGUFFile& gguf, const ModelFile& mf,
                         const std::string& name);

int load_weights(Weights& w, const Config& config, const GGUFFile& gguf,
                 const ModelFile& mf);

void free_weights(Weights& w);

int load_tokenizer(Tokenizer& t, const GGUFFile& gguf);
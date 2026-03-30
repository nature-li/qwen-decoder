#pragma once
#include <cuda_runtime.h>

#include "decoder.h"
#include "gguf_loader.h"

// GPU 上的权重，fp16 权重上传到 GPU
struct GPUWeights {
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

// GPU 上的运行时缓冲区
struct GPURunState {
  float* x;        // [dim]
  float* xb;       // [dim]
  float* xb2;      // [dim]
  float* q;        // [dim]
  float* k;        // [kv_dim]
  float* v;        // [kv_dim]
  float* att;      // [n_heads, seq_len]
  float* hb;       // [hidden_dim]
  float* hb2;      // [hidden_dim]
  float* logits;   // [vocab_size] pinned memory
  float* k_cache;  // [n_layers, seq_len, kv_dim]
  float* v_cache;  // [n_layers, seq_len, kv_dim]
};

class GPUDecoder : public Decoder {
 public:
  GPUDecoder(const std::string& model_file);
  ~GPUDecoder();

  void forward(int token, int pos) override;
  void forward_prefill(const int* tokens, int n_tokens, int start_pos) override;
  float* get_logits() override;

 private:
  ModelFile mf;
  GGUFFile gguf;
  Weights w;       // CPU 权重（mmap）
  GPUWeights gw;   // GPU 权重
  GPURunState gs;  // GPU 运行时缓冲区
};
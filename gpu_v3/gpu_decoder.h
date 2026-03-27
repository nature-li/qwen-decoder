#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "decoder.h"
#include "gguf_loader.h"

// GPU 上的权重，fp16 权重上传到 GPU
struct GPUWeights {
  __half* token_embedding;  // [vocab_size, dim] fp16
  float** rms_att;          // [n_layers][dim] fp32
  __half** wq;              // [n_layers][dim, dim] fp16
  __half** wk;              // [n_layers][kv_dim, dim] fp16
  __half** wv;              // [n_layers][kv_dim, dim] fp16
  __half** wo;              // [n_layers][dim, dim] fp16
  float** bq;               // [n_layers][dim] fp32
  float** bk;               // [n_layers][kv_dim] fp32
  float** bv;               // [n_layers][kv_dim] fp32
  float** rms_ffn;          // [n_layers][dim] fp32
  __half** w1;              // [n_layers][hidden_dim, dim] fp16
  __half** w2;              // [n_layers][dim, hidden_dim] fp16
  __half** w3;              // [n_layers][hidden_dim, dim] fp16
  float* rms_final;         // [dim] fp32
  __half* wcls;             // [vocab_size, dim] fp16
};

// GPU 上的运行时缓冲区
struct GPURunState {
  __half* x;        // [dim]
  __half* xb;       // [dim]
  __half* xb2;      // [dim]
  __half* q;        // [dim]
  __half* k;        // [kv_dim]
  __half* v;        // [kv_dim]
  float* att;       // [n_heads, seq_len] fp32 (softmax 需要精度)
  __half* hb;       // [hidden_dim]
  __half* hb2;      // [hidden_dim]
  float* logits;    // [vocab_size] pinned memory
  __half* k_cache;  // [n_layers, seq_len, kv_dim]
  __half* v_cache;  // [n_layers, seq_len, kv_dim]
  __half* logits_fp16;  // [vocab_size] 临时 fp16 logits
};

class GPUDecoder : public Decoder {
 public:
  GPUDecoder(const std::string& model_file);
  ~GPUDecoder();

  void forward(int token, int pos) override;
  float* get_logits() override;

 private:
  ModelFile mf;
  GGUFFile gguf;
  Weights w;       // CPU 权重（mmap）
  GPUWeights gw;   // GPU 权重
  GPURunState gs;  // GPU 运行时缓冲区
  cublasHandle_t cublas_handle;
};
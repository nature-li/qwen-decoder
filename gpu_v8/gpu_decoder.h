#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <queue>

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
  /**
   * decode 单请求缓冲区（prefill 内部也用）
   */
  __half* x;    // [dim]
  __half* xb;   // [dim]
  __half* xb2;  // [dim]
  __half* q;    // [dim]
  __half* k;    // [kv_dim]
  __half* v;    // [kv_dim]
  __half* hb;   // [hidden_dim]
  __half* hb2;  // [hidden_dim]

  /**
   * prefill 阶段（单请求多 token）
   */
  __half* x_pre;    // [max_prefill, dim]
  __half* xb_pre;   // [max_prefill, dim]
  __half* xb2_pre;  // [max_prefill, dim]
  __half* q_pre;    // [max_prefill, dim]
  __half* k_pre;    // [max_prefill, kv_dim]
  __half* v_pre;    // [max_prefill, kv_dim]
  __half* hb_pre;   // [max_prefill, hidden_dim]
  __half* hb2_pre;  // [max_prefill, hidden_dim]

  /**
   * batch decode 阶段（多请求并发）
   */
  __half* x_batch;    // [max_batch, dim]
  __half* xb_batch;   // [max_batch, dim]
  __half* xb2_batch;  // [max_batch, dim]
  __half* q_batch;    // [max_batch, dim]
  __half* k_batch;    // [max_batch, kv_dim]
  __half* v_batch;    // [max_batch, kv_dim]
  __half* hb_batch;   // [max_batch, hidden_dim]
  __half* hb2_batch;  // [max_batch, hidden_dim]

  /**
   * KV cache：每个请求独立
   * [max_batch, n_layers, seq_len, kv_dim]
   */
  __half* k_cache;
  __half* v_cache;

  /**
   * logits
   */
  __half* logits_fp16_batch;  // [max_batch, vocab_size]
  float* logits_batch;        // [max_batch, vocab_size] pinned memory
};

class GPUDecoder : public Decoder {
 public:
  GPUDecoder(const std::string& model_file, int max_batch);
  ~GPUDecoder();

  void forward_prefill(const int* tokens, int n_tokens, int start_pos,
                       int batch_idx) override;
  void forward_batch(const int* tokens, const int* positions,
                     int batch_size) override;
  float* get_logits_batch(int batch_idx) override;

 private:
  int max_batch;
  ModelFile mf;
  GGUFFile gguf;
  Weights w;
  GPUWeights gw;
  GPURunState gs;
  cublasHandle_t cublas_handle;
};
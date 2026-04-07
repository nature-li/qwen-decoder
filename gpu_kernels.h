#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdio.h>

constexpr int FA_BLOCK = 256;

#define CHECK_CUDA(call)                                                                      \
  {                                                                                           \
    cudaError_t _cuda_error = call;                                                           \
    if (_cuda_error != cudaSuccess) {                                                         \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(_cuda_error), __FILE__, \
              __LINE__);                                                                      \
      exit(1);                                                                                \
    }                                                                                         \
  }

#define CHECK_CUBLAS(call)                                                                \
  {                                                                                       \
    cublasStatus_t _cublas_status = call;                                                 \
    if (_cublas_status != CUBLAS_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "cuBLAS Error: %d at %s:%d\n", _cublas_status, __FILE__, __LINE__); \
      exit(1);                                                                            \
    }                                                                                     \
  }

#define CHECK_KERNEL()                                                   \
  {                                                                      \
    cudaError_t _cuda_error = cudaGetLastError();                        \
    if (_cuda_error != cudaSuccess) {                                    \
      fprintf(stderr, "Kernel error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(_cuda_error));                          \
      exit(1);                                                           \
    }                                                                    \
  }

#define CHECK_NCCL(call)                                                                       \
  {                                                                                            \
    ncclResult_t _nccl_result = call;                                                          \
    if (_nccl_result != ncclSuccess) {                                                         \
      fprintf(stderr, "NCCL Error: %s at %s:%d\n", ncclGetErrorString(_nccl_result), __FILE__, \
              __LINE__);                                                                       \
      exit(1);                                                                                 \
    }                                                                                          \
  }

__device__ float warp_reduce_sum(float val);

__device__ float warp_reduce_max(float val);

// Embedding lookup: out[i] = table[tokens[i]]
__global__ void embedding_kernel(__half* out, const __half* table, const int* tokens, int n,
                                 int dim);

// RMSNorm: out[i] = x[i] / rms(x[i]) * weight
__global__ void rmsnorm_kernel(__half* out, const __half* x, const float* weight, int dim);

// Add bias in-place: out[i] += bias
__global__ void add_bias_kernel(__half* out, const float* bias, int n_tok, int n);

// RoPE: apply rotary position embedding, each token has its own pos
__global__ void rope_kernel(__half* q, __half* k, const int* positions, int n_tok, int dim,
                            int kv_dim, int head_dim, float freq_base);

// KV cache write: slot_map[tok] = absolute slot index
// k_cache layout: [total_slots, n_layers, kv_dim]
__global__ void kvcache_write_kernel(__half* k_cache, __half* v_cache, const __half* k,
                                     const __half* v, const int* slot_map, int n_tok, int layer,
                                     int n_layers, int kv_dim);

// Residual add: x += delta
__global__ void residual_kernel(__half* x, const __half* delta, int n_tok, int dim);

// SwiGLU: hb[i] = silu(hb[i]) * hb2[i]
__global__ void swiglu_kernel(__half* hb, const __half* hb2, int n_tok, int hidden_dim);

// Extract last token feature for each request
__global__ void extract_last_token_kernel(__half* out, const __half* in, const int* last_idx,
                                          int batch_size, int dim);

// fp16 -> fp32 for logits
__global__ void fp16_to_fp32_kernel(float* out, const __half* in, int batch_size, int vocab_size);

__global__ void prefill_attention_kernel(
    const __half* q,              // [total_tokens, dim]
    const __half* k_cache,        // [total_slots, n_layers, kv_dim]
    const __half* v_cache,        // [total_slots, n_layers, kv_dim]
    __half* out,                  // [total_tokens, dim]
    const int* block_table,       // [max_batch, max_blocks_per_seq]
    const int* token_to_req,      // [total_tokens] d_token_seq
    const int* flat_positions,    // [total_tokens] d_positions
    const int* prefill_flat_idx,  // [total_prefill_tokens] prefill token 在 flat batch 里的位置
    int max_blocks_per_seq, int block_size, int n_layers, int layer, int dim, int kv_dim,
    int head_dim, int kv_mul);

__global__ void decode_attention_kernel(
    const __half* q,        // [total_tokens, dim]
    const __half* k_cache,  // [total_slots, n_layers, kv_dim]
    const __half* v_cache,
    __half* out,                // [total_tokens, dim]
    const int* block_table,     // [max_batch, max_blocks_per_seq]
    const int* token_to_req,    // [total_tokens] 复用 d_token_seq
    const int* flat_positions,  // [total_tokens] 复用 d_positions
    const int* dec_flat_idx,    // [n_decode] decode token 在 flat batch 里的位置
    int layer, int max_blocks_per_seq, int block_size, int dim, int kv_dim, int head_dim,
    int kv_mul, int n_layers);

void matmul(cublasHandle_t h, __half* out, const __half* x, const __half* w, int n, int d,
            int batch = 1);

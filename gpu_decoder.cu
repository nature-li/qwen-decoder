#include <math_constants.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gpu_decoder.h"

// ============================================================================
// 错误检查宏
// ============================================================================
#define CHECK_CUDA(call)                                                                         \
  {                                                                                              \
    cudaError_t err = call;                                                                      \
    if (err != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(1);                                                                                   \
    }                                                                                            \
  }

#define CHECK_CUBLAS(call)                                                   \
  {                                                                          \
    cublasStatus_t s = call;                                                 \
    if (s != CUBLAS_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "cuBLAS Error: %d at %s:%d\n", s, __FILE__, __LINE__); \
      exit(1);                                                               \
    }                                                                        \
  }

#define CHECK_KERNEL()                                                                             \
  {                                                                                                \
    cudaError_t err = cudaGetLastError();                                                          \
    if (err != cudaSuccess) {                                                                      \
      fprintf(stderr, "Kernel error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                                     \
    }                                                                                              \
  }

// ============================================================================
// 权重上传/释放
// ============================================================================
static void upload_weights(GPUWeights& gw, const Weights& w, const Config& cfg) {
  int head_dim = cfg.dim / cfg.n_heads;
  int kv_dim = cfg.n_kv_heads * head_dim;
  int vocab_size = cfg.vocab_size;

  auto up16 = [](__half** dst, const uint16_t* src, size_t n) {
    CHECK_CUDA(cudaMalloc(dst, n * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(*dst, src, n * sizeof(__half), cudaMemcpyHostToDevice));
  };
  auto up32 = [](float** dst, const float* src, size_t n) {
    CHECK_CUDA(cudaMalloc(dst, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
  };

  up16(&gw.token_embedding, w.token_embedding, (size_t)vocab_size * cfg.dim);
  up32(&gw.rms_final, w.rms_final, cfg.dim);
  up16(&gw.wcls, w.wcls, (size_t)vocab_size * cfg.dim);

  gw.rms_att = new float*[cfg.n_layers];
  gw.wq = new __half*[cfg.n_layers];
  gw.wk = new __half*[cfg.n_layers];
  gw.wv = new __half*[cfg.n_layers];
  gw.wo = new __half*[cfg.n_layers];
  gw.bq = new float*[cfg.n_layers];
  gw.bk = new float*[cfg.n_layers];
  gw.bv = new float*[cfg.n_layers];
  gw.rms_ffn = new float*[cfg.n_layers];
  gw.w1 = new __half*[cfg.n_layers];
  gw.w2 = new __half*[cfg.n_layers];
  gw.w3 = new __half*[cfg.n_layers];

  for (int l = 0; l < cfg.n_layers; l++) {
    up32(&gw.rms_att[l], w.rms_att[l], cfg.dim);
    up16(&gw.wq[l], w.wq[l], (size_t)cfg.dim * cfg.dim);
    up16(&gw.wk[l], w.wk[l], (size_t)kv_dim * cfg.dim);
    up16(&gw.wv[l], w.wv[l], (size_t)kv_dim * cfg.dim);
    up16(&gw.wo[l], w.wo[l], (size_t)cfg.dim * cfg.dim);
    up32(&gw.bq[l], w.bq[l], cfg.dim);
    up32(&gw.bk[l], w.bk[l], kv_dim);
    up32(&gw.bv[l], w.bv[l], kv_dim);
    up32(&gw.rms_ffn[l], w.rms_ffn[l], cfg.dim);
    up16(&gw.w1[l], w.w1[l], (size_t)cfg.hidden_dim * cfg.dim);
    up16(&gw.w2[l], w.w2[l], (size_t)cfg.dim * cfg.hidden_dim);
    up16(&gw.w3[l], w.w3[l], (size_t)cfg.hidden_dim * cfg.dim);
  }
}

static void free_gpu_weights(GPUWeights& gw, const Config& cfg) {
  cudaFree(gw.token_embedding);
  cudaFree(gw.rms_final);
  cudaFree(gw.wcls);
  for (int l = 0; l < cfg.n_layers; l++) {
    cudaFree(gw.rms_att[l]);
    cudaFree(gw.wq[l]);
    cudaFree(gw.wk[l]);
    cudaFree(gw.wv[l]);
    cudaFree(gw.wo[l]);
    cudaFree(gw.bq[l]);
    cudaFree(gw.bk[l]);
    cudaFree(gw.bv[l]);
    cudaFree(gw.rms_ffn[l]);
    cudaFree(gw.w1[l]);
    cudaFree(gw.w2[l]);
    cudaFree(gw.w3[l]);
  }
  delete[] gw.rms_att;
  delete[] gw.wq;
  delete[] gw.wk;
  delete[] gw.wv;
  delete[] gw.wo;
  delete[] gw.bq;
  delete[] gw.bk;
  delete[] gw.bv;
  delete[] gw.rms_ffn;
  delete[] gw.w1;
  delete[] gw.w2;
  delete[] gw.w3;
}

// ============================================================================
// GPURunState 分配/释放
// ============================================================================
static void alloc_run_state(GPURunState& s, const Config& cfg, int max_batch,
                            int max_blocks_per_seq, int max_flat_tokens) {
  int dim = cfg.dim;
  int kv_dim = cfg.n_kv_heads * (cfg.dim / cfg.n_heads);

  // flat batch 主缓冲区
  CHECK_CUDA(cudaMalloc(&s.x, (size_t)max_flat_tokens * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb, (size_t)max_flat_tokens * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb2, (size_t)max_flat_tokens * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.q, (size_t)max_flat_tokens * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.k, (size_t)max_flat_tokens * kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.v, (size_t)max_flat_tokens * kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb, (size_t)max_flat_tokens * cfg.hidden_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb2, (size_t)max_flat_tokens * cfg.hidden_dim * sizeof(__half)));

  // 元数据
  CHECK_CUDA(cudaMalloc(&s.d_tokens, max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_positions, max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_token_seq, max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_slot_map, max_flat_tokens * sizeof(int)));

  // block table
  CHECK_CUDA(cudaMalloc(&s.block_table, (size_t)max_batch * max_blocks_per_seq * sizeof(int)));

  // logits
  CHECK_CUDA(cudaMalloc(&s.d_last_tok_idx, max_batch * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.logits_last_tok, (size_t)max_batch * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.logits_fp16, (size_t)max_batch * cfg.vocab_size * sizeof(__half)));
  CHECK_CUDA(cudaMallocHost(&s.logits, (size_t)max_batch * cfg.vocab_size * sizeof(float)));

  // prefill 专用
  CHECK_CUDA(cudaMalloc(&s.d_prefill_flat, max_flat_tokens * sizeof(int)));

  // decode 专用
  CHECK_CUDA(cudaMalloc(&s.d_decode_flat, max_batch * sizeof(int)));
}

static void free_run_state(GPURunState& s) {
  cudaFree(s.x);
  cudaFree(s.xb);
  cudaFree(s.xb2);
  cudaFree(s.q);
  cudaFree(s.k);
  cudaFree(s.v);
  cudaFree(s.hb);
  cudaFree(s.hb2);

  cudaFree(s.d_tokens);
  cudaFree(s.d_positions);
  cudaFree(s.d_token_seq);
  cudaFree(s.d_slot_map);
  cudaFree(s.block_table);

  cudaFree(s.d_last_tok_idx);
  cudaFree(s.logits_last_tok);
  cudaFree(s.logits_fp16);
  cudaFreeHost(s.logits);

  cudaFree(s.d_prefill_flat);
  cudaFree(s.d_decode_flat);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__device__ float warp_reduce_sum(float val) {
  for (int off = 16; off > 0; off >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, off);
  }
  return val;
}

__device__ float warp_reduce_max(float val) {
  for (int off = 16; off > 0; off >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, off));
  }
  return val;
}

// Embedding lookup: out[i] = table[tokens[i]]
__global__ void embedding_kernel(__half* out, const __half* table, const int* tokens, int n,
                                 int dim) {
  int i = blockIdx.x;
  if (i >= n) {
    return;
  }

  // 每个线程计算一个 dim
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    out[i * dim + d] = table[tokens[i] * dim + d];
  }
}

// RMSNorm: out[i] = x[i] / rms(x[i]) * weight
__global__ void rmsnorm_kernel(__half* out, const __half* x, const float* weight, int dim) {
  __shared__ float ws[8];
  int tok = blockIdx.x, tid = threadIdx.x;
  int wid = tid / 32, lid = tid % 32;

  const __half* xr = x + tok * dim;
  __half* or_ = out + tok * dim;

  // 每个线程计算一个 dim
  float ss = 0.0f;
  for (int i = tid; i < dim; i += blockDim.x) {
    float xi = __half2float(xr[i]);
    ss += xi * xi;
  }
  // 块内规约
  ss = warp_reduce_sum(ss);
  if (lid == 0) {
    ws[wid] = ss;
  }
  __syncthreads();
  // 块间规约
  if (wid == 0) {
    float v = (lid < 8) ? ws[lid] : 0.0f;
    v = warp_reduce_sum(v);
    if (lid == 0) {
      ws[0] = v;
    }
  }
  __syncthreads();

  // 每个线程处理一个 dim
  float norm = rsqrtf(ws[0] / dim + 1e-6f);
  for (int i = tid; i < dim; i += blockDim.x) {
    or_[i] = __float2half(__half2float(xr[i]) * norm * weight[i]);
  }
}

// Add bias in-place: out[i] += bias
__global__ void add_bias_kernel(__half* out, const float* bias, int n_tok, int n) {
  int tok = blockIdx.x;
  if (tok >= n_tok) {
    return;
  }

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[tok * n + i] = __float2half(__half2float(out[tok * n + i]) + bias[i]);
  }
}

// RoPE: apply rotary position embedding, each token has its own pos
__global__ void rope_kernel(__half* q, __half* k, const int* positions, int n_tok, int dim,
                            int kv_dim, int head_dim, float freq_base) {
  int tok = blockIdx.x;
  if (tok >= n_tok) {
    return;
  }

  int pos = positions[tok];
  __half* qt = q + tok * dim;
  __half* kt = k + tok * kv_dim;

  // 每个线程处理一个 dim
  for (int i = threadIdx.x; i < dim / 2; i += blockDim.x) {
    int idx = i * 2, pair = (idx % head_dim) / 2;
    float theta = 1.0f / powf(freq_base, 2.0f * pair / head_dim);
    float c = cosf(theta * pos), s = sinf(theta * pos);
    float q0 = __half2float(qt[idx]), q1 = __half2float(qt[idx + 1]);
    qt[idx] = __float2half(q0 * c - q1 * s);
    qt[idx + 1] = __float2half(q0 * s + q1 * c);
  }

  // 每个线程处理一个 dim
  for (int i = threadIdx.x; i < kv_dim / 2; i += blockDim.x) {
    int idx = i * 2, pair = (idx % head_dim) / 2;
    float theta = 1.0f / powf(freq_base, 2.0f * pair / head_dim);
    float c = cosf(theta * pos), s = sinf(theta * pos);
    float k0 = __half2float(kt[idx]), k1 = __half2float(kt[idx + 1]);
    kt[idx] = __float2half(k0 * c - k1 * s);
    kt[idx + 1] = __float2half(k0 * s + k1 * c);
  }
}

// KV cache write: slot_map[tok] = absolute slot index
// k_cache layout: [total_slots, n_layers, kv_dim]
__global__ void kvcache_write_kernel(__half* k_cache, __half* v_cache, const __half* k,
                                     const __half* v, const int* slot_map, int n_tok, int layer,
                                     int n_layers, int kv_dim) {
  int tok = blockIdx.x;
  if (tok >= n_tok) {
    return;
  }

  // 获取该 token 对应的物理 slot
  int slot = slot_map[tok];
  size_t off = (size_t)slot * n_layers * kv_dim + (size_t)layer * kv_dim;

  // 每个线程处理一个 dim
  for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
    k_cache[off + i] = k[tok * kv_dim + i];
    v_cache[off + i] = v[tok * kv_dim + i];
  }
}

// Residual add: x += delta
__global__ void residual_kernel(__half* x, const __half* delta, int n_tok, int dim) {
  int tok = blockIdx.x;
  if (tok >= n_tok) {
    return;
  }

  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    int idx = tok * dim + i;
    x[idx] = __float2half(__half2float(x[idx]) + __half2float(delta[idx]));
  }
}

// SwiGLU: hb[i] = silu(hb[i]) * hb2[i]
__global__ void swiglu_kernel(__half* hb, const __half* hb2, int n_tok, int hidden_dim) {
  int tok = blockIdx.x;
  if (tok >= n_tok) {
    return;
  }

  for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    int idx = tok * hidden_dim + i;
    float x = __half2float(hb[idx]);
    hb[idx] = __float2half(x / (1.0f + expf(-x)) * __half2float(hb2[idx]));
  }
}

// Extract last token feature for each request
__global__ void extract_last_token_kernel(__half* out, const __half* in, const int* last_idx,
                                          int batch_size, int dim) {
  int b = blockIdx.x;
  if (b >= batch_size) {
    return;
  }

  int flat = last_idx[b];
  if (flat < 0) {
    return;
  }

  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    out[b * dim + i] = in[flat * dim + i];
  }
}

// fp16 -> fp32 for logits
__global__ void fp16_to_fp32_kernel(float* out, const __half* in, int batch_size, int vocab_size) {
  int b = blockIdx.x;
  if (b >= batch_size) {
    return;
  }

  for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
    out[b * vocab_size + i] = __half2float(in[b * vocab_size + i]);
  }
}

constexpr int FA_BLOCK = 256;

/**
 * Prefill attention batch 版本（Ragged Attention）
 * Grid: dim3(n_heads, total_prefill_tokens)
 * 一次调用处理所有请求的所有 prefill token
 */
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
    int head_dim, int kv_mul) {
  extern __shared__ float sm[];
  float* s = sm;
  float* o = sm + FA_BLOCK;
  float* wm = o + head_dim;
  float* wd = wm + 8;
  float* gm = wd + 8;
  float* gd = gm + 1;

  int h = blockIdx.x;
  // prefill token 编号（0..total_prefill_tokens-1）
  int p_tok = blockIdx.y;
  int tid = threadIdx.x;
  int wid = tid / 32;
  int lid = tid % 32;
  int kv_head = h / kv_mul;
  float scale = rsqrtf((float)head_dim);

  // 通过 prefill_flat_idx 找到在 flat batch 里的真实位置
  int q_tok = prefill_flat_idx[p_tok];
  int req = token_to_req[q_tok];
  int cur_pos = flat_positions[q_tok];
  int kv_end = cur_pos + 1;

  const int* bt = block_table + req * max_blocks_per_seq;
  const __half* qh = q + q_tok * dim + h * head_dim;

  // 初始化
  for (int d = tid; d < head_dim; d += blockDim.x) o[d] = 0.0f;
  if (tid == 0) {
    *gm = -CUDART_INF_F;
    *gd = 0.0f;
  }
  __syncthreads();

  // 分块遍历历史 KV，online softmax
  for (int st = 0; st < kv_end; st += FA_BLOCK) {
    int len = min(FA_BLOCK, kv_end - st);
    int t = st + tid;

    if (t < kv_end) {
      int phy = bt[t / block_size];
      int slot = phy * block_size + (t % block_size);
      const __half* kh =
          k_cache + (size_t)slot * n_layers * kv_dim + (size_t)layer * kv_dim + kv_head * head_dim;
      float sc = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        sc += __half2float(qh[d]) * __half2float(kh[d]);
      }
      s[tid] = sc * scale;
    } else {
      s[tid] = -CUDART_INF_F;
    }
    __syncthreads();

    // warp reduce max
    float lm = warp_reduce_max(s[tid]);
    if (lid == 0) wm[wid] = lm;
    __syncthreads();
    if (wid == 0) {
      float v = warp_reduce_max((lid < 8) ? wm[lid] : -CUDART_INF_F);
      if (lid == 0) wm[0] = v;
    }
    __syncthreads();

    float mn = fmaxf(*gm, wm[0]);
    float cor = expf(*gm - mn);

    // warp reduce sum
    float ld = warp_reduce_sum((t < kv_end) ? expf(s[tid] - mn) : 0.0f);
    if (lid == 0) wd[wid] = ld;
    __syncthreads();
    if (wid == 0) {
      float v = warp_reduce_sum((lid < 8) ? wd[lid] : 0.0f);
      if (lid == 0) wd[0] = v;
    }
    __syncthreads();

    if (tid == 0) {
      *gd = (*gd) * cor + wd[0];
      *gm = mn;
    }
    __syncthreads();

    // 累积 attention 输出
    for (int d = tid; d < head_dim; d += blockDim.x) {
      o[d] *= cor;
      for (int i = 0; i < len; i++) {
        int tt = st + i;
        int phy = bt[tt / block_size];
        int slot = phy * block_size + (tt % block_size);
        const __half* vh = v_cache + (size_t)slot * n_layers * kv_dim + (size_t)layer * kv_dim +
                           kv_head * head_dim;
        o[d] += expf(s[i] - mn) * __half2float(vh[d]);
      }
    }
    __syncthreads();
  }

  // 写回结果
  __half* oh = out + q_tok * dim + h * head_dim;
  float dg = *gd;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    oh[d] = __float2half(o[d] / dg);
  }
}

/**
 * Decode attention (flash attention, PagedAttention, batch)
 * Grid: dim3(n_heads, n_decode)
 * 直接读写 flat batch 的 q 和 xb，不需要 gather/scatter
 * 通过 dec_flat_idx 找到 flat batch 里的位置，复用 d_token_seq 和 d_positions
 */
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
    int kv_mul, int n_layers) {
  extern __shared__ float sm[];
  float* s = sm;
  float* o = sm + FA_BLOCK;
  float* wm = o + head_dim;
  float* wd = wm + 8;
  float* gm = wd + 8;
  float* gd = gm + 1;

  int h = blockIdx.x;
  int b = blockIdx.y;
  int tid = threadIdx.x;
  int wid = tid / 32;
  int lid = tid % 32;
  int kv_head = h / kv_mul;
  float scale = rsqrtf((float)head_dim);

  // 通过 dec_flat_idx 找到 flat batch 里的位置
  // 然后复用 d_token_seq 和 d_positions，不需要专用 buffer
  int q_tok = dec_flat_idx[b];
  int req = token_to_req[q_tok];
  int pos = flat_positions[q_tok];
  int kv_end = pos + 1;

  const __half* qh = q + q_tok * dim + h * head_dim;
  __half* oh = out + q_tok * dim + h * head_dim;
  const int* bt = block_table + req * max_blocks_per_seq;

  for (int d = tid; d < head_dim; d += blockDim.x) o[d] = 0.0f;
  if (tid == 0) {
    *gm = -CUDART_INF_F;
    *gd = 0.0f;
  }
  __syncthreads();

  for (int st = 0; st < kv_end; st += FA_BLOCK) {
    int len = min(FA_BLOCK, kv_end - st);
    int t = st + tid;

    if (t < kv_end) {
      int phy = bt[t / block_size];
      int slot = phy * block_size + (t % block_size);
      const __half* kh =
          k_cache + (size_t)slot * n_layers * kv_dim + (size_t)layer * kv_dim + kv_head * head_dim;
      float sc = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        sc += __half2float(qh[d]) * __half2float(kh[d]);
      }
      s[tid] = sc * scale;
    } else {
      s[tid] = -CUDART_INF_F;
    }
    __syncthreads();

    float lm = warp_reduce_max(s[tid]);
    if (lid == 0) wm[wid] = lm;
    __syncthreads();
    if (wid == 0) {
      float v = warp_reduce_max((lid < 8) ? wm[lid] : -CUDART_INF_F);
      if (lid == 0) wm[0] = v;
    }
    __syncthreads();

    float mn = fmaxf(*gm, wm[0]);
    float cor = expf(*gm - mn);
    float ld = warp_reduce_sum((t < kv_end) ? expf(s[tid] - mn) : 0.0f);
    if (lid == 0) wd[wid] = ld;
    __syncthreads();
    if (wid == 0) {
      float v = warp_reduce_sum((lid < 8) ? wd[lid] : 0.0f);
      if (lid == 0) wd[0] = v;
    }
    __syncthreads();

    if (tid == 0) {
      *gd = (*gd) * cor + wd[0];
      *gm = mn;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
      o[d] *= cor;
      for (int i = 0; i < len; i++) {
        int tt = st + i;
        int phy = bt[tt / block_size];
        if (phy < 0) continue;
        int slot = phy * block_size + (tt % block_size);
        const __half* vh = v_cache + (size_t)slot * n_layers * kv_dim + (size_t)layer * kv_dim +
                           kv_head * head_dim;
        o[d] += expf(s[i] - mn) * __half2float(vh[d]);
      }
    }
    __syncthreads();
  }

  float dg = *gd;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    oh[d] = __float2half(o[d] / dg);
  }
}

// ============================================================================
// cuBLAS matmul 封装
// ============================================================================
static void matmul(cublasHandle_t h, __half* out, const __half* x, const __half* w, int n, int d,
                   int batch = 1) {
  const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
  CHECK_CUBLAS(
      cublasHgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, d, batch, n, &alpha, w, n, x, n, &beta, out, d));
}

// ============================================================================
// GPUDecoder 实现
// ============================================================================
GPUDecoder::GPUDecoder(const std::string& model_file, int max_batch, int max_prefill_len,
                       int max_prefill_tokens_per_step, int max_total_tokens)
    : max_batch_(max_batch),
      max_prefill_len_(max_prefill_len),
      max_prefill_tokens_per_step_(max_prefill_tokens_per_step) {
  max_flat_tokens_ = std::max(max_total_tokens, max_prefill_tokens_per_step + max_batch);
  if (load_gguf(gguf, model_file) != 0) {
    fprintf(stderr, "load gguf failed\n");
    exit(1);
  }
  if (load_config(config, gguf) != 0) {
    fprintf(stderr, "load config failed\n");
    exit(1);
  }
  if (open_model(model_file, mf) != 0) {
    fprintf(stderr, "open model failed\n");
    exit(1);
  }
  if (load_weights(w, config, gguf, mf) != 0) {
    fprintf(stderr, "load weights failed\n");
    exit(1);
  }
  if (load_tokenizer(tokenizer, gguf) != 0) {
    fprintf(stderr, "load tokenizer failed\n");
    exit(1);
  }

  upload_weights(gw, w, config);

  max_blocks_per_seq_ = (config.seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  alloc_run_state(gs, config, max_batch_, max_blocks_per_seq_, max_flat_tokens_);

  // 权重和运行时缓冲区分配完后，查询剩余显存给 BlockPool
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  size_t budget = (size_t)(free_mem * 0.8);

  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
  size_t pg_sz = (size_t)BLOCK_SIZE * config.n_layers * kv_dim * 2 * sizeof(__half);
  int total_pgs = (int)(budget / pg_sz);

  fprintf(stderr, "GPU: total=%.2fGB free=%.2fGB kv_budget=%.2fGB blocks=%d\n",
          (float)total_mem / 1024 / 1024 / 1024, (float)free_mem / 1024 / 1024 / 1024,
          (float)budget / 1024 / 1024 / 1024, total_pgs);

  block_pool = new BlockPool(total_pgs, BLOCK_SIZE, config.n_layers, kv_dim);

  CHECK_CUBLAS(cublasCreate(&cublas_handle));
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));
}

GPUDecoder::~GPUDecoder() {
  cublasDestroy(cublas_handle);
  cudaStreamDestroy(stream);
  free_run_state(gs);
  free_gpu_weights(gw, config);
  free_weights(w);
  close_model(mf);
  delete block_pool;
}

float* GPUDecoder::get_logits_batch(int batch_idx) {
  cudaStreamSynchronize(stream);
  return gs.logits + (size_t)batch_idx * config.vocab_size;
}

void GPUDecoder::update_block_table_partial(const std::vector<Request*>& running,
                                            const std::vector<int>& changed, int max_blk) {
  for (int i : changed) {
    const Request* req = running[i];
    int n = (req != nullptr) ? (int)req->block_table.num_blocks() : 0;
    if (n == 0) {
      continue;
    }
    /**
     * 只上传实际用到的 n 个 block，不上传全部 max_blk 个
     * 节省 HtoD 带宽：实际用 n*4 bytes，全量上传需要 max_blk*4 bytes
     * 安全性:
     *  - attention kernel 只访问 0 到 pos/block_size 的逻辑块范围
     *  - ensure_pages 保证这个范围内的所有块都已分配物理块
     *  - 未分配的块（下标 >= n）不会被访问到，无需初始化
     */
    CHECK_CUDA(cudaMemcpyAsync(gs.block_table + i * max_blk, req->block_table.data(),
                               n * sizeof(int), cudaMemcpyHostToDevice, stream));
  }
}

void GPUDecoder::forward_flat(
    const std::vector<FlatRequest>& flat_reqs,
    const std::vector<int>& flat_tokens,       // 所有请求的 token id，按 flat 顺序打包
    const std::vector<int>& flat_positions,    // 每个 token 的绝对位置，用于 RoPE
    const std::vector<int>& token_to_req,      // 每个 token 属于哪个请求
    const std::vector<int>& token_slot,        // 每个 token 的绝对物理槽位，KV cache 写入用
    const std::vector<int>& last_tok_idx,      // 每个请求最后一个 token 在 flat batch 里的位置
    const std::vector<int>& prefill_flat_idx,  // decode token 在 flat batch 里的位置
    const std::vector<int>& dec_flat_idx,      // decode token 在 flat batch 里的位置
    int total_tokens)                          // flat batch 里的 token 总数
{
  int dim = config.dim;
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
  int kv_mul = config.n_heads / config.n_kv_heads;
  int T = 256;  // threads per block
  int batch = (int)flat_reqs.size();
  int n_prefill = (int)prefill_flat_idx.size();
  int n_dec = (int)dec_flat_idx.size();

  // 上传元数据
  CHECK_CUDA(cudaMemcpyAsync(gs.d_tokens, flat_tokens.data(), total_tokens * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(gs.d_positions, flat_positions.data(), total_tokens * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(gs.d_token_seq, token_to_req.data(), total_tokens * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(gs.d_slot_map, token_slot.data(), total_tokens * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(gs.d_last_tok_idx, last_tok_idx.data(), batch * sizeof(int),
                             cudaMemcpyHostToDevice, stream));

  if (n_prefill > 0) {
    CHECK_CUDA(cudaMemcpyAsync(gs.d_prefill_flat, prefill_flat_idx.data(),
                               prefill_flat_idx.size() * sizeof(int), cudaMemcpyHostToDevice,
                               stream));
  }

  if (n_dec > 0) {
    CHECK_CUDA(cudaMemcpyAsync(gs.d_decode_flat, dec_flat_idx.data(), n_dec * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
  }

  /**
   * Embedding
   * 每个 token 一个 block，每个 block 256 个线程
   *
   * IN:
   * - gs.d_tokens
   * OUT:
   * - gs.x
   */
  embedding_kernel<<<total_tokens, T, 0, stream>>>(gs.x, gw.token_embedding, gs.d_tokens,
                                                   total_tokens, dim);
  CHECK_KERNEL();

  size_t smem = (FA_BLOCK + head_dim + 8 + 8 + 2) * sizeof(float);

  for (int l = 0; l < config.n_layers; l++) {
    /**
     * RMSNorm (attention)
     * 每个 token 一个 block，每个 block 256 个线程
     *
     * IN:
     * - gs.x
     * OUT:
     * - gs.xb
     */
    rmsnorm_kernel<<<total_tokens, T, 0, stream>>>(gs.xb, gs.x, gw.rms_att[l], dim);
    CHECK_KERNEL();

    /**
     * QKV projection
     *
     * IN:
     * - gs.xb
     * OUT:
     * - gs.q
     * - gs.k
     * - gs.v
     */
    matmul(cublas_handle, gs.q, gs.xb, gw.wq[l], dim, dim, total_tokens);
    matmul(cublas_handle, gs.k, gs.xb, gw.wk[l], dim, kv_dim, total_tokens);
    matmul(cublas_handle, gs.v, gs.xb, gw.wv[l], dim, kv_dim, total_tokens);

    /**
     * QKV bias
     * 每个 token 一个 block，每个 block 256 个线程
     *
     * IN:
     * - gs.q, gs.k, gs.v
     * OUT:
     * - gs.q, gs.k, gs.v
     */
    add_bias_kernel<<<total_tokens, T, 0, stream>>>(gs.q, gw.bq[l], total_tokens, dim);
    add_bias_kernel<<<total_tokens, T, 0, stream>>>(gs.k, gw.bk[l], total_tokens, kv_dim);
    add_bias_kernel<<<total_tokens, T, 0, stream>>>(gs.v, gw.bv[l], total_tokens, kv_dim);
    CHECK_KERNEL();

    /**
     * RoPE
     * 每个 token 一个 block，每个 block 256 个线程
     *
     * IN:
     * - gs.q, gs.k
     * OUT:
     * - gs.q, gs.k
     */
    rope_kernel<<<total_tokens, T, 0, stream>>>(gs.q, gs.k, gs.d_positions, total_tokens, dim,
                                                kv_dim, head_dim, config.rope_freq_base);
    CHECK_KERNEL();

    /**
     * KV cache write (1D slot mapping)
     * 每个 token 一个 block，每个 block 256 个线程
     *
     * IN:
     * - gs.k
     * - gs.v
     *
     * OUT:
     * - k_cache
     * - v_cache
     */
    kvcache_write_kernel<<<total_tokens, T, 0, stream>>>(
        block_pool->get_k_cache(), block_pool->get_v_cache(), gs.k, gs.v, gs.d_slot_map,
        total_tokens, l, config.n_layers, kv_dim);
    CHECK_KERNEL();

    if (n_prefill > 0) {
      /**
       * 每个 block 处理一个 head, 每个 block 256 个线程
       *
       * IN:
       * - q_fr (来自 gs.q)
       * OUT:
       * - o_fr (来自 gs.xb)
       */
      dim3 grid(config.n_heads, prefill_flat_idx.size());
      prefill_attention_kernel<<<grid, T, smem, stream>>>(
          gs.q, block_pool->get_k_cache(), block_pool->get_v_cache(), gs.xb, gs.block_table,
          gs.d_token_seq, gs.d_positions, gs.d_prefill_flat, max_blocks_per_seq_, BLOCK_SIZE,
          config.n_layers, l, dim, kv_dim, head_dim, kv_mul);
      CHECK_KERNEL();
    }

    if (n_dec > 0) {
      /**
       * 每个 block 处理一个 head, 每个 head 256 个线程
       * IN:
       * - gs.q
       * OUT:
       * - gs.xb
       */
      dim3 grid_dec(config.n_heads, n_dec);
      decode_attention_kernel<<<grid_dec, T, smem, stream>>>(
          gs.q, block_pool->get_k_cache(), block_pool->get_v_cache(), gs.xb, gs.block_table,
          gs.d_token_seq, gs.d_positions, gs.d_decode_flat, l, max_blocks_per_seq_, BLOCK_SIZE, dim,
          kv_dim, head_dim, kv_mul, config.n_layers);
      CHECK_KERNEL();
    }

    /**
     * Output projection + residual
     *
     * IN:
     * - gs.xb
     * OUT:
     * - gs.xb2
     */
    matmul(cublas_handle, gs.xb2, gs.xb, gw.wo[l], dim, dim, total_tokens);
    /**
     * 残差
     * 启动 total_token 个块，每个块 256 个线程
     *
     * IN:
     * - gs.x
     * OUT:
     * - gs.x
     */
    residual_kernel<<<total_tokens, T, 0, stream>>>(gs.x, gs.xb2, total_tokens, dim);
    CHECK_KERNEL();

    /**
     * FFN RMSNorm
     * 启动 total_token 个块，每个块 256 个线程
     *
     * IN:
     * - gs.x
     * OUT:
     * - gs.xb
     */
    rmsnorm_kernel<<<total_tokens, T, 0, stream>>>(gs.xb, gs.x, gw.rms_ffn[l], dim);
    CHECK_KERNEL();

    // SwiGLU FFN
    /**
     * 升维
     *
     * IN:
     * - gs.xb
     * OUT:
     * - gs.hb
     */
    matmul(cublas_handle, gs.hb, gs.xb, gw.w1[l], dim, config.hidden_dim, total_tokens);
    /**
     * 升维
     *
     * IN:
     * - gs.xb
     * OUT:
     * - gs.hb2
     */
    matmul(cublas_handle, gs.hb2, gs.xb, gw.w3[l], dim, config.hidden_dim, total_tokens);
    /**
     * 激活
     *
     * IN:
     * - gs.hb
     * - gs.hb2
     * OUT:
     * - gs.hb
     */
    swiglu_kernel<<<total_tokens, T, 0, stream>>>(gs.hb, gs.hb2, total_tokens, config.hidden_dim);
    CHECK_KERNEL();

    /**
     * FFN output projection + residual
     * 降维
     *
     * IN:
     * - gs.hb
     * OUT:
     * - gs.xb2
     */
    matmul(cublas_handle, gs.xb2, gs.hb, gw.w2[l], config.hidden_dim, dim, total_tokens);
    /**
     * 残差
     * IN:
     * - gs.x
     * - gs.xb2
     * OUT:
     * - gs.x
     */
    residual_kernel<<<total_tokens, T, 0, stream>>>(gs.x, gs.xb2, total_tokens, dim);
    CHECK_KERNEL();
  }

  /**
   * Final RMSNorm (all tokens)
   * IN:
   * - gs.x
   * OUT:
   * - gs.xb
   */
  rmsnorm_kernel<<<total_tokens, T, 0, stream>>>(gs.xb, gs.x, gw.rms_final, dim);
  CHECK_KERNEL();

  /**
   * 从 flat batch 里提取每个请求最后一个 token 的特征，用于 logits 计算
   *
   * 推理时只需要最后一个 token 的 logits 来采样下一个 token：
   * - prefill 请求: 最后一个 prefill token 的输出
   * - decode  请求: 唯一的 decode token 的输出
   *
   * 启动 batch 个 block，每个 block 处理一个 token
   *
   * IN:
   * - gs.xb
   * - gs.d_last_tok_idx
   * OUT:
   * - gs.logits_last_tok
   */
  extract_last_token_kernel<<<batch, T, 0, stream>>>(gs.logits_last_tok, gs.xb, gs.d_last_tok_idx,
                                                     batch, dim);
  CHECK_KERNEL();

  /**
   * Logits
   * IN:
   * - logits_last_tok
   * OUT:
   * - logits_fp16
   */
  matmul(cublas_handle, gs.logits_fp16, gs.logits_last_tok, gw.wcls, dim, config.vocab_size, batch);

  // 精度转换
  fp16_to_fp32_kernel<<<batch, T, 0, stream>>>(gs.logits, gs.logits_fp16, batch, config.vocab_size);
  CHECK_KERNEL();
}

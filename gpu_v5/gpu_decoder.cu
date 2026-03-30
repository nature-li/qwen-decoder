#include <math_constants.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "gpu_decoder.h"

#define CHECK_CUDA(call)                                                      \
  {                                                                           \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), \
              __LINE__);                                                      \
      exit(1);                                                                \
    }                                                                         \
  }

#define CHECK_CUBLAS(call)                                                \
  {                                                                       \
    cublasStatus_t status = call;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                \
      fprintf(stderr, "cuBLAS Error: %d at line %d\n", status, __LINE__); \
      exit(1);                                                            \
    }                                                                     \
  }

#define CHECK_KERNEL()                                                   \
  {                                                                      \
    cudaError_t err = cudaGetLastError();                                \
    if (err != cudaSuccess) {                                            \
      fprintf(stderr, "Kernel error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                  \
      exit(1);                                                           \
    }                                                                    \
  }

// ============================================================================
// GPUWeights 上传/释放
// ============================================================================

static void upload_weights(GPUWeights& gw, const Weights& w,
                           const Config& config) {
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
  int vocab_size = config.vocab_size;

  // 上传 fp16（uint16_t* -> __half*）
  auto upload_fp16 = [](__half** dst, const uint16_t* src, size_t n) {
    CHECK_CUDA(cudaMalloc(dst, n * sizeof(__half)));
    CHECK_CUDA(
        cudaMemcpy(*dst, src, n * sizeof(__half), cudaMemcpyHostToDevice));
  };

  // 上传 fp32
  auto upload_fp32 = [](float** dst, const float* src, size_t n) {
    CHECK_CUDA(cudaMalloc(dst, n * sizeof(float)));
    CHECK_CUDA(
        cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
  };

  upload_fp16(&gw.token_embedding, w.token_embedding,
              (size_t)vocab_size * config.dim);
  upload_fp32(&gw.rms_final, w.rms_final, config.dim);
  upload_fp16(&gw.wcls, w.wcls, (size_t)vocab_size * config.dim);

  gw.rms_att = new float*[config.n_layers];
  gw.wq = new __half*[config.n_layers];
  gw.wk = new __half*[config.n_layers];
  gw.wv = new __half*[config.n_layers];
  gw.wo = new __half*[config.n_layers];
  gw.bq = new float*[config.n_layers];
  gw.bk = new float*[config.n_layers];
  gw.bv = new float*[config.n_layers];
  gw.rms_ffn = new float*[config.n_layers];
  gw.w1 = new __half*[config.n_layers];
  gw.w2 = new __half*[config.n_layers];
  gw.w3 = new __half*[config.n_layers];

  for (int l = 0; l < config.n_layers; l++) {
    upload_fp32(&gw.rms_att[l], w.rms_att[l], config.dim);
    upload_fp16(&gw.wq[l], w.wq[l], (size_t)config.dim * config.dim);
    upload_fp16(&gw.wk[l], w.wk[l], (size_t)kv_dim * config.dim);
    upload_fp16(&gw.wv[l], w.wv[l], (size_t)kv_dim * config.dim);
    upload_fp16(&gw.wo[l], w.wo[l], (size_t)config.dim * config.dim);
    upload_fp32(&gw.bq[l], w.bq[l], config.dim);
    upload_fp32(&gw.bk[l], w.bk[l], kv_dim);
    upload_fp32(&gw.bv[l], w.bv[l], kv_dim);
    upload_fp32(&gw.rms_ffn[l], w.rms_ffn[l], config.dim);
    upload_fp16(&gw.w1[l], w.w1[l], (size_t)config.hidden_dim * config.dim);
    upload_fp16(&gw.w2[l], w.w2[l], (size_t)config.dim * config.hidden_dim);
    upload_fp16(&gw.w3[l], w.w3[l], (size_t)config.hidden_dim * config.dim);
  }
}

static void free_gpu_weights(GPUWeights& gw, const Config& config) {
  cudaFree(gw.token_embedding);
  cudaFree(gw.rms_final);
  cudaFree(gw.wcls);

  for (int l = 0; l < config.n_layers; l++) {
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

static void alloc_gpu_run_state(GPURunState& s, const Config& config) {
  int dim = config.dim;
  int kv_dim = config.n_kv_heads * (config.dim / config.n_heads);
  int seq_len = config.seq_len;

  CHECK_CUDA(cudaMalloc(&s.x, dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb, dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb2, dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.q, dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.k, kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.v, kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb, config.hidden_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb2, config.hidden_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(
      &s.k_cache, (size_t)config.n_layers * seq_len * kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(
      &s.v_cache, (size_t)config.n_layers * seq_len * kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMallocHost(&s.logits, config.vocab_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&s.logits_fp16, config.vocab_size * sizeof(__half)));

  // prefill 缓冲区，最大支持 seq_len 个 token 同时处理
  int max_pre = config.seq_len;
  CHECK_CUDA(
      cudaMalloc(&s.x_pre, (size_t)max_pre * config.dim * sizeof(__half)));
  CHECK_CUDA(
      cudaMalloc(&s.xb_pre, (size_t)max_pre * config.dim * sizeof(__half)));
  CHECK_CUDA(
      cudaMalloc(&s.xb2_pre, (size_t)max_pre * config.dim * sizeof(__half)));
  CHECK_CUDA(
      cudaMalloc(&s.q_pre, (size_t)max_pre * config.dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.k_pre, (size_t)max_pre * kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.v_pre, (size_t)max_pre * kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb_pre,
                        (size_t)max_pre * config.hidden_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb2_pre,
                        (size_t)max_pre * config.hidden_dim * sizeof(__half)));
}

static void free_gpu_run_state(GPURunState& s) {
  cudaFree(s.x);
  cudaFree(s.xb);
  cudaFree(s.xb2);
  cudaFree(s.q);
  cudaFree(s.k);
  cudaFree(s.v);
  cudaFree(s.hb);
  cudaFree(s.hb2);
  cudaFree(s.k_cache);
  cudaFree(s.v_cache);
  cudaFreeHost(s.logits);
  cudaFree(s.logits_fp16);
  cudaFree(s.x_pre);
  cudaFree(s.xb_pre);
  cudaFree(s.xb2_pre);
  cudaFree(s.q_pre);
  cudaFree(s.k_pre);
  cudaFree(s.v_pre);
  cudaFree(s.hb_pre);
  cudaFree(s.hb2_pre);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * Embedding lookup kernel (fp16 -> __half)
 * Grid:  (dim + 255) / 256 个 block
 * Block: 256 个线程
 * 每个 thread 负责 out 的 1 个元素
 */
__global__ void embedding_kernel(__half* out, const __half* table, int token,
                                 int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) out[i] = table[token * dim + i];
}

/**
 * RMSNorm kernel (warp reduce)
 * 输入 x 是 fp16，权重是 fp32，输出是 fp16
 * Grid:  1 个 block
 * Block: 256 个线程
 */
__device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void rmsnorm_kernel(__half* out, const __half* x,
                               const float* weight, int dim) {
  __shared__ float warp_sums[8];
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  float local_sum = 0.0f;
  for (int i = tid; i < dim; i += blockDim.x) {
    float xi = __half2float(x[i]);
    local_sum += xi * xi;
  }

  local_sum = warp_reduce_sum(local_sum);
  if (lane_id == 0) warp_sums[warp_id] = local_sum;
  __syncthreads();

  if (warp_id == 0) {
    float val = (lane_id < 8) ? warp_sums[lane_id] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane_id == 0) warp_sums[0] = val;
  }
  __syncthreads();

  float norm = rsqrtf(warp_sums[0] / dim + 1e-6f);
  for (int i = tid; i < dim; i += blockDim.x) {
    out[i] = __float2half(__half2float(x[i]) * norm * weight[i]);
  }
}

/**
 * Add bias kernel (fp32 bias 加到 fp16 向量上)
 * Grid:  (n + 255) / 256 个 block
 * Block: 256 个线程
 */
__global__ void add_bias_kernel(__half* out, const float* bias, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(__half2float(out[i]) + bias[i]);
}

/**
 * RoPE kernel (全 fp16)
 * Grid:  (dim/2 + 255) / 256 个 block
 * Block: 256 个线程
 */
__global__ void rope_kernel(__half* q, __half* k, int pos, int dim, int kv_dim,
                            int head_dim, float rope_freq_base) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;

  if (idx < dim) {
    int pair = (idx % head_dim) / 2;
    float theta = 1.0f / powf(rope_freq_base, 2.0f * pair / head_dim);
    float cos_v = cosf(theta * pos);
    float sin_v = sinf(theta * pos);
    float q0 = __half2float(q[idx]), q1 = __half2float(q[idx + 1]);
    q[idx] = __float2half(q0 * cos_v - q1 * sin_v);
    q[idx + 1] = __float2half(q0 * sin_v + q1 * cos_v);
  }

  if (idx < kv_dim) {
    int pair = (idx % head_dim) / 2;
    float theta = 1.0f / powf(rope_freq_base, 2.0f * pair / head_dim);
    float cos_v = cosf(theta * pos);
    float sin_v = sinf(theta * pos);
    float k0 = __half2float(k[idx]), k1 = __half2float(k[idx + 1]);
    k[idx] = __float2half(k0 * cos_v - k1 * sin_v);
    k[idx + 1] = __float2half(k0 * sin_v + k1 * cos_v);
  }
}

/**
 * KV Cache 写入 kernel (fp16)
 * Grid:  (kv_dim + 255) / 256 个 block
 * Block: 256 个线程
 */
__global__ void kvcache_write_kernel(__half* k_cache, __half* v_cache,
                                     const __half* k, const __half* v,
                                     int layer, int pos, int seq_len,
                                     int kv_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kv_dim) {
    int offset = layer * seq_len * kv_dim + pos * kv_dim + i;
    k_cache[offset] = k[i];
    v_cache[offset] = v[i];
  }
}

/**
 * warp 内归约：找最大值
 * 256 个线程 = 8 个 warp，每个 warp 内 32 个线程同时执行
 * 执行完后每个 warp 的 lane 0 持有该 warp 内的最大值
 */
__device__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  return val;
}

/**
 * Flash Attention Kernel
 * Grid: n_heads 个 block
 * Block: 256 个线程
 */
// 每次处理的 KV 数量，根据 shared memory 大小调整
constexpr int FA_BLOCK_SIZE = 64;
__global__ void attention_kernel(const __half* q, const __half* k_cache,
                                 const __half* v_cache, __half* out, int pos,
                                 int seq_len, int kv_dim, int head_dim,
                                 int kv_mul) {
  extern __shared__ float smem[];
  float* s_block = smem;                 // [FA_BLOCK_SIZE] 块内 score
  float* o_smem = smem + FA_BLOCK_SIZE;  // [head_dim] 累计输出
  float* warp_m = o_smem + head_dim;     // [8]
  float* warp_d = warp_m + 8;            // [8]
  float* g_m = warp_d + 8;               // [1] 全局 m
  float* g_d = g_m + 1;                  // [1] 全局 d

  int h = blockIdx.x;
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  float scale = rsqrtf((float)head_dim);

  // Q 的 多个 head 对应 k/v 的一个 head
  // 这里计算的是第 h 个 Q head 用哪个 k/v head
  int kv_head = h / kv_mul;
  const __half* q_head = q + h * head_dim;

  // 初始化
  for (int d = tid; d < head_dim; d += blockDim.x) {
    o_smem[d] = 0.0f;
  }
  if (tid == 0) {
    *g_m = -CUDART_INF_F;
    *g_d = 0.0f;
  }
  __syncthreads();

  // 分块处理 k/v
  for (int start = 0; start <= pos; start += FA_BLOCK_SIZE) {
    int end = min(start + FA_BLOCK_SIZE, pos + 1);
    int len = end - start;

    // 1. 并行计算块内 score, 每个线程负现一个 t
    for (int i = tid; i < len; i += blockDim.x) {
      int t = start + i;
      // 跳过前 t 个 token, 再跳前过 kv_head
      const __half* k_head = k_cache + t * kv_dim + kv_head * head_dim;
      float s = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        s += __half2float(q_head[d]) * __half2float(k_head[d]);
      }
      s_block[i] = s * scale;
    }
    __syncthreads();

    // 2. 块内规约最大值
    float local_max = -CUDART_INF_F;
    for (int i = tid; i < len; i += blockDim.x) {
      local_max = fmax(local_max, s_block[i]);
    }
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) {
      warp_m[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
      float val = (lane_id < 8) ? warp_m[lane_id] : -CUDART_INF_F;
      val = warp_reduce_max(val);
      if (lane_id == 0) {
        warp_m[0] = val;
      }
    }
    __syncthreads();

    // 3.online softmax 更新 m 和 d
    float m_new = fmaxf(*g_m, warp_m[0]);
    float correction = expf(*g_m - m_new);

    float local_d = 0.0f;
    for (int i = tid; i < len; i += blockDim.x) {
      local_d += expf(s_block[i] - m_new);
    }
    local_d = warp_reduce_sum(local_d);
    if (lane_id == 0) {
      warp_d[warp_id] = local_d;
    }
    __syncthreads();

    if (warp_id == 0) {
      float val = (lane_id < 8) ? warp_d[lane_id] : 0.0f;
      val = warp_reduce_sum(val);
      if (lane_id == 0) {
        warp_d[0] = val;
      }
    }
    __syncthreads();

    if (tid == 0) {
      *g_d = (*g_d) * correction + warp_d[0];
      *g_m = m_new;
    }
    __syncthreads();

    // 4.更新输出 o = o * correction + exp(s - m_new) @ v_block
    for (int d = tid; d < head_dim; d += blockDim.x) {
      o_smem[d] *= correction;
      for (int i = 0; i < len; i++) {
        int t = start + i;
        const __half* v_head = v_cache + t * kv_dim + kv_head * head_dim;
        o_smem[d] += expf(s_block[i] - m_new) * __half2float(v_head[d]);
      }
    }
    __syncthreads();
  }

  // 5.归一化，写回
  __half* out_head = out + h * head_dim;
  float d_global = *g_d;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    out_head[d] = __float2half(o_smem[d] / d_global);
  }
}

/**
 * SwiGLU kernel (fp16)
 * Grid:  (hidden_dim + 255) / 256 个 block
 * Block: 256 个线程
 */
__global__ void swiglu_kernel(__half* hb, const __half* hb2, int hidden_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden_dim) {
    float x = __half2float(hb[i]);
    hb[i] = __float2half(x * (1.0f / (1.0f + expf(-x))) * __half2float(hb2[i]));
  }
}

/**
 * Residual add kernel (fp16)
 * Grid:  (dim + 255) / 256 个 block
 * Block: 256 个线程
 */
__global__ void residual_kernel(__half* x, const __half* delta, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) x[i] = __float2half(__half2float(x[i]) + __half2float(delta[i]));
}

/**
 * 最终 logits 输出：fp16 -> fp32
 * Grid:  (vocab_size + 255) / 256 个 block
 * Block: 256 个线程
 */
__global__ void fp16_to_fp32_kernel(float* out, const __half* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __half2float(in[i]);
}

/**
 * Flash Attention Kernel (prefill 阶段)
 * Grid: dim3(n_heads, n_tokens)
 * Block: 256 个线程
 *
 * q: [n_tokens, dim]
 * k_cache: [seq_len, kv_dim]
 * v_cache: [seq_len, kv_dim]
 * out: [n_tokens, dim]
 *
 * 每个 block 负责一个 (query_token, head) 对
 * casual mask: 第 q_tok 个 token 只能看 <= start_pos + q_tok 的 kv
 */
__global__ void flash_attention_prefill_kernel(
    const __half* q, const __half* k_cache, const __half* v_cache, __half* out,
    int start_pos, int seq_len, int dim, int kv_dim, int head_dim, int kv_mul) {
  extern __shared__ float smem[];
  float* s_block = smem;                 // [FA_BLOCK_SIZE]
  float* o_smem = smem + FA_BLOCK_SIZE;  //[head_dim]
  float* warp_m = o_smem + head_dim;     // [8]
  float* warp_d = warp_m + 8;            // [8]
  float* g_m = warp_d + 8;               // [1]
  float* g_d = g_m + 1;                  // [1]

  int h = blockIdx.x;      // head index
  int q_tok = blockIdx.y;  // query token index in this prefill batch
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  int kv_head = h / kv_mul;
  float scale = rsqrtf((float)head_dim);

  // 当前 query token 在整个序列里的绝对位置
  int cur_pos = start_pos + q_tok;

  // q 布局: [n_tokens, dim], dim = n_heads * head_dim
  const __half* q_head = q + q_tok * dim + h * head_dim;

  // 初始化
  for (int d = tid; d < head_dim; d += blockDim.x) {
    o_smem[d] = 0.0f;
  }
  if (tid == 0) {
    *g_m = -CUDART_INF_F;
    *g_d = 0.0f;
  }
  __syncthreads();

  // casual mask: 只看到 0 到 cur_pos 的 k/v
  int kv_end = cur_pos + 1;

  for (int start = 0; start < kv_end; start += FA_BLOCK_SIZE) {
    int end = min(start + FA_BLOCK_SIZE, kv_end);
    int len = end - start;

    // 1.计算块内 scores
    for (int i = tid; i < len; i += blockDim.x) {
      int t = start + i;
      const __half* k_head = k_cache + t * kv_dim + kv_head * head_dim;
      float s = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        s += __half2float(q_head[d]) * __half2float(k_head[d]);
      }
      s_block[i] = s * scale;
    }
    __syncthreads();

    // 2.块内最大规约
    float local_max = -CUDART_INF_F;
    for (int i = tid; i < len; i += blockDim.x) {
      local_max = fmaxf(local_max, s_block[i]);
    }
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) {
      warp_m[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
      float val = (lane_id < 8) ? warp_m[lane_id] : -CUDART_INF_F;
      val = warp_reduce_max(val);
      if (lane_id == 0) {
        warp_m[0] = val;
      }
    }
    __syncthreads();

    // 3.online softmax 更新 m 和 d
    float m_new = fmaxf(*g_m, warp_m[0]);
    float correction = expf(*g_m - m_new);

    float local_d = 0.0f;
    for (int i = tid; i < len; i += blockDim.x) {
      local_d += expf(s_block[i] - m_new);
    }
    local_d = warp_reduce_sum(local_d);
    if (lane_id == 0) {
      warp_d[warp_id] = local_d;
    }
    __syncthreads();

    if (warp_id == 0) {
      float val = (lane_id < 8) ? warp_d[lane_id] : 0.0f;
      val = warp_reduce_sum(val);
      if (lane_id == 0) {
        warp_d[0] = val;
      }
    }
    __syncthreads();

    if (tid == 0) {
      *g_d = (*g_d) * correction + warp_d[0];
      *g_m = m_new;
    }
    __syncthreads();

    // 4.更新输出
    for (int d = tid; d < head_dim; d += blockDim.x) {
      o_smem[d] *= correction;
      for (int i = 0; i < len; i++) {
        int t = start + i;
        const __half* v_head = v_cache + t * kv_dim + kv_head * head_dim;
        o_smem[d] += expf(s_block[i] - m_new) * __half2float(v_head[d]);
      }
    }
    __syncthreads();
  }

  // 5.归一化
  __half* out_head = out + q_tok * dim + h * head_dim;
  float d_global = *g_d;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    out_head[d] = __float2half(o_smem[d] / d_global);
  }
}

__global__ void embedding_kernel_batch(__half* out, const __half* table,
                                       const int* tokens, int n_tokens,
                                       int dim) {
  // 每个 block 负责 1 个 token
  int token_idx = blockIdx.x;
  if (token_idx > n_tokens) {
    return;
  }

  // block 内每个线程负责 blockDim 个 dim
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    out[token_idx * dim + d] = table[tokens[token_idx] * dim + d];
  }
}

/**
 * grid 大小是 n_tokens
 * x: shape[n_tokens, dim]
 * out: shape[n_tokens, dim]
 * 每个 block 处理一个 token
 */
__global__ void rmsnorm_kernel_batch(__half* out, const __half* x,
                                     const float* weight, int dim) {
  __shared__ float warp_sums[8];
  int token_idx = blockIdx.x;
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  // 计算当前 block 需要处理的 token
  const __half* x_row = x + token_idx * dim;
  __half* out_row = out + token_idx * dim;

  float local_sum = 0.0f;
  for (int i = tid; i < dim; i += blockDim.x) {
    float xi = __half2float(x_row[i]);
    local_sum += xi * xi;
  }
  local_sum = warp_reduce_sum(local_sum);
  if (lane_id == 0) {
    warp_sums[warp_id] = local_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    float val = (lane_id < 8) ? warp_sums[lane_id] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane_id == 0) {
      warp_sums[0] = val;
    }
  }
  __syncthreads();

  float norm = rsqrtf(warp_sums[0] / dim + 1e-6f);
  for (int i = tid; i < dim; i += blockDim.x) {
    out_row[i] = __float2half(__half2float(x_row[i]) * norm * weight[i]);
  }
}

__global__ void add_bias_kernel_batch(__half* out, const float* bias,
                                      int n_tokens, int n) {
  // per block per token
  int token_idx = blockIdx.x;
  if (token_idx > n_tokens) {
    return;
  }

  // per thread per elements
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[token_idx * n + i] =
        __float2half(__half2float(out[token_idx * n + i]) + bias[i]);
  }
}

/**
 * q: shape[n_tokens, dim]
 * k: [n_tokens, kv_dim]
 */
__global__ void rope_kernel_batch(__half* q, __half* k, int start_pos,
                                  int n_tokens, int dim, int kv_dim,
                                  int head_dim, float rope_freq_base) {
  // tokens in [start_pos, start_pos + gridDim.x)
  // per block per token
  int token_idx = blockIdx.x;
  if (token_idx > n_tokens) {
    return;
  }

  int pos = start_pos + token_idx;

  // 当前 block 要处理的 q_token 和 k_token
  // t_kok: shape[1, dim]
  // k_tok: shape[1, kv_dim]
  __half* q_tok = q + token_idx * dim;
  __half* k_tok = k + token_idx * kv_dim;

  // per thread per elements in one token
  for (int i = threadIdx.x; i < dim / 2; i += blockDim.x) {
    int idx = i * 2;
    int pair = (idx % head_dim) / 2;
    float theta = 1.0f / powf(rope_freq_base, 2.0f * pair / head_dim);
    float cos_v = cosf(theta * pos);
    float sin_v = sinf(theta * pos);
    float q0 = __half2float(q_tok[idx]), q1 = __half2float(q_tok[idx + 1]);
    q_tok[idx] = __float2half(q0 * cos_v - q1 * sin_v);
    q_tok[idx + 1] = __float2half(q0 * sin_v + q1 * cos_v);
  }

  // per thread per elements in one token
  for (int i = threadIdx.x; i < kv_dim / 2; i += blockDim.x) {
    int idx = i * 2;
    int pair = (idx % head_dim) / 2;
    float theta = 1.0f / powf(rope_freq_base, 2.0f * pair / head_dim);
    float cos_v = cosf(theta * pos);
    float sin_v = sinf(theta * pos);
    float k0 = __half2float(k_tok[idx]), k1 = __half2float(k_tok[idx + 1]);
    k_tok[idx] = __float2half(k0 * cos_v - k1 * sin_v);
    k_tok[idx + 1] = __float2half(k0 * sin_v + k1 * cos_v);
  }
}

__global__ void kvcache_write_kernel_batch(__half* k_cache, __half* v_cache,
                                           const __half* k, const __half* v,
                                           int layer, int start_pos,
                                           int n_tokens, int seq_len,
                                           int kv_dim) {
  // 每个 block 处理一个 token
  int token_idx = blockIdx.x;
  if (token_idx > n_tokens) {
    return;
  }

  // token 的绝对 pos
  int pos = start_pos + token_idx;
  // 每层都有 seq_len 个 token
  // 跳过前几层 + 目标层跳过前几个 token
  int offset = layer * seq_len * kv_dim + pos * kv_dim;

  // block 内多线程处理一个 token
  for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
    k_cache[offset + i] = k[token_idx * kv_dim + i];
    v_cache[offset + i] = v[token_idx * kv_dim + i];
  }
}

/**
 * x: shape[n_tokens, dim]
 * delta: shape[n_tokens, dim]
 */
__global__ void residual_kernel_batch(__half* x, const __half* delta,
                                      int n_tokens, int dim) {
  int token_idx = blockIdx.x;
  if (token_idx > n_tokens) {
    return;
  }

  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    int idx = token_idx * dim + i;
    x[idx] = __float2half(__half2float(x[idx]) + __half2float(delta[idx]));
  }
}

/**
 * hb: [n_tokens, hidden_dim]
 * hb2: [n_tokens, hidden_dim]
 */
__global__ void swiglu_kernel_batch(__half* hb, const __half* hb2, int n_tokens,
                                    int hidden_dim) {
  int token_idx = blockIdx.x;
  if (token_idx > n_tokens) {
    return;
  }

  for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    int idx = token_idx * hidden_dim + i;
    float x = __half2float(hb[idx]);
    hb[idx] =
        __float2half(x * (1.0f / (1.0f + expf(-x))) * __half2float(hb2[idx]));
  }
}

// ============================================================================
// cublasHgemm 封装
// ============================================================================

/**
 * matmul: out = x @ w^T
 * x: [n] fp16, w: [d, n] fp16, out: [d] fp16
 */
static void matmul_half(cublasHandle_t handle, __half* out, const __half* x,
                        const __half* w, int n, int d) {
  const __half alpha = __float2half(1.0f);
  const __half beta = __float2half(0.0f);
  CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, d, 1, n, &alpha, w,
                           n, x, n, &beta, out, d));
}

static void matmul_half_batch(cublasHandle_t handle, __half* out,
                              const __half* x, const __half* w, int n, int d,
                              int batch) {
  const __half alpha = __float2half(1.0f);
  const __half beta = __float2half(0.0f);
  CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, d, batch, n,
                           &alpha, w, n, x, n, &beta, out, d));
}

// ============================================================================
// GPUDecoder 实现
// ============================================================================

GPUDecoder::GPUDecoder(const std::string& model_file) {
  if (load_gguf(gguf, model_file) != 0) {
    fprintf(stderr, "failed to load gguf\n");
    exit(1);
  }
  if (load_config(config, gguf) != 0) {
    fprintf(stderr, "failed to load config\n");
    exit(1);
  }
  if (open_model(model_file, mf) != 0) {
    fprintf(stderr, "failed to open model\n");
    exit(1);
  }
  if (load_weights(w, config, gguf, mf) != 0) {
    fprintf(stderr, "failed to load weights\n");
    exit(1);
  }
  if (load_tokenizer(tokenizer, gguf) != 0) {
    fprintf(stderr, "failed to load tokenizer\n");
    exit(1);
  }

  upload_weights(gw, w, config);
  alloc_gpu_run_state(gs, config);
  CHECK_CUBLAS(cublasCreate(&cublas_handle));
}

GPUDecoder::~GPUDecoder() {
  cublasDestroy(cublas_handle);
  free_gpu_run_state(gs);
  free_gpu_weights(gw, config);
  free_weights(w);
  close_model(mf);
}

float* GPUDecoder::get_logits() {
  cudaDeviceSynchronize();
  return gs.logits;
}

void GPUDecoder::forward(int token, int pos) {
  int dim = config.dim;
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
  int kv_mul = config.n_heads / config.n_kv_heads;
  int threads = 256;

  // 1. Embedding lookup
  embedding_kernel<<<(dim + threads - 1) / threads, threads>>>(
      gs.x, gw.token_embedding, token, dim);
  CHECK_KERNEL();

  for (int l = 0; l < config.n_layers; l++) {
    // 2. RMSNorm
    rmsnorm_kernel<<<1, threads>>>(gs.xb, gs.x, gw.rms_att[l], dim);
    CHECK_KERNEL();

    // 3. QKV 投影
    matmul_half(cublas_handle, gs.q, gs.xb, gw.wq[l], dim, dim);
    matmul_half(cublas_handle, gs.k, gs.xb, gw.wk[l], dim, kv_dim);
    matmul_half(cublas_handle, gs.v, gs.xb, gw.wv[l], dim, kv_dim);

    // 4. 加 bias
    add_bias_kernel<<<(dim + threads - 1) / threads, threads>>>(gs.q, gw.bq[l],
                                                                dim);
    CHECK_KERNEL();
    add_bias_kernel<<<(kv_dim + threads - 1) / threads, threads>>>(
        gs.k, gw.bk[l], kv_dim);
    CHECK_KERNEL();
    add_bias_kernel<<<(kv_dim + threads - 1) / threads, threads>>>(
        gs.v, gw.bv[l], kv_dim);
    CHECK_KERNEL();

    // 5. RoPE
    rope_kernel<<<(dim / 2 + threads - 1) / threads, threads>>>(
        gs.q, gs.k, pos, dim, kv_dim, head_dim, config.rope_freq_base);
    CHECK_KERNEL();

    // 6. KV Cache 写入
    kvcache_write_kernel<<<(kv_dim + threads - 1) / threads, threads>>>(
        gs.k_cache, gs.v_cache, gs.k, gs.v, l, pos, config.seq_len, kv_dim);
    CHECK_KERNEL();

    // 7. Attention
    size_t smem_size = (FA_BLOCK_SIZE + head_dim + 8 + 8 + 2) * sizeof(float);
    attention_kernel<<<config.n_heads, threads, smem_size>>>(
        gs.q, gs.k_cache + (size_t)l * config.seq_len * kv_dim,
        gs.v_cache + (size_t)l * config.seq_len * kv_dim, gs.xb, pos,
        config.seq_len, kv_dim, head_dim, kv_mul);
    CHECK_KERNEL();

    // 8. 输出投影 + 残差
    matmul_half(cublas_handle, gs.xb2, gs.xb, gw.wo[l], dim, dim);
    residual_kernel<<<(dim + threads - 1) / threads, threads>>>(gs.x, gs.xb2,
                                                                dim);
    CHECK_KERNEL();

    // 9. FFN RMSNorm
    rmsnorm_kernel<<<1, threads>>>(gs.xb, gs.x, gw.rms_ffn[l], dim);
    CHECK_KERNEL();

    // 10. SwiGLU FFN
    matmul_half(cublas_handle, gs.hb, gs.xb, gw.w1[l], dim, config.hidden_dim);
    matmul_half(cublas_handle, gs.hb2, gs.xb, gw.w3[l], dim, config.hidden_dim);
    swiglu_kernel<<<(config.hidden_dim + threads - 1) / threads, threads>>>(
        gs.hb, gs.hb2, config.hidden_dim);
    CHECK_KERNEL();

    // 11. FFN 输出投影 + 残差
    matmul_half(cublas_handle, gs.xb2, gs.hb, gw.w2[l], config.hidden_dim, dim);
    residual_kernel<<<(dim + threads - 1) / threads, threads>>>(gs.x, gs.xb2,
                                                                dim);
    CHECK_KERNEL();
  }

  // 12. 最终 RMSNorm
  rmsnorm_kernel<<<1, threads>>>(gs.xb, gs.x, gw.rms_final, dim);
  CHECK_KERNEL();

  // 13. logits: fp16 matmul 然后转 fp32
  matmul_half(cublas_handle, gs.logits_fp16, gs.xb, gw.wcls, dim,
              config.vocab_size);
  fp16_to_fp32_kernel<<<(config.vocab_size + threads - 1) / threads, threads>>>(
      gs.logits, gs.logits_fp16, config.vocab_size);
  CHECK_KERNEL();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model_file>\n", argv[0]);
    return 1;
  }

  auto* decoder = new GPUDecoder(argv[1]);

  std::mt19937 rng(time(nullptr));
  float temperature = 0.7f;
  int top_k = 40;
  int max_new_tokens = 256;

  std::string user_input;
  printf("User: ");
  std::getline(std::cin, user_input);
  std::cout << user_input << std::endl;

  decoder->generate(user_input, max_new_tokens, temperature, top_k, rng);

  delete decoder;
  return 0;
}

void GPUDecoder::forward_prefill(const int* tokens, int n_tokens,
                                 int start_pos) {
  int dim = config.dim;
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
  int kv_mul = config.n_heads / config.n_kv_heads;
  int threads = 256;

  int* d_tokens;
  CHECK_CUDA(cudaMalloc(&d_tokens, n_tokens * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_tokens, tokens, n_tokens * sizeof(int),
                        cudaMemcpyHostToDevice));

  // 1.embedding lookup (batch)
  embedding_kernel_batch<<<n_tokens, threads>>>(gs.x_pre, gw.token_embedding,
                                                d_tokens, n_tokens, dim);
  CHECK_KERNEL();

  for (int l = 0; l < config.n_layers; l++) {
    // 2.RMSNorm (batch)
    rmsnorm_kernel_batch<<<n_tokens, threads>>>(gs.xb_pre, gs.x_pre,
                                                gw.rms_att[l], dim);
    CHECK_KERNEL();

    // 3.qkv 投影 (batch)
    matmul_half_batch(cublas_handle, gs.q_pre, gs.xb_pre, gw.wq[l], dim, dim,
                      n_tokens);
    matmul_half_batch(cublas_handle, gs.k_pre, gs.xb_pre, gw.wk[l], dim, kv_dim,
                      n_tokens);
    matmul_half_batch(cublas_handle, gs.v_pre, gs.xb_pre, gw.wv[l], dim, kv_dim,
                      n_tokens);

    // 4.加 bias (batch)
    add_bias_kernel_batch<<<n_tokens, threads>>>(gs.q_pre, gw.bq[l], n_tokens,
                                                 dim);
    CHECK_KERNEL();
    add_bias_kernel_batch<<<n_tokens, threads>>>(gs.k_pre, gw.bk[l], n_tokens,
                                                 kv_dim);
    CHECK_KERNEL();
    add_bias_kernel_batch<<<n_tokens, threads>>>(gs.v_pre, gw.bv[l], n_tokens,
                                                 kv_dim);
    CHECK_KERNEL();

    // 5.RoPE (batch)
    rope_kernel_batch<<<n_tokens, threads>>>(gs.q_pre, gs.k_pre, start_pos,
                                             n_tokens, dim, kv_dim, head_dim,
                                             config.rope_freq_base);
    CHECK_KERNEL();

    // 6. kv cache 写入 (batch)
    kvcache_write_kernel_batch<<<n_tokens, threads>>>(
        gs.k_cache, gs.v_cache, gs.k_pre, gs.v_pre, l, start_pos, n_tokens,
        config.seq_len, kv_dim);
    CHECK_KERNEL();

    // 7. Flash Attention (prefill)
    size_t smem_size = (FA_BLOCK_SIZE + head_dim + 8 + 8 + 2) * sizeof(float);
    dim3 grid(config.n_heads, n_tokens);
    flash_attention_prefill_kernel<<<grid, threads, smem_size>>>(
        gs.q_pre, gs.k_cache + (size_t)l * config.seq_len * kv_dim,
        gs.v_cache + (size_t)l * config.seq_len * kv_dim, gs.xb_pre, start_pos,
        config.seq_len, dim, kv_dim, head_dim, kv_mul);
    CHECK_KERNEL();

    // 8.输出投影 + 残差 (batch)
    matmul_half_batch(cublas_handle, gs.xb2_pre, gs.xb_pre, gw.wo[l], dim, dim,
                      n_tokens);
    residual_kernel_batch<<<n_tokens, threads>>>(gs.x_pre, gs.xb2_pre, n_tokens,
                                                 dim);
    CHECK_KERNEL();

    // 9.FFN RMSNorm (batch)
    rmsnorm_kernel_batch<<<n_tokens, threads>>>(gs.xb_pre, gs.x_pre,
                                                gw.rms_ffn[l], dim);
    CHECK_KERNEL();

    // 10.SwiGLU FFN (batch)
    // out: gs.hb_pre; in: gs.xb_pre
    matmul_half_batch(cublas_handle, gs.hb_pre, gs.xb_pre, gw.w1[l], dim,
                      config.hidden_dim, n_tokens);
    // out: gs.hb2_pre; in: gs.sb_pre
    matmul_half_batch(cublas_handle, gs.hb2_pre, gs.xb_pre, gw.w3[l], dim,
                      config.hidden_dim, n_tokens);

    // out: gs.hb_pre; in: gs.hb2_pre
    swiglu_kernel_batch<<<n_tokens, threads>>>(gs.hb_pre, gs.hb2_pre, n_tokens,
                                               config.hidden_dim);
    CHECK_KERNEL();

    // 11. ffn 输出投影 + 残差 (batch)
    // out: xb2_pre; in: hb_re, gw.w2[l]
    matmul_half_batch(cublas_handle, gs.xb2_pre, gs.hb_pre, gw.w2[l],
                      config.hidden_dim, dim, n_tokens);
    residual_kernel_batch<<<n_tokens, threads>>>(gs.x_pre, gs.xb2_pre, n_tokens,
                                                 dim);
    CHECK_KERNEL();
  }

  // 12. 最终 RMSNorm（只取最后一个 token）
  // out: gs.xb
  rmsnorm_kernel<<<1, threads>>>(gs.xb, gs.x_pre + (size_t)(n_tokens - 1) * dim,
                                 gw.rms_final, dim);

  // 13. logits（只算最后一个 token）
  matmul_half(cublas_handle, gs.logits_fp16, gs.xb, gw.wcls, dim,
              config.vocab_size);
  fp16_to_fp32_kernel<<<(config.vocab_size + threads - 1) / threads, threads>>>(
      gs.logits, gs.logits_fp16, config.vocab_size);
  CHECK_KERNEL();

  cudaFree(d_tokens);
}

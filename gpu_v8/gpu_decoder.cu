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

static void alloc_gpu_run_state(GPURunState& s, const Config& config,
                                int max_batch) {
  int dim = config.dim;
  int kv_dim = config.n_kv_heads * (config.dim / config.n_heads);
  int seq_len = config.seq_len;

  /**
   * decode 单请求缓冲区（prefill 内部也用）
   */
  CHECK_CUDA(cudaMalloc(&s.x, dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb, dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb2, dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.q, dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.k, kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.v, kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb, config.hidden_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb2, config.hidden_dim * sizeof(__half)));

  /**
   * prefill 缓冲区
   */
  int max_pre = config.seq_len;
  CHECK_CUDA(cudaMalloc(&s.x_pre, (size_t)max_pre * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb_pre, (size_t)max_pre * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb2_pre, (size_t)max_pre * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.q_pre, (size_t)max_pre * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.k_pre, (size_t)max_pre * kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.v_pre, (size_t)max_pre * kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb_pre,
                        (size_t)max_pre * config.hidden_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb2_pre,
                        (size_t)max_pre * config.hidden_dim * sizeof(__half)));

  /**
   * batch decode 缓冲区 [max_batch, dim]
   */
  CHECK_CUDA(cudaMalloc(&s.x_batch, (size_t)max_batch * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb_batch, (size_t)max_batch * dim * sizeof(__half)));
  CHECK_CUDA(
      cudaMalloc(&s.xb2_batch, (size_t)max_batch * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.q_batch, (size_t)max_batch * dim * sizeof(__half)));
  CHECK_CUDA(
      cudaMalloc(&s.k_batch, (size_t)max_batch * kv_dim * sizeof(__half)));
  CHECK_CUDA(
      cudaMalloc(&s.v_batch, (size_t)max_batch * kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(
      &s.hb_batch, (size_t)max_batch * config.hidden_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(
      &s.hb2_batch, (size_t)max_batch * config.hidden_dim * sizeof(__half)));

  /**
   * KV cache：每个请求独立 [max_batch, n_layers, seq_len, kv_dim]
   */
  CHECK_CUDA(cudaMalloc(&s.k_cache, (size_t)max_batch * config.n_layers *
                                        seq_len * kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.v_cache, (size_t)max_batch * config.n_layers *
                                        seq_len * kv_dim * sizeof(__half)));

  /**
   * logits [max_batch, vocab_size]
   */
  CHECK_CUDA(
      cudaMalloc(&s.logits_fp16_batch,
                 (size_t)max_batch * config.vocab_size * sizeof(__half)));
  CHECK_CUDA(cudaMallocHost(
      &s.logits_batch, (size_t)max_batch * config.vocab_size * sizeof(float)));
}

static void free_gpu_run_state(GPURunState& s) {
  // decode 单请求
  cudaFree(s.x);
  cudaFree(s.xb);
  cudaFree(s.xb2);
  cudaFree(s.q);
  cudaFree(s.k);
  cudaFree(s.v);
  cudaFree(s.hb);
  cudaFree(s.hb2);

  // prefill
  cudaFree(s.x_pre);
  cudaFree(s.xb_pre);
  cudaFree(s.xb2_pre);
  cudaFree(s.q_pre);
  cudaFree(s.k_pre);
  cudaFree(s.v_pre);
  cudaFree(s.hb_pre);
  cudaFree(s.hb2_pre);

  // batch decode
  cudaFree(s.x_batch);
  cudaFree(s.xb_batch);
  cudaFree(s.xb2_batch);
  cudaFree(s.q_batch);
  cudaFree(s.k_batch);
  cudaFree(s.v_batch);
  cudaFree(s.hb_batch);
  cudaFree(s.hb2_batch);

  // KV cache 和 logits
  cudaFree(s.k_cache);
  cudaFree(s.v_cache);
  cudaFree(s.logits_fp16_batch);
  cudaFreeHost(s.logits_batch);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

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
 * q:       [n_tokens, dim]
 * k_cache: [seq_len, kv_dim]
 * v_cache: [seq_len, kv_dim]
 * out:     [n_tokens, dim]
 *
 * 每个 block 负责一个 (query_token, head) 对
 * causal mask: 第 q_tok 个 token 只能看 <= start_pos + q_tok 的 kv
 *
 * FA_BLOCK_SIZE = blockDim.x = 256
 * 每个线程恰好负责一个 KV 元素，无空闲线程
 */
constexpr int FA_BLOCK_SIZE = 256;

__global__ void flash_attention_prefill_kernel(
    const __half* q, const __half* k_cache, const __half* v_cache, __half* out,
    int start_pos, int seq_len, int dim, int kv_dim, int head_dim, int kv_mul) {
  extern __shared__ float smem[];
  float* s_block = smem;                 // [FA_BLOCK_SIZE]
  float* o_smem = smem + FA_BLOCK_SIZE;  // [head_dim]
  float* warp_m = o_smem + head_dim;     // [8]
  float* warp_d = warp_m + 8;            // [8]
  float* g_m = warp_d + 8;               // [1]
  float* g_d = g_m + 1;                  // [1]

  int h = blockIdx.x;
  int q_tok = blockIdx.y;
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  int kv_head = h / kv_mul;
  float scale = rsqrtf((float)head_dim);

  int cur_pos = start_pos + q_tok;
  int kv_end = cur_pos + 1;

  // q 布局: [n_tokens, dim]
  const __half* q_head = q + q_tok * dim + h * head_dim;

  // 初始化
  for (int d = tid; d < head_dim; d += blockDim.x) o_smem[d] = 0.0f;
  if (tid == 0) {
    *g_m = -CUDART_INF_F;
    *g_d = 0.0f;
  }
  __syncthreads();

  for (int start = 0; start < kv_end; start += FA_BLOCK_SIZE) {
    int len = min(FA_BLOCK_SIZE, kv_end - start);
    int t = start + tid;

    // 1. 每个线程计算一个 score
    //    超出范围的填 -inf（不影响 max 归约，且不贡献到 d 和 o）
    if (t < kv_end) {
      const __half* k_head = k_cache + t * kv_dim + kv_head * head_dim;
      float s = 0.0f;
      for (int d = 0; d < head_dim; d++)
        s += __half2float(q_head[d]) * __half2float(k_head[d]);
      s_block[tid] = s * scale;
    } else {
      s_block[tid] = -CUDART_INF_F;
    }
    __syncthreads();

    // 2. 块内最大值归约
    float local_max = s_block[tid];
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) warp_m[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
      float val = (lane_id < 8) ? warp_m[lane_id] : -CUDART_INF_F;
      val = warp_reduce_max(val);
      if (lane_id == 0) warp_m[0] = val;
    }
    __syncthreads();

    // 3. online softmax 更新 m 和 d
    float m_new = fmaxf(*g_m, warp_m[0]);
    float correction = expf(*g_m - m_new);

    // 超出范围的线程 local_d = 0，不贡献
    float local_d = (t < kv_end) ? expf(s_block[tid] - m_new) : 0.0f;
    local_d = warp_reduce_sum(local_d);
    if (lane_id == 0) warp_d[warp_id] = local_d;
    __syncthreads();

    if (warp_id == 0) {
      float val = (lane_id < 8) ? warp_d[lane_id] : 0.0f;
      val = warp_reduce_sum(val);
      if (lane_id == 0) warp_d[0] = val;
    }
    __syncthreads();

    if (tid == 0) {
      *g_d = (*g_d) * correction + warp_d[0];
      *g_m = m_new;
    }
    __syncthreads();

    // 4. 更新输出，每个线程负责 out 的一个维度
    for (int d = tid; d < head_dim; d += blockDim.x) {
      o_smem[d] *= correction;
      for (int i = 0; i < len; i++) {
        int tt = start + i;
        const __half* v_head = v_cache + tt * kv_dim + kv_head * head_dim;
        o_smem[d] += expf(s_block[i] - m_new) * __half2float(v_head[d]);
      }
    }
    __syncthreads();
  }

  // 5. 归一化写回
  __half* out_head = out + q_tok * dim + h * head_dim;
  float d_global = *g_d;
  for (int d = tid; d < head_dim; d += blockDim.x)
    out_head[d] = __float2half(o_smem[d] / d_global);
}

/**
 * IN:
 * - table: [vocab_size, dim]
 * - tokens:
 * - n_tokens:
 * - dim:
 * OUT:
 * - out: [max_batch, dim]
 */
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
 * IN:
 * - x: [match_batch, dim]
 * - weight: [dim]
 * - dim:
 * OUT:
 * - out: [max_batch, dim]
 */
__global__ void rmsnorm_kernel_batch(__half* out, const __half* x,
                                     const float* weight, int dim) {
  __shared__ float warp_sums[8];
  // 每个 block 处理一个 token
  int token_idx = blockIdx.x;
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  // 跳过前 token_idx 个 dim
  const __half* x_row = x + token_idx * dim;
  __half* out_row = out + token_idx * dim;

  // block 内的每个线程处理多个 elements
  float local_sum = 0.0f;
  for (int i = tid; i < dim; i += blockDim.x) {
    float xi = __half2float(x_row[i]);
    local_sum += xi * xi;
  }
  // warp 内线程同步
  local_sum = warp_reduce_sum(local_sum);
  if (lane_id == 0) {
    warp_sums[warp_id] = local_sum;
  }
  __syncthreads();

  // block 内线程同步
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

/**
 * IN:
 * - bias: [kv_dim]
 * - n_tokens
 * - n
 * OUT:
 * - out: [max_batch, dim]
 */
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
 * x: shape[max_batch, dim]
 * delta: shape[max_batch, dim]
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
 * IN:
 * - hb: [max_batch, hidden_dim]
 * - hb2: [max_batch, hidden_dim]
 * OUT:
 * - hb: [max_batch, hidden_dim]
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

/**
 * RoPE kernel (batch, 每个请求有独立的 pos)
 *
 * IN:
 * - k: [batch, kv_dim]
 * - position: [batch_size] 每个请求当前的 pos
 * - kv_dim:
 * - head_dim
 * - rope_freq_base
 * OUT:
 * - q: [batch, dim]
 */
__global__ void rope_kernel_batch_var(__half* q, __half* k,
                                      const int* positions, int batch_size,
                                      int dim, int kv_dim, int head_dim,
                                      float rope_freq_base) {
  // 每个 block 处理一个 token
  int token_idx = blockIdx.x;
  if (token_idx >= batch_size) return;

  // 要处理的 token 对应的 pos
  int pos = positions[token_idx];
  __half* q_tok = q + token_idx * dim;
  __half* k_tok = k + token_idx * kv_dim;

  for (int i = threadIdx.x; i < dim / 2; i += blockDim.x) {
    // 两个一组
    int idx = i * 2;
    // 算 theta
    int pair = (idx % head_dim) / 2;
    float theta = 1.0f / powf(rope_freq_base, 2.0f * pair / head_dim);
    // 算 cos_v 和 sin_v
    float cos_v = cosf(theta * pos);
    float sin_v = sinf(theta * pos);
    // 位置编码
    float q0 = __half2float(q_tok[idx]), q1 = __half2float(q_tok[idx + 1]);
    q_tok[idx] = __float2half(q0 * cos_v - q1 * sin_v);
    q_tok[idx + 1] = __float2half(q0 * sin_v + q1 * cos_v);
  }

  for (int i = threadIdx.x; i < kv_dim / 2; i += blockDim.x) {
    // 两个一组
    int idx = i * 2;
    // 计算 theta
    int pair = (idx % head_dim) / 2;
    float theta = 1.0f / powf(rope_freq_base, 2.0f * pair / head_dim);
    // 计算 cos 和 sin
    float cos_v = cosf(theta * pos);
    float sin_v = sinf(theta * pos);
    // 位置编码
    float k0 = __half2float(k_tok[idx]), k1 = __half2float(k_tok[idx + 1]);
    k_tok[idx] = __float2half(k0 * cos_v - k1 * sin_v);
    k_tok[idx + 1] = __float2half(k0 * sin_v + k1 * cos_v);
  }
}

/**
 * KV cache 写入 kernel (batch, 每个请求有独立的 pos 和 KV cache 槽)
 *
 * OUT:
 * - k_cache: [max_batch, n_layers, seq_len, kv_dim]
 * - v_cache: [max_batch, n_layers, seq_len, kv_dim]
 * IN:
 * - k: [max_batch, kv_dim]
 * - v: [max_batch, kv_dim]
 * - positions: [batch_size]
 * - batch_size
 * - seq_len
 * - kv_dim
 * - n_layers
 */
__global__ void kvcache_write_kernel_batch_var(__half* k_cache, __half* v_cache,
                                               const __half* k, const __half* v,
                                               int layer, const int* positions,
                                               int batch_size, int seq_len,
                                               int kv_dim, int n_layers) {
  // 每个 block 处理一个 token
  int token_idx = blockIdx.x;
  if (token_idx >= batch_size) return;

  // 当前 block 要处理的 token 的 pos
  int pos = positions[token_idx];

  /**
   * 跳过 token_idx
   * 跳过 layer
   * 跳过 pos
   * 得到当前 kv_dim 对应的 offset
   */
  size_t offset = (size_t)token_idx * n_layers * seq_len * kv_dim +
                  (size_t)layer * seq_len * kv_dim + (size_t)pos * kv_dim;

  // block 内多线程分工
  for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
    k_cache[offset + i] = k[token_idx * kv_dim + i];
    v_cache[offset + i] = v[token_idx * kv_dim + i];
  }
}
/**
 * Attention kernel (batch decode，每个请求只看自己的 KV cache)
 *
 * Grid: dim3(n_heads, batch_size)
 * Block: 256 个线程
 *
 * q:        [batch_size, dim]
 * k_cache:  [max_batch, n_layers, seq_len, kv_dim]
 * v_cache:  [max_batch, n_layers, seq_len, kv_dim]
 * out:      [batch_size, dim]
 * positions:[batch_size] 每个请求当前的 pos
 *
 * FA_BLOCK_SIZE = blockDim.x = 256
 * 每个线程恰好负责一个 KV 元素，无空闲线程
 */
__global__ void attention_batch_kernel(const __half* q, const __half* k_cache,
                                       const __half* v_cache, __half* out,
                                       const int* positions, int layer,
                                       int seq_len, int dim, int kv_dim,
                                       int head_dim, int kv_mul, int n_layers) {
  extern __shared__ float smem[];
  float* s_block = smem;
  float* o_smem = smem + FA_BLOCK_SIZE;
  float* warp_m = o_smem + head_dim;
  float* warp_d = warp_m + 8;
  float* g_m = warp_d + 8;
  float* g_d = g_m + 1;

  int h = blockIdx.x;
  int batch_idx = blockIdx.y;
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  int kv_head = h / kv_mul;
  float scale = rsqrtf((float)head_dim);

  int pos = positions[batch_idx];
  int kv_end = pos + 1;

  const __half* q_head = q + batch_idx * dim + h * head_dim;

  const __half* k_layer = k_cache +
                          (size_t)batch_idx * n_layers * seq_len * kv_dim +
                          (size_t)layer * seq_len * kv_dim;
  const __half* v_layer = v_cache +
                          (size_t)batch_idx * n_layers * seq_len * kv_dim +
                          (size_t)layer * seq_len * kv_dim;

  // 初始化
  for (int d = tid; d < head_dim; d += blockDim.x) o_smem[d] = 0.0f;
  if (tid == 0) {
    *g_m = -CUDART_INF_F;
    *g_d = 0.0f;
  }
  __syncthreads();

  for (int start = 0; start < kv_end; start += FA_BLOCK_SIZE) {
    int len = min(FA_BLOCK_SIZE, kv_end - start);
    int t = start + tid;

    // 1. 每个线程计算一个 score
    if (t < kv_end) {
      const __half* k_head = k_layer + t * kv_dim + kv_head * head_dim;
      float s = 0.0f;
      for (int d = 0; d < head_dim; d++)
        s += __half2float(q_head[d]) * __half2float(k_head[d]);
      s_block[tid] = s * scale;
    } else {
      s_block[tid] = -CUDART_INF_F;
    }
    __syncthreads();

    // 2. 块内最大值归约
    float local_max = s_block[tid];
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) warp_m[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
      float val = (lane_id < 8) ? warp_m[lane_id] : -CUDART_INF_F;
      val = warp_reduce_max(val);
      if (lane_id == 0) warp_m[0] = val;
    }
    __syncthreads();

    // 3. online softmax 更新 m 和 d
    float m_new = fmaxf(*g_m, warp_m[0]);
    float correction = expf(*g_m - m_new);

    float local_d = (t < kv_end) ? expf(s_block[tid] - m_new) : 0.0f;
    local_d = warp_reduce_sum(local_d);
    if (lane_id == 0) warp_d[warp_id] = local_d;
    __syncthreads();

    if (warp_id == 0) {
      float val = (lane_id < 8) ? warp_d[lane_id] : 0.0f;
      val = warp_reduce_sum(val);
      if (lane_id == 0) warp_d[0] = val;
    }
    __syncthreads();

    if (tid == 0) {
      *g_d = (*g_d) * correction + warp_d[0];
      *g_m = m_new;
    }
    __syncthreads();

    // 4. 更新输出，每个线程负责 out 的一个维度
    for (int d = tid; d < head_dim; d += blockDim.x) {
      o_smem[d] *= correction;
      for (int i = 0; i < len; i++) {
        int tt = start + i;
        const __half* v_head = v_layer + tt * kv_dim + kv_head * head_dim;
        o_smem[d] += expf(s_block[i] - m_new) * __half2float(v_head[d]);
      }
    }
    __syncthreads();
  }

  // 5. 归一化写回
  __half* out_head = out + batch_idx * dim + h * head_dim;
  float d_global = *g_d;
  for (int d = tid; d < head_dim; d += blockDim.x)
    out_head[d] = __float2half(o_smem[d] / d_global);
}

/**
 * fp16 -> fp32 (batch)
 * in:  [batch_size, vocab_size]
 * out: [batch_size, vocab_size]
 */
__global__ void fp16_to_fp32_batch_kernel(float* out, const __half* in,
                                          int batch_size, int vocab_size) {
  int batch_idx = blockIdx.x;
  if (batch_idx >= batch_size) return;
  for (int i = threadIdx.x; i < vocab_size; i += blockDim.x)
    out[batch_idx * vocab_size + i] =
        __half2float(in[batch_idx * vocab_size + i]);
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

/**
 * 输出参数
 * out: shape[max_batch, dim]
 * 输入参数
 * x: shape[max_batch, dim]
 * w: shape[ouput_dim, input_dim]
 * n: input dim
 * d: output dim
 * batch
 */
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
GPUDecoder::GPUDecoder(const std::string& model_file, int max_batch)
    : max_batch(max_batch) {
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
  alloc_gpu_run_state(gs, config, max_batch);
  CHECK_CUBLAS(cublasCreate(&cublas_handle));
}

GPUDecoder::~GPUDecoder() {
  cublasDestroy(cublas_handle);
  free_gpu_run_state(gs);
  free_gpu_weights(gw, config);
  free_weights(w);
  close_model(mf);
}

float* GPUDecoder::get_logits_batch(int batch_idx) {
  cudaDeviceSynchronize();
  return gs.logits_batch + (size_t)batch_idx * config.vocab_size;
}

void GPUDecoder::forward_prefill(const int* tokens, int n_tokens, int start_pos,
                                 int batch_idx) {
  int dim = config.dim;
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
  int kv_mul = config.n_heads / config.n_kv_heads;
  int threads = 256;

  // KV cache 偏移：跳过前 batch_idx 个请求
  size_t batch_kv_offset =
      (size_t)batch_idx * config.n_layers * config.seq_len * kv_dim;
  __half* k_cache = gs.k_cache + batch_kv_offset;
  __half* v_cache = gs.v_cache + batch_kv_offset;

  int* d_tokens;
  CHECK_CUDA(cudaMalloc(&d_tokens, n_tokens * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_tokens, tokens, n_tokens * sizeof(int),
                        cudaMemcpyHostToDevice));

  // 1. embedding lookup (batch)
  embedding_kernel_batch<<<n_tokens, threads>>>(gs.x_pre, gw.token_embedding,
                                                d_tokens, n_tokens, dim);
  CHECK_KERNEL();

  for (int l = 0; l < config.n_layers; l++) {
    // 2. RMSNorm (batch)
    rmsnorm_kernel_batch<<<n_tokens, threads>>>(gs.xb_pre, gs.x_pre,
                                                gw.rms_att[l], dim);
    CHECK_KERNEL();

    // 3. QKV 投影 (batch gemm)
    matmul_half_batch(cublas_handle, gs.q_pre, gs.xb_pre, gw.wq[l], dim, dim,
                      n_tokens);
    matmul_half_batch(cublas_handle, gs.k_pre, gs.xb_pre, gw.wk[l], dim, kv_dim,
                      n_tokens);
    matmul_half_batch(cublas_handle, gs.v_pre, gs.xb_pre, gw.wv[l], dim, kv_dim,
                      n_tokens);

    // 4. 加 bias (batch)
    add_bias_kernel_batch<<<n_tokens, threads>>>(gs.q_pre, gw.bq[l], n_tokens,
                                                 dim);
    add_bias_kernel_batch<<<n_tokens, threads>>>(gs.k_pre, gw.bk[l], n_tokens,
                                                 kv_dim);
    add_bias_kernel_batch<<<n_tokens, threads>>>(gs.v_pre, gw.bv[l], n_tokens,
                                                 kv_dim);
    CHECK_KERNEL();

    // 5. RoPE (batch)
    rope_kernel_batch<<<n_tokens, threads>>>(gs.q_pre, gs.k_pre, start_pos,
                                             n_tokens, dim, kv_dim, head_dim,
                                             config.rope_freq_base);
    CHECK_KERNEL();

    // 6. KV cache 写入 (batch)，用 batch 偏移后的指针
    kvcache_write_kernel_batch<<<n_tokens, threads>>>(
        k_cache, v_cache, gs.k_pre, gs.v_pre, l, start_pos, n_tokens,
        config.seq_len, kv_dim);
    CHECK_KERNEL();

    // 7. Flash Attention (prefill)
    size_t smem_size = (FA_BLOCK_SIZE + head_dim + 8 + 8 + 2) * sizeof(float);
    dim3 grid(config.n_heads, n_tokens);
    flash_attention_prefill_kernel<<<grid, threads, smem_size>>>(
        gs.q_pre, k_cache + (size_t)l * config.seq_len * kv_dim,
        v_cache + (size_t)l * config.seq_len * kv_dim, gs.xb_pre, start_pos,
        config.seq_len, dim, kv_dim, head_dim, kv_mul);
    CHECK_KERNEL();

    // 8. 输出投影 + 残差 (batch)
    matmul_half_batch(cublas_handle, gs.xb2_pre, gs.xb_pre, gw.wo[l], dim, dim,
                      n_tokens);
    residual_kernel_batch<<<n_tokens, threads>>>(gs.x_pre, gs.xb2_pre, n_tokens,
                                                 dim);
    CHECK_KERNEL();

    // 9. FFN RMSNorm (batch)
    rmsnorm_kernel_batch<<<n_tokens, threads>>>(gs.xb_pre, gs.x_pre,
                                                gw.rms_ffn[l], dim);
    CHECK_KERNEL();

    // 10. SwiGLU FFN (batch)
    matmul_half_batch(cublas_handle, gs.hb_pre, gs.xb_pre, gw.w1[l], dim,
                      config.hidden_dim, n_tokens);
    matmul_half_batch(cublas_handle, gs.hb2_pre, gs.xb_pre, gw.w3[l], dim,
                      config.hidden_dim, n_tokens);
    swiglu_kernel_batch<<<n_tokens, threads>>>(gs.hb_pre, gs.hb2_pre, n_tokens,
                                               config.hidden_dim);
    CHECK_KERNEL();

    // 11. FFN 输出投影 + 残差 (batch)
    matmul_half_batch(cublas_handle, gs.xb2_pre, gs.hb_pre, gw.w2[l],
                      config.hidden_dim, dim, n_tokens);
    residual_kernel_batch<<<n_tokens, threads>>>(gs.x_pre, gs.xb2_pre, n_tokens,
                                                 dim);
    CHECK_KERNEL();
  }

  // 12. 最终 RMSNorm（只取最后一个 token）
  rmsnorm_kernel<<<1, threads>>>(gs.xb, gs.x_pre + (size_t)(n_tokens - 1) * dim,
                                 gw.rms_final, dim);
  CHECK_KERNEL();

  // 13. logits 写到 batch_idx 对应的槽
  matmul_half(cublas_handle,
              gs.logits_fp16_batch + (size_t)batch_idx * config.vocab_size,
              gs.xb, gw.wcls, dim, config.vocab_size);
  fp16_to_fp32_kernel<<<(config.vocab_size + threads - 1) / threads, threads>>>(
      gs.logits_batch + (size_t)batch_idx * config.vocab_size,
      gs.logits_fp16_batch + (size_t)batch_idx * config.vocab_size,
      config.vocab_size);
  CHECK_KERNEL();

  cudaFree(d_tokens);
}

void GPUDecoder::forward_batch(const int* tokens, const int* positions,
                               int batch_size) {
  int dim = config.dim;
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
  int kv_mul = config.n_heads / config.n_kv_heads;
  int threads = 256;

  // tokens 上传到 GPU
  int* d_tokens;
  CHECK_CUDA(cudaMalloc(&d_tokens, batch_size * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_tokens, tokens, batch_size * sizeof(int),
                        cudaMemcpyHostToDevice));

  // positions 上传到 GPU
  int* d_positions;
  CHECK_CUDA(cudaMalloc(&d_positions, batch_size * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_positions, positions, batch_size * sizeof(int),
                        cudaMemcpyHostToDevice));

  // 1. Embedding lookup (batch)
  embedding_kernel_batch<<<batch_size, threads>>>(
      gs.x_batch, gw.token_embedding, d_tokens, batch_size, dim);
  CHECK_KERNEL();

  for (int l = 0; l < config.n_layers; l++) {
    // 2. RMSNorm (batch)
    rmsnorm_kernel_batch<<<batch_size, threads>>>(gs.xb_batch, gs.x_batch,
                                                  gw.rms_att[l], dim);
    CHECK_KERNEL();

    // 3. QKV 投影 (batch gemm)
    matmul_half_batch(cublas_handle, gs.q_batch, gs.xb_batch, gw.wq[l], dim,
                      dim, batch_size);
    matmul_half_batch(cublas_handle, gs.k_batch, gs.xb_batch, gw.wk[l], dim,
                      kv_dim, batch_size);
    matmul_half_batch(cublas_handle, gs.v_batch, gs.xb_batch, gw.wv[l], dim,
                      kv_dim, batch_size);

    // 4. 加 bias (batch)
    add_bias_kernel_batch<<<batch_size, threads>>>(gs.q_batch, gw.bq[l],
                                                   batch_size, dim);
    add_bias_kernel_batch<<<batch_size, threads>>>(gs.k_batch, gw.bk[l],
                                                   batch_size, kv_dim);
    add_bias_kernel_batch<<<batch_size, threads>>>(gs.v_batch, gw.bv[l],
                                                   batch_size, kv_dim);
    CHECK_KERNEL();

    // 5. RoPE (batch)，每个请求用自己的 pos
    rope_kernel_batch_var<<<batch_size, threads>>>(
        gs.q_batch, gs.k_batch, d_positions, batch_size, dim, kv_dim, head_dim,
        config.rope_freq_base);
    CHECK_KERNEL();

    // 6. KV cache 写入 (batch)，每个请求写到自己的 KV cache 槽
    kvcache_write_kernel_batch_var<<<batch_size, threads>>>(
        gs.k_cache, gs.v_cache, gs.k_batch, gs.v_batch, l, d_positions,
        batch_size, config.seq_len, kv_dim, config.n_layers);
    CHECK_KERNEL();

    // 7. Attention (batch)，每个请求只看自己的 KV cache
    size_t smem_size = (FA_BLOCK_SIZE + head_dim + 8 + 8 + 2) * sizeof(float);
    dim3 grid(config.n_heads, batch_size);
    attention_batch_kernel<<<grid, threads, smem_size>>>(
        gs.q_batch, gs.k_cache, gs.v_cache, gs.xb_batch, d_positions, l,
        config.seq_len, dim, kv_dim, head_dim, kv_mul, config.n_layers);
    CHECK_KERNEL();

    // 8. 输出投影 + 残差 (batch)
    matmul_half_batch(cublas_handle, gs.xb2_batch, gs.xb_batch, gw.wo[l], dim,
                      dim, batch_size);
    residual_kernel_batch<<<batch_size, threads>>>(gs.x_batch, gs.xb2_batch,
                                                   batch_size, dim);
    CHECK_KERNEL();

    // 9. FFN RMSNorm (batch)
    rmsnorm_kernel_batch<<<batch_size, threads>>>(gs.xb_batch, gs.x_batch,
                                                  gw.rms_ffn[l], dim);
    CHECK_KERNEL();

    // 10. SwiGLU FFN (batch)
    matmul_half_batch(cublas_handle, gs.hb_batch, gs.xb_batch, gw.w1[l], dim,
                      config.hidden_dim, batch_size);
    matmul_half_batch(cublas_handle, gs.hb2_batch, gs.xb_batch, gw.w3[l], dim,
                      config.hidden_dim, batch_size);
    swiglu_kernel_batch<<<batch_size, threads>>>(gs.hb_batch, gs.hb2_batch,
                                                 batch_size, config.hidden_dim);
    CHECK_KERNEL();

    // 11. FFN 输出投影 + 残差 (batch)
    matmul_half_batch(cublas_handle, gs.xb2_batch, gs.hb_batch, gw.w2[l],
                      config.hidden_dim, dim, batch_size);
    residual_kernel_batch<<<batch_size, threads>>>(gs.x_batch, gs.xb2_batch,
                                                   batch_size, dim);
    CHECK_KERNEL();
  }

  // 12. 最终 RMSNorm (batch)
  rmsnorm_kernel_batch<<<batch_size, threads>>>(gs.xb_batch, gs.x_batch,
                                                gw.rms_final, dim);
  CHECK_KERNEL();

  // 13. logits (batch)
  matmul_half_batch(cublas_handle, gs.logits_fp16_batch, gs.xb_batch, gw.wcls,
                    dim, config.vocab_size, batch_size);
  fp16_to_fp32_batch_kernel<<<batch_size, threads>>>(
      gs.logits_batch, gs.logits_fp16_batch, batch_size, config.vocab_size);
  CHECK_KERNEL();

  cudaFree(d_tokens);
  cudaFree(d_positions);
}

// int main(int argc, char** argv) {
//   if (argc < 2) {
//     fprintf(stderr, "Usage: %s <model_file> [batch_size]\n", argv[0]);
//     return 1;
//   }

//   int max_batch = 4;
//   if (argc >= 3) max_batch = atoi(argv[2]);

//   auto* decoder = new GPUDecoder(argv[1], max_batch);

//   std::mt19937 rng(time(nullptr));
//   float temperature = 0.7f;
//   int top_k = 40;
//   int max_new_tokens = 256;

//   // 构造 batch_size 个请求
//   std::vector<std::string> user_inputs;
//   for (int i = 0; i < max_batch; i++) {
//     std::string input;
//     printf("User[%d]: ", i);
//     std::getline(std::cin, input);
//     user_inputs.push_back(input);
//   }

//   decoder->generate_continuous(user_inputs, max_batch, max_new_tokens,
//                                temperature, top_k, rng);

//   delete decoder;
//   return 0;
// }

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model_file> [max_batch]\n", argv[0]);
    return 1;
  }

  int max_batch = 4;
  if (argc >= 3) max_batch = atoi(argv[2]);

  auto* decoder = new GPUDecoder(argv[1], max_batch);

  std::mt19937 rng(time(nullptr));
  float temperature = 0.7f;
  int top_k = 30;
  int max_new_tokens = 256;

  // 读取任意数量的请求，直到 EOF
  std::vector<std::string> user_inputs;
  std::string line;
  int idx = 0;
  while (true) {
    printf("User[%d] (empty to start): ", idx);
    if (!std::getline(std::cin, line) || line.empty()) break;
    user_inputs.push_back(line);
    idx++;
  }

  if (user_inputs.empty()) {
    fprintf(stderr, "no input\n");
    delete decoder;
    return 1;
  }

  fprintf(stderr, "%d requests, max_batch=%d\n", (int)user_inputs.size(),
          max_batch);

  decoder->generate_continuous(user_inputs, max_batch, max_new_tokens,
                               temperature, top_k, rng);

  delete decoder;
  return 0;
}
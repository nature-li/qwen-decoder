#include "gpu_kernels.h"

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

void matmul(cublasHandle_t h, __half* out, const __half* x, const __half* w, int n, int d,
            int batch) {
  const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
  CHECK_CUBLAS(
      cublasHgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, d, batch, n, &alpha, w, n, x, n, &beta, out, d));
}

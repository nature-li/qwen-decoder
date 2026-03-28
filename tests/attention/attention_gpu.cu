#include <cuda_runtime.h>
#include <math_constants.h>

#include <iostream>

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
 * warp 内归约：求和
 * 执行完后每个 warp 的 lane 0 持有该 warp 内的总和
 */
__device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

/**
 * softmax kernel（GPU）
 * Grid:  1 个 block
 * Block: 256 个线程（= 8 个 warp）
 * 每个线程跨步处理多个元素（stride = blockDim.x）
 *
 * 两级归约：
 *   第一级：warp 内用 __shfl_down_sync 归约（无需 shared memory）
 *   第二级：warp 间用 shared memory 归约
 */
__global__ void softmax_kernel(float* x, int n) {
  __shared__ float warp_max[8];  // 存 8 个 warp 各自的最大值
  __shared__ float warp_sum[8];  // 存 8 个 warp 各自的总和

  int tid = threadIdx.x;
  int warp_id = tid / 32;  // 第几个 warp（0-7）
  int lane_id = tid % 32;  // warp 内第几个线程（0-31）

  // =========================================================================
  // 1. 找全局 max（两级归约）
  // =========================================================================

  // 1.1 每个线程跨步扫描，找本线程负责的元素中的最大值
  float local_max = -CUDART_INF_F;
  for (int i = tid; i < n; i += blockDim.x) local_max = fmaxf(local_max, x[i]);

  // 1.2 warp 内归约，执行完后每个 warp 的 lane 0 是该 warp 内最大值
  local_max = warp_reduce_max(local_max);
  if (lane_id == 0) warp_max[warp_id] = local_max;
  __syncthreads();

  // 1.3 warp 0 对 8 个 warp 的结果再做一次归约，得到全局最大值
  if (warp_id == 0) {
    float val = (lane_id < 8) ? warp_max[lane_id] : -CUDART_INF_F;
    val = warp_reduce_max(val);
    if (lane_id == 0) warp_max[0] = val;  // 全局最大值写到 warp_max[0]
  }
  __syncthreads();

  float max_val = warp_max[0];  // 所有线程读全局最大值

  // =========================================================================
  // 2. 算 exp(x - max) 并求 sum（两级归约）
  // =========================================================================

  // 2.1 每个线程跨步计算 exp，累加本地 sum
  float local_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    x[i] = expf(x[i] - max_val);
    local_sum += x[i];
  }

  // 2.2 warp 内归约
  local_sum = warp_reduce_sum(local_sum);
  if (lane_id == 0) warp_sum[warp_id] = local_sum;
  __syncthreads();

  // 2.3 warp 0 对 8 个 warp 的结果再做一次归约，得到全局 sum
  if (warp_id == 0) {
    float val = (lane_id < 8) ? warp_sum[lane_id] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane_id == 0) warp_sum[0] = val;
  }
  __syncthreads();

  float sum = warp_sum[0];  // 所有线程读全局 sum

  // =========================================================================
  // 3. 归一化
  // =========================================================================
  for (int i = tid; i < n; i += blockDim.x) x[i] /= sum;
}

__global__ void online_softmax_kernel(float* x, int n) {
  __shared__ float warp_m[8];
  __shared__ float warp_d[8];
  __shared__ float global_m;
  __shared__ float global_d;

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  float m = -CUDART_INF_F;
  float d = 0.0f;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float m_new = fmaxf(m, x[i]);
    d = d * expf(m - m_new) + expf(x[i] - m_new);
    m = m_new;
  }
  // warp 内规约
  float m_warp = warp_reduce_max(m);
  if (lane_id == 0) {
    warp_m[warp_id] = m_warp;
  }

  // warp 间规约
  __syncthreads();
  if (warp_id == 0) {
    float val = (lane_id < 8) ? warp_m[lane_id] : -CUDART_INF_F;
    val = warp_reduce_max(val);
    if (lane_id == 0) {
      global_m = val;
    }
  }

  __syncthreads();
  float m_global = global_m;
  d = d * expf(m - m_global);

  float d_warp = warp_reduce_sum(d);
  if (lane_id == 0) {
    warp_d[warp_id] = d_warp;
  }

  __syncthreads();
  if (warp_id == 0) {
    float val = (lane_id < 8) ? warp_d[lane_id] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane_id == 0) {
      global_d = val;
    }
  }

  __syncthreads();
  float d_global = global_d;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    x[i] = expf(x[i] - m_global) / d_global;
  }
}

__global__ void standard_attention_kernel(const float* q, const float* k,
                                          const float* v, float* scores,
                                          float* out, int seq_len,
                                          int head_dim) {
  __shared__ float warp_max[8];
  __shared__ float warp_sum[8];

  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  float scale = rsqrtf((float)head_dim);

  // 1.scores = q @ k^T * scale (写入显存)
  for (int t = tid; t < seq_len; t += blockDim.x) {
    float s = 0.0f;
    for (int d = 0; d < head_dim; d++) {
      s += q[d] * k[t * head_dim + d];
    }
    scores[t] = s * scale;
  }
  __syncthreads();

  // 2. softmax(scores) 读显存，写显存
  float local_max = -CUDART_INF_F;
  for (int t = tid; t < seq_len; t += blockDim.x) {
    local_max = fmaxf(local_max, scores[t]);
  }
  local_max = warp_reduce_max(local_max);
  if (lane_id == 0) {
    warp_max[warp_id] = local_max;
  }
  __syncthreads();

  if (warp_id == 0) {
    float val = (lane_id < 8) ? warp_max[lane_id] : -CUDART_INF_F;
    val = warp_reduce_max(val);
    if (lane_id == 0) {
      warp_max[0] = val;
    }
  }
  __syncthreads();

  float max_val = warp_max[0];
  float local_sum = 0.0f;
  for (int t = tid; t < seq_len; t += blockDim.x) {
    scores[t] = expf(scores[t] - max_val);
    local_sum += scores[t];
  }
  local_sum = warp_reduce_sum(local_sum);
  if (lane_id == 0) {
    warp_sum[warp_id] = local_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    float val = (lane_id < 8) ? warp_sum[lane_id] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane_id == 0) {
      warp_sum[0] = val;
    }
  }
  __syncthreads();

  float sum = warp_sum[0];
  for (int t = tid; t < seq_len; t += blockDim.x) {
    scores[t] /= sum;
  }
  __syncthreads();

  // 3. out = probs @ v (从显存读取 scores 和 v)
  for (int d = tid; d < head_dim; d += blockDim.x) {
    float val = 0.0f;
    for (int t = 0; t < seq_len; t++) {
      val += scores[t] * v[t * head_dim + d];
    }
    out[d] = val;
  }
}

constexpr int BLOCK_SIZE = 32;
__global__ void flash_attention_kernel(const float* q, const float* k,
                                       const float* v, float* out, int seq_len,
                                       int head_dim) {
  extern __shared__ float smem[];
  float* s_block = smem;                         // [BLOCK_SIZE]
  float* o_smem = smem + BLOCK_SIZE;             // [head_dim]
  float* warp_m = smem + BLOCK_SIZE + head_dim;  // [8]
  float* warp_d = warp_m + 8;                    // [8]
  float* g_m = warp_d + 8;                       // [1]
  float* g_d = g_m + 1;                          // [1]

  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  float scale = rsqrtf((float)head_dim);

  // 初始化 o 和全局 m, d
  for (int d = tid; d < head_dim; d += blockDim.x) {
    o_smem[d] = 0.0f;
  }
  if (tid == 0) {
    *g_m = -CUDART_INF_F;
    *g_d = 0.0f;
  }
  __syncthreads();

  // 分块处理 k/v
  for (int start = 0; start < seq_len; start += BLOCK_SIZE) {
    int end = min(start + BLOCK_SIZE, seq_len);
    int len = end - start;

    // 1.并行计算块内 score, 每个线程负责一个 t
    // scores 存在 shared memory 里，不写显存
    for (int t = tid; t < len; t += blockDim.x) {
      float s = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        s += q[d] * k[(start + t) * head_dim + d];
      }
      s_block[t] = s * scale;
    }
    __syncthreads();

    // 2.规约得到块内最大值
    float local_max = -CUDART_INF_F;
    for (int t = tid; t < len; t += blockDim.x) {
      local_max = fmaxf(local_max, s_block[t]);
    }
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) {
      warp_m[warp_id] = local_max;
    }
    __syncthreads();

    // 块间规约
    if (warp_id == 0) {
      float val = (lane_id < 8) ? warp_m[lane_id] : -CUDART_INF_F;
      val = warp_reduce_max(val);
      if (lane_id == 0) {
        warp_m[0] = val;
      }
    }
    __syncthreads();

    float m_new = fmaxf(*g_m, warp_m[0]);

    // 3. 计算 correction，更新全局 d
    float correction = expf(*g_m - m_new);

    float local_d = 0.0f;
    for (int t = tid; t < len; t += blockDim.x) {
      local_d += expf(s_block[t] - m_new);
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

    // 4.更新输出: o = o * correction + exp(s - m_new) @ v_block
    for (int d = tid; d < head_dim; d += blockDim.x) {
      o_smem[d] *= correction;
      for (int t = 0; t < len; t++) {
        o_smem[d] += expf(s_block[t] - m_new) * v[(start + t) * head_dim + d];
      }
    }
    __syncthreads();
  }

  // 5.归一化，定回 HBM
  float d_global = *g_d;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    out[d] = o_smem[d] / d_global;
  }
}

int main() {
  const int head_dim = 4;
  const int seq_len = 4;

  float h_q[head_dim] = {0.1f, 0.2f, 0.3f, 0.4f};
  float h_k[seq_len * head_dim] = {
      0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
      0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f,
  };
  float h_v[seq_len * head_dim] = {
      1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
  };

  float *d_q, *d_k, *d_v, *d_scores, *d_out_std, *d_out_flash;
  cudaMalloc(&d_q, head_dim * sizeof(float));
  cudaMalloc(&d_k, seq_len * head_dim * sizeof(float));
  cudaMalloc(&d_v, seq_len * head_dim * sizeof(float));
  cudaMalloc(&d_scores, seq_len * sizeof(float));
  cudaMalloc(&d_out_std, head_dim * sizeof(float));
  cudaMalloc(&d_out_flash, head_dim * sizeof(float));

  cudaMemcpy(d_q, h_q, head_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k, seq_len * head_dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v, seq_len * head_dim * sizeof(float),
             cudaMemcpyHostToDevice);

  // softmax 测试
  {
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float* d_x;
    cudaMalloc(&d_x, 4 * sizeof(float));
    cudaMemcpy(d_x, h_x, 4 * sizeof(float), cudaMemcpyHostToDevice);
    softmax_kernel<<<1, 256>>>(d_x, 4);
    cudaDeviceSynchronize();
    cudaMemcpy(h_x, d_x, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    printf("=== softmax ===\n");
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
      printf("y[%d] = %.6f\n", i, h_x[i]);
      sum += h_x[i];
    }
    printf("sum = %.6f\n\n", sum);
  }

  // online softmax 测试
  {
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float* d_x;
    cudaMalloc(&d_x, 4 * sizeof(float));
    cudaMemcpy(d_x, h_x, 4 * sizeof(float), cudaMemcpyHostToDevice);
    online_softmax_kernel<<<1, 4>>>(d_x, 4);
    cudaDeviceSynchronize();
    cudaMemcpy(h_x, d_x, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    printf("=== online_softmax ===\n");
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
      printf("y[%d] = %.6f\n", i, h_x[i]);
      sum += h_x[i];
    }
    printf("sum = %.6f\n\n", sum);
  }

  // standard attention
  standard_attention_kernel<<<1, 256>>>(d_q, d_k, d_v, d_scores, d_out_std,
                                        seq_len, head_dim);
  cudaDeviceSynchronize();

  // flash attention
  size_t smem_size = (BLOCK_SIZE + head_dim + 8 + 8 + 2) * sizeof(float);
  flash_attention_kernel<<<1, 256, smem_size>>>(d_q, d_k, d_v, d_out_flash,
                                                seq_len, head_dim);
  cudaDeviceSynchronize();

  float h_out_std[head_dim], h_out_flash[head_dim];
  cudaMemcpy(h_out_std, d_out_std, head_dim * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out_flash, d_out_flash, head_dim * sizeof(float),
             cudaMemcpyDeviceToHost);

  printf("=== standard_attention ===\n");
  for (int i = 0; i < head_dim; i++)
    printf("out[%d] = %.6f\n", i, h_out_std[i]);

  printf("\n=== flash_attention ===\n");
  for (int i = 0; i < head_dim; i++)
    printf("out[%d] = %.6f\n", i, h_out_flash[i]);

  bool match = true;
  for (int i = 0; i < head_dim; i++)
    if (fabsf(h_out_std[i] - h_out_flash[i]) > 1e-4f) match = false;
  printf("\nmatch: %s\n", match ? "true" : "false");

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_scores);
  cudaFree(d_out_std);
  cudaFree(d_out_flash);
  return 0;

  return 0;
}

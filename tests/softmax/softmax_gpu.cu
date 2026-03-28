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
  for (int i = threadIdx.x; i < n; i+= blockDim.x) {
    x[i] = expf(x[i] - m_global) / d_global;
  }
}

int main() {
  float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};

  float* d_x;
  cudaMalloc(&d_x, sizeof(float) * 4);
  cudaMemcpy(d_x, x, sizeof(float) * 4, cudaMemcpyHostToDevice);

  online_softmax_kernel<<<1, 4>>>(d_x, 4);
  cudaDeviceSynchronize();

  cudaMemcpy(x, d_x, sizeof(float) * 4, cudaMemcpyDeviceToHost);
  cudaFree(d_x);

  float sum = 0.0f;
  for (int i = 0; i < 4; i++) {
    printf("x[%d] = %.6f\n", i, x[i]);
    sum += x[i];
  }
  printf("sum  = %.6f\n", sum);

  return 0;
}

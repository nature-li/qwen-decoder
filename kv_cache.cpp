#include "kv_cache.h"

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                                 \
  {                                                                                      \
    cudaError_t err = call;                                                              \
    if (err != cudaSuccess) {                                                            \
      fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
      exit(1);                                                                           \
    }                                                                                    \
  }

BlockPool::BlockPool(int total_blocks, int block_size, int n_layers, int kv_dim)
    : total_blocks_(total_blocks), block_size_(block_size), n_layers_(n_layers), kv_dim_(kv_dim) {
  // k_cache 大小 和 v_cache 大小
  size_t size = (size_t)total_blocks * block_size * n_layers * kv_dim * sizeof(__half);
  // 分配 k_cache
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&k_cache_), size));
  // 分配 v_cache
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&v_cache_), size));
  // 初始化时所有 block 为空闲状态
  for (int i = 0; i < total_blocks; i++) {
    free_blocks_.push(i);
  }
  // 记录分配了多少 blocks
  fprintf(stdout, "BlockPool: %d blocks, %.2fMB each, %.2fGB total\n", total_blocks,
          (float)(block_size * n_layers * kv_dim * 2 * sizeof(__half)) / 1024 / 1024,
          (float)(size * 2) / 1024 / 1024 / 1024);
}

BlockPool::~BlockPool() {
  cudaFree(k_cache_);
  cudaFree(v_cache_);
}

int BlockPool::allocate() {
  if (free_blocks_.empty()) {
    return -1;
  }
  int id = free_blocks_.front();
  free_blocks_.pop();
  return id;
}

void BlockPool::free(int block_id) { free_blocks_.push(block_id); }

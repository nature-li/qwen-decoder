#include "kv_cache.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call)                                                                 \
  {                                                                                      \
    cudaError_t err = call;                                                              \
    if (err != cudaSuccess) {                                                            \
      fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
      exit(1);                                                                           \
    }                                                                                    \
  }

BlockPool::BlockPool(int total_blocks, int block_size, int n_layers, int kv_dim)
    : total_blocks_(total_blocks),
      block_size_(block_size),
      n_layers_(n_layers),
      kv_dim_(kv_dim),
      ref_counts_(total_blocks, 0) {
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
  ref_counts_[id] = 1;
  return id;
}

void BlockPool::free(int block_id) { dec_ref(block_id); }

void BlockPool::inc_ref(int block_id) {
  ref_counts_[block_id]++;
}

void BlockPool::dec_ref(int block_id) {
  ref_counts_[block_id]--;
  if (ref_counts_[block_id] == 0) {
    // 引用计数降到 0，真正释放
    free_blocks_.push(block_id);
  }
}

int BlockPool::get_ref(int block_id) const { return ref_counts_[block_id]; }


#include "paged_kv_cache.h"

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                      \
  {                                                                           \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), \
              __LINE__);                                                      \
      exit(1);                                                                \
    }                                                                         \
  }

PagePool::PagePool(int total_pages, int block_size, int n_layers, int kv_dim)
    : total_pages(total_pages),
      block_size(block_size),
      n_layers(n_layers),
      kv_dim(kv_dim) {
  // 分配显存
  size_t size =
      (size_t)total_pages * block_size * n_layers * kv_dim * sizeof(__half);

  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&k_cache), size));
  fprintf(stderr,
          "k_cache allocated: total_size=%0.2fGB total_pages=%d "
          "page_size=%0.2fMB block_size=%d "
          "n_layers=%d kv_dim=%d\n",
          (float)size / 1024 / 1024 / 1024, total_pages,
          (float)(block_size * n_layers * kv_dim * 2 * sizeof(__half)) / 1024 /
              1024,
          block_size, n_layers, kv_dim);
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&v_cache), size));
  fprintf(stderr,
          "v_cache allocated: total_size=%0.2fGB total_pages=%d "
          "page_size=%0.2fMB block_size=%d "
          "n_layers=%d kv_dim=%d\n",
          (float)size / 1024 / 1024 / 1024, total_pages,
          (float)(block_size * n_layers * kv_dim * 2 * sizeof(__half)) / 1024 /
              1024,
          block_size, n_layers, kv_dim);

  // 所有块初始都是空闲的
  for (int i = 0; i < total_pages; i++) free_pages.push(i);
}

PagePool::~PagePool() {
  cudaFree(k_cache);
  cudaFree(v_cache);
}

int PagePool::allocate() {
  if (free_pages.empty()) return -1;
  int id = free_pages.front();
  free_pages.pop();
  return id;
}

void PagePool::free(int page_id) { free_pages.push(page_id); }
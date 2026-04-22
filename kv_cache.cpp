#include "kv_cache.h"

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

BlockPool::BlockPool(int total_blocks, int block_size, int n_layers, int kv_dim)
    : total_blocks_(total_blocks),
      block_size_(block_size),
      n_layers_(n_layers),
      kv_dim_(kv_dim),
      free_iter_(total_blocks),
      is_free_(total_blocks, true),
      ref_counts_(total_blocks, 0),
      block_hash_(total_blocks, 0) {
  // k_cache / v_cache 单独分配，大小各为 total_blocks * block_size * n_layers *
  // kv_dim * fp16
  size_t size =
      (size_t)total_blocks * block_size * n_layers * kv_dim * sizeof(__half);
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&k_cache_), size));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&v_cache_), size));

  // 初始化所有 block 入 free_list
  // 初始顺序无所谓（所有 block 的 block_hash_ 都是 0，都没价值）
  // push_back 顺序入：0,1,2,...,N-1；第一次 pop_back 拿 N-1
  for (int i = 0; i < total_blocks; i++) {
    free_list_.push_back(i);
    free_iter_[i] = std::prev(free_list_.end());
  }

  fprintf(stdout, "BlockPool: %d blocks, %.2fMB each, %.2fGB total\n",
          total_blocks,
          (float)(block_size * n_layers * kv_dim * 2 * sizeof(__half)) / 1024 /
              1024,
          (float)(size * 2) / 1024 / 1024 / 1024);
}

BlockPool::~BlockPool() {
  cudaFree(k_cache_);
  cudaFree(v_cache_);
}

int BlockPool::allocate() {
  if (free_list_.empty()) {
    return -1;
  }

  // 从尾部取最老的（hash 最不值钱，清理的"伤害"最小）
  int id = free_list_.back();
  free_list_.pop_back();
  // 不再空闲
  is_free_[id] = false;
  // 引用计数
  ref_counts_[id] = 1;

  // 如果这个 block 之前关联过 hash，必须清理
  // 因为 block 即将被覆盖写新 KV，旧 hash 不再对应真实内容
  uint64_t old_hash = block_hash_[id];
  if (old_hash != 0) {
    block_hash_[id] = 0;
    if (on_allocate_) {
      // 通知 PrefixCache 从 hash_to_block_ 中 erase 这个 hash
      on_allocate_(id, old_hash);
    }
  }

  return id;
}

void BlockPool::free(int block_id) { dec_ref(block_id); }

void BlockPool::inc_ref(int block_id) { ref_counts_[block_id]++; }

void BlockPool::dec_ref(int block_id) {
  ref_counts_[block_id]--;
  if (ref_counts_[block_id] == 0) {
    // ref 归 0，回到 free_list 头部（最新释放，hash 最值钱）
    // 关键：不清除 block_hash_，让 hash 继续有效
    // 这样之后同样 prefix 的请求来还能命中这个 block
    free_list_.push_front(block_id);
    free_iter_[block_id] = free_list_.begin();
    is_free_[block_id] = true;
  }
}

void BlockPool::take_from_free_list(int block_id) {
  // match 命中且 block 在 free_list 里：从中间取走
  // block_hash_ 保持不变（hash 依然有效，内容没变）
  free_list_.erase(free_iter_[block_id]);
  is_free_[block_id] = false;
  ref_counts_[block_id] = 1;
}
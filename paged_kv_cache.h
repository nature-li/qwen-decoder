#pragma once

#include <cuda_fp16.h>

#include <queue>
#include <vector>

// 每个 page 存多少个 token
constexpr int BLOCK_SIZE = 16;

/**
 * 管理所有物理块的分配和释放
 * k_cache: [total_pages, block_size, n_layers, kv_dim]
 * v_cache: [total_pages, block_size, n_layers, kv_dim]
 */
class PagePool {
 public:
  PagePool(int total_pages, int block_size, int n_layers, int kv_dim);
  ~PagePool();

  // 分配一个空闲物理块，返回物理块 id，没有空闲块返回 -1
  int allocate();

  // 释放一个物理块
  void free(int page_id);

  // 剩余空闲块数量
  int free_count() const { return (int)free_pages.size(); }

  __half* get_k_cache() const { return k_cache; }
  __half* get_v_cache() const { return v_cache; }

  int get_total_pages() const { return total_pages; }
  int get_block_size() const { return block_size; }
  int get_n_layers() const { return n_layers; }
  int get_kv_dim() const { return kv_dim; }

 private:
  int total_pages;
  int block_size;
  int n_layers;
  int kv_dim;

  /**
   * shape: [total_pages, block_size, n_layers, kv_dim]
   * shape: [total_pages, block_size, n_layers, kv_dim]
   */
  __half* k_cache;
  __half* v_cache;
  // 空闲块 id 列表
  std::queue<int> free_pages;
};

/**
 * 单个请求的逻辑块→物理块映射
 * block_table[i] = 第 i 个逻辑块对应的物理块 id
 * 例如：[3, 7, 12] 表示该请求用了 3 个物理块
 */
struct BlockTable {
  // 下标是逻辑块，值是物理块
  std::vector<int> table;  // 逻辑块 id → 物理块 id

  // 当前已分配的逻辑块数量
  int num_blocks() const { return (int)table.size(); }

  // 第 token_idx 个 token 在哪个物理块
  int physical_block(int token_idx) const {
    return table[token_idx / BLOCK_SIZE];
  }

  // 第 token_idx 个 token 在块内的偏移
  int block_offset(int token_idx) const { return token_idx % BLOCK_SIZE; }
};

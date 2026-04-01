#pragma once

#include <cuda_fp16.h>

#include <functional>
#include <queue>
#include <vector>

// 每个 block 存多少个 token
constexpr int BLOCK_SIZE = 16;

/**
 * BlockPool：管理所有物理块的分配和释放
 *
 * KV cache 布局:
 * [total_blocks, block_size, n_layers, kv_dim]
 * 等价于:
 * [total_slots, n_layers, kv_dim]
 *
 * 其中:
 * total_slots = total_blocks * block_size
 *
 * 访问方式:
 * token 对应的 slot
 * slot = physical_block_id * block_size + block_offset
 * token 对应的指定层 cache
 * offset = slot * n_layers * kv_dim + layer * kv_dim
 */
class BlockPool {
 public:
  BlockPool(int total_blocks, int block_size, int n_layers, int kv_dim);
  ~BlockPool();

  /**
   * 分配一个空闲物理块
   * @return 物理块 id, OOM 时返回 -1
   */
  int allocate();

  /**
   * 释放一个物理块
   * @block_id 物理块 id
   */
  void free(int block_id);

  /**
   * 获取当前空闲的物理块数量
   * @return 当前空闲的物理块数量
   */
  int free_count() const { return (int)free_blocks_.size(); }

  /**
   * 获取 k_cache
   * @return k_cache
   */
  __half* get_k_cache() const { return k_cache_; }

  /**
   * 获取 v_cache
   * @return v_cache
   */
  __half* get_v_cache() const { return v_cache_; }

  /**
   * 获取 cuda 上分配出的块的数量
   * @return cuda 上分配出的块的数量
   */
  int get_total_blocks() const { return total_blocks_; }

  /**
   * 获取单个块的大小
   * @return 单个块的大小
   */
  int get_block_size() const { return block_size_; }

  /**
   * 获取模型层数
   * @return 模型层数
   */
  int get_n_layers() const { return n_layers_; }

  /**
   * 获取模型维度
   * @return 模型维度
   */
  int get_kv_dim() const { return kv_dim_; }

 private:
  // cuda 上 malloc 出的 blocks 总数
  int total_blocks_;
  // 单个 block 大小，可以存放多少个 token
  int block_size_;
  // kv_cache 层数
  int n_layers_;
  // kv_cache 维度
  int kv_dim_;
  // [total_blocks, block_size, n_layers, kv_dim]
  __half* k_cache_;
  // [total_blocks, block_size, n_layers, kv_dim]
  __half* v_cache_;
  // 当前空闲的 block 列表
  std::queue<int> free_blocks_;
};

/**
 * BlockTable：单个请求的逻辑块 -> 物理块映射
 */
struct BlockTable {
 public:
  /**
   * 反回底层存储数据指针，上传 cuda 用
   * @return 底层连续内存地址
   */
  const int* data() const { return table_.data(); }

  /**
   * 添加 block
   * @param block_id 物理块 id
   */
  void add_block(int block_id) { table_.push_back(block_id); }

  /**
   * 获取第 pos 个 token 所在的物理块
   * @param pos token 的 offset
   * @return token 所在的物理块编号
   */
  int physical_idx(int pos) { return table_[pos / BLOCK_SIZE]; }

  /**
   * 获取第 pos 个 token 所在块的偏移
   * @param pos token 的 offset
   * @return token 所在(逻辑/物理)块的偏移
   */
  int block_offset(int pos) { return pos % BLOCK_SIZE; }

  /**
   * 获取 table 中分配到的块数量
   * @return table 中分配到的块数量
   */
  int num_blocks() const { return (int)table_.size(); }

  /**
   * 释放当前请求分配到的所有块
   * @param pool BlockPool
   */
  void free_blocks(std::function<void(int)> free_func) {
    for (int block_id : table_) {
      free_func(block_id);
    }
    table_.clear();
  }

 private:
  /**
   * table[i]: 第 i 个逻辑块对应的物理块 id
   */
  std::vector<int> table_;
};

#pragma once

#include <cuda_fp16.h>

#include <cstdint>
#include <functional>
#include <list>
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
 *
 *
 * 新设计要点（相比旧版的 PrefixCache+BlockPool 双层结构）:
 *
 * 1. free_list 用 std::list，支持从中间取走
 *    - dec_ref 归 0 时：push_front（最新的在头部）
 *    - allocate 时：pop_back（最老的在尾部，hash 最不值钱）
 *    - match 命中且 block 空闲时：take_from_free_list（中间取走）
 *
 * 2. 每个 block 记录自己当前关联的 hash（block_hash_）
 *    - 只有被 allocate 覆盖写时才清除
 *    - 在 free_list 里时 hash 依然有效，可被 match 命中复用
 *
 * 3. PrefixCache 通过回调清理自己的 hash map
 *    - BlockPool 不知道 hash map 的存在，职责清晰
 *    - block 被 allocate 覆盖时触发 on_allocate_ 回调，PrefixCache erase
 *
 * 优势:
 *   - prefix cache 容量 = 整个 BlockPool 的空闲量（动态、无上限）
 *   - hash 有效期最大化（block 只有真被覆盖才失效）
 *   - 引用计数单一持有（只有活跃请求持有，cache 本身不持有）
 */
class BlockPool {
 public:
  BlockPool(int total_blocks, int block_size, int n_layers, int kv_dim);
  ~BlockPool();

  /**
   * 分配一个空闲物理块（从 free_list 尾部取最老的）
   * 如果该 block 之前关联了 hash，会通过 on_allocate_ 回调
   * 通知 PrefixCache 清理对应条目（因为 block 即将被覆盖写新 KV）
   *
   * @return 物理块 id，OOM 时返回 -1
   */
  int allocate();

  /**
   * 释放一个物理块（等价于 dec_ref）
   * ref 归 0 时 push_front 到 free_list（最新释放的在头部）
   * 注意：不清除 block_hash_，让 hash 继续有效以便后续 match 命中复用
   */
  void free(int block_id);

  /**
   * 增加物理块的引用计数
   * match 命中且 block 正在被其他请求使用（不在 free_list）时调用
   */
  void inc_ref(int block_id);

  /**
   * 减少物理块的引用计数
   * 归 0 时 push_front 到 free_list（hash 保留）
   */
  void dec_ref(int block_id);

  /**
   * 获取引用计数，调试用
   */
  int get_ref(int block_id) const { return ref_counts_[block_id]; }

  /**
   * 判断 block 当前是否在 free_list 中
   * 等价于 ref_counts_[block_id] == 0
   */
  bool is_free(int block_id) const { return is_free_[block_id]; }

  /**
   * 把指定 block 从 free_list 中间取走（prefix cache 命中复用时用）
   * 调用后 ref_count 变为 1，block_hash_ 保持不变
   *
   * 前置条件：is_free(block_id) == true
   */
  void take_from_free_list(int block_id);

  /**
   * 记录 block 关联的 hash（PrefixCache 在 insert 时调用）
   * h 不能为 0（0 是"无 hash"的哨兵值；FNV-1a 永远不会生成 0）
   */
  void set_block_hash(int block_id, uint64_t h) { block_hash_[block_id] = h; }

  /**
   * 获取 block 当前关联的 hash，0 表示无映射
   */
  uint64_t get_block_hash(int block_id) const { return block_hash_[block_id]; }

  /**
   * 注册回调：block 被 allocate（即将被覆盖写）时调用
   * PrefixCache 在这里 erase 自己的 hash map
   *
   * 只会在构造 PrefixCache 时设置一次
   */
  void set_on_allocate_callback(
      std::function<void(int /*block_id*/, uint64_t /*old_hash*/)> cb) {
    on_allocate_ = std::move(cb);
  }

  int free_count() const { return (int)free_list_.size(); }
  __half* get_k_cache() const { return k_cache_; }
  __half* get_v_cache() const { return v_cache_; }
  int get_total_blocks() const { return total_blocks_; }
  int get_block_size() const { return block_size_; }
  int get_n_layers() const { return n_layers_; }
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

  // 空闲物理块链表
  //   - push_front：新释放（最新的，hash 最值钱）
  //   - pop_back：allocate 时取（最老的，hash 最不值钱）
  //   - 中间 erase：match 命中复用时
  std::list<int> free_list_;

  // block_id -> 它在 free_list 中的迭代器（O(1) 中间删除用）
  // 仅当 is_free_[block_id] == true 时有效
  std::vector<std::list<int>::iterator> free_iter_;

  // block_id -> 是否在 free_list 里（等价于 ref_counts_[id] == 0）
  std::vector<bool> is_free_;

  // 每个物理块的引用计数，表示当前有多少活跃请求持有它
  std::vector<int> ref_counts_;

  // block_id -> 当前关联的 hash，0 表示无映射
  // FNV-1a 的种子非 0 且永远不会输出 0，所以 0 可以安全作为哨兵
  std::vector<uint64_t> block_hash_;

  // block 被 allocate（即将覆盖写入）时的回调
  // 让 PrefixCache 清理 hash_to_block_ 中的过期条目
  std::function<void(int, uint64_t)> on_allocate_;
};

/**
 * BlockTable：单个请求的逻辑块 -> 物理块映射
 */
struct BlockTable {
 public:
  /**
   * 返回底层存储数据指针，上传 cuda 用
   */
  const int* data() const { return table_.data(); }

  /**
   * 添加 block
   */
  void add_block(int block_id) { table_.push_back(block_id); }

  /**
   * 获取第 block_idx 个逻辑块对应的物理块 id
   */
  int get_block_id(int block_idx) const { return table_[block_idx]; }

  /**
   * 获取第 pos 个 token 所在的物理块 id
   */
  int physical_idx(int pos) { return table_[pos / BLOCK_SIZE]; }

  /**
   * 获取第 pos 个 token 所在块的偏移
   */
  int block_offset(int pos) { return pos % BLOCK_SIZE; }

  /**
   * 获取 table 中分配到的块数量
   */
  int num_blocks() const { return (int)table_.size(); }

  /**
   * 释放当前请求分配到的所有块
   *
   * 反向遍历（从 table_[N-1] 到 table_[0]），配合 BlockPool::dec_ref 的
   * push_front:
   *
   * 一个重要的不变量（ref_count 单调性）:
   *   ref_count(p0) >= ref_count(p1) >= ... >= ref_count(pN-1)
   *   因为任何命中 p_k 的请求都必然命中 p0..p_{k-1}（前缀匹配的连续性）
   *
   * 这导致 dec_ref 后归 0 的 block 形成一个"从右到左的连续段":
   *   如果 p_k 归 0，则 p_{k+1}..p_{N-1} 也归 0
   *   如果 p_k 未归 0，则 p_0..p_{k-1} 也未归 0
   *
   * 反向遍历让这个连续段按如下顺序进 free_list 头部:
   *   最先 push_front: p_{N-1}（最长 prefix，进组的末尾）
   *   最后 push_front: p_k（最短归零的 prefix，停在组头部）
   *
   * allocate 从尾部取 -> 老块先回收 -> 长 prefix 先回收 -> 短 prefix 最晚回收
   *
   * 业务依据:
   *   短 prefix（如 system prompt 前缀）被多个请求共享的概率
   *   远高于长 prefix（长 prefix 是当前请求独有），短 prefix 复用价值高。
   */
  void free_blocks(std::function<void(int)> free_func) {
    for (auto it = table_.rbegin(); it != table_.rend(); ++it) {
      free_func(*it);
    }
    table_.clear();
  }

  size_t size() { return table_.size(); }

 private:
  // table[i]: 第 i 个逻辑块对应的物理块 id
  std::vector<int> table_;
};
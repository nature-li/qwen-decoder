#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kv_cache.h"

/**
 * Prefix Cache
 *
 * 以 block 为粒度缓存历史请求的 KV，key 是前缀 token 序列的累积 hash
 * 支持最长前缀匹配
 *
 * 设计:
 *   这是一个轻量级索引层，只维护 hash -> block_id 映射，不持有引用
 *
 *   block 的生命周期由 BlockPool 的 free_list 管理:
 *     - block ref 归 0 -> 进入 free_list（hash 仍有效，可被命中复用）
 *     - block 被 match 命中 -> 从 free_list 取走（hash 仍有效）
 *     - block 被 allocate 覆盖 -> BlockPool 回调通知，erase hash 映射
 *
 *   容量 = 整个 BlockPool，自动随空闲 block 数量伸缩，没有 max_entries 限制
 *
 * 例:
 *   prompt = [t0..t15, t16..t31, t32..t47]  (BLOCK_SIZE=16)
 *
 *   insert 时建立映射:
 *     hash([t0..t15])      -> 物理块 id=5
 *     hash([t0..t31])      -> 物理块 id=12
 *     hash([t0..t47])      -> 物理块 id=3
 *
 *   match 时:
 *     新请求 prompt = [t0..t15, t16..t31, t32..t63]
 *     -> 命中 block0(5) 和 block1(12)
 *     -> req->pos = 32，跳过前 32 个 token 的 prefill
 *     -> 只需要 prefill t32..t63
 */
class PrefixCache {
 public:
  /**
   * 构造时向 BlockPool 注册回调
   * 当 BlockPool 的 block 被 allocate 覆盖写时，回调会 erase 对应 hash
   */
  PrefixCache(BlockPool* pool);

  /**
   * 最长前缀匹配
   * 从第一个 block 开始逐块匹配，遇到未命中立即停止
   * 命中的物理块写入 block_table，并根据当前状态处理引用:
   *   - 在 free_list 里：take_from_free_list（ref 变 1）
   *   - 被其他请求持有：inc_ref（ref + 1）
   *
   * @param tokens 请求的完整 prompt token 序列
   * @param block_table 命中的物理块写入这里
   * @return 命中的 block 数量（0 表示未命中）
   */
  int match(const std::vector<int>& tokens, BlockTable& block_table);

  /**
   * prefill 完成后建立 hash -> block_id 映射
   * 只处理完整的 block（token 数不足 BLOCK_SIZE 的最后一个不建立映射）
   *
   * 不持有引用，不做 LRU 淘汰
   * 冲突（hash 已存在）时跳过，保持已有映射
   *
   * @param tokens 请求的完整 prompt token 序列
   * @param block_table 请求当前的 block table
   */
  void insert(const std::vector<int>& tokens, const BlockTable& block_table);

 private:
  /**
   * 计算前 n 个 token 的累积 hash（n 必须是 BLOCK_SIZE 的整数倍）
   * FNV-1a 哈希，逐 token 累积，相同前缀产生相同 hash 路径
   *
   * 注意：种子是 0xcbf29ce484222325，非 0，且乘法不会让 h 变 0
   *       所以 BlockPool 里用 0 做"无 hash"哨兵是安全的
   */
  uint64_t hash_prefix(const std::vector<int>& tokens, int n);

  BlockPool* pool_;

  // hash -> block_id 正向映射
  // 反向映射（block_id -> hash）存在 BlockPool::block_hash_ 里，
  // 让 BlockPool 在 allocate 时能 O(1) 查到要 erase 的 hash
  std::unordered_map<uint64_t, int> hash_to_block_;
};
#pragma once

#include <list>
#include <unordered_map>
#include <vector>
#include <cstdint>

#include "kv_cache.h"

/**
 * Prefix Cache
 *
 * 以 block 为粒度缓存历史请求的 KV，key 是前缀 token 序列的累积 hash
 * 支持最长前缀匹配和 LRU 淘汰
 *
 * 例:
 *   prompt = [t0..t15, t16..t31, t32..t47]  (BLOCK_SIZE=16)
 *
 *   insert 时存入:
 *     hash([t0..t15])      → 物理块 id=5
 *     hash([t0..t31])      → 物理块 id=12
 *     hash([t0..t47])      → 物理块 id=3   (最后一个完整块)
 *
 *   match 时:
 *     新请求 prompt=[t0..t15, t16..t31, t32..t63]
 *     → 命中 block0(5) 和 block1(12)
 *     → req->pos = 32，跳过前 32 个 token 的 prefill
 *     → 只需要 prefill t32..t63
 */
class PrefixCache {
 public:
  /**
   * @param pool BlockPool，用于管理物理块的引用计数
   * @param max_entries LRU cache 最大条目数，超出时淘汰最久未使用的
   */
  PrefixCache(BlockPool* pool, int max_entries = 4096);

  /**
   * 最长前缀匹配
   * 从第一个 block 开始逐块匹配，遇到未命中立即停止
   * 命中的物理块引用计数 +1，并写入 block_table
   *
   * @param tokens     请求的完整 prompt token 序列
   * @param block_table 命中的物理块写入这里
   * @return 命中的 block 数量（0 表示未命中）
   */
  int match(const std::vector<int>& tokens, BlockTable& block_table);

  /**
   * prefill 完成后写入 cache
   * 只存完整的 block（token 数不足 BLOCK_SIZE 的最后一个 block 不存）
   * 存入的物理块引用计数 +1
   *
   * @param tokens     请求的完整 prompt token 序列
   * @param block_table 请求当前的 block table
   */
  void insert(const std::vector<int>& tokens, const BlockTable& block_table);

 private:
  /**
   * 计算前 n 个 token 的累积 hash（n 必须是 BLOCK_SIZE 的整数倍）
   * 用 FNV-1a 哈希，逐 token 累积，保证前缀相同的 hash 路径一致
   */
  uint64_t hash_prefix(const std::vector<int>& tokens, int n);

  /**
   * LRU 淘汰一个最久未使用的条目
   * 淘汰时物理块引用计数 -1
   */
  void evict_one();

  BlockPool* pool_;
  int max_entries_;

  // LRU 链表，存 hash key，最近访问的在链表头部
  std::list<uint64_t> lru_list_;

  struct CacheEntry {
    int block_id;                          // 对应的物理块 id
    std::list<uint64_t>::iterator lru_it;  // 在 lru_list_ 中的位置，用于 O(1) 移到头部
  };

  // hash → CacheEntry
  std::unordered_map<uint64_t, CacheEntry> cache_;
};
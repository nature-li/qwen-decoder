#include "prefix_cache.h"

#include <cstdio>

PrefixCache::PrefixCache(BlockPool* pool, int max_entries)
    : pool_(pool), max_entries_(max_entries) {}

uint64_t PrefixCache::hash_prefix(const std::vector<int>& tokens, int n) {
  // FNV-1a hash，逐 token 累积
  uint64_t h = 14695981039346656037ULL;
  for (int i = 0; i < n; i++) {
    h ^= (uint64_t)tokens[i];
    h *= 1099511628211ULL;
  }
  return h;
}

int PrefixCache::match(const std::vector<int>& tokens, BlockTable& block_table) {
  int n_full_blocks = (int)tokens.size() / BLOCK_SIZE;
  int hit_blocks = 0;

  for (int b = 0; b < n_full_blocks; b++) {
    uint64_t h = hash_prefix(tokens, (b + 1) * BLOCK_SIZE);
    auto it = cache_.find(h);
    if (it == cache_.end()) {
      // 未命中，停止匹配
      break;
    }

    // 命中，移到 LRU 头部
    lru_list_.erase(it->second.lru_it);
    lru_list_.push_front(h);
    it->second.lru_it = lru_list_.begin();

    // 物理块引用计数 +1，写入 block_table
    int block_id = it->second.block_id;
    pool_->inc_ref(block_id);
    block_table.add_block(block_id);
    hit_blocks++;
  }

//   if (hit_blocks > 0) {
//     fprintf(stderr, "prefix cache hit: %d blocks (%d tokens)\n", hit_blocks,
//             hit_blocks * BLOCK_SIZE);
//   }

  return hit_blocks;
}

void PrefixCache::insert(const std::vector<int>& tokens, const BlockTable& block_table) {
  int n_full_blocks = (int)tokens.size() / BLOCK_SIZE;

  // 只存完整的 block
  for (int b = 0; b < n_full_blocks && b < block_table.num_blocks(); b++) {
    uint64_t h = hash_prefix(tokens, (b + 1) * BLOCK_SIZE);

    // 已经在 cache 里了，不重复存
    if (cache_.find(h) != cache_.end()) {
      continue;
    }

    // cache 满了，先淘汰一个
    if ((int)cache_.size() >= max_entries_) {
      evict_one();
    }

    // cache 持有一份引用
    int block_id = block_table.get_block_id(b);
    pool_->inc_ref(block_id);

    lru_list_.push_front(h);
    cache_[h] = {block_id, lru_list_.begin()};
  }
}

void PrefixCache::evict_one() {
  if (lru_list_.empty()) return;

  uint64_t h = lru_list_.back();
  lru_list_.pop_back();

  auto it = cache_.find(h);
  if (it != cache_.end()) {
    // cache 释放引用
    pool_->dec_ref(it->second.block_id);
    cache_.erase(it);
  }
}
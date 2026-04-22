#include "prefix_cache.h"

#include <cstdio>

PrefixCache::PrefixCache(BlockPool* pool) : pool_(pool) {
  // 向 BlockPool 注册回调:
  // 当 block 被 allocate（即将被覆盖写）时，清理对应的 hash 映射
  // 这样 hash_to_block_ 和 pool.block_hash_ 始终保持一致
  pool_->set_on_allocate_callback([this](int /*block_id*/, uint64_t old_hash) {
    hash_to_block_.erase(old_hash);
  });
}

uint64_t PrefixCache::hash_prefix(const std::vector<int>& tokens, int n) {
  // FNV-1a 64-bit
  // 种子: 0xcbf29ce484222325 (非 0)
  // 乘数: 0x100000001b3
  uint64_t h = 14695981039346656037ULL;
  for (int i = 0; i < n; i++) {
    h ^= (uint64_t)tokens[i];
    h *= 1099511628211ULL;
  }
  return h;
}

int PrefixCache::match(const std::vector<int>& tokens,
                       BlockTable& block_table) {
  int n_full_blocks = (int)tokens.size() / BLOCK_SIZE;
  int hit_blocks = 0;

  for (int b = 0; b < n_full_blocks; b++) {
    uint64_t h = hash_prefix(tokens, (b + 1) * BLOCK_SIZE);
    auto it = hash_to_block_.find(h);
    if (it == hash_to_block_.end()) {
      // 未命中，停止（前缀匹配需要连续，中间断了就不再往后找）
      break;
    }

    int block_id = it->second;

    // 根据 block 当前状态处理引用:
    //   - 在 free_list 里（空闲）：从 free_list 取走，ref 变 1
    //   - 不在 free_list（被其他请求持有）：inc_ref
    if (pool_->is_free(block_id)) {
      pool_->take_from_free_list(block_id);
    } else {
      pool_->inc_ref(block_id);
    }

    block_table.add_block(block_id);
    hit_blocks++;
  }

  fprintf(stderr, "命中了 %d 个 blocks\n", hit_blocks);
  return hit_blocks;
}

void PrefixCache::insert(const std::vector<int>& tokens,
                         const BlockTable& block_table) {
  int n_full_blocks = (int)tokens.size() / BLOCK_SIZE;

  // 只处理完整的 block
  // 不完整的最后一个 block（< BLOCK_SIZE tokens）不建立映射
  // 因为它的内容还会继续追加，hash 会失效
  for (int b = 0; b < n_full_blocks && b < block_table.num_blocks(); b++) {
    uint64_t h = hash_prefix(tokens, (b + 1) * BLOCK_SIZE);

    // hash 已存在：说明有另一个 block 已经缓存了这段前缀
    // 跳过，保持已有映射（不覆盖）
    if (hash_to_block_.count(h) > 0) {
      continue;
    }

    int block_id = block_table.get_block_id(b);

    // 建立双向映射:
    //   hash_to_block_[h] = block_id   （PrefixCache 持有）
    //   pool.block_hash_[block_id] = h （BlockPool 持有，allocate 时用）
    hash_to_block_[h] = block_id;
    pool_->set_block_hash(block_id, h);
  }
}
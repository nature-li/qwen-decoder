#include "scheduler.h"

#include <iostream>

Scheduler::Scheduler(int max_batch, int block_size, BlockPool* pool, bool enable_prefix_cache)
    : max_batch_(max_batch),
      block_size_(block_size),
      pool_(pool),
      enable_prefix_cache_(enable_prefix_cache),
      prefix_cache_(pool) {
  running_.resize(max_batch, nullptr);
}

void Scheduler::add_request(Request* req) { waiting_.push(req); }

void Scheduler::fill_slots() {
  for (int i = 0; i < max_batch_; i++) {
    if (waiting_.empty()) {
      break;
    }

    if (running_[i] != nullptr) {
      continue;
    }

    Request* req = waiting_.front();
    waiting_.pop();

    if (enable_prefix_cache_) {
      // 查 prefix cache，命中则复用物理块跳过 prefill
      int hit_blocks = prefix_cache_.match(req->prompt_tokens, req->block_table);
      if (hit_blocks > 0) {
        // 跳过已命中的 token，直接从命中位置开始 prefill
        req->pos = hit_blocks * BLOCK_SIZE;
        req->prefill_offset = hit_blocks * BLOCK_SIZE;
      }
    }

    running_[i] = req;
  }
}

void Scheduler::release_finished() {
  for (int i = 0; i < max_batch_; i++) {
    if (running_[i] == nullptr || !running_[i]->finished) {
      continue;
    }

    running_[i]->block_table.free_blocks([&](int block_id) { pool_->free(block_id); });
    running_[i] = nullptr;
  }
}

bool Scheduler::all_done() const {
  if (!waiting_.empty()) {
    return false;
  }

  for (auto* r : running_) {
    if (r != nullptr) {
      return false;
    }
  }
  return true;
}

bool Scheduler::ensure_blocks(Request* req, int n_tokens) {
  // 计算该请求至少分配多少块
  int need_blocks = (req->pos + n_tokens - 1) / block_size_ + 1;
  // 一直分到足够的数据为止
  while ((int)req->block_table.num_blocks() < need_blocks) {
    int block_id = pool_->allocate();
    if (block_id == -1) {
      fprintf(stderr, "OOM: no free blocks\n");
      return false;
    }
    req->block_table.add_block(block_id);
  }
  return true;
}

void Scheduler::on_prefill_done(Request* req) {
  if (enable_prefix_cache_) {
    prefix_cache_.insert(req->prompt_tokens, req->block_table);
  }
}

#include "scheduler.h"

Scheduler::Scheduler(int max_batch, int block_size, BlockPool* pool)
    : max_batch_(max_batch), block_size_(block_size), pool_(pool) {
  running_.resize(max_batch, nullptr);
}

void Scheduler::add_request(Request* req) { waiting_.push(req); }

void Scheduler::fill_slots() {
  for (int i = 0; i < max_batch_; i++) {
    if (running_[i] == nullptr && !waiting_.empty()) {
      running_[i] = waiting_.front();
      waiting_.pop();
    }
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

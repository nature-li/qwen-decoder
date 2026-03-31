#include "scheduler.h"

Scheduler::Scheduler(int max_batch, int block_size, PagePool* pool)
    : max_batch(max_batch), block_size(block_size), pool(pool) {
  running.resize(max_batch, nullptr);
}

void Scheduler::add_request(Request* req) { waiting.push(req); }

void Scheduler::fill_slots() {
  for (int i = 0; i < max_batch; i++) {
    if (running[i] == nullptr && !waiting.empty()) {
      running[i] = waiting.front();
      waiting.pop();
    }
  }
}

void Scheduler::release_finished() {
  for (int i = 0; i < max_batch; i++) {
    if (running[i] != nullptr && running[i]->finished) {
      for (int page_id : running[i]->block_table.table) {
        pool->free(page_id);
      }
      running[i]->block_table.table.clear();
      running[i] = nullptr;
    }
  }
}

bool Scheduler::all_done() const {
  if (!waiting.empty()) return false;
  for (auto* r : running)
    if (r != nullptr) return false;
  return true;
}

bool Scheduler::ensure_pages(Request* req, int n_tokens) {
  int need_until = req->pos + n_tokens - 1;
  int need_blocks = need_until / block_size + 1;

  // fprintf(stderr,
  //         "ensure_pages: req%d pos=%d n_tokens=%d need_blocks=%d current=%d\n",
  //         req->id, req->pos, n_tokens, need_blocks,
  //         (int)req->block_table.table.size());

  while ((int)req->block_table.table.size() < need_blocks) {
    int page_id = pool->allocate();
    if (page_id == -1) {
      fprintf(stderr, "OOM: no free pages\n");
      return false;
    }
    req->block_table.table.push_back(page_id);
  }

  return true;
}
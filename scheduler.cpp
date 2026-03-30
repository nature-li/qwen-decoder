#include "scheduler.h"

Scheduler::Scheduler(int max_batch) : max_batch(max_batch) {
  running.resize(max_batch, nullptr);
}

void Scheduler::add_request(Request* req) {
  waiting.push(req);
}

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
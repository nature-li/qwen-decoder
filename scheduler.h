#pragma once

#include <queue>
#include <string>
#include <vector>

#include "paged_kv_cache.h"

struct Request {
  int id;
  std::string input;
  std::vector<int> prompt_tokens;
  int pos = 0;
  int cur_token = 0;
  bool prefill_done = false;
  bool finished = false;
  int n_generated = 0;
  std::string output;

  // PagedAttention：每个请求有自己的 BlockTable
  BlockTable block_table;
};

class Scheduler {
 public:
  Scheduler(int max_batch, int block_size, PagePool* pool);

  // 添加新请求到等待队列
  void add_request(Request* req);

  // 把等待队列里的请求填入空槽
  void fill_slots();

  // 释放完成的请求槽位
  void release_finished();

  // 是否全部完成
  bool all_done() const;

  // 为请求分配 page，确保 start_pos 到 start_pos + n_tokens 的 KV
  bool ensure_pages(Request* req, int n_tokens);

  int max_batch;
  int block_size;
  PagePool* pool;
  std::queue<Request*> waiting;
  std::vector<Request*> running;  // nullptr 表示空槽
};
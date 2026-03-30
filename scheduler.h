#pragma once

#include <queue>
#include <string>
#include <vector>

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
};

class Scheduler {
 public:
  Scheduler(int max_batch);

  // 添加新请求到等待队列
  void add_request(Request* req);

  // 把等待队列里的请求填入空槽
  void fill_slots();

  // 释放完成的请求槽位
  void release_finished();

  // 是否全部完成
  bool all_done() const;

  int max_batch;
  std::queue<Request*> waiting;
  std::vector<Request*> running;  // nullptr 表示空槽
};
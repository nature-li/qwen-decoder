#pragma once

#include <queue>
#include <string>
#include <vector>

#include "kv_cache.h"
#include "prefix_cache.h"

struct Request {
  // 请求 id
  int id = -1;
  // 原始 prompt
  std::string input;
  // prompt tokens
  std::vector<int> prompt_tokens;
  // token 的 offset
  int pos = 0;
  // 当前 step 预测出的 token
  int cur_token = 0;
  // 已经 prefill 了多少个 token（chunked prefill 用）
  int prefill_offset = 0;
  // 生成的 token 数量
  int n_generated = 0;
  // prefill 阶段是否完成
  bool prefill_done = false;
  // 请求是否处理完成
  bool finished = false;
  // decode 输出的输出
  std::string output;
  // 该请求的逻辑块 -> 物理块映射
  BlockTable block_table;
};

class Scheduler {
 public:
  Scheduler(int max_batch, int block_size, BlockPool* pool, bool enable_prefix_cache = true);

  /**
   * 添加新请求到等待队列
   * @param req
   */
  void add_request(Request* req);

  /**
   * 把等待队列里的请求填入空槽
   */
  void fill_slots();

  /**
   * 释放完成请求的槽位和 block
   */
  void release_finished();

  /**
   * 所有请求是否完成
   * @return true: 完成; false: 未全部会成
   */
  bool all_done() const;

  /**
   * 为请求分配足够的 block
   * @param req:
   * @param n_tokens: 为多少 tokens 分配 block
   */
  bool ensure_blocks(Request* req, int n_tokens);

  /**
   * prefill 完成时调用，把该请求的 prefix block 写入 cache
   * 供后续相同 prompt 的请求复用，跳过 prefill 计算
   */
  void on_prefill_done(Request* req);

  /**
   * 获取同时处理的请求数据
   * @return 同时处理的请求数量
   */
  int get_max_batch() const { return max_batch_; }

  /**
   * 获取正在运行的 Request
   * @param id: running 的槽位 id
   * @return
   */
  Request* get_running_req(int id) { return running_[id]; }

  /**
   * 获取所有正在运行的 Request
   */
  const std::vector<Request*>& running() const { return running_; }

  // 本步 block_table 发生变化的槽位下标
  std::vector<int> last_changed;

 private:
  // 控制同时有多少个请求在 running 队列里
  int max_batch_;
  // 块大小
  int block_size_;
  // 内存池
  BlockPool* pool_;
  // waiting 队列
  std::queue<Request*> waiting_;
  // 正在运行的请求，nullptr 表示空槽
  std::vector<Request*> running_;
  // prefix cache 开关
  bool enable_prefix_cache_;
  // Prefix Cache，以 block 为粒度缓存历史请求的 KV
  PrefixCache prefix_cache_;
};

#pragma once

#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "scheduler.h"

/**
 * FlatRequest：描述 flat batch 里每个请求的元信息
 *
 * flat batch 是把所有请求的 token 打包成一个连续数组:
 *   [req0_tokens..., req1_tokens..., req2_tokens..., ...]
 *
 * prefill 请求贡献 n_tokens 个 token（chunked）
 * decode  请求贡献 1 个 token
 */
struct FlatRequest {
  // 对应 scheduler.running 里的下标
  int req_idx;
  // 在 flat batch 里的起始 token 位置
  int flat_offset;
  // 这个请求贡献几个 token
  int n_tokens;
  // KV cache 里的起始位置（=req->pos）
  int start_pos;
  // true=prefill，false=decode
  bool is_prefill;
};

class Decoder {
 public:
  virtual ~Decoder() = default;

  /**
   * 1-flat 推理：把所有请求的 token 打包成一个 flat batch，一次 forward 处理
   *
   * 设计思路：
   * - matmul/ffn：所有 token 一起处理，batch 越大 GPU 利用率越高
   * - prefill attention：每个请求单独调用，串行处理（causal mask 不同）
   * - decode attention：所有 decode token 打包并行处理
   * - KV cache 写入：通过 slot_mapping 直接定位物理槽位
   *
   * chunked prefill：
   * - 每步 prefill token 总数不超过 max_prefill_tokens_per_step
   * - 避免长 prompt 导致 total_tokens 突然变大，保持每步 GPU 负载稳定
   *
   * 调用方负责：
   * - 1. 调用前通过 update_block_table_partial 更新有变化的槽位
   * - 2. 按 flat_requests 顺序调用 get_logits_batch(fi) 获取采样结果
   */
  virtual void forward_flat(
      const std::vector<FlatRequest>& flat_requests,
      // [total_tokens] 所有请求的 token id, 按 flat 顺序打包
      const std::vector<int>& flat_tokens,
      // [total_tokens] 每个 token 的绝对位置（pos），用于 RoPE
      const std::vector<int>& flat_positions,
      // [total_tokens] 每个 token 属于哪个请求（查 block_table 用）
      const std::vector<int>& token_to_seq,
      // [total_tokens] 每个 token 对应的绝对物理槽位，用于 KV cache 写入
      const std::vector<int>& slot_mapping,
      // [batch_size] 每个请求最后一个 token 在 flat batch 里的位置，用于提取 logits
      const std::vector<int>& last_token_indices,
      // [n_decode] decode token 在 flat batch 里的位置，用于 gather/scatter
      const std::vector<int>& decode_flat_indices,
      // [n_decode] decode token 的绝对位置，用于 decode attention
      const std::vector<int>& decode_positions,
      // [n_decode] decode token 属于哪个请求，用于查 block_table
      const std::vector<int>& decode_req_indices,
      // flat batch 里的 token 总数
      int total_tokens) = 0;

  /**
   * 获取第 batch_idx 个请求的 logits
   * batch_idx 对应 flat_requests 的下标，不是 scheduler.running 的下标
   * 返回的指针指向 pinned host memory，大小为 vocab_size 个 float
   */
  virtual float* get_logits_batch(int batch_idx) = 0;

  /**
   * 返回 KV cache 物理块管理器
   * generate_continuous 通过它初始化 Scheduler，用于分配和释放物理块
   */
  virtual BlockPool* get_block_pool() { return nullptr; }

  /**
   * 返回最大并发数
   */
  virtual int get_max_batch() const {return 0;}

  /**
   * 返回单个请求最多占用多少个逻辑块
   * 由 seq_len 和 BLOCK_SIZE 决定：(seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE
   * 用于分配 block_table_gpu 的每行大小
   */
  virtual int get_max_blocks_per_seq() const { return 0; }

  /**
   * 每步允许处理的最大 prefill token 数量（全局预算，不是单个请求）
   * 作用：
   *  控制每步 total_tokens 的上限，避免长 prompt 导致 GPU 负载突变
   *  prefill token 和 decode token 共享 total_tokens 预算：
   *   - total_tokens = prefill_tokens + n_decode
   *   - prefill_tokens <= get_max_prefill_tokens_per_step()
   *
   * 影响:
   * - 值越大：TTFT 越低（prefill 更快完成），但每步 GPU 负载波动更大
   * - 值越小：每步负载更稳定，但 TTFT 更高（prefill 需要更多步才能完成）
   */
  virtual int get_max_prefill_tokens_per_step() const { return 4096; }

  /**
   * 增量更新 GPU 上的 block_table，只更新有变化的槽位
   * 相比全量更新的优化：
   * - 全量：每步上传 max_batch * max_blocks_per_seq * 4 bytes
   * - 增量：只上传 changed_indices 里的槽位，且每个槽位只传实际用到的 block 数
   * - 显著减少每步的 HtoD memcpy 开销
   *
   * 什么时候需要更新：
   * - 1. 新请求填入空槽（prefill_offset == 0）
   * - 2. ensure_pages 为请求分配了新的物理块
   */
  virtual void update_block_table_partial(
      const std::vector<Request*>& running,  // 所有槽位的请求指针
      // 本步发生变化的槽位下标（由 generate_continuous 维护）
      const std::vector<int>& changed_indices,
      int max_blocks_per_seq)  // 每行的大小
  {}

  /**
   * 连续批处理推理主循环
   *
   * 整体流程：
   * - 1. 把所有请求加入 waiting 队列
   * - 2. 每步：fill_slots → 组装 flat batch → forward_flat → 采样 → 释放完成请求
   * - 3. 所有请求完成后退出
   *
   * 调度策略：
   * - continuous batching：请求完成后立刻让出槽位给 waiting 队列里的新请求
   * - chunked prefill：每步 prefill token 总数不超过 max_prefill_tokens_per_step
   * - 多个请求共享 prefill 预算，避免长 prompt 独占 GPU
   * - 增量 block_table 更新：只更新有变化的槽位，减少 HtoD memcpy
   *
   * flat batch 组装规则：
   * - prefill 请求：贡献 min(remaining, prefill_budget) 个 token
   * - decode  请求：贡献 1 个 token
   * - decode token 单独收集，用于并行 attention
   */
  void generate_continuous(std::vector<std::string>& user_inputs,  // 所有请求的输入文本
                           int max_batch,       // 最大并发请求数，决定 running 队列大小
                           int max_new_tokens,  // 每个请求最多生成多少个 token
                           float temperature,   // 采样温度，0 表示 greedy decoding
                           int top_k,           // top-k 采样
                           std::mt19937& rng,  // 随机数生成器
                           bool enable_prefix_cache = true); // 是否开启 prefix kv_cache

  const Config& get_config() const { return config; }

  std::string apply_chat_template_pub(const std::string& input) {
    return apply_chat_template(input);
  }

  void encode_pub(const std::string& prompt, std::vector<int>& tokens) {
    encode(tokenizer, prompt, tokens);
  }

  Tokenizer& get_tokenizer() { return tokenizer; }

 protected:
  // 模型配置
  Config config;
  // token 编解码
  Tokenizer tokenizer;
};

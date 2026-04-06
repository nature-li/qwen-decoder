#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "decoder.h"
#include "gguf_loader.h"

// ============================================================================
// GPU 权重：fp16 上传到 GPU，每层单独分配
// ============================================================================
struct GPUWeights {
  __half* token_embedding;  // [vocab_size, dim]
  float* rms_final;         // [dim]
  __half* wcls;             // [vocab_size, dim]

  float** rms_att;  // [n_layers][dim]
  __half** wq;      // [n_layers][dim, dim]
  __half** wk;      // [n_layers][kv_dim, dim]
  __half** wv;      // [n_layers][kv_dim, dim]
  __half** wo;      // [n_layers][dim, dim]
  float** bq;       // [n_layers][dim]
  float** bk;       // [n_layers][kv_dim]
  float** bv;       // [n_layers][kv_dim]
  float** rms_ffn;  // [n_layers][dim]
  __half** w1;      // [n_layers][hidden_dim, dim]
  __half** w2;      // [n_layers][dim, hidden_dim]
  __half** w3;      // [n_layers][hidden_dim, dim]
};

// ============================================================================
// GPU 运行时缓冲区
// ============================================================================
struct GPURunState {
  // flat batch 主缓冲区 [max_flat_tokens, *]
  __half* x;    // hidden state      [max_flat_tokens, dim]
  __half* xb;   // after attention   [max_flat_tokens, dim]
  __half* xb2;  // after ffn proj    [max_flat_tokens, dim]
  __half* q;    // query             [max_flat_tokens, dim]
  __half* k;    // key               [max_flat_tokens, kv_dim]
  __half* v;    // value             [max_flat_tokens, kv_dim]
  __half* hb;   // ffn hidden 1      [max_flat_tokens, hidden_dim]
  __half* hb2;  // ffn hidden 2      [max_flat_tokens, hidden_dim]

  // flat batch 元数据 [max_flat_tokens]
  int* d_tokens;     // token id
  int* d_positions;  // 每个 token 的绝对 pos
  int* d_token_seq;  // 每个 token 属于哪个请求（查 block_table 用）
  int* d_slot_map;   // 每个 token 的绝对物理槽位（KV cache 写入用）

  // PagedAttention block table [max_batch, max_blocks_per_seq]
  int* block_table;

  // logits 计算
  int* d_last_tok_idx;      // [max_batch] 每个请求最后一个 token 的 flat 下标
  __half* logits_last_tok;  // [max_batch, dim] 提取出的最后 token 特征
  __half* logits_fp16;      // [max_batch, vocab_size]
  float* logits;            // [max_batch, vocab_size] pinned host memory

  // prefill 专用缓冲区
  int* d_prefill_flat;
  // decode 专用缓冲区（所有 decode token 并行 attention 用）
  int* d_decode_flat;
};

// ============================================================================
// GPUDecoder
// ============================================================================
class GPUDecoder : public Decoder {
 public:
  GPUDecoder(const std::string& model_file,
             int max_concurrent_seqs,  // 最大并发请求数（原 max_batch）
             int max_prefill_len, int max_prefill_tokens_per_step,
             int max_total_tokens);  // 每步最多处理多少 token（新增）
  ~GPUDecoder();

  void forward_flat(const std::vector<FlatRequest>& flat_requests,
                    const std::vector<int>& flat_tokens, const std::vector<int>& flat_positions,
                    const std::vector<int>& token_to_seq, const std::vector<int>& slot_mapping,
                    const std::vector<int>& last_token_indices,
                    const std::vector<int>& decode_flat_indices, int total_tokens) override;

  float* get_logits_batch(int batch_idx) override;
  BlockPool* get_block_pool() override { return block_pool; }
  int get_max_batch() const { return max_batch_; }
  int get_max_blocks_per_seq() const override { return max_blocks_per_seq_; }
  int get_max_prefill_tokens_per_step() const override { return max_prefill_tokens_per_step_; }

  void update_block_table_partial(const std::vector<Request*>& running,
                                  const std::vector<int>& changed_indices,
                                  int max_blocks_per_seq) override;

 private:
  /**
   * 最大并发请求数
   * 决定 running 队列大小、block_table/logits 等缓冲区的行数
   * decode 阶段 total_tokens ≈ max_batch，max_batch 越大 GPU 利用率越高
   */
  int max_batch_;

  /**
   * flat batch 缓冲区大小（x_flat/q_flat/k_flat 等的第一维）
   * = max_prefill_tokens_per_step + max_batch
   * 最坏情况:
   * 本步所有 prefill 预算都用完(max_prefill_tokens_per_step 个 prefill token)
   * +
   * 所有请求各出一个 decode token(max_batch 个 decode token)
   */
  int max_flat_tokens_;  // max_prefill_tokens_per_step + max_batch

  /**
   * 单个请求 prompt 的最大长度
   * 超过此长度的请求会被截断
   * 影响:
   * chunked prefill 的分片次数 = ceil(prompt_len / max_prefill_tokens_per_step)
   */
  int max_prefill_len_;

  /**
   * 单个请求最多占用多少个逻辑块
   * = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE
   * 决定 block_table_gpu 每行的大小
   */
  int max_blocks_per_seq_;

  /**
   * 每步允许处理的最大 prefill token 数（所有请求共享的预算）
   * 同时控制两件事:
   * - 1. max_flat_tokens 的上限（每步 GPU 处理多少 token）
   * - 2. TTFT：值越大 prefill 越快完成，但每步负载波动越大
   */
  int max_prefill_tokens_per_step_;

  ModelFile mf;
  GGUFFile gguf;
  Weights w;
  GPUWeights gw;
  GPURunState gs;
  BlockPool* block_pool;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
};

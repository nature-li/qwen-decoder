#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "decoder.h"
#include "gguf_loader.h"
#include "gpu_decoder.h"
#include "gpu_kernels.h"
#include "scheduler.h"

// ============================================================
// TP 权重结构
//
// GGUF 权重物理存储：[out_features, in_features]
//
// 切分规则（Megatron-LM 列并行）：
//   wq/wk/wv/w1/w3：按输出维度切，物理内存外层，连续 memcpy
//   wo/w2：          按输入维度切，物理内存内层，需逐行拷贝
// ============================================================
struct TPWeights {
  __half* token_embedding;
  float* rms_final;
  __half* wcls;

  float** rms_att;
  __half** wq;  // 按 out 切 [dim/n_ranks, dim]
  __half** wk;  // 按 out 切 [kv_dim/n_ranks, dim]
  __half** wv;  // 按 out 切 [kv_dim/n_ranks, dim]
  __half** wo;  // 按 in  切 [dim, dim/n_ranks]
  float** bq;
  float** bk;
  float** bv;
  float** rms_ffn;
  __half** w1;  // 按 out 切 [hidden/n_ranks, dim]
  __half** w2;  // 按 in  切 [hidden_dim, dim/n_ranks]
  __half** w3;  // 按 out 切 [hidden/n_ranks, dim]
};

// ============================================================
// TP 运行时状态
// ============================================================
struct TPRunState {
  __half* x;    // [max_flat_tokens, dim]
  __half* xb;   // [max_flat_tokens, dim/n_ranks]
  __half* xb2;  // [max_flat_tokens, dim]
  __half* q;    // [max_flat_tokens, dim/n_ranks]
  __half* k;    // [max_flat_tokens, kv_dim/n_ranks]
  __half* v;    // [max_flat_tokens, kv_dim/n_ranks]
  __half* hb;   // [max_flat_tokens, hidden/n_ranks]
  __half* hb2;  // [max_flat_tokens, hidden/n_ranks]

  int* d_tokens;
  int* d_positions;
  int* d_token_seq;
  int* d_slot_map;
  int* block_table;  // [max_batch, max_blocks_per_seq]
  int* d_last_tok_idx;
  int* d_prefill_flat;
  int* d_decode_flat;

  __half* logits_last_tok;
  __half* logits_fp16;
  float* logits;  // pinned，[max_batch, vocab_size]
};

// ============================================================
// NCCL 上下文
// ============================================================
struct NCCLContext {
  ncclComm_t comm;
  cudaStream_t stream;
  int rank;
  int n_ranks;
};

// ============================================================
// 权重切分上传
// ============================================================
static void upload_tp_weights(TPWeights& tw, const Weights& w, const Config& cfg, int rank,
                              int n_ranks) {
  int dim = cfg.dim;
  int kv_dim = cfg.n_kv_heads * (cfg.dim / cfg.n_heads);
  int hidden_dim = cfg.hidden_dim;
  int vocab_size = cfg.vocab_size;

  // 按 out 切：物理内存 [out, in]，切外层 → 连续内存，直接 memcpy
  auto up16_col = [&](__half** dst, const uint16_t* src, int rows, int cols) {
    int col_size = cols / n_ranks;
    int col_start = rank * col_size;
    size_t offset = (size_t)col_start * rows;
    size_t size = (size_t)col_size * rows;
    CHECK_CUDA(cudaMalloc(dst, size * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(*dst, (const __half*)src + offset, size * sizeof(__half),
                          cudaMemcpyHostToDevice));
  };

  // 按 in 切：物理内存 [out, in]，切内层 → 不连续，逐行拷贝
  auto up16_row = [&](__half** dst, const uint16_t* src, int rows, int cols) {
    int row_size = rows / n_ranks;
    int row_start = rank * row_size;
    std::vector<__half> tmp((size_t)cols * row_size);
    for (int c = 0; c < cols; c++) {
      memcpy(tmp.data() + (size_t)c * row_size, (const __half*)src + (size_t)c * rows + row_start,
             row_size * sizeof(__half));
    }
    CHECK_CUDA(cudaMalloc(dst, (size_t)cols * row_size * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(*dst, tmp.data(), (size_t)cols * row_size * sizeof(__half),
                          cudaMemcpyHostToDevice));
  };

  auto up32_col = [&](float** dst, const float* src, int size) {
    int col_start = rank * (size / n_ranks);
    int col_size = size / n_ranks;
    CHECK_CUDA(cudaMalloc(dst, col_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(*dst, src + col_start, col_size * sizeof(float), cudaMemcpyHostToDevice));
  };

  auto up32 = [](float** dst, const float* src, size_t n) {
    CHECK_CUDA(cudaMalloc(dst, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
  };

  auto up16 = [](__half** dst, const uint16_t* src, size_t n) {
    CHECK_CUDA(cudaMalloc(dst, n * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(*dst, src, n * sizeof(__half), cudaMemcpyHostToDevice));
  };

  up16(&tw.token_embedding, w.token_embedding, (size_t)vocab_size * dim);
  up32(&tw.rms_final, w.rms_final, dim);
  up16(&tw.wcls, w.wcls, (size_t)vocab_size * dim);

  tw.rms_att = new float*[cfg.n_layers];
  tw.wq = new __half*[cfg.n_layers];
  tw.wk = new __half*[cfg.n_layers];
  tw.wv = new __half*[cfg.n_layers];
  tw.wo = new __half*[cfg.n_layers];
  tw.bq = new float*[cfg.n_layers];
  tw.bk = new float*[cfg.n_layers];
  tw.bv = new float*[cfg.n_layers];
  tw.rms_ffn = new float*[cfg.n_layers];
  tw.w1 = new __half*[cfg.n_layers];
  tw.w2 = new __half*[cfg.n_layers];
  tw.w3 = new __half*[cfg.n_layers];

  for (int l = 0; l < cfg.n_layers; l++) {
    up32(&tw.rms_att[l], w.rms_att[l], dim);
    up16_col(&tw.wq[l], w.wq[l], dim, dim);
    up16_col(&tw.wk[l], w.wk[l], dim, kv_dim);
    up16_col(&tw.wv[l], w.wv[l], dim, kv_dim);
    up16_row(&tw.wo[l], w.wo[l], dim, dim);
    up32_col(&tw.bq[l], w.bq[l], dim);
    up32_col(&tw.bk[l], w.bk[l], kv_dim);
    up32_col(&tw.bv[l], w.bv[l], kv_dim);
    up32(&tw.rms_ffn[l], w.rms_ffn[l], dim);
    up16_col(&tw.w1[l], w.w1[l], dim, hidden_dim);
    up16_row(&tw.w2[l], w.w2[l], hidden_dim, dim);
    up16_col(&tw.w3[l], w.w3[l], dim, hidden_dim);
  }

  fprintf(stderr, "rank=%d 权重切分上传完成\n", rank);
}

// ============================================================
// 运行时状态分配
// ============================================================
static void alloc_tp_run_state(TPRunState& s, const Config& cfg, int max_batch,
                               int max_blocks_per_seq, int max_flat_tokens, int n_ranks) {
  int dim = cfg.dim;
  int kv_dim = cfg.n_kv_heads * (cfg.dim / cfg.n_heads);
  int hidden_dim = cfg.hidden_dim;
  int dim_per_rank = dim / n_ranks;
  int kv_per_rank = kv_dim / n_ranks;
  int hidden_per_rank = hidden_dim / n_ranks;

  CHECK_CUDA(cudaMalloc(&s.x, (size_t)max_flat_tokens * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb2, (size_t)max_flat_tokens * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb, (size_t)max_flat_tokens * dim_per_rank * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.q, (size_t)max_flat_tokens * dim_per_rank * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.k, (size_t)max_flat_tokens * kv_per_rank * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.v, (size_t)max_flat_tokens * kv_per_rank * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb, (size_t)max_flat_tokens * hidden_per_rank * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb2, (size_t)max_flat_tokens * hidden_per_rank * sizeof(__half)));

  CHECK_CUDA(cudaMalloc(&s.d_tokens, max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_positions, max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_token_seq, max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_slot_map, max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.block_table, (size_t)max_batch * max_blocks_per_seq * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_last_tok_idx, max_batch * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_prefill_flat, max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_decode_flat, max_batch * sizeof(int)));

  CHECK_CUDA(cudaMalloc(&s.logits_last_tok, (size_t)max_batch * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.logits_fp16, (size_t)max_batch * cfg.vocab_size * sizeof(__half)));
  CHECK_CUDA(cudaMallocHost(&s.logits, (size_t)max_batch * cfg.vocab_size * sizeof(float)));
}

// ============================================================
// matmul 辅助函数
// ============================================================

static void tp_matmul(cublasHandle_t h, __half* out, const __half* x, const __half* w, int n, int d,
                      int batch = 1) {
  const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
  CHECK_CUBLAS(
      cublasHgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, d, batch, n, &alpha, w, n, x, n, &beta, out, d));
}

static void tp_matmul_col(cublasHandle_t h, __half* out, const __half* x, const __half* w, int rows,
                          int col_size, int batch = 1) {
  const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
  CHECK_CUBLAS(cublasHgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, col_size, batch, rows, &alpha, w, rows, x,
                           rows, &beta, out, col_size));
}

static void tp_matmul_row(cublasHandle_t h, __half* out, const __half* x, const __half* w,
                          int row_size, int cols, int batch = 1) {
  const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
  CHECK_CUBLAS(cublasHgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, cols, batch, row_size, &alpha, w, row_size,
                           x, row_size, &beta, out, cols));
}

// ============================================================
// forward_tp
// ============================================================
static void forward_tp(TPWeights& tw, TPRunState& s, BlockPool* pool, const Config& cfg,
                       const NCCLContext& nccl, cublasHandle_t cublas,
                       const std::vector<FlatRequest>& flat_reqs,
                       const std::vector<int>& flat_tokens, const std::vector<int>& flat_positions,
                       const std::vector<int>& token_to_req, const std::vector<int>& token_slot,
                       const std::vector<int>& last_tok_idx,
                       const std::vector<int>& prefill_flat_idx,
                       const std::vector<int>& dec_flat_idx, int total_tokens) {
  // cudaEvent_t ev_start, ev_end;
  // cudaEventCreate(&ev_start);
  // cudaEventCreate(&ev_end);
  // cudaEventRecord(ev_start, nccl.stream);

  int dim = cfg.dim;
  int head_dim = cfg.dim / cfg.n_heads;
  int kv_dim = cfg.n_kv_heads * head_dim;
  int kv_mul = cfg.n_heads / cfg.n_kv_heads;

  int dim_per_rank = dim / nccl.n_ranks;
  int kv_per_rank = kv_dim / nccl.n_ranks;
  int hidden_per_rank = cfg.hidden_dim / nccl.n_ranks;
  int heads_per_rank = cfg.n_heads / nccl.n_ranks;

  int T = 256;
  int batch = (int)flat_reqs.size();
  int n_prefill = (int)prefill_flat_idx.size();
  int n_dec = (int)dec_flat_idx.size();
  int max_blocks_per_seq = (cfg.seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
  size_t smem = (FA_BLOCK + head_dim + 8 + 8 + 2) * sizeof(float);

  cudaStream_t stream = nccl.stream;

  CHECK_CUDA(cudaMemcpyAsync(s.d_tokens, flat_tokens.data(), total_tokens * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(s.d_positions, flat_positions.data(), total_tokens * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(s.d_token_seq, token_to_req.data(), total_tokens * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(s.d_slot_map, token_slot.data(), total_tokens * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(s.d_last_tok_idx, last_tok_idx.data(), batch * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  if (n_prefill > 0) {
    CHECK_CUDA(cudaMemcpyAsync(s.d_prefill_flat, prefill_flat_idx.data(), n_prefill * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
  }
  if (n_dec > 0) {
    CHECK_CUDA(cudaMemcpyAsync(s.d_decode_flat, dec_flat_idx.data(), n_dec * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
  }

  embedding_kernel<<<total_tokens, T, 0, stream>>>(s.x, tw.token_embedding, s.d_tokens,
                                                   total_tokens, dim);
  CHECK_KERNEL();

  for (int l = 0; l < cfg.n_layers; l++) {
    rmsnorm_kernel<<<total_tokens, T, 0, stream>>>(s.xb2, s.x, tw.rms_att[l], dim);
    CHECK_KERNEL();

    tp_matmul_col(cublas, s.q, s.xb2, tw.wq[l], dim, dim_per_rank, total_tokens);
    tp_matmul_col(cublas, s.k, s.xb2, tw.wk[l], dim, kv_per_rank, total_tokens);
    tp_matmul_col(cublas, s.v, s.xb2, tw.wv[l], dim, kv_per_rank, total_tokens);

    add_bias_kernel<<<total_tokens, T, 0, stream>>>(s.q, tw.bq[l], total_tokens, dim_per_rank);
    add_bias_kernel<<<total_tokens, T, 0, stream>>>(s.k, tw.bk[l], total_tokens, kv_per_rank);
    add_bias_kernel<<<total_tokens, T, 0, stream>>>(s.v, tw.bv[l], total_tokens, kv_per_rank);
    CHECK_KERNEL();

    rope_kernel<<<total_tokens, T, 0, stream>>>(s.q, s.k, s.d_positions, total_tokens, dim_per_rank,
                                                kv_per_rank, head_dim, cfg.rope_freq_base);
    CHECK_KERNEL();

    kvcache_write_kernel<<<total_tokens, T, 0, stream>>>(pool->get_k_cache(), pool->get_v_cache(),
                                                         s.k, s.v, s.d_slot_map, total_tokens, l,
                                                         cfg.n_layers, kv_per_rank);
    CHECK_KERNEL();

    if (n_prefill > 0) {
      dim3 grid(heads_per_rank, n_prefill);
      prefill_attention_kernel<<<grid, T, smem, stream>>>(
          s.q, pool->get_k_cache(), pool->get_v_cache(), s.xb, s.block_table, s.d_token_seq,
          s.d_positions, s.d_prefill_flat, max_blocks_per_seq, BLOCK_SIZE, cfg.n_layers, l,
          dim_per_rank, kv_per_rank, head_dim, kv_mul);
      CHECK_KERNEL();
    }

    if (n_dec > 0) {
      dim3 grid_dec(heads_per_rank, n_dec);
      decode_attention_kernel<<<grid_dec, T, smem, stream>>>(
          s.q, pool->get_k_cache(), pool->get_v_cache(), s.xb, s.block_table, s.d_token_seq,
          s.d_positions, s.d_decode_flat, l, max_blocks_per_seq, BLOCK_SIZE, dim_per_rank,
          kv_per_rank, head_dim, kv_mul, cfg.n_layers);
      CHECK_KERNEL();
    }

    tp_matmul_row(cublas, s.xb2, s.xb, tw.wo[l], dim_per_rank, dim, total_tokens);

    if (nccl.n_ranks > 1) {
      CHECK_NCCL(ncclAllReduce(s.xb2, s.xb2, (size_t)total_tokens * dim, ncclFloat16, ncclSum,
                               nccl.comm, stream));
    }

    residual_kernel<<<total_tokens, T, 0, stream>>>(s.x, s.xb2, total_tokens, dim);
    CHECK_KERNEL();

    rmsnorm_kernel<<<total_tokens, T, 0, stream>>>(s.xb2, s.x, tw.rms_ffn[l], dim);
    CHECK_KERNEL();

    tp_matmul_col(cublas, s.hb, s.xb2, tw.w1[l], dim, hidden_per_rank, total_tokens);
    tp_matmul_col(cublas, s.hb2, s.xb2, tw.w3[l], dim, hidden_per_rank, total_tokens);

    swiglu_kernel<<<total_tokens, T, 0, stream>>>(s.hb, s.hb2, total_tokens, hidden_per_rank);
    CHECK_KERNEL();

    tp_matmul_row(cublas, s.xb2, s.hb, tw.w2[l], hidden_per_rank, dim, total_tokens);

    if (nccl.n_ranks > 1) {
      CHECK_NCCL(ncclAllReduce(s.xb2, s.xb2, (size_t)total_tokens * dim, ncclFloat16, ncclSum,
                               nccl.comm, stream));
    }

    residual_kernel<<<total_tokens, T, 0, stream>>>(s.x, s.xb2, total_tokens, dim);
    CHECK_KERNEL();
  }

  rmsnorm_kernel<<<total_tokens, T, 0, stream>>>(s.xb2, s.x, tw.rms_final, dim);
  CHECK_KERNEL();

  extract_last_token_kernel<<<batch, T, 0, stream>>>(s.logits_last_tok, s.xb2, s.d_last_tok_idx,
                                                     batch, dim);
  CHECK_KERNEL();

  tp_matmul(cublas, s.logits_fp16, s.logits_last_tok, tw.wcls, dim, cfg.vocab_size, batch);

  fp16_to_fp32_kernel<<<batch, T, 0, stream>>>(s.logits, s.logits_fp16, batch, cfg.vocab_size);
  CHECK_KERNEL();

  // cudaEventRecord(ev_end, stream);
  // cudaEventSynchronize(ev_end);
  // float gpu_ms = 0;
  // cudaEventElapsedTime(&gpu_ms, ev_start, ev_end);
  // fprintf(stderr, "rank=%d gpu_ms=%.2f\n", nccl.rank, gpu_ms);
}

// ============================================================
// 增量更新 GPU block_table（搬自 decoder.cpp）
// ============================================================
static void update_block_table_partial(const std::vector<Request*>& running,
                                       const std::vector<int>& changed, int* d_block_table,
                                       int max_blocks_per_seq) {
  for (int i : changed) {
    if (i >= (int)running.size() || running[i] == nullptr) continue;
    Request* req = running[i];
    int n = req->block_table.num_blocks();
    CHECK_CUDA(cudaMemcpy(d_block_table + (size_t)i * max_blocks_per_seq, req->block_table.data(),
                          n * sizeof(int), cudaMemcpyHostToDevice));
  }
}

// ============================================================
// generate_continuous：多请求 continuous batching
// ============================================================
static void generate_continuous(TPWeights& tw, TPRunState& s, BlockPool* pool, const Config& cfg,
                                Tokenizer& tokenizer, const NCCLContext& nccl,
                                cublasHandle_t cublas, std::vector<std::string>& user_inputs,
                                int max_batch, int max_new_tokens, float temperature, int top_k,
                                std::mt19937& rng, int max_flat_tokens, int max_blocks_per_seq,
                                bool enable_prefix_cache) {
  int rank = nccl.rank;

  Scheduler scheduler(max_batch, BLOCK_SIZE, pool, enable_prefix_cache);

  // 初始化所有请求
  std::vector<Request*> all_requests;
  for (int i = 0; i < (int)user_inputs.size(); i++) {
    auto* req = new Request();
    req->id = i;
    req->input = user_inputs[i];
    std::string prompt = apply_chat_template(user_inputs[i]);
    encode(tokenizer, prompt, req->prompt_tokens);
    req->prompt_tokens.insert(req->prompt_tokens.begin(), tokenizer.bos_token_id);
    scheduler.add_request(req);
    all_requests.push_back(req);
  }

  auto t_start = std::chrono::steady_clock::now();
  auto last_step_time = t_start;
  int output_tokens = 0;
  int step = 0;
  // chunked prefill 每步预算（和单机版保持一致）
  int max_pps = max_flat_tokens / 2;

  while (!scheduler.all_done()) {
    scheduler.fill_slots();
    scheduler.last_changed.clear();

    std::vector<FlatRequest> flat_requests;
    std::vector<int> flat_tokens;
    std::vector<int> flat_positions;
    std::vector<int> token_to_req;
    std::vector<int> token_slot;
    std::vector<int> last_token_indices;
    std::vector<int> prefill_flat_indices;
    std::vector<int> decode_flat_indices;

    int flat_offset = 0;
    int prefill_budget = max_pps;

    for (int i = 0; i < max_batch; i++) {
      Request* req = scheduler.get_running_req(i);
      if (req == nullptr || req->finished) continue;

      FlatRequest fr;
      fr.req_idx = i;
      fr.flat_offset = flat_offset;
      fr.start_pos = req->pos;

      if (!req->prefill_done) {
        fr.is_prefill = true;
        if (req->prefill_offset == 0) scheduler.last_changed.push_back(i);

        int remaining = (int)req->prompt_tokens.size() - req->prefill_offset;
        fr.n_tokens = std::min(remaining, prefill_budget);
        if (fr.n_tokens <= 0) continue;

        if (req->pos + fr.n_tokens >= cfg.seq_len) {
          fprintf(stderr, "request %d: prompt too long\n", req->id);
          req->finished = true;
          continue;
        }

        int old_blocks = req->block_table.num_blocks();
        if (!scheduler.ensure_blocks(req, fr.n_tokens)) {
          req->finished = true;
          continue;
        }
        if (req->block_table.num_blocks() != old_blocks) scheduler.last_changed.push_back(i);

        for (int j = 0; j < fr.n_tokens; j++) {
          int tok = req->prompt_tokens[req->prefill_offset + j];
          int pos = req->pos + j;
          int phy = req->block_table.physical_idx(pos);
          flat_tokens.push_back(tok);
          flat_positions.push_back(pos);
          token_to_req.push_back(i);
          token_slot.push_back(phy * BLOCK_SIZE + pos % BLOCK_SIZE);
          prefill_flat_indices.push_back(flat_offset + j);
        }
        prefill_budget -= fr.n_tokens;

      } else {
        fr.is_prefill = false;
        fr.n_tokens = 1;

        int old_blocks = req->block_table.num_blocks();
        if (!scheduler.ensure_blocks(req, 1)) {
          req->finished = true;
          continue;
        }
        if (req->block_table.num_blocks() != old_blocks) scheduler.last_changed.push_back(i);

        int phy = req->block_table.physical_idx(req->pos);
        flat_tokens.push_back(req->cur_token);
        flat_positions.push_back(req->pos);
        token_to_req.push_back(i);
        token_slot.push_back(phy * BLOCK_SIZE + req->pos % BLOCK_SIZE);
        decode_flat_indices.push_back(flat_offset);
      }

      last_token_indices.push_back(flat_offset + fr.n_tokens - 1);
      flat_offset += fr.n_tokens;
      flat_requests.push_back(fr);
    }

    if (flat_requests.empty()) continue;

    // 增量更新 GPU block_table
    std::sort(scheduler.last_changed.begin(), scheduler.last_changed.end());
    scheduler.last_changed.erase(
        std::unique(scheduler.last_changed.begin(), scheduler.last_changed.end()),
        scheduler.last_changed.end());
    update_block_table_partial(scheduler.running(), scheduler.last_changed, s.block_table,
                               max_blocks_per_seq);

    auto now = std::chrono::steady_clock::now();
    double between = std::chrono::duration<double, std::milli>(now - last_step_time).count();
    last_step_time = now;
    if (rank == 0) {
      fprintf(stderr,
              "between %.2f, step %d: total_tokens=%d n_decode=%d n_prefill=%d prefill_tokens=%d\n",
              between, step, (int)flat_tokens.size(), (int)decode_flat_indices.size(),
              (int)flat_requests.size() - (int)decode_flat_indices.size(),
              (int)flat_tokens.size() - (int)decode_flat_indices.size());
    }

    forward_tp(tw, s, pool, cfg, nccl, cublas, flat_requests, flat_tokens, flat_positions,
               token_to_req, token_slot, last_token_indices, prefill_flat_indices,
               decode_flat_indices, (int)flat_tokens.size());

    // rank=0 采样，广播所有请求的 next_token
    int n_reqs = (int)flat_requests.size();
    std::vector<int> next_tokens(n_reqs, 0);
    if (rank == 0) {
      cudaStreamSynchronize(nccl.stream);
      for (int fi = 0; fi < n_reqs; fi++) {
        float* logits = s.logits + (size_t)fi * cfg.vocab_size;
        next_tokens[fi] = sample_topk(logits, cfg.vocab_size, top_k, temperature, rng);
      }
    }
    MPI_Bcast(next_tokens.data(), n_reqs, MPI_INT, 0, MPI_COMM_WORLD);

    // 更新请求状态
    for (int fi = 0; fi < n_reqs; fi++) {
      auto& fr = flat_requests[fi];
      Request* req = scheduler.get_running_req(fr.req_idx);
      if (req == nullptr || req->finished) continue;

      int next_token = next_tokens[fi];

      if (!req->prefill_done) {
        req->pos += fr.n_tokens;
        req->prefill_offset += fr.n_tokens;
        if (req->prefill_offset >= (int)req->prompt_tokens.size()) {
          req->prefill_done = true;
          req->cur_token = next_token;
          scheduler.on_prefill_done(req);
        }
      } else {
        if (rank == 0) {
          const char* piece = decode(tokenizer, req->cur_token);
          if (piece) req->output += piece;
        }
        req->pos++;
        req->cur_token = next_token;
        req->n_generated++;
        output_tokens++;

        if (next_token == tokenizer.eos_token_id || next_token == tokenizer.bos_token_id ||
            tokenizer.vocab[next_token] == "<|im_end|>" || req->n_generated >= max_new_tokens ||
            req->pos >= cfg.seq_len) {
          req->finished = true;
        }
      }
    }

    scheduler.release_finished();
    step++;
  }

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();

  if (rank == 0) {
    for (auto* req : all_requests) {
      printf("request id: %d\nprompt: %s\noutput: %s\n--------------------\n", req->id,
             req->input.c_str(), req->output.c_str());
    }
    fprintf(stderr, "total: %d tokens in %.2fs (%.1f tokens/s)\n", output_tokens, elapsed,
            output_tokens / elapsed);
  }

  for (auto* req : all_requests) delete req;
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, n_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  if (argc < 2) {
    if (rank == 0) fprintf(stderr, "Usage: %s <model_file>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  bool enable_prefix_cache = true;
  if (argc >= 4) atoi(argv[3]) != 0;
  ;

  cudaSetDevice(0);

  // 1. NCCL 初始化
  NCCLContext nccl;
  nccl.rank = rank;
  nccl.n_ranks = n_ranks;
  {
    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&nccl.comm, n_ranks, nccl_id, rank);
    cudaStreamCreate(&nccl.stream);
  }
  fprintf(stderr, "rank=%d NCCL 初始化完成\n", rank);

  // 2. 加载模型
  GGUFFile gguf;
  Config cfg;
  ModelFile mf;
  Weights w;
  Tokenizer tokenizer;
  if (load_gguf(gguf, argv[1]) != 0) {
    MPI_Finalize();
    return 1;
  }
  if (load_config(cfg, gguf) != 0) {
    MPI_Finalize();
    return 1;
  }
  if (open_model(argv[1], mf) != 0) {
    MPI_Finalize();
    return 1;
  }
  if (load_weights(w, cfg, gguf, mf) != 0) {
    MPI_Finalize();
    return 1;
  }
  if (load_tokenizer(tokenizer, gguf) != 0) {
    MPI_Finalize();
    return 1;
  }

  // 3. 切分权重上传 GPU
  TPWeights tw;
  upload_tp_weights(tw, w, cfg, rank, n_ranks);

  // 4. cublas
  cublasHandle_t cublas;
  CHECK_CUBLAS(cublasCreate(&cublas));
  CHECK_CUBLAS(cublasSetStream(cublas, nccl.stream));

  // 5. BlockPool + RunState
  int max_batch = 8;
  int max_prefill_len = 4096;
  int max_flat_tokens = max_prefill_len + max_batch;
  int max_blocks_per_seq = (cfg.seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  size_t budget = (size_t)(free_mem * 0.8);
  int head_dim = cfg.dim / cfg.n_heads;
  int kv_dim = cfg.n_kv_heads * head_dim;
  int kv_per_rank = kv_dim / n_ranks;
  size_t pg_sz = (size_t)BLOCK_SIZE * cfg.n_layers * kv_per_rank * 2 * sizeof(__half);
  int total_pgs = (int)(budget / pg_sz);

  BlockPool* pool = new BlockPool(total_pgs, BLOCK_SIZE, cfg.n_layers, kv_per_rank);
  fprintf(stderr, "rank=%d BlockPool: %d blocks\n", rank, total_pgs);

  TPRunState s;
  alloc_tp_run_state(s, cfg, max_batch, max_blocks_per_seq, max_flat_tokens, n_ranks);

  // 6. 推理参数
  std::mt19937 rng(42);
  float temperature = 0.0f;
  int top_k = 30;
  int max_new_toks = 256;

  // 7. 收集用户输入（rank=0 收集，广播给所有 rank）
  std::vector<std::string> inputs;
  if (rank == 0) {
    std::string line;
    int idx = 0;
    while (true) {
      printf("User[%d] (empty to start): ", idx++);
      if (!std::getline(std::cin, line) || line.empty()) break;
      inputs.push_back(line);
    }
    if (inputs.empty()) {
      fprintf(stderr, "no input\n");
      MPI_Finalize();
      return 1;
    }
    fprintf(stderr, "%d requests\n", (int)inputs.size());
  }

  // 广播 inputs
  int n_inputs = (rank == 0) ? (int)inputs.size() : 0;
  MPI_Bcast(&n_inputs, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) inputs.resize(n_inputs);
  for (int i = 0; i < n_inputs; i++) {
    int len = (rank == 0) ? (int)inputs[i].size() : 0;
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    inputs[i].resize(len);
    MPI_Bcast(inputs[i].data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  // 8. 批量推理
  generate_continuous(tw, s, pool, cfg, tokenizer, nccl, cublas, inputs, max_batch, max_new_toks,
                      temperature, top_k, rng, max_flat_tokens, max_blocks_per_seq,
                      enable_prefix_cache);

  delete pool;
  cublasDestroy(cublas);
  ncclCommDestroy(nccl.comm);
  cudaStreamDestroy(nccl.stream);
  MPI_Finalize();
  return 0;
}

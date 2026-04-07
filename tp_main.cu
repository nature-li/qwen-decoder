#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mpi.h>
#include <nccl.h>

#include <chrono>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "gguf_loader.h"
#include "gpu_decoder.h"
#include "gpu_kernels.h"

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
  float*  rms_final;
  __half* wcls;

  float**  rms_att;
  __half** wq;  // 按 out 切 [dim/n_ranks, dim]
  __half** wk;  // 按 out 切 [kv_dim/n_ranks, dim]
  __half** wv;  // 按 out 切 [kv_dim/n_ranks, dim]
  __half** wo;  // 按 in  切 [dim, dim/n_ranks]
  float**  bq;
  float**  bk;
  float**  bv;
  float**  rms_ffn;
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
  int* block_table;
  int* d_last_tok_idx;
  int* d_prefill_flat;
  int* d_decode_flat;

  __half* logits_last_tok;
  __half* logits_fp16;
  float*  logits;
};

// ============================================================
// NCCL 上下文
// ============================================================
struct NCCLContext {
  ncclComm_t   comm;
  cudaStream_t stream;
  int rank;
  int n_ranks;
};

// ============================================================
// 权重切分上传
// ============================================================
static void upload_tp_weights(TPWeights& tw, const Weights& w, const Config& cfg,
                              int rank, int n_ranks) {
  int dim        = cfg.dim;
  int kv_dim     = cfg.n_kv_heads * (cfg.dim / cfg.n_heads);
  int hidden_dim = cfg.hidden_dim;
  int vocab_size = cfg.vocab_size;

  // 按 out 切：物理内存 [out, in]，切外层 → 连续内存，直接 memcpy
  auto up16_col = [&](__half** dst, const uint16_t* src, int rows, int cols) {
    int col_size  = cols / n_ranks;
    int col_start = rank * col_size;
    size_t offset = (size_t)col_start * rows;
    size_t size   = (size_t)col_size * rows;
    CHECK_CUDA(cudaMalloc(dst, size * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(*dst, (const __half*)src + offset,
                          size * sizeof(__half), cudaMemcpyHostToDevice));
  };

  // 按 in 切：物理内存 [out, in]，切内层 → 不连续，逐行拷贝
  auto up16_row = [&](__half** dst, const uint16_t* src, int rows, int cols) {
    int row_size  = rows / n_ranks;
    int row_start = rank * row_size;
    std::vector<__half> tmp((size_t)cols * row_size);
    for (int c = 0; c < cols; c++) {
      memcpy(tmp.data() + (size_t)c * row_size,
             (const __half*)src + (size_t)c * rows + row_start,
             row_size * sizeof(__half));
    }
    CHECK_CUDA(cudaMalloc(dst, (size_t)cols * row_size * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(*dst, tmp.data(),
                          (size_t)cols * row_size * sizeof(__half), cudaMemcpyHostToDevice));
  };

  auto up32_col = [&](float** dst, const float* src, int size) {
    int col_start = rank * (size / n_ranks);
    int col_size  = size / n_ranks;
    CHECK_CUDA(cudaMalloc(dst, col_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(*dst, src + col_start,
                          col_size * sizeof(float), cudaMemcpyHostToDevice));
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
  tw.wq      = new __half*[cfg.n_layers];
  tw.wk      = new __half*[cfg.n_layers];
  tw.wv      = new __half*[cfg.n_layers];
  tw.wo      = new __half*[cfg.n_layers];
  tw.bq      = new float*[cfg.n_layers];
  tw.bk      = new float*[cfg.n_layers];
  tw.bv      = new float*[cfg.n_layers];
  tw.rms_ffn = new float*[cfg.n_layers];
  tw.w1      = new __half*[cfg.n_layers];
  tw.w2      = new __half*[cfg.n_layers];
  tw.w3      = new __half*[cfg.n_layers];

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
  int dim         = cfg.dim;
  int kv_dim      = cfg.n_kv_heads * (cfg.dim / cfg.n_heads);
  int hidden_dim  = cfg.hidden_dim;
  int dim_per_rank    = dim / n_ranks;
  int kv_per_rank     = kv_dim / n_ranks;
  int hidden_per_rank = hidden_dim / n_ranks;

  CHECK_CUDA(cudaMalloc(&s.x,   (size_t)max_flat_tokens * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb2, (size_t)max_flat_tokens * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb,  (size_t)max_flat_tokens * dim_per_rank * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.q,   (size_t)max_flat_tokens * dim_per_rank * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.k,   (size_t)max_flat_tokens * kv_per_rank  * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.v,   (size_t)max_flat_tokens * kv_per_rank  * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb,  (size_t)max_flat_tokens * hidden_per_rank * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb2, (size_t)max_flat_tokens * hidden_per_rank * sizeof(__half)));

  CHECK_CUDA(cudaMalloc(&s.d_tokens,       max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_positions,    max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_token_seq,    max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_slot_map,     max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.block_table,
      (size_t)max_batch * max_blocks_per_seq * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_last_tok_idx, max_batch * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_prefill_flat, max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_decode_flat,  max_batch * sizeof(int)));

  CHECK_CUDA(cudaMalloc(&s.logits_last_tok,
      (size_t)max_batch * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.logits_fp16,
      (size_t)max_batch * cfg.vocab_size * sizeof(__half)));
  CHECK_CUDA(cudaMallocHost(&s.logits,
      (size_t)max_batch * cfg.vocab_size * sizeof(float)));
}

// ============================================================
// matmul 辅助函数
// ============================================================

static void tp_matmul(cublasHandle_t h, __half* out, const __half* x, const __half* w,
                      int n, int d, int batch = 1) {
  const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
  CHECK_CUBLAS(cublasHgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, d, batch, n,
                            &alpha, w, n, x, n, &beta, out, d));
}

static void tp_matmul_col(cublasHandle_t h, __half* out, const __half* x, const __half* w,
                          int rows, int col_size, int batch = 1) {
  const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
  CHECK_CUBLAS(cublasHgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, col_size, batch, rows,
                            &alpha, w, rows, x, rows, &beta, out, col_size));
}

static void tp_matmul_row(cublasHandle_t h, __half* out, const __half* x, const __half* w,
                          int row_size, int cols, int batch = 1) {
  const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
  CHECK_CUBLAS(cublasHgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, cols, batch, row_size,
                            &alpha, w, row_size, x, row_size, &beta, out, cols));
}

// ============================================================
// forward_tp
// ============================================================
static void forward_tp(TPWeights& tw, TPRunState& s, BlockPool* pool,
                       const Config& cfg, const NCCLContext& nccl,
                       cublasHandle_t cublas,
                       const std::vector<FlatRequest>& flat_reqs,
                       const std::vector<int>& flat_tokens,
                       const std::vector<int>& flat_positions,
                       const std::vector<int>& token_to_req,
                       const std::vector<int>& token_slot,
                       const std::vector<int>& last_tok_idx,
                       const std::vector<int>& prefill_flat_idx,
                       const std::vector<int>& dec_flat_idx,
                       int total_tokens) {
  int dim        = cfg.dim;
  int head_dim   = cfg.dim / cfg.n_heads;
  int kv_dim     = cfg.n_kv_heads * head_dim;
  int kv_mul     = cfg.n_heads / cfg.n_kv_heads;

  int dim_per_rank    = dim / nccl.n_ranks;
  int kv_per_rank     = kv_dim / nccl.n_ranks;
  int hidden_per_rank = cfg.hidden_dim / nccl.n_ranks;
  int heads_per_rank  = cfg.n_heads / nccl.n_ranks;

  int T         = 256;
  int batch     = (int)flat_reqs.size();
  int n_prefill = (int)prefill_flat_idx.size();
  int n_dec     = (int)dec_flat_idx.size();
  int max_blocks_per_seq = (cfg.seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
  size_t smem   = (FA_BLOCK + head_dim + 8 + 8 + 2) * sizeof(float);

  cudaStream_t stream = nccl.stream;

  CHECK_CUDA(cudaMemcpyAsync(s.d_tokens, flat_tokens.data(),
      total_tokens * sizeof(int), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(s.d_positions, flat_positions.data(),
      total_tokens * sizeof(int), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(s.d_token_seq, token_to_req.data(),
      total_tokens * sizeof(int), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(s.d_slot_map, token_slot.data(),
      total_tokens * sizeof(int), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(s.d_last_tok_idx, last_tok_idx.data(),
      batch * sizeof(int), cudaMemcpyHostToDevice, stream));
  if (n_prefill > 0) {
    CHECK_CUDA(cudaMemcpyAsync(s.d_prefill_flat, prefill_flat_idx.data(),
        n_prefill * sizeof(int), cudaMemcpyHostToDevice, stream));
  }
  if (n_dec > 0) {
    CHECK_CUDA(cudaMemcpyAsync(s.d_decode_flat, dec_flat_idx.data(),
        n_dec * sizeof(int), cudaMemcpyHostToDevice, stream));
  }

  embedding_kernel<<<total_tokens, T, 0, stream>>>(
      s.x, tw.token_embedding, s.d_tokens, total_tokens, dim);
  CHECK_KERNEL();

  for (int l = 0; l < cfg.n_layers; l++) {
    rmsnorm_kernel<<<total_tokens, T, 0, stream>>>(s.xb2, s.x, tw.rms_att[l], dim);
    CHECK_KERNEL();

    tp_matmul_col(cublas, s.q, s.xb2, tw.wq[l], dim, dim_per_rank,    total_tokens);
    tp_matmul_col(cublas, s.k, s.xb2, tw.wk[l], dim, kv_per_rank,     total_tokens);
    tp_matmul_col(cublas, s.v, s.xb2, tw.wv[l], dim, kv_per_rank,     total_tokens);

    add_bias_kernel<<<total_tokens, T, 0, stream>>>(s.q, tw.bq[l], total_tokens, dim_per_rank);
    add_bias_kernel<<<total_tokens, T, 0, stream>>>(s.k, tw.bk[l], total_tokens, kv_per_rank);
    add_bias_kernel<<<total_tokens, T, 0, stream>>>(s.v, tw.bv[l], total_tokens, kv_per_rank);
    CHECK_KERNEL();

    rope_kernel<<<total_tokens, T, 0, stream>>>(
        s.q, s.k, s.d_positions, total_tokens,
        dim_per_rank, kv_per_rank, head_dim, cfg.rope_freq_base);
    CHECK_KERNEL();

    kvcache_write_kernel<<<total_tokens, T, 0, stream>>>(
        pool->get_k_cache(), pool->get_v_cache(),
        s.k, s.v, s.d_slot_map, total_tokens, l, cfg.n_layers, kv_per_rank);
    CHECK_KERNEL();

    if (n_prefill > 0) {
      dim3 grid(heads_per_rank, n_prefill);
      prefill_attention_kernel<<<grid, T, smem, stream>>>(
          s.q, pool->get_k_cache(), pool->get_v_cache(),
          s.xb, s.block_table,
          s.d_token_seq, s.d_positions, s.d_prefill_flat,
          max_blocks_per_seq, BLOCK_SIZE, cfg.n_layers, l,
          dim_per_rank, kv_per_rank, head_dim, kv_mul);
      CHECK_KERNEL();
    }

    if (n_dec > 0) {
      dim3 grid_dec(heads_per_rank, n_dec);
      decode_attention_kernel<<<grid_dec, T, smem, stream>>>(
          s.q, pool->get_k_cache(), pool->get_v_cache(),
          s.xb, s.block_table,
          s.d_token_seq, s.d_positions, s.d_decode_flat,
          l, max_blocks_per_seq, BLOCK_SIZE,
          dim_per_rank, kv_per_rank, head_dim, kv_mul, cfg.n_layers);
      CHECK_KERNEL();
    }

    tp_matmul_row(cublas, s.xb2, s.xb, tw.wo[l], dim_per_rank, dim, total_tokens);

    if (nccl.n_ranks > 1) {
      CHECK_NCCL(ncclAllReduce(s.xb2, s.xb2,
          (size_t)total_tokens * dim, ncclFloat16, ncclSum, nccl.comm, stream));
    }

    residual_kernel<<<total_tokens, T, 0, stream>>>(s.x, s.xb2, total_tokens, dim);
    CHECK_KERNEL();

    rmsnorm_kernel<<<total_tokens, T, 0, stream>>>(s.xb2, s.x, tw.rms_ffn[l], dim);
    CHECK_KERNEL();

    tp_matmul_col(cublas, s.hb,  s.xb2, tw.w1[l], dim, hidden_per_rank, total_tokens);
    tp_matmul_col(cublas, s.hb2, s.xb2, tw.w3[l], dim, hidden_per_rank, total_tokens);

    swiglu_kernel<<<total_tokens, T, 0, stream>>>(s.hb, s.hb2, total_tokens, hidden_per_rank);
    CHECK_KERNEL();

    tp_matmul_row(cublas, s.xb2, s.hb, tw.w2[l], hidden_per_rank, dim, total_tokens);

    if (nccl.n_ranks > 1) {
      CHECK_NCCL(ncclAllReduce(s.xb2, s.xb2,
          (size_t)total_tokens * dim, ncclFloat16, ncclSum, nccl.comm, stream));
    }

    residual_kernel<<<total_tokens, T, 0, stream>>>(s.x, s.xb2, total_tokens, dim);
    CHECK_KERNEL();
  }

  rmsnorm_kernel<<<total_tokens, T, 0, stream>>>(s.xb2, s.x, tw.rms_final, dim);
  CHECK_KERNEL();

  extract_last_token_kernel<<<batch, T, 0, stream>>>(
      s.logits_last_tok, s.xb2, s.d_last_tok_idx, batch, dim);
  CHECK_KERNEL();

  tp_matmul(cublas, s.logits_fp16, s.logits_last_tok, tw.wcls, dim, cfg.vocab_size, batch);

  fp16_to_fp32_kernel<<<batch, T, 0, stream>>>(
      s.logits, s.logits_fp16, batch, cfg.vocab_size);
  CHECK_KERNEL();
}

// ============================================================
// 单次推理：prefill + decode 循环，返回性能统计
// ============================================================
static void run_inference(TPWeights& tw, TPRunState& s, BlockPool* pool,
                          const Config& cfg, Tokenizer& tokenizer,
                          const NCCLContext& nccl, cublasHandle_t cublas,
                          const std::vector<int>& token_ids) {
  using clock = std::chrono::steady_clock;
  int rank = nccl.rank;

  // 分配 KV cache block
  Request req;
  req.id  = 0;
  req.pos = 0;
  int need_blocks = ((int)token_ids.size() - 1) / BLOCK_SIZE + 1;
  for (int b = 0; b < need_blocks; b++) req.block_table.add_block(pool->allocate());

  std::mt19937 rng(42);
  int  cur_token  = -1;
  bool is_prefill = true;

  int n_prefill_tokens = (int)token_ids.size();
  int n_decode_tokens  = 0;

  auto t_prefill_start = clock::now();
  auto t_decode_start  = clock::now();
  auto t_decode_end    = clock::now();

  if (rank == 0) printf("Assistant: ");

  for (int step = 0; step < 4096; step++) {
    std::vector<FlatRequest> flat_reqs;
    std::vector<int> flat_tokens;
    std::vector<int> flat_positions;
    std::vector<int> token_to_req;
    std::vector<int> token_slot;
    std::vector<int> last_tok_idx;
    std::vector<int> prefill_flat_idx;
    std::vector<int> dec_flat_idx;
    int total_tokens = 0;

    if (is_prefill) {
      t_prefill_start = clock::now();

      FlatRequest fr;
      fr.req_idx    = 0;
      fr.flat_offset = 0;
      fr.n_tokens   = (int)token_ids.size();
      fr.start_pos  = 0;
      fr.is_prefill = true;
      flat_reqs.push_back(fr);

      flat_tokens = token_ids;
      flat_positions.resize(token_ids.size());
      token_to_req.resize(token_ids.size(), 0);
      token_slot.resize(token_ids.size());
      std::iota(flat_positions.begin(), flat_positions.end(), 0);
      for (int i = 0; i < (int)token_ids.size(); i++) {
        int phy = req.block_table.physical_idx(i);
        int off = req.block_table.block_offset(i);
        token_slot[i] = phy * BLOCK_SIZE + off;
      }
      last_tok_idx = {(int)token_ids.size() - 1};
      prefill_flat_idx.resize(token_ids.size());
      std::iota(prefill_flat_idx.begin(), prefill_flat_idx.end(), 0);
      total_tokens = (int)token_ids.size();
      req.pos      = total_tokens;

    } else {
      if (cur_token == tokenizer.eos_token_id ||
          tokenizer.vocab[cur_token] == "<|im_end|>") break;

      if (rank == 0) {
        const char* piece = decode(tokenizer, cur_token);
        if (piece) { printf("%s", piece); fflush(stdout); }
      }

      int need = req.pos / BLOCK_SIZE + 1;
      while (req.block_table.num_blocks() < need) {
        int block_id = pool->allocate();
        if (block_id < 0) { fprintf(stderr, "rank=%d OOM\n", rank); exit(1); }
        req.block_table.add_block(block_id);
      }

      FlatRequest dfr;
      dfr.req_idx    = 0;
      dfr.flat_offset = 0;
      dfr.n_tokens   = 1;
      dfr.start_pos  = req.pos;
      dfr.is_prefill = false;
      flat_reqs.push_back(dfr);

      flat_tokens    = {cur_token};
      flat_positions = {req.pos};
      token_to_req   = {0};
      {
        int phy = req.block_table.physical_idx(req.pos);
        int off = req.block_table.block_offset(req.pos);
        token_slot = {phy * BLOCK_SIZE + off};
      }
      last_tok_idx = {0};
      dec_flat_idx = {0};
      total_tokens = 1;
    }

    int n_blocks = req.block_table.num_blocks();
    CHECK_CUDA(cudaMemcpy(s.block_table, req.block_table.data(),
        n_blocks * sizeof(int), cudaMemcpyHostToDevice));

    forward_tp(tw, s, pool, cfg, nccl, cublas,
               flat_reqs, flat_tokens, flat_positions, token_to_req,
               token_slot, last_tok_idx, prefill_flat_idx, dec_flat_idx,
               total_tokens);

    if (rank == 0) {
      cudaStreamSynchronize(nccl.stream);
      cur_token = sample_topk(s.logits, cfg.vocab_size, 30, 0.0f, rng);
    }
    MPI_Bcast(&cur_token, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (is_prefill) {
      t_decode_start = clock::now();
      is_prefill     = false;
    } else {
      t_decode_end = clock::now();
      n_decode_tokens++;
      req.pos++;
    }
  }

  if (rank == 0) {
    printf("\n");

    double prefill_ms = std::chrono::duration<double, std::milli>(
        t_decode_start - t_prefill_start).count();
    double decode_ms  = std::chrono::duration<double, std::milli>(
        t_decode_end - t_decode_start).count();

    double prefill_tps = n_prefill_tokens / (prefill_ms / 1000.0);
    double decode_tps  = n_decode_tokens  / (decode_ms  / 1000.0);

    fprintf(stderr, "----\n");
    fprintf(stderr, "prefill: %d tokens in %.2f ms (%.1f tokens/s)\n",
            n_prefill_tokens, prefill_ms, prefill_tps);
    fprintf(stderr, "decode:  %d tokens in %.2f ms (%.1f tokens/s)\n",
            n_decode_tokens, decode_ms, decode_tps);
  }

  // 释放这轮的 KV cache block
  req.block_table.free_blocks([pool](int id) { pool->free(id); });
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

  cudaSetDevice(0);

  // 1. NCCL 初始化
  NCCLContext nccl;
  nccl.rank    = rank;
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
  GGUFFile  gguf;
  Config    cfg;
  ModelFile mf;
  Weights   w;
  Tokenizer tokenizer;
  if (load_gguf(gguf, argv[1]) != 0)       { MPI_Finalize(); return 1; }
  if (load_config(cfg, gguf) != 0)          { MPI_Finalize(); return 1; }
  if (open_model(argv[1], mf) != 0)         { MPI_Finalize(); return 1; }
  if (load_weights(w, cfg, gguf, mf) != 0)  { MPI_Finalize(); return 1; }
  if (load_tokenizer(tokenizer, gguf) != 0) { MPI_Finalize(); return 1; }

  // 3. 切分权重上传 GPU
  TPWeights tw;
  upload_tp_weights(tw, w, cfg, rank, n_ranks);

  // 4. cublas
  cublasHandle_t cublas;
  CHECK_CUBLAS(cublasCreate(&cublas));
  CHECK_CUBLAS(cublasSetStream(cublas, nccl.stream));

  // 5. BlockPool + RunState
  int max_batch          = 1;
  int max_prefill_len    = 4096;
  int max_flat_tokens    = max_prefill_len + max_batch;
  int max_blocks_per_seq = (cfg.seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  size_t budget    = (size_t)(free_mem * 0.8);
  int    head_dim  = cfg.dim / cfg.n_heads;
  int    kv_dim    = cfg.n_kv_heads * head_dim;
  int    kv_per_rank = kv_dim / n_ranks;
  size_t pg_sz     = (size_t)BLOCK_SIZE * cfg.n_layers * kv_per_rank * 2 * sizeof(__half);
  int    total_pgs = (int)(budget / pg_sz);

  BlockPool* pool = new BlockPool(total_pgs, BLOCK_SIZE, cfg.n_layers, kv_per_rank);
  fprintf(stderr, "rank=%d BlockPool: %d blocks\n", rank, total_pgs);

  TPRunState s;
  alloc_tp_run_state(s, cfg, max_batch, max_blocks_per_seq, max_flat_tokens, n_ranks);

  // 6. 多轮对话循环
  while (true) {
    std::string input;
    if (rank == 0) {
      printf("User: ");
      if (!std::getline(std::cin, input) || input.empty()) break;
    }

    int input_len = (rank == 0) ? (int)input.size() : 0;
    MPI_Bcast(&input_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (input_len == 0) break;
    input.resize(input_len);
    MPI_Bcast(input.data(), input_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    std::string      prompt = apply_chat_template(input);
    std::vector<int> token_ids;
    encode(tokenizer, prompt, token_ids);
    token_ids.insert(token_ids.begin(), tokenizer.bos_token_id);

    run_inference(tw, s, pool, cfg, tokenizer, nccl, cublas, token_ids);
  }

  delete pool;
  cublasDestroy(cublas);
  ncclCommDestroy(nccl.comm);
  cudaStreamDestroy(nccl.stream);
  MPI_Finalize();
  return 0;
}

#include <math_constants.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gpu_decoder.h"
#include "gpu_kernels.h"

// ============================================================================
// 权重上传/释放
// ============================================================================
static void upload_weights(GPUWeights& gw, const Weights& w, const Config& cfg) {
  int head_dim = cfg.dim / cfg.n_heads;
  int kv_dim = cfg.n_kv_heads * head_dim;
  int vocab_size = cfg.vocab_size;

  auto up16 = [](__half** dst, const uint16_t* src, size_t n) {
    CHECK_CUDA(cudaMalloc(dst, n * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(*dst, src, n * sizeof(__half), cudaMemcpyHostToDevice));
  };
  auto up32 = [](float** dst, const float* src, size_t n) {
    CHECK_CUDA(cudaMalloc(dst, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
  };

  up16(&gw.token_embedding, w.token_embedding, (size_t)vocab_size * cfg.dim);
  up32(&gw.rms_final, w.rms_final, cfg.dim);

  up16(&gw.wcls, w.wcls, (size_t)vocab_size * cfg.dim);

  gw.rms_att = new float*[cfg.n_layers];
  gw.wq = new __half*[cfg.n_layers];
  gw.wk = new __half*[cfg.n_layers];
  gw.wv = new __half*[cfg.n_layers];
  gw.wo = new __half*[cfg.n_layers];
  gw.bq = new float*[cfg.n_layers];
  gw.bk = new float*[cfg.n_layers];
  gw.bv = new float*[cfg.n_layers];
  gw.rms_ffn = new float*[cfg.n_layers];
  gw.w1 = new __half*[cfg.n_layers];
  gw.w2 = new __half*[cfg.n_layers];
  gw.w3 = new __half*[cfg.n_layers];

  for (int l = 0; l < cfg.n_layers; l++) {
    up32(&gw.rms_att[l], w.rms_att[l], cfg.dim);
    up16(&gw.wq[l], w.wq[l], (size_t)cfg.dim * cfg.dim);
    up16(&gw.wk[l], w.wk[l], (size_t)kv_dim * cfg.dim);
    up16(&gw.wv[l], w.wv[l], (size_t)kv_dim * cfg.dim);
    up16(&gw.wo[l], w.wo[l], (size_t)cfg.dim * cfg.dim);
    up32(&gw.bq[l], w.bq[l], cfg.dim);
    up32(&gw.bk[l], w.bk[l], kv_dim);
    up32(&gw.bv[l], w.bv[l], kv_dim);
    up32(&gw.rms_ffn[l], w.rms_ffn[l], cfg.dim);
    up16(&gw.w1[l], w.w1[l], (size_t)cfg.hidden_dim * cfg.dim);
    up16(&gw.w2[l], w.w2[l], (size_t)cfg.dim * cfg.hidden_dim);
    up16(&gw.w3[l], w.w3[l], (size_t)cfg.hidden_dim * cfg.dim);
  }
}

static void free_gpu_weights(GPUWeights& gw, const Config& cfg) {
  cudaFree(gw.token_embedding);
  cudaFree(gw.rms_final);
  cudaFree(gw.wcls);
  for (int l = 0; l < cfg.n_layers; l++) {
    cudaFree(gw.rms_att[l]);
    cudaFree(gw.wq[l]);
    cudaFree(gw.wk[l]);
    cudaFree(gw.wv[l]);
    cudaFree(gw.wo[l]);
    cudaFree(gw.bq[l]);
    cudaFree(gw.bk[l]);
    cudaFree(gw.bv[l]);
    cudaFree(gw.rms_ffn[l]);
    cudaFree(gw.w1[l]);
    cudaFree(gw.w2[l]);
    cudaFree(gw.w3[l]);
  }
  delete[] gw.rms_att;
  delete[] gw.wq;
  delete[] gw.wk;
  delete[] gw.wv;
  delete[] gw.wo;
  delete[] gw.bq;
  delete[] gw.bk;
  delete[] gw.bv;
  delete[] gw.rms_ffn;
  delete[] gw.w1;
  delete[] gw.w2;
  delete[] gw.w3;
}

// ============================================================================
// GPURunState 分配/释放
// ============================================================================
static void alloc_run_state(GPURunState& s, const Config& cfg, int max_batch,
                            int max_blocks_per_seq, int max_flat_tokens) {
  int dim = cfg.dim;
  int kv_dim = cfg.n_kv_heads * (cfg.dim / cfg.n_heads);

  // flat batch 主缓冲区
  CHECK_CUDA(cudaMalloc(&s.x, (size_t)max_flat_tokens * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb, (size_t)max_flat_tokens * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb2, (size_t)max_flat_tokens * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.q, (size_t)max_flat_tokens * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.k, (size_t)max_flat_tokens * kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.v, (size_t)max_flat_tokens * kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb, (size_t)max_flat_tokens * cfg.hidden_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb2, (size_t)max_flat_tokens * cfg.hidden_dim * sizeof(__half)));

  // 元数据
  CHECK_CUDA(cudaMalloc(&s.d_tokens, max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_positions, max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_token_seq, max_flat_tokens * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.d_slot_map, max_flat_tokens * sizeof(int)));

  // block table
  CHECK_CUDA(cudaMalloc(&s.block_table, (size_t)max_batch * max_blocks_per_seq * sizeof(int)));

  // logits
  CHECK_CUDA(cudaMalloc(&s.d_last_tok_idx, max_batch * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&s.logits_last_tok, (size_t)max_batch * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.logits_fp16, (size_t)max_batch * cfg.vocab_size * sizeof(__half)));
  CHECK_CUDA(cudaMallocHost(&s.logits, (size_t)max_batch * cfg.vocab_size * sizeof(float)));

  // prefill 专用
  CHECK_CUDA(cudaMalloc(&s.d_prefill_flat, max_flat_tokens * sizeof(int)));

  // decode 专用
  CHECK_CUDA(cudaMalloc(&s.d_decode_flat, max_batch * sizeof(int)));
}

static void free_run_state(GPURunState& s) {
  cudaFree(s.x);
  cudaFree(s.xb);
  cudaFree(s.xb2);
  cudaFree(s.q);
  cudaFree(s.k);
  cudaFree(s.v);
  cudaFree(s.hb);
  cudaFree(s.hb2);

  cudaFree(s.d_tokens);
  cudaFree(s.d_positions);
  cudaFree(s.d_token_seq);
  cudaFree(s.d_slot_map);
  cudaFree(s.block_table);

  cudaFree(s.d_last_tok_idx);
  cudaFree(s.logits_last_tok);
  cudaFree(s.logits_fp16);
  cudaFreeHost(s.logits);

  cudaFree(s.d_prefill_flat);
  cudaFree(s.d_decode_flat);
}

// ============================================================================
// GPUDecoder 实现
// ============================================================================
GPUDecoder::GPUDecoder(const std::string& model_file, int max_batch, int max_prefill_len,
                       int max_prefill_tokens_per_step, int max_total_tokens)
    : max_batch_(max_batch),
      max_prefill_len_(max_prefill_len),
      max_prefill_tokens_per_step_(max_prefill_tokens_per_step) {
  max_flat_tokens_ = std::max(max_total_tokens, max_prefill_tokens_per_step + max_batch);
  if (load_gguf(gguf, model_file) != 0) {
    fprintf(stderr, "load gguf failed\n");
    exit(1);
  }
  if (load_config(config, gguf) != 0) {
    fprintf(stderr, "load config failed\n");
    exit(1);
  }
  if (open_model(model_file, mf) != 0) {
    fprintf(stderr, "open model failed\n");
    exit(1);
  }
  if (load_weights(w, config, gguf, mf) != 0) {
    fprintf(stderr, "load weights failed\n");
    exit(1);
  }
  if (load_tokenizer(tokenizer, gguf) != 0) {
    fprintf(stderr, "load tokenizer failed\n");
    exit(1);
  }

  upload_weights(gw, w, config);

  max_blocks_per_seq_ = (config.seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  alloc_run_state(gs, config, max_batch_, max_blocks_per_seq_, max_flat_tokens_);

  // 权重和运行时缓冲区分配完后，查询剩余显存给 BlockPool
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  size_t budget = (size_t)(free_mem * 0.8);

  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
  size_t pg_sz = (size_t)BLOCK_SIZE * config.n_layers * kv_dim * 2 * sizeof(__half);
  int total_pgs = (int)(budget / pg_sz);

  fprintf(stderr, "GPU: total=%.2fGB free=%.2fGB kv_budget=%.2fGB blocks=%d\n",
          (float)total_mem / 1024 / 1024 / 1024, (float)free_mem / 1024 / 1024 / 1024,
          (float)budget / 1024 / 1024 / 1024, total_pgs);

  block_pool = new BlockPool(total_pgs, BLOCK_SIZE, config.n_layers, kv_dim);

  CHECK_CUBLAS(cublasCreate(&cublas_handle));
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));
}

GPUDecoder::~GPUDecoder() {
  cublasDestroy(cublas_handle);
  cudaStreamDestroy(stream);
  free_run_state(gs);
  free_gpu_weights(gw, config);
  free_weights(w);
  close_model(mf);
  delete block_pool;
}

float* GPUDecoder::get_logits_batch(int batch_idx) {
  cudaStreamSynchronize(stream);
  return gs.logits + (size_t)batch_idx * config.vocab_size;
}

void GPUDecoder::update_block_table_partial(const std::vector<Request*>& running,
                                            const std::vector<int>& changed, int max_blk) {
  for (int i : changed) {
    const Request* req = running[i];
    int n = (req != nullptr) ? (int)req->block_table.num_blocks() : 0;
    if (n == 0) {
      continue;
    }
    /**
     * 只上传实际用到的 n 个 block，不上传全部 max_blk 个
     * 节省 HtoD 带宽：实际用 n*4 bytes，全量上传需要 max_blk*4 bytes
     * 安全性:
     *  - attention kernel 只访问 0 到 pos/block_size 的逻辑块范围
     *  - ensure_pages 保证这个范围内的所有块都已分配物理块
     *  - 未分配的块（下标 >= n）不会被访问到，无需初始化
     */
    CHECK_CUDA(cudaMemcpyAsync(gs.block_table + i * max_blk, req->block_table.data(),
                               n * sizeof(int), cudaMemcpyHostToDevice, stream));
  }
}

void GPUDecoder::forward_flat(
    const std::vector<FlatRequest>& flat_reqs,
    const std::vector<int>& flat_tokens,       // 所有请求的 token id，按 flat 顺序打包
    const std::vector<int>& flat_positions,    // 每个 token 的绝对位置，用于 RoPE
    const std::vector<int>& token_to_req,      // 每个 token 属于哪个请求
    const std::vector<int>& token_slot,        // 每个 token 的绝对物理槽位，KV cache 写入用
    const std::vector<int>& last_tok_idx,      // 每个请求最后一个 token 在 flat batch 里的位置
    const std::vector<int>& prefill_flat_idx,  // decode token 在 flat batch 里的位置
    const std::vector<int>& dec_flat_idx,      // decode token 在 flat batch 里的位置
    int total_tokens)                          // flat batch 里的 token 总数
{
  int dim = config.dim;
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
  int kv_mul = config.n_heads / config.n_kv_heads;
  int T = 256;  // threads per block
  int batch = (int)flat_reqs.size();
  int n_prefill = (int)prefill_flat_idx.size();
  int n_dec = (int)dec_flat_idx.size();

  // 上传元数据
  CHECK_CUDA(cudaMemcpyAsync(gs.d_tokens, flat_tokens.data(), total_tokens * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(gs.d_positions, flat_positions.data(), total_tokens * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(gs.d_token_seq, token_to_req.data(), total_tokens * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(gs.d_slot_map, token_slot.data(), total_tokens * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(gs.d_last_tok_idx, last_tok_idx.data(), batch * sizeof(int),
                             cudaMemcpyHostToDevice, stream));

  if (n_prefill > 0) {
    CHECK_CUDA(cudaMemcpyAsync(gs.d_prefill_flat, prefill_flat_idx.data(),
                               prefill_flat_idx.size() * sizeof(int), cudaMemcpyHostToDevice,
                               stream));
  }

  if (n_dec > 0) {
    CHECK_CUDA(cudaMemcpyAsync(gs.d_decode_flat, dec_flat_idx.data(), n_dec * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
  }

  /**
   * Embedding
   * 每个 token 一个 block，每个 block 256 个线程
   *
   * IN:
   * - gs.d_tokens
   * OUT:
   * - gs.x
   */
  embedding_kernel<<<total_tokens, T, 0, stream>>>(gs.x, gw.token_embedding, gs.d_tokens,
                                                   total_tokens, dim);
  CHECK_KERNEL();

  size_t smem = (FA_BLOCK + head_dim + 8 + 8 + 2) * sizeof(float);

  for (int l = 0; l < config.n_layers; l++) {
    /**
     * RMSNorm (attention)
     * 每个 token 一个 block，每个 block 256 个线程
     *
     * IN:
     * - gs.x
     * OUT:
     * - gs.xb
     */
    rmsnorm_kernel<<<total_tokens, T, 0, stream>>>(gs.xb, gs.x, gw.rms_att[l], dim);
    CHECK_KERNEL();

    /**
     * QKV projection
     *
     * IN:
     * - gs.xb
     * OUT:
     * - gs.q
     * - gs.k
     * - gs.v
     */
    matmul(cublas_handle, gs.q, gs.xb, gw.wq[l], dim, dim, total_tokens);
    matmul(cublas_handle, gs.k, gs.xb, gw.wk[l], dim, kv_dim, total_tokens);
    matmul(cublas_handle, gs.v, gs.xb, gw.wv[l], dim, kv_dim, total_tokens);

    /**
     * QKV bias
     * 每个 token 一个 block，每个 block 256 个线程
     *
     * IN:
     * - gs.q, gs.k, gs.v
     * OUT:
     * - gs.q, gs.k, gs.v
     */
    add_bias_kernel<<<total_tokens, T, 0, stream>>>(gs.q, gw.bq[l], total_tokens, dim);
    add_bias_kernel<<<total_tokens, T, 0, stream>>>(gs.k, gw.bk[l], total_tokens, kv_dim);
    add_bias_kernel<<<total_tokens, T, 0, stream>>>(gs.v, gw.bv[l], total_tokens, kv_dim);
    CHECK_KERNEL();

    /**
     * RoPE
     * 每个 token 一个 block，每个 block 256 个线程
     *
     * IN:
     * - gs.q, gs.k
     * OUT:
     * - gs.q, gs.k
     */
    rope_kernel<<<total_tokens, T, 0, stream>>>(gs.q, gs.k, gs.d_positions, total_tokens, dim,
                                                kv_dim, head_dim, config.rope_freq_base);
    CHECK_KERNEL();

    /**
     * KV cache write (1D slot mapping)
     * 每个 token 一个 block，每个 block 256 个线程
     *
     * IN:
     * - gs.k
     * - gs.v
     *
     * OUT:
     * - k_cache
     * - v_cache
     */
    kvcache_write_kernel<<<total_tokens, T, 0, stream>>>(
        block_pool->get_k_cache(), block_pool->get_v_cache(), gs.k, gs.v, gs.d_slot_map,
        total_tokens, l, config.n_layers, kv_dim);
    CHECK_KERNEL();

    if (n_prefill > 0) {
      /**
       * 每个 block 处理一个 head, 每个 block 256 个线程
       *
       * IN:
       * - q_fr (来自 gs.q)
       * OUT:
       * - o_fr (来自 gs.xb)
       */
      dim3 grid(config.n_heads, prefill_flat_idx.size());
      prefill_attention_kernel<<<grid, T, smem, stream>>>(
          gs.q, block_pool->get_k_cache(), block_pool->get_v_cache(), gs.xb, gs.block_table,
          gs.d_token_seq, gs.d_positions, gs.d_prefill_flat, max_blocks_per_seq_, BLOCK_SIZE,
          config.n_layers, l, dim, kv_dim, head_dim, kv_mul);
      CHECK_KERNEL();
    }

    if (n_dec > 0) {
      /**
       * 每个 block 处理一个 head, 每个 head 256 个线程
       * IN:
       * - gs.q
       * OUT:
       * - gs.xb
       */
      dim3 grid_dec(config.n_heads, n_dec);
      decode_attention_kernel<<<grid_dec, T, smem, stream>>>(
          gs.q, block_pool->get_k_cache(), block_pool->get_v_cache(), gs.xb, gs.block_table,
          gs.d_token_seq, gs.d_positions, gs.d_decode_flat, l, max_blocks_per_seq_, BLOCK_SIZE, dim,
          kv_dim, head_dim, kv_mul, config.n_layers);
      CHECK_KERNEL();
    }

    /**
     * Output projection + residual
     *
     * IN:
     * - gs.xb
     * OUT:
     * - gs.xb2
     */
    matmul(cublas_handle, gs.xb2, gs.xb, gw.wo[l], dim, dim, total_tokens);
    /**
     * 残差
     * 启动 total_token 个块，每个块 256 个线程
     *
     * IN:
     * - gs.x
     * OUT:
     * - gs.x
     */
    residual_kernel<<<total_tokens, T, 0, stream>>>(gs.x, gs.xb2, total_tokens, dim);
    CHECK_KERNEL();

    /**
     * FFN RMSNorm
     * 启动 total_token 个块，每个块 256 个线程
     *
     * IN:
     * - gs.x
     * OUT:
     * - gs.xb
     */
    rmsnorm_kernel<<<total_tokens, T, 0, stream>>>(gs.xb, gs.x, gw.rms_ffn[l], dim);
    CHECK_KERNEL();

    // SwiGLU FFN
    /**
     * 升维
     *
     * IN:
     * - gs.xb
     * OUT:
     * - gs.hb
     */
    matmul(cublas_handle, gs.hb, gs.xb, gw.w1[l], dim, config.hidden_dim, total_tokens);
    /**
     * 升维
     *
     * IN:
     * - gs.xb
     * OUT:
     * - gs.hb2
     */
    matmul(cublas_handle, gs.hb2, gs.xb, gw.w3[l], dim, config.hidden_dim, total_tokens);
    /**
     * 激活
     *
     * IN:
     * - gs.hb
     * - gs.hb2
     * OUT:
     * - gs.hb
     */
    swiglu_kernel<<<total_tokens, T, 0, stream>>>(gs.hb, gs.hb2, total_tokens, config.hidden_dim);
    CHECK_KERNEL();

    /**
     * FFN output projection + residual
     * 降维
     *
     * IN:
     * - gs.hb
     * OUT:
     * - gs.xb2
     */
    matmul(cublas_handle, gs.xb2, gs.hb, gw.w2[l], config.hidden_dim, dim, total_tokens);
    /**
     * 残差
     * IN:
     * - gs.x
     * - gs.xb2
     * OUT:
     * - gs.x
     */
    residual_kernel<<<total_tokens, T, 0, stream>>>(gs.x, gs.xb2, total_tokens, dim);
    CHECK_KERNEL();
  }

  /**
   * Final RMSNorm (all tokens)
   * IN:
   * - gs.x
   * OUT:
   * - gs.xb
   */
  rmsnorm_kernel<<<total_tokens, T, 0, stream>>>(gs.xb, gs.x, gw.rms_final, dim);
  CHECK_KERNEL();

  /**
   * 从 flat batch 里提取每个请求最后一个 token 的特征，用于 logits 计算
   *
   * 推理时只需要最后一个 token 的 logits 来采样下一个 token：
   * - prefill 请求: 最后一个 prefill token 的输出
   * - decode  请求: 唯一的 decode token 的输出
   *
   * 启动 batch 个 block，每个 block 处理一个 token
   *
   * IN:
   * - gs.xb
   * - gs.d_last_tok_idx
   * OUT:
   * - gs.logits_last_tok
   */
  extract_last_token_kernel<<<batch, T, 0, stream>>>(gs.logits_last_tok, gs.xb, gs.d_last_tok_idx,
                                                     batch, dim);
  CHECK_KERNEL();

  /**
   * Logits
   * IN:
   * - logits_last_tok
   * OUT:
   * - logits_fp16
   */
  matmul(cublas_handle, gs.logits_fp16, gs.logits_last_tok, gw.wcls, dim, config.vocab_size, batch);

  // 精度转换
  fp16_to_fp32_kernel<<<batch, T, 0, stream>>>(gs.logits, gs.logits_fp16, batch, config.vocab_size);
  CHECK_KERNEL();
}

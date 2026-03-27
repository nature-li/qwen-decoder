#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "gpu_decoder.h"

#define CHECK_CUDA(call)                                                      \
  {                                                                           \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), \
              __LINE__);                                                      \
      exit(1);                                                                \
    }                                                                         \
  }

#define CHECK_CUBLAS(call)                                                \
  {                                                                       \
    cublasStatus_t status = call;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                \
      fprintf(stderr, "cuBLAS Error: %d at line %d\n", status, __LINE__); \
      exit(1);                                                            \
    }                                                                     \
  }

// ============================================================================
// GPUWeights 上传/释放
// ============================================================================

static void upload_weights(GPUWeights& gw, const Weights& w,
                           const Config& config) {
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
  int vocab_size = config.vocab_size;

  // 上传 fp16（uint16_t* -> __half*）
  auto upload_fp16 = [](__half** dst, const uint16_t* src, size_t n) {
    CHECK_CUDA(cudaMalloc(dst, n * sizeof(__half)));
    CHECK_CUDA(
        cudaMemcpy(*dst, src, n * sizeof(__half), cudaMemcpyHostToDevice));
  };

  // 上传 fp32
  auto upload_fp32 = [](float** dst, const float* src, size_t n) {
    CHECK_CUDA(cudaMalloc(dst, n * sizeof(float)));
    CHECK_CUDA(
        cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
  };

  upload_fp16(&gw.token_embedding, w.token_embedding,
              (size_t)vocab_size * config.dim);
  upload_fp32(&gw.rms_final, w.rms_final, config.dim);
  upload_fp16(&gw.wcls, w.wcls, (size_t)vocab_size * config.dim);

  gw.rms_att = new float*[config.n_layers];
  gw.wq = new __half*[config.n_layers];
  gw.wk = new __half*[config.n_layers];
  gw.wv = new __half*[config.n_layers];
  gw.wo = new __half*[config.n_layers];
  gw.bq = new float*[config.n_layers];
  gw.bk = new float*[config.n_layers];
  gw.bv = new float*[config.n_layers];
  gw.rms_ffn = new float*[config.n_layers];
  gw.w1 = new __half*[config.n_layers];
  gw.w2 = new __half*[config.n_layers];
  gw.w3 = new __half*[config.n_layers];

  for (int l = 0; l < config.n_layers; l++) {
    upload_fp32(&gw.rms_att[l], w.rms_att[l], config.dim);
    upload_fp16(&gw.wq[l], w.wq[l], (size_t)config.dim * config.dim);
    upload_fp16(&gw.wk[l], w.wk[l], (size_t)kv_dim * config.dim);
    upload_fp16(&gw.wv[l], w.wv[l], (size_t)kv_dim * config.dim);
    upload_fp16(&gw.wo[l], w.wo[l], (size_t)config.dim * config.dim);
    upload_fp32(&gw.bq[l], w.bq[l], config.dim);
    upload_fp32(&gw.bk[l], w.bk[l], kv_dim);
    upload_fp32(&gw.bv[l], w.bv[l], kv_dim);
    upload_fp32(&gw.rms_ffn[l], w.rms_ffn[l], config.dim);
    upload_fp16(&gw.w1[l], w.w1[l], (size_t)config.hidden_dim * config.dim);
    upload_fp16(&gw.w2[l], w.w2[l], (size_t)config.dim * config.hidden_dim);
    upload_fp16(&gw.w3[l], w.w3[l], (size_t)config.hidden_dim * config.dim);
  }
}

static void free_gpu_weights(GPUWeights& gw, const Config& config) {
  cudaFree(gw.token_embedding);
  cudaFree(gw.rms_final);
  cudaFree(gw.wcls);

  for (int l = 0; l < config.n_layers; l++) {
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

static void alloc_gpu_run_state(GPURunState& s, const Config& config) {
  int dim = config.dim;
  int kv_dim = config.n_kv_heads * (config.dim / config.n_heads);
  int seq_len = config.seq_len;

  CHECK_CUDA(cudaMalloc(&s.x, dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb, dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.xb2, dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.q, dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.k, kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.v, kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.att, config.n_heads * seq_len * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&s.hb, config.hidden_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&s.hb2, config.hidden_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(
      &s.k_cache, (size_t)config.n_layers * seq_len * kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(
      &s.v_cache, (size_t)config.n_layers * seq_len * kv_dim * sizeof(__half)));
  CHECK_CUDA(cudaMallocHost(&s.logits, config.vocab_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&s.logits_fp16, config.vocab_size * sizeof(__half)));
}

static void free_gpu_run_state(GPURunState& s) {
  cudaFree(s.x);
  cudaFree(s.xb);
  cudaFree(s.xb2);
  cudaFree(s.q);
  cudaFree(s.k);
  cudaFree(s.v);
  cudaFree(s.att);
  cudaFree(s.hb);
  cudaFree(s.hb2);
  cudaFree(s.k_cache);
  cudaFree(s.v_cache);
  cudaFreeHost(s.logits);
  cudaFree(s.logits_fp16);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * Embedding lookup kernel (fp16 -> __half)
 * Grid:  (dim + 255) / 256 个 block
 * Block: 256 个线程
 * 每个 thread 负责 out 的 1 个元素
 */
__global__ void embedding_kernel(__half* out, const __half* table, int token,
                                 int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) out[i] = table[token * dim + i];
}

/**
 * RMSNorm kernel (warp reduce)
 * 输入 x 是 fp16，权重是 fp32，输出是 fp16
 * Grid:  1 个 block
 * Block: 256 个线程
 */
__device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void rmsnorm_kernel(__half* out, const __half* x,
                               const float* weight, int dim) {
  __shared__ float warp_sums[8];
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  float local_sum = 0.0f;
  for (int i = tid; i < dim; i += blockDim.x) {
    float xi = __half2float(x[i]);
    local_sum += xi * xi;
  }

  local_sum = warp_reduce_sum(local_sum);
  if (lane_id == 0) warp_sums[warp_id] = local_sum;
  __syncthreads();

  if (warp_id == 0) {
    float val = (lane_id < 8) ? warp_sums[lane_id] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane_id == 0) warp_sums[0] = val;
  }
  __syncthreads();

  float norm = rsqrtf(warp_sums[0] / dim + 1e-6f);
  for (int i = tid; i < dim; i += blockDim.x) {
    out[i] = __float2half(__half2float(x[i]) * norm * weight[i]);
  }
}

/**
 * Add bias kernel (fp32 bias 加到 fp16 向量上)
 * Grid:  (n + 255) / 256 个 block
 * Block: 256 个线程
 */
__global__ void add_bias_kernel(__half* out, const float* bias, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(__half2float(out[i]) + bias[i]);
}

/**
 * RoPE kernel (全 fp16)
 * Grid:  (dim/2 + 255) / 256 个 block
 * Block: 256 个线程
 */
__global__ void rope_kernel(__half* q, __half* k, int pos, int dim, int kv_dim,
                            int head_dim, float rope_freq_base) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;

  if (idx < dim) {
    int pair = (idx % head_dim) / 2;
    float theta = 1.0f / powf(rope_freq_base, 2.0f * pair / head_dim);
    float cos_v = cosf(theta * pos);
    float sin_v = sinf(theta * pos);
    float q0 = __half2float(q[idx]), q1 = __half2float(q[idx + 1]);
    q[idx] = __float2half(q0 * cos_v - q1 * sin_v);
    q[idx + 1] = __float2half(q0 * sin_v + q1 * cos_v);
  }

  if (idx < kv_dim) {
    int pair = (idx % head_dim) / 2;
    float theta = 1.0f / powf(rope_freq_base, 2.0f * pair / head_dim);
    float cos_v = cosf(theta * pos);
    float sin_v = sinf(theta * pos);
    float k0 = __half2float(k[idx]), k1 = __half2float(k[idx + 1]);
    k[idx] = __float2half(k0 * cos_v - k1 * sin_v);
    k[idx + 1] = __float2half(k0 * sin_v + k1 * cos_v);
  }
}

/**
 * KV Cache 写入 kernel (fp16)
 * Grid:  (kv_dim + 255) / 256 个 block
 * Block: 256 个线程
 */
__global__ void kvcache_write_kernel(__half* k_cache, __half* v_cache,
                                     const __half* k, const __half* v,
                                     int layer, int pos, int seq_len,
                                     int kv_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kv_dim) {
    int offset = layer * seq_len * kv_dim + pos * kv_dim + i;
    k_cache[offset] = k[i];
    v_cache[offset] = v[i];
  }
}

/**
 * Attention kernel (fp16 kv cache，fp32 scores)
 * Grid:  n_heads 个 block
 * Block: 256 个线程
 * shared memory: [seq_len] fp32 scores
 */
__global__ void attention_kernel(const __half* q, const __half* k_cache,
                                 const __half* v_cache, __half* out, float* att,
                                 int pos, int seq_len, int kv_dim, int head_dim,
                                 int kv_mul) {
  int h = blockIdx.x;
  int tid = threadIdx.x;
  float scale = rsqrtf((float)head_dim);

  // 每个 head 用自己的 att 区域
  float* scores = att + h * seq_len;

  const __half* q_head = q + h * head_dim;

  // Q @ K^T
  for (int t = tid; t <= pos; t += blockDim.x) {
    const __half* k_head = k_cache + t * kv_dim + (h / kv_mul) * head_dim;
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++)
      score += __half2float(q_head[d]) * __half2float(k_head[d]);
    scores[t] = score * scale;
  }
  __syncthreads();

  // softmax
  if (tid == 0) {
    float max_val = scores[0];
    for (int t = 1; t <= pos; t++) max_val = fmaxf(max_val, scores[t]);
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
      scores[t] = expf(scores[t] - max_val);
      sum += scores[t];
    }
    for (int t = 0; t <= pos; t++) scores[t] /= sum;
  }
  __syncthreads();

  // scores @ V
  __half* out_head = out + h * head_dim;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    float val = 0.0f;
    for (int t = 0; t <= pos; t++) {
      const __half* v_head = v_cache + t * kv_dim + (h / kv_mul) * head_dim;
      val += scores[t] * __half2float(v_head[d]);
    }
    out_head[d] = __float2half(val);
  }
}

/**
 * SwiGLU kernel (fp16)
 * Grid:  (hidden_dim + 255) / 256 个 block
 * Block: 256 个线程
 */
__global__ void swiglu_kernel(__half* hb, const __half* hb2, int hidden_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden_dim) {
    float x = __half2float(hb[i]);
    hb[i] = __float2half(x * (1.0f / (1.0f + expf(-x))) * __half2float(hb2[i]));
  }
}

/**
 * Residual add kernel (fp16)
 * Grid:  (dim + 255) / 256 个 block
 * Block: 256 个线程
 */
__global__ void residual_kernel(__half* x, const __half* delta, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) x[i] = __float2half(__half2float(x[i]) + __half2float(delta[i]));
}

/**
 * 最终 logits 输出：fp16 -> fp32
 * Grid:  (vocab_size + 255) / 256 个 block
 * Block: 256 个线程
 */
__global__ void fp16_to_fp32_kernel(float* out, const __half* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __half2float(in[i]);
}

// ============================================================================
// cublasHgemm 封装
// ============================================================================

/**
 * matmul: out = x @ w^T
 * x: [n] fp16, w: [d, n] fp16, out: [d] fp16
 */
static void matmul_half(cublasHandle_t handle, __half* out, const __half* x,
                        const __half* w, int n, int d) {
  const __half alpha = __float2half(1.0f);
  const __half beta = __float2half(0.0f);
  CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, d, 1, n, &alpha, w,
                           n, x, n, &beta, out, d));
}

// ============================================================================
// GPUDecoder 实现
// ============================================================================

GPUDecoder::GPUDecoder(const std::string& model_file) {
  if (load_gguf(gguf, model_file) != 0) {
    fprintf(stderr, "failed to load gguf\n");
    exit(1);
  }
  if (load_config(config, gguf) != 0) {
    fprintf(stderr, "failed to load config\n");
    exit(1);
  }
  if (open_model(model_file, mf) != 0) {
    fprintf(stderr, "failed to open model\n");
    exit(1);
  }
  if (load_weights(w, config, gguf, mf) != 0) {
    fprintf(stderr, "failed to load weights\n");
    exit(1);
  }
  if (load_tokenizer(tokenizer, gguf) != 0) {
    fprintf(stderr, "failed to load tokenizer\n");
    exit(1);
  }

  upload_weights(gw, w, config);
  alloc_gpu_run_state(gs, config);
  CHECK_CUBLAS(cublasCreate(&cublas_handle));
}

GPUDecoder::~GPUDecoder() {
  cublasDestroy(cublas_handle);
  free_gpu_run_state(gs);
  free_gpu_weights(gw, config);
  free_weights(w);
  close_model(mf);
}

float* GPUDecoder::get_logits() {
  cudaDeviceSynchronize();
  return gs.logits;
}

void GPUDecoder::forward(int token, int pos) {
  int dim = config.dim;
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
  int kv_mul = config.n_heads / config.n_kv_heads;
  int threads = 256;

  // 1. Embedding lookup
  embedding_kernel<<<(dim + threads - 1) / threads, threads>>>(
      gs.x, gw.token_embedding, token, dim);

  for (int l = 0; l < config.n_layers; l++) {
    // 2. RMSNorm
    rmsnorm_kernel<<<1, threads>>>(gs.xb, gs.x, gw.rms_att[l], dim);

    // 3. QKV 投影
    matmul_half(cublas_handle, gs.q, gs.xb, gw.wq[l], dim, dim);
    matmul_half(cublas_handle, gs.k, gs.xb, gw.wk[l], dim, kv_dim);
    matmul_half(cublas_handle, gs.v, gs.xb, gw.wv[l], dim, kv_dim);

    // 4. 加 bias
    add_bias_kernel<<<(dim + threads - 1) / threads, threads>>>(gs.q, gw.bq[l],
                                                                dim);
    add_bias_kernel<<<(kv_dim + threads - 1) / threads, threads>>>(
        gs.k, gw.bk[l], kv_dim);
    add_bias_kernel<<<(kv_dim + threads - 1) / threads, threads>>>(
        gs.v, gw.bv[l], kv_dim);

    // 5. RoPE
    rope_kernel<<<(dim / 2 + threads - 1) / threads, threads>>>(
        gs.q, gs.k, pos, dim, kv_dim, head_dim, config.rope_freq_base);

    // 6. KV Cache 写入
    kvcache_write_kernel<<<(kv_dim + threads - 1) / threads, threads>>>(
        gs.k_cache, gs.v_cache, gs.k, gs.v, l, pos, config.seq_len, kv_dim);

    // 7. Attention
    attention_kernel<<<config.n_heads, threads>>>(
        gs.q, gs.k_cache + (size_t)l * config.seq_len * kv_dim,
        gs.v_cache + (size_t)l * config.seq_len * kv_dim, gs.xb, gs.att, pos,
        config.seq_len, kv_dim, head_dim, kv_mul);

    // 8. 输出投影 + 残差
    matmul_half(cublas_handle, gs.xb2, gs.xb, gw.wo[l], dim, dim);
    residual_kernel<<<(dim + threads - 1) / threads, threads>>>(gs.x, gs.xb2,
                                                                dim);

    // 9. FFN RMSNorm
    rmsnorm_kernel<<<1, threads>>>(gs.xb, gs.x, gw.rms_ffn[l], dim);

    // 10. SwiGLU FFN
    matmul_half(cublas_handle, gs.hb, gs.xb, gw.w1[l], dim, config.hidden_dim);
    matmul_half(cublas_handle, gs.hb2, gs.xb, gw.w3[l], dim, config.hidden_dim);
    swiglu_kernel<<<(config.hidden_dim + threads - 1) / threads, threads>>>(
        gs.hb, gs.hb2, config.hidden_dim);

    // 11. FFN 输出投影 + 残差
    matmul_half(cublas_handle, gs.xb2, gs.hb, gw.w2[l], config.hidden_dim, dim);
    residual_kernel<<<(dim + threads - 1) / threads, threads>>>(gs.x, gs.xb2,
                                                                dim);
  }

  // 12. 最终 RMSNorm
  rmsnorm_kernel<<<1, threads>>>(gs.xb, gs.x, gw.rms_final, dim);

  // 13. logits: fp16 matmul 然后转 fp32
  matmul_half(cublas_handle, gs.logits_fp16, gs.xb, gw.wcls, dim,
              config.vocab_size);
  fp16_to_fp32_kernel<<<(config.vocab_size + threads - 1) / threads, threads>>>(
      gs.logits, gs.logits_fp16, config.vocab_size);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model_file>\n", argv[0]);
    return 1;
  }

  auto* decoder = new GPUDecoder(argv[1]);

  std::mt19937 rng(time(nullptr));
  float temperature = 0.7f;
  int top_k = 40;
  int max_new_tokens = 256;

  std::string user_input;
  printf("User: ");
  std::getline(std::cin, user_input);
  std::cout << user_input << std::endl;

  decoder->generate(user_input, max_new_tokens, temperature, top_k, rng);

  delete decoder;
  return 0;
}
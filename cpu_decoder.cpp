#include "cpu_decoder.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

void alloc_run_state(RunState& s, const Config& config) {
  int dim = config.dim;
  int kv_dim = config.n_kv_heads * (config.dim / config.n_heads);
  int seq_len = config.seq_len;

  s.x = new float[dim];
  s.xb = new float[dim];
  s.xb2 = new float[dim];
  s.q = new float[dim];
  s.k = new float[kv_dim];
  s.v = new float[kv_dim];
  s.att = new float[config.n_heads * seq_len];
  s.hb = new float[config.hidden_dim];
  s.hb2 = new float[config.hidden_dim];
  s.logits = new float[config.vocab_size];
  s.k_cache = new float[config.n_layers * seq_len * kv_dim];
  s.v_cache = new float[config.n_layers * seq_len * kv_dim];
}

void free_run_state(RunState& s) {
  delete[] s.x;
  delete[] s.xb;
  delete[] s.xb2;
  delete[] s.q;
  delete[] s.k;
  delete[] s.v;
  delete[] s.att;
  delete[] s.hb;
  delete[] s.hb2;
  delete[] s.logits;
  delete[] s.k_cache;
  delete[] s.v_cache;
}

// fp16 矩阵乘向量: out = x @ w^T
// w 是 fp16，计算时转成 float
void matmul_fp16(float* out, const float* x, const uint16_t* w, int n, int d) {
  for (int i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += x[j] * fp16_to_float(w[i * n + j]);
    }
    out[i] = val;
  }
}

// RMSNorm
void rmsnorm(float* out, const float* x, const float* weight, int dim) {
  float ss = 0.0f;
  for (int i = 0; i < dim; i++) ss += x[i] * x[i];
  ss = 1.0f / sqrtf(ss / dim + 1e-6f);
  for (int i = 0; i < dim; i++) out[i] = x[i] * ss * weight[i];
}

// SiLU
float silu(float x) { return x * (1.0f / (1.0f + expf(-x))); }

// fp16 embedding lookup，转成 float
void embedding_lookup(float* out, const uint16_t* table, int token, int dim) {
  const uint16_t* row = table + token * dim;
  for (int i = 0; i < dim; i++) out[i] = fp16_to_float(row[i]);
}

void forward(const Config& config, const Weights& w, RunState& s, int token,
             int pos) {
  int dim = config.dim;
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
  int kv_mul = config.n_heads / config.n_kv_heads;

  // 1. Embedding lookup (fp16 -> float)
  embedding_lookup(s.x, w.token_embedding, token, dim);

  for (int l = 0; l < config.n_layers; l++) {
    // 2. Attention 前 RMSNorm
    rmsnorm(s.xb, s.x, w.rms_att[l], dim);

    // 3. QKV 投影 (fp16 matmul)
    matmul_fp16(s.q, s.xb, w.wq[l], dim, dim);
    matmul_fp16(s.k, s.xb, w.wk[l], dim, kv_dim);
    matmul_fp16(s.v, s.xb, w.wv[l], dim, kv_dim);

    // 4. 加 bias
    for (int i = 0; i < dim; i++) s.q[i] += w.bq[l][i];
    for (int i = 0; i < kv_dim; i++) s.k[i] += w.bk[l][i];
    for (int i = 0; i < kv_dim; i++) s.v[i] += w.bv[l][i];

    // 5. RoPE 动态计算
    for (int h = 0; h < config.n_heads; h++) {
      float* q_head = s.q + h * head_dim;
      for (int i = 0; i < head_dim / 2; i++) {
        float theta = 1.0f / powf(config.rope_freq_base, 2.0f * i / head_dim);
        float cos_val = cosf(theta * pos);
        float sin_val = sinf(theta * pos);
        float q0 = q_head[2 * i], q1 = q_head[2 * i + 1];
        q_head[2 * i] = q0 * cos_val - q1 * sin_val;
        q_head[2 * i + 1] = q0 * sin_val + q1 * cos_val;
      }
    }
    for (int h = 0; h < config.n_kv_heads; h++) {
      float* k_head = s.k + h * head_dim;
      for (int i = 0; i < head_dim / 2; i++) {
        float theta = 1.0f / powf(config.rope_freq_base, 2.0f * i / head_dim);
        float cos_val = cosf(theta * pos);
        float sin_val = sinf(theta * pos);
        float k0 = k_head[2 * i], k1 = k_head[2 * i + 1];
        k_head[2 * i] = k0 * cos_val - k1 * sin_val;
        k_head[2 * i + 1] = k0 * sin_val + k1 * cos_val;
      }
    }

    // 6. KV Cache 写入
    float* k_cache_layer = s.k_cache + l * config.seq_len * kv_dim;
    float* v_cache_layer = s.v_cache + l * config.seq_len * kv_dim;
    memcpy(k_cache_layer + pos * kv_dim, s.k, kv_dim * sizeof(float));
    memcpy(v_cache_layer + pos * kv_dim, s.v, kv_dim * sizeof(float));

    // 7. Attention
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int h = 0; h < config.n_heads; h++) {
      float* q_head = s.q + h * head_dim;
      float* att_head = s.att + h * config.seq_len;

      for (int t = 0; t <= pos; t++) {
        float* k_head = k_cache_layer + t * kv_dim + (h / kv_mul) * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) score += q_head[d] * k_head[d];
        att_head[t] = score * scale;
      }

      float max_val = att_head[0];
      for (int t = 1; t <= pos; t++) max_val = fmaxf(max_val, att_head[t]);
      float sum = 0.0f;
      for (int t = 0; t <= pos; t++) {
        att_head[t] = expf(att_head[t] - max_val);
        sum += att_head[t];
      }
      for (int t = 0; t <= pos; t++) att_head[t] /= sum;

      float* out_head = s.xb + h * head_dim;
      memset(out_head, 0, head_dim * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        float* v_head = v_cache_layer + t * kv_dim + (h / kv_mul) * head_dim;
        for (int d = 0; d < head_dim; d++)
          out_head[d] += att_head[t] * v_head[d];
      }
    }

    // 8. Attention 输出投影 + 残差
    matmul_fp16(s.xb2, s.xb, w.wo[l], dim, dim);
    for (int i = 0; i < dim; i++) s.x[i] += s.xb2[i];

    // 9. FFN 前 RMSNorm
    rmsnorm(s.xb, s.x, w.rms_ffn[l], dim);

    // 10. SwiGLU FFN
    matmul_fp16(s.hb, s.xb, w.w1[l], dim, config.hidden_dim);
    matmul_fp16(s.hb2, s.xb, w.w3[l], dim, config.hidden_dim);
    for (int i = 0; i < config.hidden_dim; i++)
      s.hb[i] = silu(s.hb[i]) * s.hb2[i];

    // 11. FFN 输出投影 + 残差
    matmul_fp16(s.xb2, s.hb, w.w2[l], config.hidden_dim, dim);
    for (int i = 0; i < dim; i++) s.x[i] += s.xb2[i];
  }

  // 12. 最终 RMSNorm
  rmsnorm(s.xb, s.x, w.rms_final, dim);

  // 13. 输出 logits
  matmul_fp16(s.logits, s.xb, w.wcls, dim, config.vocab_size);
}
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

/**
 * softmax (CPU)
 * 原地修改 x，结果写回 x
 * 两遍扫描：第一遍找 max，第二遍算 exp 和 sum，第三遍归一化
 */
void softmax(float* x, int n) {
  // 1. 找 max（数值稳定）
  float max_val = *std::max_element(x, x + n);

  // 2. exp(x - max) 并累加 sum
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }

  // 3. 归一化
  for (int i = 0; i < n; i++) {
    x[i] /= sum;
  }
}

void online_softmax(float* x, int n, int block_size) {
  float m = -INFINITY;  // 当前全局最大值
  float d = 0.0f;       // 当前分母 sum(exp)

  for (int start = 0; start < n; start += block_size) {
    // 更新全局最大值
    float m_new =
        *std::max_element(x + start, x + std::min(start + block_size, n));
    m_new = std::max(m_new, m);

    // 修正旧的 d, 换算到新的 base
    float correction = expf(m - m_new);

    // 旧的 d 修正后加上新块的贡献
    d = d * correction;
    for (int i = 0; i < block_size; i++) {
      if (start + i < n) {
        d += expf(x[start + i] - m_new);
      }
    }
    m = m_new;
  }

  // 最终归一化
  for (int i = 0; i < n; i++) {
    x[i] = expf(x[i] - m) / d;
  }
}

void standard_attention(const float* q, const float* k, const float* v,
                        float* out, float* scores, int seq_len, int head_dim) {
  float scale = 1.0f / sqrtf((float)head_dim);

  // 1. scores = q @ k^T * scale
  for (int t = 0; t < seq_len; t++) {
    float s = 0.0f;
    for (int d = 0; d < head_dim; d++) {
      s += q[d] * k[t * head_dim + d];
    }
    scores[t] = s * scale;
  }

  // 2. softmax
  softmax(scores, seq_len);

  // 3. out = probs @ v
  memset(out, 0, head_dim * sizeof(float));
  for (int t = 0; t < seq_len; t++) {
    for (int d = 0; d < head_dim; d++) {
      out[d] += scores[t] * v[t * head_dim + d];
    }
  }
}

void flash_attention(const float* q, const float* k, const float* v, float* out,
                     int seq_len, int head_dim, int block_size) {
  float scale = 1.0f / sqrtf((float)head_dim);

  float m = -INFINITY;
  float d = 0.0f;
  std::vector<float> o(head_dim, 0.0f);

  for (int start = 0; start < seq_len; start += block_size) {
    int end = std::min(start + block_size, seq_len);
    int len = end - start;

    // 1. 计算块内 scores
    std::vector<float> s(len);
    for (int i = 0; i < len; i++) {
      float val = 0.0f;
      for (int dd = 0; dd < head_dim; dd++) {
        val += q[dd] * k[(start + i) * head_dim + dd];
      }
      s[i] = val * scale;
    }

    // 2. online softmax 更新 m 和 d
    float m_new = *std::max_element(s.begin(), s.end());
    m_new = std::max(m_new, m);

    float corrention = expf(m - m_new);
    d = d * corrention;
    for (int i = 0; i < len; i++) {
      d += expf(s[i] - m_new);
    }

    // 3. 更新输出: 修正历史贡献 + 新块贡献
    for (int dd = 0; dd < head_dim; dd++) {
      o[dd] *= corrention;
      for (int i = 0; i < len; i++) {
        o[dd] += expf(s[i] - m_new) * v[(start + i) * head_dim + dd];
      }
    }
    m = m_new;
  }

  // 最终归一化
  for (int dd = 0; dd < head_dim; dd++) {
    out[dd] = o[dd] / d;
  }
}

void test_softmax() {
  float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  softmax(x, 4);

  float sum = 0.0f;
  for (int i = 0; i < 4; i++) {
    std::cout << "y[" << i << "] = " << x[i] << "\n";
    sum += x[i];
  }
  std::cout << "sum  = " << sum << "\n";
}

void test_online_softmax() {
  float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  online_softmax(x, 4, 3);

  float sum = 0.0f;
  for (int i = 0; i < 4; i++) {
    std::cout << "y[" << i << "] = " << x[i] << "\n";
    sum += x[i];
  }
  std::cout << "sum  = " << sum << "\n";
}

void test_attention() {
  std::cout << "=== attention ===\n";
  const int head_dim = 4;
  const int seq_len = 4;

  float q[head_dim] = {0.1f, 0.2f, 0.3f, 0.4f};
  float k[seq_len * head_dim] = {
      0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
      0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f,
  };
  float v[seq_len * head_dim] = {
      1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
  };

  float out_std[head_dim];
  float out_flash[head_dim];
  float scores[seq_len];

  standard_attention(q, k, v, out_std, scores, seq_len, head_dim);
  flash_attention(q, k, v, out_flash, seq_len, head_dim, 2);

  std::cout << "standard:  ";
  for (int i = 0; i < head_dim; i++) std::cout << out_std[i] << " ";
  std::cout << "\n";

  std::cout << "flash:     ";
  for (int i = 0; i < head_dim; i++) std::cout << out_flash[i] << " ";
  std::cout << "\n";

  bool match = true;
  for (int i = 0; i < head_dim; i++)
    if (fabsf(out_std[i] - out_flash[i]) > 1e-5f) match = false;
  std::cout << "match: " << (match ? "true" : "false") << "\n";
}

int main() {
  test_softmax();
  test_online_softmax();
  test_attention();
  return 0;
}

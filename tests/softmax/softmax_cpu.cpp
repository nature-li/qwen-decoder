#include <algorithm>
#include <cmath>
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

int main() {
  float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  softmax(x, 4);

  float sum = 0.0f;
  for (int i = 0; i < 4; i++) {
    std::cout << "y[" << i << "] = " << x[i] << "\n";
    sum += x[i];
  }
  std::cout << "sum  = " << sum << "\n";

  return 0;
}

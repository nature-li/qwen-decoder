#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUDA(call)                                                                         \
  {                                                                                              \
    cudaError_t err = call;                                                                      \
    if (err != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(1);                                                                                   \
    }                                                                                            \
  }
#define CHECK_CUBLAS(call)                                                   \
  {                                                                          \
    cublasStatus_t s = call;                                                 \
    if (s != CUBLAS_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "cuBLAS Error: %d at %s:%d\n", s, __FILE__, __LINE__); \
      exit(1);                                                               \
    }                                                                        \
  }

// seed=42 生成的数据，和 numpy 一致
// X[2,4]
static float X_data[] = {0.304717f, -1.039984f, -2.420264f, 1.394428f,
                         0.777027f, 2.240665f,  -1.598986f, -0.318218f};
// Wq[4,4]
static float Wq_data[] = {0.314247f,  -0.908024f, -1.412304f, 1.465649f, -0.225776f, 0.067528f,
                          -1.424748f, -0.544383f, 0.110923f,  0.375698f, -0.600639f, 0.291787f,
                          1.852278f,  0.179252f,  -0.058020f, -1.057711f};
// Wo[4,4]（Wo 按行切，Wo0=Wo[:2,:], Wo1=Wo[2:,:]）
static float Wo_data[] = {-0.545156f, 0.484853f,  -0.843953f, 0.399303f,  -0.370250f, 1.257611f,
                          -0.308286f, -1.742371f, 0.184559f,  -0.993263f, 0.313424f,  -0.148683f,
                          -0.494835f, 0.313008f,  0.456550f,  0.227906f};
// W1[4,6]（W1 按列切，W1_0=W1[:,:3], W1_1=W1[:,3:]）
static float W1_data[] = {0.051945f, 1.131401f,  0.729091f,  0.278646f,  -1.132420f, -0.573918f,
                          0.414893f, 0.219572f,  0.024055f,  -0.028763f, 0.374736f,  -0.476220f,
                          0.286963f, -0.116680f, -1.320516f, 0.301999f,  0.078874f,  -0.306419f,
                          1.234291f, -0.398897f, 0.131798f,  0.150230f,  -0.135789f, 0.148149f};
// W2[6,4]（W2 按行切，W2_0=W2[:3,:], W2_1=W2[3:,:]）
static float W2_data[] = {-0.856421f, 0.218569f,  0.978736f,  -0.358901f, 0.484366f,  -0.234125f,
                          0.148765f,  0.812345f,  -0.123456f, 0.567890f,  -0.789012f, 0.345678f,
                          0.901234f,  -0.567890f, 0.234567f,  -0.890123f, -0.456789f, 0.123456f,
                          -0.678901f, 0.345678f,  0.789012f,  -0.234567f, 0.567890f,  -0.123456f};

static void to_fp16(const float* src, __half* dst, int n) {
  for (int i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}
static void to_fp32(const __half* src, float* dst, int n) {
  for (int i = 0; i < n; i++) dst[i] = __half2float(src[i]);
}
static void print_mat(const char* name, const float* m, int rows, int cols) {
  printf("%s [%d,%d]:\n", name, rows, cols);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) printf("%8.4f ", m[r * cols + c]);
    printf("\n");
  }
}
static float max_err(const float* a, const float* b, int n) {
  float e = 0;
  for (int i = 0; i < n; i++) e = fmaxf(e, fabsf(a[i] - b[i]));
  return e;
}
// 矩阵乘法参考实现（行主序）
static void matmul_ref(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      float s = 0;
      for (int k = 0; k < K; k++) s += A[m * K + k] * B[k * N + n];
      C[m * N + n] = s;
    }
}

int main() {
  int T = 2, dim = 4, dim2 = dim / 2;
  int hidden = 6, hidden2 = hidden / 2;

  cublasHandle_t cublas;
  CHECK_CUBLAS(cublasCreate(&cublas));

  // ===== 1. Q 投影：Wq 按列切 =====
  printf("\n========== 1. Q 投影：Wq 按列切 ==========\n");
  // Wq0 = Wq[:, :2]，Wq1 = Wq[:, 2:]
  float Wq0[8], Wq1[8];
  for (int r = 0; r < dim; r++) {
    Wq0[r * dim2 + 0] = Wq_data[r * dim + 0];
    Wq0[r * dim2 + 1] = Wq_data[r * dim + 1];
    Wq1[r * dim2 + 0] = Wq_data[r * dim + 2];
    Wq1[r * dim2 + 1] = Wq_data[r * dim + 3];
  }

  // 参考值
  float Q_ref[8], Q0_ref[4], Q1_ref[4];
  matmul_ref(X_data, Wq_data, Q_ref, T, dim, dim);
  matmul_ref(X_data, Wq0, Q0_ref, T, dim2, dim);
  matmul_ref(X_data, Wq1, Q1_ref, T, dim2, dim);

  // GPU
  __half *d_X, *d_Wq0, *d_Wq1, *d_Q0, *d_Q1;
  __half tmp[64];
  CHECK_CUDA(cudaMalloc(&d_X, T * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_Wq0, dim * dim2 * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_Wq1, dim * dim2 * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_Q0, T * dim2 * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_Q1, T * dim2 * sizeof(__half)));

  to_fp16(X_data, tmp, T * dim);
  CHECK_CUDA(cudaMemcpy(d_X, tmp, T * dim * sizeof(__half), cudaMemcpyHostToDevice));
  to_fp16(Wq0, tmp, dim * dim2);
  CHECK_CUDA(cudaMemcpy(d_Wq0, tmp, dim * dim2 * sizeof(__half), cudaMemcpyHostToDevice));
  to_fp16(Wq1, tmp, dim * dim2);
  CHECK_CUDA(cudaMemcpy(d_Wq1, tmp, dim * dim2 * sizeof(__half), cudaMemcpyHostToDevice));

  const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);

  // Wq0 行主序[dim, dim2]，列主序看[dim2, dim]，lda=dim2
  // Q0 = X @ Wq0: m=dim2, n=T, k=dim
  //   CHECK_CUBLAS(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dim2, T, dim, &alpha, d_Wq0, dim,
  //   d_X,
  //                            dim, &beta, d_Q0, dim2));
  CHECK_CUBLAS(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, dim2, T, dim, &alpha, d_Wq0, dim2, d_X,
                           dim, &beta, d_Q0, dim2));
  // CHECK_CUBLAS(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dim2, T, dim, &alpha, d_Wq0, dim, d_X,
  //                          dim, &beta, d_Q0, dim2));
  CHECK_CUBLAS(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, dim2, T, dim, &alpha, d_Wq1, dim2, d_X,
                           dim, &beta, d_Q1, dim2));

  float Q0_out[4], Q1_out[4];
  CHECK_CUDA(cudaMemcpy(tmp, d_Q0, T * dim2 * sizeof(__half), cudaMemcpyDeviceToHost));
  to_fp32(tmp, Q0_out, T * dim2);
  CHECK_CUDA(cudaMemcpy(tmp, d_Q1, T * dim2 * sizeof(__half), cudaMemcpyDeviceToHost));
  to_fp32(tmp, Q1_out, T * dim2);

  print_mat("Q0 ref", Q0_ref, T, dim2);
  print_mat("Q0 cublas", Q0_out, T, dim2);
  printf("Q0 误差: %f\n", max_err(Q0_out, Q0_ref, T * dim2));
  print_mat("Q1 ref", Q1_ref, T, dim2);
  print_mat("Q1 cublas", Q1_out, T, dim2);
  printf("Q1 误差: %f\n", max_err(Q1_out, Q1_ref, T * dim2));

  // ===== 2. O 投影：Wo 按行切 =====
  printf("\n========== 2. O 投影：Wo 按行切 ==========\n");
  // Wo0 = Wo[:2, :]，Wo1 = Wo[2:, :]
  // 假设 Attn0[T, dim2] 和 Attn1[T, dim2] 已知（用随机数模拟）
  float Attn0_data[] = {0.5f, -0.3f, 0.1f, 0.8f};  // [2,2]
  float Attn1_data[] = {-0.2f, 0.4f, 0.6f, -0.7f};

  float Wo0[8], Wo1[8];
  memcpy(Wo0, Wo_data, dim2 * dim * sizeof(float));
  memcpy(Wo1, Wo_data + 8, dim2 * dim * sizeof(float));

  // 参考：Out_full = [Attn0|Attn1] @ Wo
  float Attn_full[8] = {0.5f, -0.3f, -0.2f, 0.4f, 0.1f, 0.8f, 0.6f, -0.7f};
  float Out_ref[8];
  matmul_ref(Attn_full, Wo_data, Out_ref, T, dim, dim);

  float Out0_ref[8], Out1_ref[8], Out_tp_ref[8];
  matmul_ref(Attn0_data, Wo0, Out0_ref, T, dim, dim2);
  matmul_ref(Attn1_data, Wo1, Out1_ref, T, dim, dim2);
  for (int i = 0; i < T * dim; i++) Out_tp_ref[i] = Out0_ref[i] + Out1_ref[i];

  __half *d_Attn0, *d_Attn1, *d_Wo0, *d_Wo1, *d_Out0, *d_Out1;
  CHECK_CUDA(cudaMalloc(&d_Attn0, T * dim2 * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_Attn1, T * dim2 * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_Wo0, dim2 * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_Wo1, dim2 * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_Out0, T * dim * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_Out1, T * dim * sizeof(__half)));

  to_fp16(Attn0_data, tmp, T * dim2);
  CHECK_CUDA(cudaMemcpy(d_Attn0, tmp, T * dim2 * sizeof(__half), cudaMemcpyHostToDevice));
  to_fp16(Attn1_data, tmp, T * dim2);
  CHECK_CUDA(cudaMemcpy(d_Attn1, tmp, T * dim2 * sizeof(__half), cudaMemcpyHostToDevice));
  to_fp16(Wo0, tmp, dim2 * dim);
  CHECK_CUDA(cudaMemcpy(d_Wo0, tmp, dim2 * dim * sizeof(__half), cudaMemcpyHostToDevice));
  to_fp16(Wo1, tmp, dim2 * dim);
  CHECK_CUDA(cudaMemcpy(d_Wo1, tmp, dim2 * dim * sizeof(__half), cudaMemcpyHostToDevice));

  // Wo0 行主序[dim2, dim]，列主序看[dim, dim2]，lda=dim
  // Out0 = Attn0 @ Wo0: m=dim, n=T, k=dim2
  CHECK_CUBLAS(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, dim, T, dim2, &alpha, d_Wo0, dim,
                           d_Attn0, dim2, &beta, d_Out0, dim));
  CHECK_CUBLAS(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, dim, T, dim2, &alpha, d_Wo1, dim,
                           d_Attn1, dim2, &beta, d_Out1, dim));

  float Out0_out[8], Out1_out[8], Out_tp[8];
  CHECK_CUDA(cudaMemcpy(tmp, d_Out0, T * dim * sizeof(__half), cudaMemcpyDeviceToHost));
  to_fp32(tmp, Out0_out, T * dim);
  CHECK_CUDA(cudaMemcpy(tmp, d_Out1, T * dim * sizeof(__half), cudaMemcpyDeviceToHost));
  to_fp32(tmp, Out1_out, T * dim);
  for (int i = 0; i < T * dim; i++) Out_tp[i] = Out0_out[i] + Out1_out[i];

  print_mat("Out ref (full)", Out_ref, T, dim);
  print_mat("Out_tp (cublas)", Out_tp, T, dim);
  printf("Out 误差: %f\n", max_err(Out_tp, Out_tp_ref, T * dim));

  cublasDestroy(cublas);
  printf("\n全部完成\n");
  return 0;
}

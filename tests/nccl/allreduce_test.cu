#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call)                                                            \
  {                                                                                 \
    cudaError_t err = call;                                                         \
    if (err != cudaSuccess) {                                                       \
      fprintf(stderr, "CUDA error: %s at %d\n", cudaGetErrorString(err), __LINE__); \
      exit(1);                                                                      \
    }                                                                               \
  }

#define CHECK_NCCL(call)                                                            \
  {                                                                                 \
    ncclResult_t err = call;                                                        \
    if (err != ncclSuccess) {                                                       \
      fprintf(stderr, "NCCL error: %s at %d\n", ncclGetErrorString(err), __LINE__); \
      exit(1);                                                                      \
    }                                                                               \
  }

// rank0 是 root，负责生成 ncclUniqueId 并广播给 rank1
// 用简单的 TCP 传递 uniqueId
ncclUniqueId exchange_unique_id(int rank, const char* peer_ip) {
  ncclUniqueId id;

  if (rank == 0) {
    // rank0 生成 id，通过 TCP 发给 rank1
    CHECK_NCCL(ncclGetUniqueId(&id));

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(12345);
    bind(server_fd, (sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 1);

    fprintf(stderr, "[rank0] 等待 rank1 连接...\n");
    int client_fd = accept(server_fd, nullptr, nullptr);
    send(client_fd, &id, sizeof(id), 0);
    close(client_fd);
    close(server_fd);
    fprintf(stderr, "[rank0] uniqueId 已发送\n");

  } else {
    // rank1 连接 rank0，接收 id
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(peer_ip);
    addr.sin_port = htons(12345);

    fprintf(stderr, "[rank1] 连接 rank0...\n");
    while (connect(sock, (sockaddr*)&addr, sizeof(addr)) != 0) {
      usleep(100000);
    }
    recv(sock, &id, sizeof(id), MSG_WAITALL);
    close(sock);
    fprintf(stderr, "[rank1] uniqueId 已接收\n");
  }

  return id;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <rank> [peer_ip]\n", argv[0]);
    fprintf(stderr, "  rank0: %s 0\n", argv[0]);
    fprintf(stderr, "  rank1: %s 1 <rank0_ip>\n", argv[0]);
    return 1;
  }

  int rank = atoi(argv[1]);
  const char* peer_ip = (argc >= 3) ? argv[2] : "127.0.0.1";
  int n_ranks = 2;
  int n = 10 * 1024 * 1024;  // 测试向量大小

  // 1. 交换 uniqueId，初始化 NCCL
  ncclUniqueId id = exchange_unique_id(rank, peer_ip);

  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, n_ranks, id, rank));
  fprintf(stderr, "[rank%d] NCCL 初始化完成\n", rank);

  // 2. 分配 GPU buffer
  float* d_buf;
  CHECK_CUDA(cudaMalloc(&d_buf, n * sizeof(float)));

  // 每个 rank 填入不同的值
  // rank0 填 1.0，rank1 填 2.0
  // AllReduce sum 之后应该都是 3.0
  float val = (float)(rank + 1);
  std::vector<float> h_buf(n, val);
  CHECK_CUDA(cudaMemcpy(d_buf, h_buf.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  fprintf(stderr, "[rank%d] 输入值: %.1f\n", rank, val);

  // 3. AllReduce
  CHECK_NCCL(ncclAllReduce(d_buf, d_buf, n, ncclFloat, ncclSum, comm, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  // 4. 验证结果
  CHECK_CUDA(cudaMemcpy(h_buf.data(), d_buf, n * sizeof(float), cudaMemcpyDeviceToHost));
  fprintf(stderr, "[rank%d] AllReduce 结果: %.1f（期望 3.0）%s\n", rank, h_buf[0],
          h_buf[0] == 3.0f ? " ✓" : " ✗");

  // 5. 测速
  int iters = 100;
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  cudaEventRecord(t0);
  for (int i = 0; i < iters; i++) {
    ncclAllReduce(d_buf, d_buf, n, ncclFloat, ncclSum, comm, 0);
  }
  cudaEventRecord(t1);
  cudaDeviceSynchronize();

  float ms;
  cudaEventElapsedTime(&ms, t0, t1);
  double bytes = (double)n * sizeof(float) * 2;  // AllReduce 流量 = 2 * data_size
  double bandwidth = bytes * iters / (ms / 1000) / 1e9;
  fprintf(stderr, "[rank%d] 平均延迟: %.3f ms，带宽: %.2f GB/s\n", rank, ms / iters, bandwidth);

  cudaFree(d_buf);
  ncclCommDestroy(comm);
  return 0;
}
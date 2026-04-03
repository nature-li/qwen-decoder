#include "pd_comm.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <vector>

// ============================================================================
// TCP 工具函数
// ============================================================================

bool tcp_send(int fd, const void* buf, size_t size) {
  const char* p = (const char*)buf;
  size_t remaining = size;
  while (remaining > 0) {
    ssize_t n = write(fd, p, remaining);
    if (n <= 0) {
      fprintf(stderr, "tcp_send failed: %s\n", strerror(errno));
      return false;
    }
    p += n;
    remaining -= n;
  }
  return true;
}

bool tcp_recv(int fd, void* buf, size_t size) {
  char* p = (char*)buf;
  size_t remaining = size;
  while (remaining > 0) {
    ssize_t n = read(fd, p, remaining);
    if (n <= 0) {
      fprintf(stderr, "tcp_recv failed: %s\n", strerror(errno));
      return false;
    }
    p += n;
    remaining -= n;
  }
  return true;
}

bool tcp_send_msg(int fd, MsgType type, const void* body, int32_t body_size) {
  MsgHeader header;
  header.type = type;
  header.body_size = body_size;
  if (!tcp_send(fd, &header, sizeof(header))) return false;
  if (body_size > 0 && !tcp_send(fd, body, body_size)) return false;
  return true;
}

bool tcp_recv_header(int fd, MsgHeader& header) { return tcp_recv(fd, &header, sizeof(header)); }

// ============================================================================
// NcclComm 析构
// ============================================================================

NcclComm::~NcclComm() {
  if (comm) ncclCommDestroy(comm);
  if (stream) cudaStreamDestroy(stream);
}

// ============================================================================
// NCCL 初始化
// ============================================================================

bool nccl_init_prefill(NcclComm& nccl, int tcp_fd) {
  // P节点（rank=0）生成 uniqueId，通过 TCP 发给 D节点
  ncclUniqueId uid;
  ncclGetUniqueId(&uid);

  // 发给 D节点
  if (!tcp_send(tcp_fd, &uid, sizeof(uid))) {
    fprintf(stderr, "nccl_init_prefill: failed to send uniqueId\n");
    return false;
  }

  // 初始化 NCCL
  cudaStreamCreate(&nccl.stream);
  nccl.rank = 0;
  nccl.n_ranks = 2;
  ncclResult_t ret = ncclCommInitRank(&nccl.comm, 2, uid, 0);
  if (ret != ncclSuccess) {
    fprintf(stderr, "ncclCommInitRank failed: %s\n", ncclGetErrorString(ret));
    return false;
  }

  fprintf(stderr, "P节点 NCCL 初始化完成\n");
  return true;
}

bool nccl_init_decode(NcclComm& nccl, int tcp_fd) {
  // D节点（rank=1）从 TCP 接收 uniqueId
  ncclUniqueId uid;
  if (!tcp_recv(tcp_fd, &uid, sizeof(uid))) {
    fprintf(stderr, "nccl_init_decode: failed to recv uniqueId\n");
    return false;
  }

  // 初始化 NCCL
  cudaStreamCreate(&nccl.stream);
  nccl.rank = 1;
  nccl.n_ranks = 2;
  ncclResult_t ret = ncclCommInitRank(&nccl.comm, 2, uid, 1);
  if (ret != ncclSuccess) {
    fprintf(stderr, "ncclCommInitRank failed: %s\n", ncclGetErrorString(ret));
    return false;
  }

  fprintf(stderr, "D节点 NCCL 初始化完成\n");
  return true;
}

// ============================================================================
// KV cache 传输
//
// 布局：k_cache/v_cache [total_slots, n_layers, kv_dim]
// 每个 slot 对应一个 token 的 KV
// 只传该请求实际用到的 slots，不传整个 cache
//
// 发送顺序：
//   for each layer:
//     send k[slots, layer, :]  ← 该请求所有 token 在第 layer 层的 K
//     send v[slots, layer, :]  ← 该请求所有 token 在第 layer 层的 V
// ============================================================================

bool nccl_send_kv(NcclComm& nccl, const __half* k_cache, const __half* v_cache,
                  const std::vector<int>& slots, int n_tokens, int n_layers, int kv_dim) {
  // 分配临时 buffer，把不连续的 slots gather 到连续内存再发送
  // [n_tokens, kv_dim]
  size_t buf_size = (size_t)n_tokens * kv_dim * sizeof(__half);
  __half* d_buf;
  cudaMalloc(&d_buf, buf_size);

  for (int l = 0; l < n_layers; l++) {
    // gather K：把该请求所有 token 在第 l 层的 K 收集到 d_buf
    for (int i = 0; i < n_tokens; i++) {
      int slot = slots[i];
      const __half* src = k_cache + (size_t)slot * n_layers * kv_dim + (size_t)l * kv_dim;
      cudaMemcpyAsync(d_buf + (size_t)i * kv_dim, src, kv_dim * sizeof(__half),
                      cudaMemcpyDeviceToDevice, nccl.stream);
    }
    // 发送 K
    ncclSend(d_buf, n_tokens * kv_dim, ncclFloat16, 1, nccl.comm, nccl.stream);

    // gather V
    for (int i = 0; i < n_tokens; i++) {
      int slot = slots[i];
      const __half* src = v_cache + (size_t)slot * n_layers * kv_dim + (size_t)l * kv_dim;
      cudaMemcpyAsync(d_buf + (size_t)i * kv_dim, src, kv_dim * sizeof(__half),
                      cudaMemcpyDeviceToDevice, nccl.stream);
    }
    // 发送 V
    ncclSend(d_buf, n_tokens * kv_dim, ncclFloat16, 1, nccl.comm, nccl.stream);
  }

  cudaStreamSynchronize(nccl.stream);
  cudaFree(d_buf);
  return true;
}

bool nccl_recv_kv(NcclComm& nccl, __half* k_cache, __half* v_cache, const std::vector<int>& slots,
                  int n_tokens, int n_layers, int kv_dim) {
  // 分配临时 buffer 接收，再 scatter 到对应 slots
  size_t buf_size = (size_t)n_tokens * kv_dim * sizeof(__half);
  __half* d_buf;
  cudaMalloc(&d_buf, buf_size);

  for (int l = 0; l < n_layers; l++) {
    // 接收 K
    ncclRecv(d_buf, n_tokens * kv_dim, ncclFloat16, 0, nccl.comm, nccl.stream);
    cudaStreamSynchronize(nccl.stream);

    // scatter K：把 d_buf 写回对应 slots
    for (int i = 0; i < n_tokens; i++) {
      int slot = slots[i];
      __half* dst = k_cache + (size_t)slot * n_layers * kv_dim + (size_t)l * kv_dim;
      cudaMemcpyAsync(dst, d_buf + (size_t)i * kv_dim, kv_dim * sizeof(__half),
                      cudaMemcpyDeviceToDevice, nccl.stream);
    }

    // 接收 V
    ncclRecv(d_buf, n_tokens * kv_dim, ncclFloat16, 0, nccl.comm, nccl.stream);
    cudaStreamSynchronize(nccl.stream);

    // scatter V
    for (int i = 0; i < n_tokens; i++) {
      int slot = slots[i];
      __half* dst = v_cache + (size_t)slot * n_layers * kv_dim + (size_t)l * kv_dim;
      cudaMemcpyAsync(dst, d_buf + (size_t)i * kv_dim, kv_dim * sizeof(__half),
                      cudaMemcpyDeviceToDevice, nccl.stream);
    }
    cudaStreamSynchronize(nccl.stream);
  }

  cudaFree(d_buf);
  return true;
}
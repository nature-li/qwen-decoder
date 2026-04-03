#include "pd_comm.h"

#include <cstdio>

// ============================================================================
// NCCL 初始化：通过 ZMQ 交换 uniqueId
// ============================================================================

bool nccl_init_prefill(NcclComm& nccl, PNode& pnode) {
  // P节点生成 uniqueId，通过 ZMQ push 发给 D节点
  ncclUniqueId uid;
  ncclGetUniqueId(&uid);

  zmq::message_t msg(sizeof(uid));
  memcpy(msg.data(), &uid, sizeof(uid));
  pnode.push.send(msg, zmq::send_flags::none);

  cudaStreamCreate(&nccl.stream);
  nccl.rank    = 0;
  nccl.n_ranks = 2;

  ncclResult_t ret = ncclCommInitRank(&nccl.comm, 2, uid, 0);
  if (ret != ncclSuccess) {
    fprintf(stderr, "ncclCommInitRank failed: %s\n", ncclGetErrorString(ret));
    return false;
  }
  fprintf(stderr, "P节点 NCCL 初始化完成\n");
  return true;
}

bool nccl_init_decode(NcclComm& nccl, DNode& dnode) {
  // D节点通过 ZMQ pull 接收 uniqueId
  zmq::message_t msg;
  dnode.pull.recv(msg, zmq::recv_flags::none);

  ncclUniqueId uid;
  memcpy(&uid, msg.data(), sizeof(uid));

  cudaStreamCreate(&nccl.stream);
  nccl.rank    = 1;
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
// KV cache 传输（一次性传输，不分层）
// ============================================================================

bool nccl_send_kv(NcclComm& nccl,
                   const __half* k_cache, const __half* v_cache,
                   const std::vector<int>& slots,
                   int n_tokens, int n_layers, int kv_dim) {
  // 分配 buffer：[n_layers, 2, n_tokens, kv_dim]
  size_t total = (size_t)n_layers * 2 * n_tokens * kv_dim;
  __half* d_buf;
  cudaMalloc(&d_buf, total * sizeof(__half));

  // gather 所有层的 K 和 V 到连续 buffer
  for (int l = 0; l < n_layers; l++) {
    for (int i = 0; i < n_tokens; i++) {
      int slot = slots[i];
      // K
      const __half* k_src = k_cache
          + (size_t)slot * n_layers * kv_dim
          + (size_t)l * kv_dim;
      cudaMemcpyAsync(
          d_buf + (size_t)(l * 2 * n_tokens + i) * kv_dim,
          k_src, kv_dim * sizeof(__half),
          cudaMemcpyDeviceToDevice, nccl.stream);
      // V
      const __half* v_src = v_cache
          + (size_t)slot * n_layers * kv_dim
          + (size_t)l * kv_dim;
      cudaMemcpyAsync(
          d_buf + (size_t)((l * 2 + 1) * n_tokens + i) * kv_dim,
          v_src, kv_dim * sizeof(__half),
          cudaMemcpyDeviceToDevice, nccl.stream);
    }
  }
  cudaStreamSynchronize(nccl.stream);

  // 一次性发送
  ncclSend(d_buf, total, ncclFloat16, 1, nccl.comm, nccl.stream);
  cudaStreamSynchronize(nccl.stream);

  cudaFree(d_buf);
  return true;
}

bool nccl_recv_kv(NcclComm& nccl,
                   __half* k_cache, __half* v_cache,
                   const std::vector<int>& slots,
                   int n_tokens, int n_layers, int kv_dim) {
  size_t total = (size_t)n_layers * 2 * n_tokens * kv_dim;
  __half* d_buf;
  cudaMalloc(&d_buf, total * sizeof(__half));

  // 一次性接收
  ncclRecv(d_buf, total, ncclFloat16, 0, nccl.comm, nccl.stream);
  cudaStreamSynchronize(nccl.stream);

  // scatter 回对应 slots
  for (int l = 0; l < n_layers; l++) {
    for (int i = 0; i < n_tokens; i++) {
      int slot = slots[i];
      // K
      __half* k_dst = k_cache
          + (size_t)slot * n_layers * kv_dim
          + (size_t)l * kv_dim;
      cudaMemcpyAsync(
          k_dst,
          d_buf + (size_t)(l * 2 * n_tokens + i) * kv_dim,
          kv_dim * sizeof(__half),
          cudaMemcpyDeviceToDevice, nccl.stream);
      // V
      __half* v_dst = v_cache
          + (size_t)slot * n_layers * kv_dim
          + (size_t)l * kv_dim;
      cudaMemcpyAsync(
          v_dst,
          d_buf + (size_t)((l * 2 + 1) * n_tokens + i) * kv_dim,
          kv_dim * sizeof(__half),
          cudaMemcpyDeviceToDevice, nccl.stream);
    }
  }
  cudaStreamSynchronize(nccl.stream);

  cudaFree(d_buf);
  return true;
}
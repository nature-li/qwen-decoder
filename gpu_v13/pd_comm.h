#pragma once

#include <cuda_fp16.h>
#include <nccl.h>

#include <string>
#include <vector>

// ============================================================================
// 通信协议
//
// 控制面（TCP）：
//   D节点 → P节点：PrefillRequest（prompt 文本）
//   P节点 → D节点：PrefillResponse（prefill 完成，告知 KV cache 信息）
//
// 数据面（NCCL）：
//   P节点 → D节点：KV cache（每一层的 K 和 V）
//
// 流程：
//   1. 启动时 TCP 握手 + NCCL 初始化
//   2. D节点发 PrefillRequest 给 P节点
//   3. P节点做 prefill，完成后发 PrefillResponse
//   4. P节点通过 NCCL 把 KV cache 传给 D节点
//   5. D节点收到 KV cache，开始 decode
// ============================================================================

// TCP 端口
constexpr int PD_TCP_PORT = 9527;

// 控制消息类型
enum class MsgType : int32_t {
  PREFILL_REQUEST = 1,   // D → P：发送 prompt
  PREFILL_RESPONSE = 2,  // P → D：prefill 完成
  ALL_SENT = 3,          // D节点通知所有请求已发完
  SHUTDOWN = 4,
};

// 消息头（所有 TCP 消息都有）
struct MsgHeader {
  MsgType type;
  int32_t body_size;  // body 的字节数
};

// D → P：prefill 请求
struct PrefillRequest {
  int32_t req_id;
  int32_t n_tokens;  // prompt token 数量
  // 后面跟 n_tokens 个 int32_t（token ids）
};

// P → D：prefill 完成
struct PrefillResponse {
  int32_t req_id;
  int32_t n_tokens;     // prefill 了多少个 token
  int32_t n_layers;     // KV cache 层数
  int32_t kv_dim;       // KV cache 维度
  int32_t first_token;  // P节点采样的第一个 token
  // 后续 NCCL 传输 KV cache
};

// ============================================================================
// TCP 工具函数
// ============================================================================

// 发送完整数据（处理 partial write）
bool tcp_send(int fd, const void* buf, size_t size);

// 接收完整数据（处理 partial read）
bool tcp_recv(int fd, void* buf, size_t size);

// 发送消息（header + body）
bool tcp_send_msg(int fd, MsgType type, const void* body, int32_t body_size);

// 接收消息头
bool tcp_recv_header(int fd, MsgHeader& header);

// ============================================================================
// NCCL 通信
// ============================================================================

struct NcclComm {
  ncclComm_t comm;
  int rank;     // 0=P节点，1=D节点
  int n_ranks;  // 2
  cudaStream_t stream;

  NcclComm() : comm(nullptr), rank(-1), n_ranks(2), stream(nullptr) {}
  ~NcclComm();
};

/**
 * P节点初始化 NCCL（rank=0）
 * 通过 TCP 和 D节点交换 ncclUniqueId
 * @param tcp_fd  和 D节点的 TCP 连接
 */
bool nccl_init_prefill(NcclComm& comm, int tcp_fd);

/**
 * D节点初始化 NCCL（rank=1）
 * 通过 TCP 和 P节点交换 ncclUniqueId
 * @param tcp_fd  和 P节点的 TCP 连接
 */
bool nccl_init_decode(NcclComm& comm, int tcp_fd);

/**
 * P节点发送 KV cache 给 D节点
 * 逐层发送，每层发 K 和 V
 *
 * @param k_cache    [total_slots, n_layers, kv_dim]
 * @param v_cache    [total_slots, n_layers, kv_dim]
 * @param slots      该请求用到的所有 slot（按 pos 顺序）
 * @param n_tokens   token 数量
 * @param n_layers   层数
 * @param kv_dim     KV 维度
 */
bool nccl_send_kv(NcclComm& comm, const __half* k_cache, const __half* v_cache,
                  const std::vector<int>& slots, int n_tokens, int n_layers, int kv_dim);

/**
 * D节点接收 KV cache
 * 写入 D节点自己的 BlockPool 对应槽位
 */
bool nccl_recv_kv(NcclComm& comm, __half* k_cache, __half* v_cache, const std::vector<int>& slots,
                  int n_tokens, int n_layers, int kv_dim);
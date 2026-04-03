#pragma once

#include <cuda_fp16.h>
#include <nccl.h>

#include <cstring>
#include <string>
#include <vector>
#include <zmq.hpp>

// ============================================================================
// ZMQ 端口
// ============================================================================
constexpr int PD_ZMQ_REQ_PORT = 9527;   // D → P：发请求
constexpr int PD_ZMQ_RESP_PORT = 9528;  // P → D：发 response

// ============================================================================
// 控制消息
// ============================================================================
struct PrefillRequest {
  int32_t req_id;
  int32_t n_tokens;
  // 后面跟 n_tokens 个 int32_t token ids
};

struct PrefillResponse {
  int32_t req_id;
  int32_t n_tokens;
  int32_t n_layers;
  int32_t kv_dim;
  int32_t first_token;
};

// ============================================================================
// P节点 ZMQ：PULL 收请求，PUSH 发 response
// ============================================================================
struct PNode {
  zmq::context_t ctx{1};
  zmq::socket_t pull{ctx, ZMQ_PULL};  // 收 D节点的请求
  zmq::socket_t push{ctx, ZMQ_PUSH};  // 发 response 给 D节点

  bool init() {
    try {
      pull.bind("tcp://*:" + std::to_string(PD_ZMQ_REQ_PORT));
      push.bind("tcp://*:" + std::to_string(PD_ZMQ_RESP_PORT));
      fprintf(stderr, "P节点 ZMQ 初始化完成 req_port=%d resp_port=%d\n", PD_ZMQ_REQ_PORT,
              PD_ZMQ_RESP_PORT);
      return true;
    } catch (zmq::error_t& e) {
      fprintf(stderr, "P节点 ZMQ 初始化失败: %s\n", e.what());
      return false;
    }
  }

  // 接收一个 PrefillRequest（阻塞）
  bool recv_request(PrefillRequest& req, std::vector<int>& token_ids) {
    zmq::message_t msg;
    auto ret = pull.recv(msg, zmq::recv_flags::none);
    if (!ret) return false;
    const char* data = (const char*)msg.data();
    memcpy(&req, data, sizeof(PrefillRequest));
    token_ids.resize(req.n_tokens);
    memcpy(token_ids.data(), data + sizeof(PrefillRequest), req.n_tokens * sizeof(int));
    return true;
  }

  // 发送 PrefillResponse（非阻塞）
  bool send_response(const PrefillResponse& resp) {
    zmq::message_t msg(sizeof(PrefillResponse));
    memcpy(msg.data(), &resp, sizeof(PrefillResponse));
    auto ret = push.send(msg, zmq::send_flags::none);
    return ret.has_value();
  }
};

// ============================================================================
// D节点 ZMQ：PUSH 发请求，PULL 收 response
// ============================================================================
struct DNode {
  zmq::context_t ctx{1};
  zmq::socket_t push{ctx, ZMQ_PUSH};  // 发请求给 P节点
  zmq::socket_t pull{ctx, ZMQ_PULL};  // 收 P节点的 response

  bool init(const char* prefill_ip) {
    try {
      push.connect("tcp://" + std::string(prefill_ip) + ":" + std::to_string(PD_ZMQ_REQ_PORT));
      pull.connect("tcp://" + std::string(prefill_ip) + ":" + std::to_string(PD_ZMQ_RESP_PORT));
      fprintf(stderr, "D节点 ZMQ 初始化完成\n");
      return true;
    } catch (zmq::error_t& e) {
      fprintf(stderr, "D节点 ZMQ 初始化失败: %s\n", e.what());
      return false;
    }
  }

  // 发送 PrefillRequest
  bool send_request(const PrefillRequest& req, const std::vector<int>& token_ids) {
    int msg_size = sizeof(PrefillRequest) + req.n_tokens * sizeof(int);
    zmq::message_t msg(msg_size);
    char* data = (char*)msg.data();
    memcpy(data, &req, sizeof(PrefillRequest));
    memcpy(data + sizeof(PrefillRequest), token_ids.data(), req.n_tokens * sizeof(int));
    auto ret = push.send(msg, zmq::send_flags::none);
    return ret.has_value();
  }

  // 接收 PrefillResponse（阻塞）
  bool recv_response(PrefillResponse& resp) {
    zmq::message_t msg;
    auto ret = pull.recv(msg, zmq::recv_flags::none);
    if (!ret) return false;
    memcpy(&resp, msg.data(), sizeof(PrefillResponse));
    return true;
  }
};

// ============================================================================
// NCCL
// ============================================================================
struct NcclComm {
  ncclComm_t comm = nullptr;
  int rank = -1;
  int n_ranks = 2;
  cudaStream_t stream = nullptr;

  // 预分配 KV cache 传输 buffer
  __half* d_buf = nullptr;
  size_t buf_size = 0;

  // 按需扩容
  void ensure_buf(size_t size) {
    if (size <= buf_size) return;
    if (d_buf) cudaFree(d_buf);
    cudaMalloc(&d_buf, size);
    buf_size = size;
  }

  ~NcclComm() {
    if (comm) ncclCommDestroy(comm);
    if (stream) cudaStreamDestroy(stream);
    if (d_buf) cudaFree(d_buf);
  }
};

// P节点 NCCL 初始化：通过 ZMQ 交换 uniqueId
bool nccl_init_prefill(NcclComm& nccl, PNode& pnode);

// D节点 NCCL 初始化：通过 ZMQ 交换 uniqueId
bool nccl_init_decode(NcclComm& nccl, DNode& dnode);

// KV cache 传输（一次性传输所有层）
bool nccl_send_kv(NcclComm& nccl, const __half* k_cache, const __half* v_cache,
                  const std::vector<int>& slots, int n_tokens, int n_layers, int kv_dim);

bool nccl_recv_kv(NcclComm& nccl, __half* k_cache, __half* v_cache, const std::vector<int>& slots,
                  int n_tokens, int n_layers, int kv_dim);
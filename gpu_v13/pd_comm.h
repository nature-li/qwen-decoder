#pragma once
#include <cuda_fp16.h>
#include <nccl.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <zmq.hpp>

constexpr int PD_ZMQ_REQ_PORT = 9527;  // D → P：prefill 请求 + 心跳（DEALER→ROUTER）
constexpr int PD_ZMQ_REG_PORT = 9529;  // D → P：注册专用
constexpr int HEARTBEAT_INTERVAL_S = 5;
constexpr int HEARTBEAT_TIMEOUT_S = 15;
constexpr int NCCL_INIT_TIMEOUT_S = 30;

enum MsgType : int32_t {
  MSG_PREFILL_REQUEST = 0,
  MSG_HEARTBEAT = 1,
  MSG_SHUTDOWN = 2,
};

struct MsgHeader {
  MsgType msg_type;
};

struct PrefillRequest {
  int32_t d_node_id;
  int32_t req_id;
  int32_t n_tokens;
};

struct PrefillResponse {
  int32_t req_id;
  int32_t n_tokens;
  int32_t n_layers;
  int32_t kv_dim;
  int32_t first_token;
};

struct NcclComm {
  ncclComm_t comm = nullptr;
  int rank = -1;
  int n_ranks = 2;
  cudaStream_t stream = nullptr;
  __half* d_buf = nullptr;
  size_t buf_size = 0;

  void ensure_buf(size_t size);
  ~NcclComm();
};

struct PNode {
  zmq::context_t ctx{1};
  zmq::socket_t router{ctx, ZMQ_ROUTER};      // 接收请求 + 发 response，精确定向
  zmq::socket_t reg_router{ctx, ZMQ_ROUTER};  // 注册专用，p_register_thread 专用

  // key: d_node_id
  std::unordered_map<int, NcclComm*> nccl_comms;
  std::mutex nccl_mu;

  // key: d_node_id → ZMQ identity，用于定向发 response
  std::unordered_map<int, std::string> d_node_identity;
  std::mutex identity_mu;

  std::unordered_map<int, std::chrono::steady_clock::time_point> last_heartbeat;
  std::mutex heartbeat_mu;

  std::atomic<int> next_d_node_id{0};

  bool init();
  int alloc_dnode(ncclUniqueId& uid);
  void register_dnode(int d_node_id, NcclComm* nccl);
  void cleanup_failed_nccl(NcclComm* nccl);
  void cleanup_dnode(int d_node_id);
  void update_heartbeat(int d_node_id);
  std::vector<int> check_heartbeat_timeout();

  bool send_kv(int d_node_id, const __half* k_cache, const __half* v_cache,
               const std::vector<int>& slots, int n_tokens, int n_layers, int kv_dim);

  /**
   * 接收消息，同时记录 D节点的 ZMQ identity
   */
  bool recv_request(PrefillRequest& req, std::vector<int>& token_ids, MsgType& msg_type);

  /**
   * 精确发 response 给指定 D节点
   */
  bool send_response(int d_node_id, const PrefillResponse& resp);

  ~PNode();
};

struct DNode {
  int d_node_id = -1;
  zmq::context_t ctx{1};
  zmq::socket_t dealer{ctx, ZMQ_DEALER};  // 发请求 + 收 response
  zmq::socket_t reg_dealer{ctx, ZMQ_DEALER};
  NcclComm nccl;

  bool init(const char* prefill_ip);
  bool register_and_init_nccl();
  bool send_heartbeat();
  bool send_shutdown();
  bool send_request(const PrefillRequest& req, const std::vector<int>& token_ids);
  bool recv_response(PrefillResponse& resp);
};

bool nccl_recv_kv(NcclComm& nccl, __half* k_cache, __half* v_cache, const std::vector<int>& slots,
                  int n_tokens, int n_layers, int kv_dim);
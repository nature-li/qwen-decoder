#pragma once
#include <cuda_fp16.h>
#include <grpcpp/grpcpp.h>
#include <nccl.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "pd_service.grpc.pb.h"

constexpr int NCCL_INIT_TIMEOUT_S = 30;

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

// 前向声明
struct PNode;

// ============================================================================
// PrefillReactor：每个 D节点一个，管理 Prefill 双向流
// ============================================================================
class PrefillReactor : public grpc::ServerBidiReactor<pd::PrefillRequest, pd::PrefillResponse> {
 public:
  PrefillReactor(PNode* pnode);

  void OnReadDone(bool ok) override;
  void OnWriteDone(bool ok) override;
  void OnDone() override;

  /**
   * 发送 response 给 D节点
   */
  void WriteResponse(const pd::PrefillResponse& resp);

 private:
  PNode* pnode_;
  int d_node_id_;
  pd::PrefillRequest req_;
  pd::PrefillResponse pending_resp_;
  std::mutex write_mu_;
  bool writing_ = false;
};

// ============================================================================
// P节点：gRPC 异步服务端
// ============================================================================
struct PNode final : public pd::PrefillService::CallbackService {
  // 收到 prefill 请求时的回调
  std::function<void(pd::PrefillRequest)> on_request;

  // D节点断开时的回调
  std::function<void(int d_node_id)> on_disconnect;

  // 每个 D节点一个 NcclComm，shared_ptr 保证并发安全
  std::unordered_map<int, std::shared_ptr<NcclComm>> nccl_comms;
  std::mutex nccl_mu;

  // 每个 D节点的 reactor，key 是 d_node_id
  std::unordered_map<int, PrefillReactor*> streams;
  std::mutex streams_mu;

  std::atomic<int> next_d_node_id{0};
  std::unique_ptr<grpc::Server> server;

  bool init(const std::string& addr = "0.0.0.0:9527");

  /**
   * Register：异步，NCCL 初始化完成后才返回 OK 给 D节点
   */
  grpc::ServerUnaryReactor* Register(grpc::CallbackServerContext* ctx,
                                     const pd::RegisterRequest* req,
                                     pd::RegisterResponse* resp) override;

  /**
   * Prefill：异步双向流，每个 D节点一个 PrefillReactor
   */
  grpc::ServerBidiReactor<pd::PrefillRequest, pd::PrefillResponse>* Prefill(
      grpc::CallbackServerContext* ctx) override;

  void cleanup_dnode(int d_node_id);

  bool send_response(int d_node_id, const pd::PrefillResponse& resp);

  bool send_kv(int d_node_id, const __half* k_cache, const __half* v_cache,
               const std::vector<int>& slots, int n_tokens, int n_layers, int kv_dim);

  bool is_alive(int d_node_id) {
    std::lock_guard<std::mutex> lock(nccl_mu);
    return nccl_comms.count(d_node_id) > 0;
  }

  ~PNode();
};

// ============================================================================
// D节点：gRPC 客户端
// ============================================================================
struct DNode {
  int d_node_id = -1;
  std::unique_ptr<pd::PrefillService::Stub> stub;
  std::unique_ptr<grpc::ClientContext> prefill_ctx;
  std::unique_ptr<grpc::ClientReaderWriter<pd::PrefillRequest, pd::PrefillResponse>> prefill_stream;
  NcclComm nccl;

  bool init(const std::string& prefill_ip, int port = 9527);
  bool register_and_init_nccl();
  bool open_prefill_stream();
  bool send_request(const pd::PrefillRequest& req);
  bool recv_response(pd::PrefillResponse& resp);
  bool send_shutdown();
};

bool nccl_recv_kv(NcclComm& nccl, __half* k_cache, __half* v_cache, const std::vector<int>& slots,
                  int n_tokens, int n_layers, int kv_dim);
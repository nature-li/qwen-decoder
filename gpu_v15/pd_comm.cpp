#include "pd_comm.h"

#include <cstdio>
#include <thread>

// ============================================================================
// NcclComm
// ============================================================================

void NcclComm::ensure_buf(size_t size) {
  if (size <= buf_size) return;
  if (d_buf) cudaFree(d_buf);
  cudaMalloc(&d_buf, size);
  buf_size = size;
}

NcclComm::~NcclComm() {
  if (comm) {
    ncclCommAbort(comm);
    comm = nullptr;
  }
  if (stream) cudaStreamDestroy(stream);
  if (d_buf) cudaFree(d_buf);
}

// ============================================================================
// PrefillReactor
// ============================================================================

PrefillReactor::PrefillReactor(PNode* pnode) : pnode_(pnode), d_node_id_(-1), writing_(false) {
  StartRead(&req_);
}

void PrefillReactor::OnReadDone(bool ok) {
  if (!ok) {
    // 流关闭，清理资源
    if (d_node_id_ != -1) {
      fprintf(stderr, "P节点：D节点[%d] 连接断开，清理资源\n", d_node_id_);
      {
        std::lock_guard<std::mutex> lock(pnode_->streams_mu);
        pnode_->streams.erase(d_node_id_);
      }
      pnode_->cleanup_dnode(d_node_id_);
      if (pnode_->on_disconnect) pnode_->on_disconnect(d_node_id_);
    }
    Finish(grpc::Status::OK);
    return;
  }

  if (d_node_id_ == -1) {
    // 第一条消息，注册 stream
    d_node_id_ = req_.d_node_id();
    {
      std::lock_guard<std::mutex> lock(pnode_->streams_mu);
      pnode_->streams[d_node_id_] = this;
    }
    fprintf(stderr, "P节点：D节点[%d] Prefill stream 建立\n", d_node_id_);
  }

  // 通过回调通知外部
  if (pnode_->on_request) pnode_->on_request(req_);

  // 继续读下一条消息
  StartRead(&req_);
}

void PrefillReactor::OnWriteDone(bool ok) {
  if (!ok) {
    fprintf(stderr, "P节点：D节点[%d] Write 失败\n", d_node_id_);
  }
  std::lock_guard<std::mutex> lock(write_mu_);
  writing_ = false;
}

void PrefillReactor::OnDone() { delete this; }

void PrefillReactor::WriteResponse(const pd::PrefillResponse& resp) {
  std::lock_guard<std::mutex> lock(write_mu_);
  pending_resp_ = resp;
  if (!writing_) {
    writing_ = true;
    StartWrite(&pending_resp_);
  }
}

// ============================================================================
// PNode
// ============================================================================

bool PNode::init(const std::string& addr) {
  grpc::ServerBuilder builder;
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(this);
  server = builder.BuildAndStart();
  if (!server) {
    fprintf(stderr, "P节点：gRPC server 启动失败\n");
    return false;
  }
  fprintf(stderr, "P节点：gRPC server 启动成功，监听 %s\n", addr.c_str());
  return true;
}

grpc::ServerUnaryReactor* PNode::Register(grpc::CallbackServerContext* ctx,
                                          const pd::RegisterRequest* req,
                                          pd::RegisterResponse* resp) {
  int d_node_id = next_d_node_id++;
  fprintf(stderr, "P节点：新 D节点连接，分配 id=%d\n", d_node_id);

  ncclUniqueId uid;
  ncclGetUniqueId(&uid);
  resp->set_d_node_id(d_node_id);
  resp->set_nccl_uid(uid.internal, sizeof(uid.internal));

  // 先返回 OK，让 D节点拿到 uid 开始初始化 NCCL
  auto* reactor = ctx->DefaultReactor();
  reactor->Finish(grpc::Status::OK);

  // 新线程初始化 NCCL，和 D节点 ncclCommInitRank 同步握手
  std::thread([this, d_node_id, uid]() mutable {
    auto nccl = std::make_shared<NcclComm>();
    cudaStreamCreate(&nccl->stream);
    nccl->rank = 0;
    nccl->n_ranks = 2;

    ncclResult_t ret = ncclCommInitRank(&nccl->comm, 2, uid, 0);
    if (ret != ncclSuccess) {
      fprintf(stderr, "P节点：D节点[%d] NCCL 初始化失败: %s，释放资源\n", d_node_id,
              ncclGetErrorString(ret));
      // shared_ptr 析构自动释放
      return;
    }

    {
      std::lock_guard<std::mutex> lock(nccl_mu);
      nccl_comms[d_node_id] = nccl;
    }
    fprintf(stderr, "P节点：D节点[%d] NCCL 初始化完成\n", d_node_id);
  }).detach();

  return reactor;
}

grpc::ServerBidiReactor<pd::PrefillRequest, pd::PrefillResponse>* PNode::Prefill(
    grpc::CallbackServerContext* ctx) {
  return new PrefillReactor(this);
}

void PNode::cleanup_dnode(int d_node_id) {
  fprintf(stderr, "P节点：释放 D节点[%d] 资源\n", d_node_id);

  std::shared_ptr<NcclComm> nccl;
  {
    std::lock_guard<std::mutex> lock(nccl_mu);
    auto it = nccl_comms.find(d_node_id);
    if (it == nccl_comms.end()) {
      fprintf(stderr, "P节点：D节点[%d] 已清理，跳过\n", d_node_id);
      return;
    }
    nccl = it->second;
    nccl_comms.erase(it);
  }
  // shared_ptr 析构，引用计数 -1，send_kv 还在用时延迟析构
  fprintf(stderr, "P节点：D节点[%d] 资源释放完成\n", d_node_id);
}

bool PNode::send_response(int d_node_id, const pd::PrefillResponse& resp) {
  std::lock_guard<std::mutex> lock(streams_mu);
  auto it = streams.find(d_node_id);
  if (it == streams.end()) {
    fprintf(stderr, "P节点：D节点[%d] stream 不存在\n", d_node_id);
    return false;
  }
  it->second->WriteResponse(resp);
  return true;
}

bool PNode::send_kv(int d_node_id, const __half* k_cache, const __half* v_cache,
                    const std::vector<int>& slots, int n_tokens, int n_layers, int kv_dim) {
  std::shared_ptr<NcclComm> nccl;
  {
    std::lock_guard<std::mutex> lock(nccl_mu);
    auto it = nccl_comms.find(d_node_id);
    if (it == nccl_comms.end()) {
      fprintf(stderr, "P节点：D节点[%d] NCCL 未初始化\n", d_node_id);
      return false;
    }
    nccl = it->second;
  }

  size_t total = (size_t)n_layers * 2 * n_tokens * kv_dim;
  nccl->ensure_buf(total * sizeof(__half));

  for (int l = 0; l < n_layers; l++) {
    for (int i = 0; i < n_tokens; i++) {
      int slot = slots[i];
      const __half* k_src = k_cache + (size_t)slot * n_layers * kv_dim + (size_t)l * kv_dim;
      cudaMemcpyAsync(nccl->d_buf + (size_t)(l * 2 * n_tokens + i) * kv_dim, k_src,
                      kv_dim * sizeof(__half), cudaMemcpyDeviceToDevice, nccl->stream);
      const __half* v_src = v_cache + (size_t)slot * n_layers * kv_dim + (size_t)l * kv_dim;
      cudaMemcpyAsync(nccl->d_buf + (size_t)((l * 2 + 1) * n_tokens + i) * kv_dim, v_src,
                      kv_dim * sizeof(__half), cudaMemcpyDeviceToDevice, nccl->stream);
    }
  }
  cudaStreamSynchronize(nccl->stream);

  ncclResult_t ret = ncclSend(nccl->d_buf, total, ncclFloat16, 1, nccl->comm, nccl->stream);
  if (ret != ncclSuccess) {
    fprintf(stderr, "P节点：D节点[%d] ncclSend 失败: %s，触发清理\n", d_node_id,
            ncclGetErrorString(ret));
    std::thread([this, d_node_id]() { cleanup_dnode(d_node_id); }).detach();
    return false;
  }
  cudaStreamSynchronize(nccl->stream);
  return true;
}

PNode::~PNode() {
  if (server) server->Shutdown();
}

// ============================================================================
// DNode
// ============================================================================

bool DNode::init(const std::string& prefill_ip, int port) {
  std::string addr = prefill_ip + ":" + std::to_string(port);
  auto channel = grpc::CreateChannel(addr, grpc::InsecureChannelCredentials());
  stub = pd::PrefillService::NewStub(channel);
  fprintf(stderr, "D节点：gRPC 连接 %s\n", addr.c_str());
  return true;
}

bool DNode::register_and_init_nccl() {
  grpc::ClientContext ctx;
  pd::RegisterRequest req;
  pd::RegisterResponse resp;

  // Register 会等 P节点 NCCL 初始化完才返回
  grpc::Status status = stub->Register(&ctx, req, &resp);
  if (!status.ok()) {
    fprintf(stderr, "D节点：Register 失败: %s\n", status.error_message().c_str());
    return false;
  }

  d_node_id = resp.d_node_id();
  fprintf(stderr, "D节点：P节点分配 id=%d，开始 NCCL 初始化\n", d_node_id);

  ncclUniqueId uid;
  memcpy(uid.internal, resp.nccl_uid().data(), sizeof(uid.internal));

  cudaStreamCreate(&nccl.stream);
  nccl.rank = 1;
  nccl.n_ranks = 2;

  // 直接等，和 P节点 ncclCommInitRank 同步握手
  ncclResult_t ret = ncclCommInitRank(&nccl.comm, 2, uid, 1);
  if (ret != ncclSuccess) {
    fprintf(stderr, "D节点[%d] NCCL 初始化失败: %s\n", d_node_id, ncclGetErrorString(ret));
    if (nccl.stream) {
      cudaStreamDestroy(nccl.stream);
      nccl.stream = nullptr;
    }
    nccl.comm = nullptr;
    return false;
  }

  fprintf(stderr, "D节点[%d] NCCL 初始化完成\n", d_node_id);
  return true;
}

bool DNode::open_prefill_stream() {
  prefill_ctx = std::make_unique<grpc::ClientContext>();
  prefill_stream = stub->Prefill(prefill_ctx.get());
  if (!prefill_stream) {
    fprintf(stderr, "D节点[%d] Prefill stream 建立失败\n", d_node_id);
    return false;
  }
  fprintf(stderr, "D节点[%d] Prefill stream 建立成功\n", d_node_id);
  return true;
}

bool DNode::send_request(const pd::PrefillRequest& req) { return prefill_stream->Write(req); }

bool DNode::recv_response(pd::PrefillResponse& resp) { return prefill_stream->Read(&resp); }

bool DNode::send_shutdown() {
  if (prefill_stream) {
    prefill_stream->WritesDone();
  }
  return true;
}

// ============================================================================
// nccl_recv_kv
// ============================================================================

bool nccl_recv_kv(NcclComm& nccl, __half* k_cache, __half* v_cache, const std::vector<int>& slots,
                  int n_tokens, int n_layers, int kv_dim) {
  size_t total = (size_t)n_layers * 2 * n_tokens * kv_dim;
  nccl.ensure_buf(total * sizeof(__half));

  ncclResult_t ret = ncclRecv(nccl.d_buf, total, ncclFloat16, 0, nccl.comm, nccl.stream);
  if (ret != ncclSuccess) {
    fprintf(stderr, "D节点：ncclRecv 失败: %s\n", ncclGetErrorString(ret));
    return false;
  }
  cudaStreamSynchronize(nccl.stream);

  for (int l = 0; l < n_layers; l++) {
    for (int i = 0; i < n_tokens; i++) {
      int slot = slots[i];
      __half* k_dst = k_cache + (size_t)slot * n_layers * kv_dim + (size_t)l * kv_dim;
      cudaMemcpyAsync(k_dst, nccl.d_buf + (size_t)(l * 2 * n_tokens + i) * kv_dim,
                      kv_dim * sizeof(__half), cudaMemcpyDeviceToDevice, nccl.stream);
      __half* v_dst = v_cache + (size_t)slot * n_layers * kv_dim + (size_t)l * kv_dim;
      cudaMemcpyAsync(v_dst, nccl.d_buf + (size_t)((l * 2 + 1) * n_tokens + i) * kv_dim,
                      kv_dim * sizeof(__half), cudaMemcpyDeviceToDevice, nccl.stream);
    }
  }
  cudaStreamSynchronize(nccl.stream);
  return true;
}
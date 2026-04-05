#include "pd_comm.h"

#include <cstdio>
#include <future>
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
  if (comm) ncclCommDestroy(comm);
  if (stream) cudaStreamDestroy(stream);
  if (d_buf) cudaFree(d_buf);
}

// ============================================================================
// PNode
// ============================================================================

bool PNode::init() {
  try {
    router.bind("tcp://*:" + std::to_string(PD_ZMQ_REQ_PORT));
    reg_pull.bind("tcp://*:" + std::to_string(PD_ZMQ_REG_PORT));
    reg_push.bind("tcp://*:" + std::to_string(PD_ZMQ_REG_RESP_PORT));
    fprintf(stderr, "P节点 ZMQ 初始化完成\n");
    return true;
  } catch (zmq::error_t& e) {
    fprintf(stderr, "P节点 ZMQ 初始化失败: %s\n", e.what());
    return false;
  }
}

int PNode::alloc_dnode(ncclUniqueId& uid) {
  zmq::message_t reg_msg;
  reg_pull.recv(reg_msg, zmq::recv_flags::none);

  int d_node_id = next_d_node_id++;
  fprintf(stderr, "P节点：新 D节点连接，分配 id=%d\n", d_node_id);

  ncclGetUniqueId(&uid);

  struct {
    int32_t d_node_id;
    ncclUniqueId uid;
  } resp;
  resp.d_node_id = d_node_id;
  resp.uid = uid;
  zmq::message_t resp_msg(sizeof(resp));
  memcpy(resp_msg.data(), &resp, sizeof(resp));
  reg_push.send(resp_msg, zmq::send_flags::none);

  return d_node_id;
}

void PNode::register_dnode(int d_node_id, NcclComm* nccl) {
  {
    std::lock_guard<std::mutex> lock(heartbeat_mu);
    last_heartbeat[d_node_id] = std::chrono::steady_clock::now();
  }
  {
    std::lock_guard<std::mutex> lock(nccl_mu);
    nccl_comms[d_node_id] = nccl;
  }
  fprintf(stderr, "P节点 D节点[%d] 注册完成\n", d_node_id);
}

void PNode::cleanup_failed_nccl(NcclComm* nccl) {
  if (nccl->stream) {
    cudaStreamDestroy(nccl->stream);
    nccl->stream = nullptr;
  }
  nccl->comm = nullptr;
  delete nccl;
}

void PNode::cleanup_dnode(int d_node_id) {
  fprintf(stderr, "P节点：释放 D节点[%d] 资源\n", d_node_id);

  NcclComm* nccl = nullptr;
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
  delete nccl;

  {
    std::lock_guard<std::mutex> lock(heartbeat_mu);
    last_heartbeat.erase(d_node_id);
  }
  {
    std::lock_guard<std::mutex> lock(identity_mu);
    d_node_identity.erase(d_node_id);
  }

  fprintf(stderr, "P节点：D节点[%d] 资源释放完成\n", d_node_id);
}

void PNode::update_heartbeat(int d_node_id) {
  std::lock_guard<std::mutex> lock(heartbeat_mu);
  last_heartbeat[d_node_id] = std::chrono::steady_clock::now();
}

std::vector<int> PNode::check_heartbeat_timeout() {
  std::vector<int> timeout_nodes;
  auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(heartbeat_mu);
  for (auto& kv : last_heartbeat) {
    double elapsed = std::chrono::duration<double>(now - kv.second).count();
    if (elapsed > HEARTBEAT_TIMEOUT_S) {
      timeout_nodes.push_back(kv.first);
    }
  }
  return timeout_nodes;
}

bool PNode::recv_request(PrefillRequest& req, std::vector<int>& token_ids, MsgType& msg_type) {
  // ROUTER 收消息格式：[identity][空帧][消息体]
  zmq::message_t identity_msg;
  zmq::message_t empty_msg;
  zmq::message_t body_msg;

  router.recv(identity_msg, zmq::recv_flags::none);
  router.recv(empty_msg, zmq::recv_flags::none);
  router.recv(body_msg, zmq::recv_flags::none);

  const char* data = (const char*)body_msg.data();
  MsgHeader header;
  memcpy(&header, data, sizeof(MsgHeader));
  msg_type = header.msg_type;

  if (msg_type == MSG_HEARTBEAT || msg_type == MSG_SHUTDOWN) {
    memcpy(&req.d_node_id, data + sizeof(MsgHeader), sizeof(int32_t));
    // 更新 identity 映射
    std::lock_guard<std::mutex> lock(identity_mu);
    d_node_identity[req.d_node_id] = std::string((char*)identity_msg.data(), identity_msg.size());
    return true;
  }

  // prefill 请求
  memcpy(&req, data + sizeof(MsgHeader), sizeof(PrefillRequest));
  token_ids.resize(req.n_tokens);
  memcpy(token_ids.data(), data + sizeof(MsgHeader) + sizeof(PrefillRequest),
         req.n_tokens * sizeof(int));

  // 记录 identity
  {
    std::lock_guard<std::mutex> lock(identity_mu);
    d_node_identity[req.d_node_id] = std::string((char*)identity_msg.data(), identity_msg.size());
  }
  return true;
}

bool PNode::send_response(int d_node_id, const PrefillResponse& resp) {
  std::string identity;
  {
    std::lock_guard<std::mutex> lock(identity_mu);
    auto it = d_node_identity.find(d_node_id);
    if (it == d_node_identity.end()) {
      fprintf(stderr, "P节点：D节点[%d] identity 未知，无法发 response\n", d_node_id);
      return false;
    }
    identity = it->second;
  }

  // ROUTER 发消息格式：[identity][空帧][消息体]
  zmq::message_t identity_msg(identity.size());
  memcpy(identity_msg.data(), identity.data(), identity.size());

  zmq::message_t empty_msg(0);
  zmq::message_t body_msg(sizeof(PrefillResponse));
  memcpy(body_msg.data(), &resp, sizeof(PrefillResponse));

  router.send(identity_msg, zmq::send_flags::sndmore);
  router.send(empty_msg, zmq::send_flags::sndmore);
  router.send(body_msg, zmq::send_flags::none);
  return true;
}

bool PNode::send_kv(int d_node_id, const __half* k_cache, const __half* v_cache,
                    const std::vector<int>& slots, int n_tokens, int n_layers, int kv_dim) {
  NcclComm* nccl = nullptr;
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
  ncclSend(nccl->d_buf, total, ncclFloat16, 1, nccl->comm, nccl->stream);
  cudaStreamSynchronize(nccl->stream);
  return true;
}

PNode::~PNode() {
  std::lock_guard<std::mutex> lock(nccl_mu);
  for (auto& kv : nccl_comms) delete kv.second;
}

// ============================================================================
// DNode
// ============================================================================

bool DNode::init(const char* prefill_ip) {
  try {
    std::string ip = prefill_ip;
    dealer.connect("tcp://" + ip + ":" + std::to_string(PD_ZMQ_REQ_PORT));
    reg_push.connect("tcp://" + ip + ":" + std::to_string(PD_ZMQ_REG_PORT));
    reg_pull.connect("tcp://" + ip + ":" + std::to_string(PD_ZMQ_REG_RESP_PORT));
    fprintf(stderr, "D节点 ZMQ 初始化完成\n");
    return true;
  } catch (zmq::error_t& e) {
    fprintf(stderr, "D节点 ZMQ 初始化失败: %s\n", e.what());
    return false;
  }
}

bool DNode::register_and_init_nccl() {
  zmq::message_t reg_msg(0);
  reg_push.send(reg_msg, zmq::send_flags::none);

  zmq::pollitem_t items[] = {{static_cast<void*>(reg_pull), 0, ZMQ_POLLIN, 0}};
  int rc = zmq::poll(items, 1, std::chrono::milliseconds(10000));
  if (rc == 0) {
    fprintf(stderr, "D节点：等待 P节点注册响应超时\n");
    return false;
  }

  zmq::message_t msg;
  reg_pull.recv(msg, zmq::recv_flags::none);

  struct {
    int32_t d_node_id;
    ncclUniqueId uid;
  } resp;
  memcpy(&resp, msg.data(), sizeof(resp));
  d_node_id = resp.d_node_id;
  fprintf(stderr, "D节点：P节点分配 id=%d，开始 NCCL 初始化\n", d_node_id);

  cudaStreamCreate(&nccl.stream);
  nccl.rank = 1;
  nccl.n_ranks = 2;

  ncclUniqueId uid = resp.uid;
  auto fut = std::async(std::launch::async,
                        [this, uid]() mutable { return ncclCommInitRank(&nccl.comm, 2, uid, 1); });

  if (fut.wait_for(std::chrono::seconds(NCCL_INIT_TIMEOUT_S)) == std::future_status::timeout) {
    fprintf(stderr, "D节点[%d] NCCL 初始化超时\n", d_node_id);
    if (nccl.stream) {
      cudaStreamDestroy(nccl.stream);
      nccl.stream = nullptr;
    }
    nccl.comm = nullptr;
    return false;
  }

  ncclResult_t ret = fut.get();
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

bool DNode::send_heartbeat() {
  struct {
    MsgHeader header;
    int32_t d_node_id;
  } msg;
  msg.header.msg_type = MSG_HEARTBEAT;
  msg.d_node_id = d_node_id;

  // DEALER 发消息格式：[空帧][消息体]
  zmq::message_t empty(0);
  zmq::message_t body(sizeof(msg));
  memcpy(body.data(), &msg, sizeof(msg));

  dealer.send(empty, zmq::send_flags::sndmore);
  dealer.send(body, zmq::send_flags::none);
  return true;
}

bool DNode::send_shutdown() {
  struct {
    MsgHeader header;
    int32_t d_node_id;
  } msg;
  msg.header.msg_type = MSG_SHUTDOWN;
  msg.d_node_id = d_node_id;

  zmq::message_t empty(0);
  zmq::message_t body(sizeof(msg));
  memcpy(body.data(), &msg, sizeof(msg));

  dealer.send(empty, zmq::send_flags::sndmore);
  dealer.send(body, zmq::send_flags::none);
  return true;
}

bool DNode::send_request(const PrefillRequest& req, const std::vector<int>& token_ids) {
  int msg_size = sizeof(MsgHeader) + sizeof(PrefillRequest) + req.n_tokens * sizeof(int);

  zmq::message_t empty(0);
  zmq::message_t body(msg_size);
  char* data = (char*)body.data();

  MsgHeader header{MSG_PREFILL_REQUEST};
  memcpy(data, &header, sizeof(MsgHeader));
  memcpy(data + sizeof(MsgHeader), &req, sizeof(PrefillRequest));
  memcpy(data + sizeof(MsgHeader) + sizeof(PrefillRequest), token_ids.data(),
         req.n_tokens * sizeof(int));

  dealer.send(empty, zmq::send_flags::sndmore);
  dealer.send(body, zmq::send_flags::none);
  return true;
}

bool DNode::recv_response(PrefillResponse& resp) {
  // DEALER 收消息格式：[空帧][消息体]
  zmq::message_t empty;
  zmq::message_t body;

  dealer.recv(empty, zmq::recv_flags::none);
  dealer.recv(body, zmq::recv_flags::none);

  memcpy(&resp, body.data(), sizeof(PrefillResponse));
  return true;
}

// ============================================================================
// nccl_recv_kv
// ============================================================================

bool nccl_recv_kv(NcclComm& nccl, __half* k_cache, __half* v_cache, const std::vector<int>& slots,
                  int n_tokens, int n_layers, int kv_dim) {
  size_t total = (size_t)n_layers * 2 * n_tokens * kv_dim;
  nccl.ensure_buf(total * sizeof(__half));

  ncclRecv(nccl.d_buf, total, ncclFloat16, 0, nccl.comm, nccl.stream);
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
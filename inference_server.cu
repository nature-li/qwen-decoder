#include <drogon/drogon.h>
#include <json/json.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "gpu_decoder.h"
#include "pd_comm.h"

// ============================================================================
// 请求结构
// ============================================================================
struct InferRequest {
  int req_id;
  std::vector<int> token_ids;
  int max_new_tokens = 256;
  float temperature = 0.0f;
  int top_k = 30;
  bool stream = false;

  std::function<void(const std::string& token, bool finished)> on_token;
};

// ============================================================================
// 请求队列
// ============================================================================
struct RequestQueue {
  std::mutex mu;
  std::condition_variable cv;
  std::queue<std::shared_ptr<InferRequest>> q;
  bool closed = false;

  void push(std::shared_ptr<InferRequest> req) {
    std::lock_guard<std::mutex> lock(mu);
    q.push(std::move(req));
    cv.notify_one();
  }

  bool pop(std::shared_ptr<InferRequest>& req) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return !q.empty() || closed; });
    if (q.empty()) return false;
    req = std::move(q.front());
    q.pop();
    return true;
  }

  bool try_pop(std::shared_ptr<InferRequest>& req) {
    std::lock_guard<std::mutex> lock(mu);
    if (q.empty()) return false;
    req = std::move(q.front());
    q.pop();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu);
    closed = true;
    cv.notify_all();
  }
};

// ============================================================================
// 推理线程：支持单机和 P/D 分离两种模式
// ============================================================================
void infer_thread(GPUDecoder* decoder, BlockPool* pool, RequestQueue& req_queue, DNode* dnode) {
  const Config& cfg = decoder->get_config();
  int max_batch = decoder->get_max_batch();
  int max_pps = decoder->get_max_prefill_tokens_per_step();
  int max_blk = decoder->get_max_blocks_per_seq();
  int n_layers = cfg.n_layers;
  int kv_dim = cfg.n_kv_heads * (cfg.dim / cfg.n_heads);
  std::mt19937 rng(42);
  bool pd_mode = (dnode != nullptr);

  struct RunningRequest {
    std::shared_ptr<InferRequest> req;
    Request* r = nullptr;
    int prefill_offset = 0;
    bool prefill_done = false;
  };

  std::vector<RunningRequest> running(max_batch);
  std::vector<Request*> running_ptrs(max_batch, nullptr);
  int next_req_id = 0;

  // P/D 分离模式：启动 recv 线程接收 P节点的 response
  // key: req_id → RunningRequest slot
  std::unordered_map<int, int> req_id_to_slot;
  std::mutex slot_mu;

  // P/D 分离模式的 recv 线程
  std::thread pd_recv_th;
  if (pd_mode) {
    pd_recv_th = std::thread([&]() {
      while (true) {
        pd::PrefillResponse grpc_resp;
        if (!dnode->recv_response(grpc_resp)) {
          fprintf(stderr, "[infer] P节点连接断开\n");
          break;
        }

        int slot = -1;
        {
          std::lock_guard<std::mutex> lock(slot_mu);
          auto it = req_id_to_slot.find(grpc_resp.req_id());
          if (it != req_id_to_slot.end()) {
            slot = it->second;
            req_id_to_slot.erase(it);
          }
        }
        if (slot < 0) continue;

        auto& rr = running[slot];
        Request* r = rr.r;
        if (!r) continue;

        // 收 KV cache
        std::vector<int> slots_vec((int)rr.req->token_ids.size());
        for (int j = 0; j < (int)rr.req->token_ids.size(); j++) {
          slots_vec[j] =
              r->block_table.physical_idx(j) * BLOCK_SIZE + r->block_table.block_offset(j);
        }
        nccl_recv_kv(dnode->nccl, pool->get_k_cache(), pool->get_v_cache(), slots_vec,
                     rr.req->token_ids.size(), n_layers, kv_dim);

        // 设置 first_token，标记 prefill 完成
        r->cur_token = grpc_resp.first_token();
        r->pos = (int)rr.req->token_ids.size();
        rr.prefill_done = true;
      }
    });
  }

  while (true) {
    std::vector<int> changed;

    // 填充空槽
    for (int i = 0; i < max_batch; i++) {
      if (running[i].req != nullptr) continue;

      std::shared_ptr<InferRequest> req;
      bool got = req_queue.try_pop(req);

      if (!got) {
        bool any = false;
        for (int j = 0; j < max_batch; j++) {
          if (running[j].req) {
            any = true;
            break;
          }
        }
        if (!any) {
          if (!req_queue.pop(req)) goto done;
        } else {
          break;
        }
      }

      // 分配 block
      auto* r = new Request();
      r->id = next_req_id++;
      r->pos = 0;

      int need_blocks = ((int)req->token_ids.size() - 1) / BLOCK_SIZE + 1;
      bool oom = false;
      for (int b = 0; b < need_blocks; b++) {
        int block_id = pool->allocate();
        if (block_id < 0) {
          oom = true;
          break;
        }
        r->block_table.add_block(block_id);
      }

      if (oom) {
        delete r;
        req_queue.push(req);
        break;
      }

      running[i].req = req;
      running[i].r = r;
      running[i].prefill_offset = 0;
      running[i].prefill_done = false;
      running_ptrs[i] = r;
      changed.push_back(i);

      if (pd_mode) {
        // P/D 分离：发 prefill 请求给 P节点
        pd::PrefillRequest grpc_req;
        grpc_req.set_d_node_id(dnode->d_node_id);
        grpc_req.set_req_id(r->id);
        for (int tok : req->token_ids) grpc_req.add_token_ids(tok);
        dnode->send_request(grpc_req);

        // 记录 slot，等 recv 线程收到 response 后填入
        {
          std::lock_guard<std::mutex> lock(slot_mu);
          req_id_to_slot[r->id] = i;
        }
        // 标记 prefill 未完成，等 recv 线程设置
        running[i].prefill_done = false;
      }
    }

    // 构建 flat batch
    {
      std::vector<FlatRequest> flat_reqs;
      std::vector<int> flat_tokens, flat_positions, token_to_req, token_slot;
      std::vector<int> last_tok_idx;
      std::vector<int> prefill_flat;
      std::vector<int> dec_flat;
      int flat_offset = 0;
      int prefill_budget = max_pps;

      for (int i = 0; i < max_batch; i++) {
        auto& rr = running[i];
        if (!rr.req) continue;
        Request* r = rr.r;

        if (!rr.prefill_done) {
          if (pd_mode) {
            // P/D 分离：等 recv 线程设置 prefill_done，跳过
            continue;
          }

          // 单机模式：本地 prefill
          int& poff = rr.prefill_offset;
          int remaining = (int)rr.req->token_ids.size() - poff;
          int n_tok = std::min(remaining, prefill_budget);
          if (n_tok <= 0) continue;

          int old_blocks = r->block_table.num_blocks();
          int need = (r->pos + n_tok - 1) / BLOCK_SIZE + 1;
          while (r->block_table.num_blocks() < need) {
            int block_id = pool->allocate();
            if (block_id < 0) goto done;
            r->block_table.add_block(block_id);
          }
          if (r->block_table.num_blocks() != old_blocks) changed.push_back(i);

          FlatRequest fr;
          fr.req_idx = i;
          fr.flat_offset = flat_offset;
          fr.n_tokens = n_tok;
          fr.start_pos = r->pos;
          fr.is_prefill = true;
          flat_reqs.push_back(fr);

          for (int j = 0; j < n_tok; j++) {
            int tok = rr.req->token_ids[poff + j];
            int pos = r->pos + j;
            int phy = r->block_table.physical_idx(pos);
            int off = r->block_table.block_offset(pos);
            flat_tokens.push_back(tok);
            flat_positions.push_back(pos);
            token_to_req.push_back(i);
            token_slot.push_back(phy * BLOCK_SIZE + off);
            prefill_flat.push_back(flat_offset + j);
          }
          last_tok_idx.push_back(flat_offset + n_tok - 1);
          flat_offset += n_tok;
          prefill_budget -= n_tok;

        } else {
          // decode 阶段（单机和 P/D 分离共用）
          int need = r->pos / BLOCK_SIZE + 1;
          int old_blocks = r->block_table.num_blocks();
          while (r->block_table.num_blocks() < need) {
            int block_id = pool->allocate();
            if (block_id < 0) {
              r->finished = true;
              break;
            }
            r->block_table.add_block(block_id);
          }
          if (r->finished) continue;
          if (r->block_table.num_blocks() != old_blocks) changed.push_back(i);

          FlatRequest fr;
          fr.req_idx = i;
          fr.flat_offset = flat_offset;
          fr.n_tokens = 1;
          fr.start_pos = r->pos;
          fr.is_prefill = false;
          flat_reqs.push_back(fr);

          int phy = r->block_table.physical_idx(r->pos);
          int off = r->block_table.block_offset(r->pos);
          flat_tokens.push_back(r->cur_token);
          flat_positions.push_back(r->pos);
          token_to_req.push_back(i);
          token_slot.push_back(phy * BLOCK_SIZE + off);
          last_tok_idx.push_back(flat_offset);
          dec_flat.push_back(flat_offset);
          flat_offset++;
        }
      }

      if (flat_reqs.empty()) continue;

      std::sort(changed.begin(), changed.end());
      changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
      if (!changed.empty()) {
        for (int i = 0; i < max_batch; i++) running_ptrs[i] = running[i].r;
        decoder->update_block_table_partial(running_ptrs, changed, max_blk);
      }

      decoder->forward_flat(flat_reqs, flat_tokens, flat_positions, token_to_req, token_slot,
                            last_tok_idx, prefill_flat, dec_flat, flat_offset);

      // 处理结果
      for (int fi = 0; fi < (int)flat_reqs.size(); fi++) {
        auto& fr = flat_reqs[fi];
        int i = fr.req_idx;
        auto& rr = running[i];
        Request* r = rr.r;
        if (!r) continue;

        if (fr.is_prefill) {
          // 单机模式 prefill 完成
          r->pos += fr.n_tokens;
          rr.prefill_offset += fr.n_tokens;

          if (rr.prefill_offset >= (int)rr.req->token_ids.size()) {
            float* logits = decoder->get_logits_batch(fi);
            int first_token =
                sample_topk(logits, cfg.vocab_size, rr.req->top_k, rr.req->temperature, rng);
            r->cur_token = first_token;
            rr.prefill_done = true;
          }

        } else {
          // decode 结果（单机和 P/D 分离共用）
          float* logits = decoder->get_logits_batch(fi);
          int next_token =
              sample_topk(logits, cfg.vocab_size, rr.req->top_k, rr.req->temperature, rng);
          const char* piece = decode(decoder->get_tokenizer(), r->cur_token);
          std::string token_str = piece ? piece : "";

          r->pos++;
          r->cur_token = next_token;
          r->n_generated++;

          bool finished = (next_token == decoder->get_tokenizer().eos_token_id ||
                           decoder->get_tokenizer().vocab[next_token] == "<|im_end|>" ||
                           r->n_generated >= rr.req->max_new_tokens || r->pos >= cfg.seq_len);
          if (finished) r->finished = true;

          if (rr.req->on_token) rr.req->on_token(token_str, finished);
        }

        // 请求完成，释放资源
        if (r->finished) {
          r->block_table.free_blocks([pool](int id) { pool->free(id); });
          delete r;
          rr.req = nullptr;
          rr.r = nullptr;
          running_ptrs[i] = nullptr;
        }
      }
    }
  }

done:
  if (pd_mode && pd_recv_th.joinable()) {
    dnode->send_shutdown();
    pd_recv_th.join();
  }
  fprintf(stderr, "[infer] 线程退出\n");
}

// ============================================================================
// HTTP 处理：POST /v1/chat/completions
// ============================================================================
void handle_chat_completions(const drogon::HttpRequestPtr& req,
                             std::function<void(const drogon::HttpResponsePtr&)>&& callback,
                             GPUDecoder* decoder, RequestQueue& req_queue,
                             std::atomic<int>& req_id_counter) {
  Json::Value body;
  Json::Reader reader;
  std::string body_str(req->getBody());
  if (!reader.parse(body_str, body)) {
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(drogon::k400BadRequest);
    resp->setBody("{\"error\": \"invalid json\"}");
    callback(resp);
    return;
  }

  std::string prompt;
  if (body.isMember("messages") && body["messages"].isArray()) {
    for (auto& msg : body["messages"]) {
      if (msg["role"].asString() == "user") {
        prompt = msg["content"].asString();
        break;
      }
    }
  } else if (body.isMember("prompt")) {
    prompt = body["prompt"].asString();
  }

  if (prompt.empty()) {
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(drogon::k400BadRequest);
    resp->setBody("{\"error\": \"prompt is empty\"}");
    callback(resp);
    return;
  }

  int max_new_tokens = body.get("max_tokens", 256).asInt();
  float temperature = body.get("temperature", 0.0f).asFloat();
  int top_k = body.get("top_k", 30).asInt();
  bool stream = body.get("stream", false).asBool();

  std::string formatted = apply_chat_template(prompt);
  std::vector<int> token_ids;
  decoder->encode_pub(formatted, token_ids);
  token_ids.insert(token_ids.begin(), decoder->get_tokenizer().bos_token_id);

  int req_id = req_id_counter++;

  if (stream) {
    auto infer_req = std::make_shared<InferRequest>();
    infer_req->req_id = req_id;
    infer_req->token_ids = token_ids;
    infer_req->max_new_tokens = max_new_tokens;
    infer_req->temperature = temperature;
    infer_req->top_k = top_k;
    infer_req->stream = true;

    auto resp = drogon::HttpResponse::newAsyncStreamResponse(
        [infer_req, &req_queue](drogon::ResponseStreamPtr stream_ptr) {
          auto shared_stream = std::make_shared<drogon::ResponseStreamPtr>(std::move(stream_ptr));

          infer_req->on_token = [shared_stream, rid = infer_req->req_id](const std::string& token,
                                                                         bool finished) {
            Json::Value chunk;
            chunk["id"] = "chatcmpl-" + std::to_string(rid);
            chunk["object"] = "chat.completion.chunk";
            chunk["choices"][0]["delta"]["content"] = token;
            chunk["choices"][0]["finish_reason"] = finished ? Json::Value("stop") : Json::Value();

            Json::StreamWriterBuilder builder;
            builder["emitUTF8"] = true;
            std::string data = "data: " + Json::writeString(builder, chunk) + "\n";
            (*shared_stream)->send(data);

            if (finished) {
              (*shared_stream)->send("data: [DONE]\n\n");
              (*shared_stream)->close();
            }
          };

          req_queue.push(infer_req);
        });

    resp->addHeader("Content-Type", "text/event-stream");
    resp->addHeader("Cache-Control", "no-cache");
    resp->addHeader("Connection", "keep-alive");
    callback(resp);

  } else {
    auto full_output = std::make_shared<std::string>();
    auto cb =
        std::make_shared<std::function<void(const drogon::HttpResponsePtr&)>>(std::move(callback));

    auto infer_req = std::make_shared<InferRequest>();
    infer_req->req_id = req_id;
    infer_req->token_ids = token_ids;
    infer_req->max_new_tokens = max_new_tokens;
    infer_req->temperature = temperature;
    infer_req->top_k = top_k;
    infer_req->stream = false;

    infer_req->on_token = [full_output, cb, rid = req_id](const std::string& token,
                                                          bool finished) mutable {
      *full_output += token;
      if (finished) {
        Json::Value resp_json;
        resp_json["id"] = "chatcmpl-" + std::to_string(rid);
        resp_json["object"] = "chat.completion";
        resp_json["choices"][0]["message"]["role"] = "assistant";
        resp_json["choices"][0]["message"]["content"] = *full_output;
        resp_json["choices"][0]["finish_reason"] = "stop";

        Json::StreamWriterBuilder builder;
        builder["emitUTF8"] = true;
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k200OK);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(Json::writeString(builder, resp_json));
        (*cb)(resp);
      }
    };

    req_queue.push(infer_req);
  }
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model_file> [max_batch] [port] [--prefill-node <ip>]\n", argv[0]);
    return 1;
  }

  int max_batch = (argc >= 3) ? atoi(argv[2]) : 64;
  int port = (argc >= 4) ? atoi(argv[3]) : 8080;

  // 解析 --prefill-node 参数
  std::string prefill_ip;
  for (int i = 1; i < argc - 1; i++) {
    if (std::string(argv[i]) == "--prefill-node") {
      prefill_ip = argv[i + 1];
      break;
    }
  }

  bool pd_mode = !prefill_ip.empty();
  fprintf(stderr, "模式: %s\n", pd_mode ? "P/D 分离" : "单机");

  auto* decoder = new GPUDecoder(argv[1], max_batch, 4096, 4096, 4096);
  fprintf(stderr, "模型加载完成，max_batch=%d port=%d\n", max_batch, port);

  BlockPool* pool = decoder->get_block_pool();
  RequestQueue req_queue;
  std::atomic<int> req_id_counter{0};

  // P/D 分离模式：初始化 DNode
  DNode* dnode = nullptr;
  if (pd_mode) {
    dnode = new DNode();
    if (!dnode->init(prefill_ip)) return 1;
    if (!dnode->register_and_init_nccl()) return 1;
    if (!dnode->open_prefill_stream()) return 1;
    fprintf(stderr, "D节点[%d] 初始化完成，连接 P节点 %s\n", dnode->d_node_id, prefill_ip.c_str());
  }

  // 启动推理线程
  std::thread infer_th(infer_thread, decoder, pool, std::ref(req_queue), dnode);

  // 配置 Drogon
  drogon::app()
      .setLogLevel(trantor::Logger::kWarn)
      .addListener("0.0.0.0", port)
      .setThreadNum(4)
      .registerHandler("/v1/chat/completions",
                       [decoder, &req_queue, &req_id_counter](
                           const drogon::HttpRequestPtr& req,
                           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
                         handle_chat_completions(req, std::move(callback), decoder, req_queue,
                                                 req_id_counter);
                       },
                       {drogon::Post})
      .run();

  req_queue.close();
  infer_th.join();
  if (dnode) delete dnode;
  delete decoder;
  return 0;
}
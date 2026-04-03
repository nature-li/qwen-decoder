#include "common.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <climits>
#include <cstdio>
#include <cstring>
#include <limits>
#include <list>
#include <queue>

#include "gguf_loader.h"

int open_model(const std::string& path, ModelFile& mf) {
  mf.fd = open(path.c_str(), O_RDONLY);
  if (mf.fd < 0) {
    fprintf(stderr, "failed to open: %s\n", path.c_str());
    return -1;
  }
  mf.size = lseek(mf.fd, 0, SEEK_END);
  mf.data = mmap(nullptr, mf.size, PROT_READ, MAP_PRIVATE, mf.fd, 0);
  if (mf.data == MAP_FAILED) {
    fprintf(stderr, "mmap failed\n");
    close(mf.fd);
    return -1;
  }
  return 0;
}

void close_model(ModelFile& mf) {
  munmap(mf.data, mf.size);
  close(mf.fd);
}

// 从 GGUF 元数据里读 Config
int load_config(Config& config, const GGUFFile& gguf) {
  auto get_u32 = [&](const std::string& key) -> uint32_t {
    auto it = gguf.metadata.find(key);
    if (it == gguf.metadata.end()) {
      fprintf(stderr, "missing key: %s\n", key.c_str());
      return 0;
    }
    if (it->second.type != GGUF_TYPE_UINT32) {
      fprintf(stderr, "key %s type=%d, expected UINT32\n", key.c_str(), it->second.type);
      return 0;
    }
    return it->second.u32;
  };

  auto get_f32 = [&](const std::string& key) -> float {
    auto it = gguf.metadata.find(key);
    if (it == gguf.metadata.end()) {
      fprintf(stderr, "missing key: %s\n", key.c_str());
      return 0.0f;
    }
    if (it->second.type != GGUF_TYPE_FLOAT32) {
      fprintf(stderr, "key %s type=%d, expected FLOAT32\n", key.c_str(), it->second.type);
      return 0.0f;
    }
    return it->second.f32;
  };

  config.dim = get_u32("qwen2.embedding_length");
  config.hidden_dim = get_u32("qwen2.feed_forward_length");
  config.n_layers = get_u32("qwen2.block_count");
  config.n_heads = get_u32("qwen2.attention.head_count");
  config.n_kv_heads = get_u32("qwen2.attention.head_count_kv");
  config.seq_len = get_u32("qwen2.context_length");
  config.rope_freq_base = get_f32("qwen2.rope.freq_base");

  // 从 token_embd.weight 的 shape 读 vocab_size
  for (auto& t : gguf.tensors) {
    if (t.name == "token_embd.weight") {
      config.vocab_size = (int)t.shape[1];
      break;
    }
  }

  // printf("dim            = %d\n", config.dim);
  // printf("hidden_dim     = %d\n", config.hidden_dim);
  // printf("n_layers       = %d\n", config.n_layers);
  // printf("n_heads        = %d\n", config.n_heads);
  // printf("n_kv_heads     = %d\n", config.n_kv_heads);
  // printf("vocab_size     = %d\n", config.vocab_size);
  // printf("seq_len        = %d\n", config.seq_len);
  // printf("rope_freq_base = %.1f\n", config.rope_freq_base);

  return 0;
}

// 根据张量名找到数据指针
static void* find_tensor(const GGUFFile& gguf, const ModelFile& mf, const std::string& name) {
  for (auto& t : gguf.tensors) {
    if (t.name == name) {
      return (char*)mf.data + gguf.data_offset + t.offset;
    }
  }
  fprintf(stderr, "tensor not found: %s\n", name.c_str());
  return nullptr;
}

/**
 * 注意：当前仅支持 fp16 模型（文件名含 fp16 的 GGUF）
 * 权重类型硬编码如下：
 *   RMSNorm 权重 (rms_att, rms_ffn, rms_final): float32
 *   Attention/FFN 权重矩阵 (wq/wk/wv/wo/w1/w2/w3): float16 (uint16_t)
 *   Q/K/V bias (bq/bk/bv): float32
 *
 * 如需支持量化模型 (q4/q8)，应根据 TensorInfo.type 动态判断类型
 */
int load_weights(Weights& w, const Config& config, const GGUFFile& gguf, const ModelFile& mf) {
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;

  // Embedding 和输出层
  w.token_embedding = (uint16_t*)find_tensor(gguf, mf, "token_embd.weight");
  w.wcls = (uint16_t*)find_tensor(gguf, mf, "output.weight");
  w.rms_final = (float*)find_tensor(gguf, mf, "output_norm.weight");

  // 每层权重，分配数组
  w.rms_att = new float*[config.n_layers];
  w.wq = new uint16_t*[config.n_layers];
  w.wk = new uint16_t*[config.n_layers];
  w.wv = new uint16_t*[config.n_layers];
  w.wo = new uint16_t*[config.n_layers];
  w.bq = new float*[config.n_layers];
  w.bk = new float*[config.n_layers];
  w.bv = new float*[config.n_layers];
  w.rms_ffn = new float*[config.n_layers];
  w.w1 = new uint16_t*[config.n_layers];
  w.w2 = new uint16_t*[config.n_layers];
  w.w3 = new uint16_t*[config.n_layers];

  char name[128];
  for (int l = 0; l < config.n_layers; l++) {
    snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
    w.rms_att[l] = (float*)find_tensor(gguf, mf, name);

    snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
    w.wq[l] = (uint16_t*)find_tensor(gguf, mf, name);

    snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
    w.wk[l] = (uint16_t*)find_tensor(gguf, mf, name);

    snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
    w.wv[l] = (uint16_t*)find_tensor(gguf, mf, name);

    snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
    w.wo[l] = (uint16_t*)find_tensor(gguf, mf, name);

    snprintf(name, sizeof(name), "blk.%d.attn_q.bias", l);
    w.bq[l] = (float*)find_tensor(gguf, mf, name);

    snprintf(name, sizeof(name), "blk.%d.attn_k.bias", l);
    w.bk[l] = (float*)find_tensor(gguf, mf, name);

    snprintf(name, sizeof(name), "blk.%d.attn_v.bias", l);
    w.bv[l] = (float*)find_tensor(gguf, mf, name);

    snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
    w.rms_ffn[l] = (float*)find_tensor(gguf, mf, name);

    snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
    w.w1[l] = (uint16_t*)find_tensor(gguf, mf, name);

    snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
    w.w2[l] = (uint16_t*)find_tensor(gguf, mf, name);

    snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
    w.w3[l] = (uint16_t*)find_tensor(gguf, mf, name);
  }

  return 0;
}

void free_weights(Weights& w) {
  delete[] w.rms_att;
  delete[] w.wq;
  delete[] w.wk;
  delete[] w.wv;
  delete[] w.wo;
  delete[] w.bq;
  delete[] w.bk;
  delete[] w.bv;
  delete[] w.rms_ffn;
  delete[] w.w1;
  delete[] w.w2;
  delete[] w.w3;
}

/**
 * fp16 转 float
 *
 * 内存布局
 * fp16 (16位):
 * [15]    [14:10]    [9:0]
 * 符号位   指数(5位)   尾数(10位)
 *
 * fp32 (32位):
 * [31]    [30:23]    [22:0]
 * 符号位   指数(8位)   尾数(23位)
 */
float fp16_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exponent = (h >> 10) & 0x1f;
  uint32_t mantissa = h & 0x3ff;

  uint32_t f;
  if (exponent == 0) {
    // 非规格化数（很小的数接近0）
    f = (sign << 31) | (mantissa << 13);
  } else if (exponent == 31) {
    // inf 或 nan（特殊值）
    f = (sign << 31) | (0xff << 23) | (mantissa << 13);
  } else {
    // 正常数，指数需要转换偏置
    // fp16 偏置是 15，fp32 偏置是 127
    // 所以 exponent + 112 (= 127 - 15)
    f = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
  }

  float result;
  memcpy(&result, &f, sizeof(float));
  return result;
}

int load_tokenizer(Tokenizer& t, const GGUFFile& gguf) {
  // 读 bos/eos token id
  auto it_bos = gguf.metadata.find("tokenizer.ggml.bos_token_id");
  auto it_eos = gguf.metadata.find("tokenizer.ggml.eos_token_id");
  if (it_bos == gguf.metadata.end() || it_eos == gguf.metadata.end()) {
    fprintf(stderr, "missing bos/eos token id\n");
    return -1;
  }
  t.bos_token_id = (int)it_bos->second.u32;
  t.eos_token_id = (int)it_eos->second.u32;

  // 读 token 列表
  auto it_tokens = gguf.metadata.find("tokenizer.ggml.tokens");
  if (it_tokens == gguf.metadata.end()) {
    fprintf(stderr, "missing tokenizer.ggml.tokens\n");
    return -1;
  }
  const auto& arr = it_tokens->second.arr;
  t.vocab_size = (int)arr.size();
  t.vocab.resize(t.vocab_size);
  for (int i = 0; i < t.vocab_size; i++) {
    t.vocab[i] = arr[i].str;
  }

  auto it_merges = gguf.metadata.find("tokenizer.ggml.merges");
  if (it_merges != gguf.metadata.end()) {
    const auto& arr = it_merges->second.arr;
    t.merges.resize(arr.size());
    for (int i = 0; i < (int)arr.size(); i++) {
      t.merges[i] = arr[i].str;
    }
  }

  // 读 token 类型
  auto it_types = gguf.metadata.find("tokenizer.ggml.token_type");
  if (it_types != gguf.metadata.end()) {
    const auto& type_arr = it_types->second.arr;
    t.token_type.resize(type_arr.size());
    for (int i = 0; i < (int)type_arr.size(); i++) {
      t.token_type[i] = (int)type_arr[i].i32;
    }
  }

  // printf("vocab_size     = %d\n", t.vocab_size);

  // 预建 vocab_map
  for (int i = 0; i < t.vocab_size; i++) {
    t.vocab_map[t.vocab[i]] = i;
  }

  // 预建 merge_rank
  for (int i = 0; i < (int)t.merges.size(); i++) {
    t.merge_rank[t.merges[i]] = i;
  }
  return 0;
}

static std::string cp_to_utf8(int cp) {
  std::string s;
  if (cp < 0x80) {
    s += (char)cp;
  } else if (cp < 0x800) {
    s += (char)(0xC0 | (cp >> 6));
    s += (char)(0x80 | (cp & 0x3F));
  }
  return s;
}

static void build_maps(std::unordered_map<char, std::string>& b2u,
                       std::unordered_map<std::string, char>& u2b) {
  // 可打印字节直接映射，其余映射到 256+
  std::vector<int> bs;
  for (int i = 33; i <= 126; i++) bs.push_back(i);
  for (int i = 161; i <= 172; i++) bs.push_back(i);
  for (int i = 174; i <= 255; i++) bs.push_back(i);

  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; b++) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n++);
    }
  }

  for (int i = 0; i < (int)bs.size(); i++) {
    std::string utf8 = cp_to_utf8(cs[i]);
    b2u[(char)bs[i]] = utf8;
    u2b[utf8] = (char)bs[i];
  }
}

const std::unordered_map<char, std::string>& get_byte_to_unicode() {
  static std::unordered_map<char, std::string> b2u;
  static std::unordered_map<std::string, char> u2b;
  static bool initialized = false;
  if (!initialized) {
    build_maps(b2u, u2b);
    initialized = true;
  }
  return b2u;
}

const std::unordered_map<std::string, char>& get_unicode_to_byte() {
  static std::unordered_map<char, std::string> b2u;
  static std::unordered_map<std::string, char> u2b;
  static bool initialized = false;
  if (!initialized) {
    build_maps(b2u, u2b);
    initialized = true;
  }
  return u2b;
}

const char* decode(Tokenizer& t, int token) {
  if (t.token_type[token] == 3) return "";

  static char byte_piece[2];
  unsigned char byte_val;
  if (sscanf(t.vocab[token].c_str(), "<0x%02hhX>", &byte_val) == 1) {
    byte_piece[0] = (char)byte_val;
    byte_piece[1] = '\0';
    return byte_piece;
  }

  // 把 GPT2 unicode 字符转回原始字节
  const auto& u2b = get_unicode_to_byte();
  static std::string result;
  result.clear();

  const std::string& piece = t.vocab[token];
  int i = 0;
  while (i < (int)piece.size()) {
    // 判断 UTF-8 字符长度
    unsigned char c = (unsigned char)piece[i];
    int char_len = 1;
    if ((c & 0x80) == 0x00)
      char_len = 1;
    else if ((c & 0xE0) == 0xC0)
      char_len = 2;
    else if ((c & 0xF0) == 0xE0)
      char_len = 3;
    else if ((c & 0xF8) == 0xF0)
      char_len = 4;

    std::string ch = piece.substr(i, char_len);
    auto it = u2b.find(ch);
    if (it != u2b.end()) {
      result += it->second;  // 转回原始字节
    } else {
      result += ch;  // 找不到就原样输出
    }
    i += char_len;
  }

  return result.c_str();
}

// 在词表里查找字符串，返回 token id，找不到返回 -1
int vocab_lookup(const Tokenizer& t, const std::string& str) {
  for (int i = 0; i < t.vocab_size; i++) {
    if (t.vocab[i] == str) return i;
  }
  return -1;
}

int encode(Tokenizer& t, const std::string& text, std::vector<int>& tokens) {
  tokens.clear();

  // 1. 收集特殊 token，按长度从长到短排序（只做一次，但这里每次都排，可以预建）
  std::vector<std::pair<std::string, int>> special_tokens;
  for (int i = 0; i < t.vocab_size; i++) {
    if (t.token_type[i] == 3) {
      special_tokens.push_back({t.vocab[i], i});
    }
  }
  std::sort(special_tokens.begin(), special_tokens.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

  // 2. 把文本按特殊 token 分割成多段
  std::vector<std::string> segments;
  std::vector<bool> is_special;

  std::string remaining = text;
  while (!remaining.empty()) {
    bool found = false;
    for (auto& [sp, id] : special_tokens) {
      if (remaining.size() >= sp.size() && remaining.substr(0, sp.size()) == sp) {
        segments.push_back(sp);
        is_special.push_back(true);
        remaining = remaining.substr(sp.size());
        found = true;
        break;
      }
    }
    if (!found) {
      size_t next_special = remaining.size();
      for (auto& [sp, id] : special_tokens) {
        size_t pos = remaining.find(sp);
        if (pos != std::string::npos && pos < next_special) {
          next_special = pos;
        }
      }
      segments.push_back(remaining.substr(0, next_special));
      is_special.push_back(false);
      remaining = remaining.substr(next_special);
    }
  }

  // 3. 对每段分别处理
  const auto& b2u = get_byte_to_unicode();

  for (int si = 0; si < (int)segments.size(); si++) {
    if (is_special[si]) {
      // 特殊 token 直接查 vocab_map
      auto it = t.vocab_map.find(segments[si]);
      if (it != t.vocab_map.end()) tokens.push_back(it->second);
      continue;
    }

    // 普通文本：bytes → unicode → UTF-8 字符初始化 token ids
    std::string processed;
    for (unsigned char c : segments[si]) {
      auto it = b2u.find((char)c);
      if (it != b2u.end()) processed += it->second;
    }

    // 按 UTF-8 字符初始化 ids
    std::vector<int> ids;
    int i = 0;
    while (i < (int)processed.size()) {
      unsigned char c = (unsigned char)processed[i];
      int char_len = 1;
      if ((c & 0x80) == 0x00)
        char_len = 1;
      else if ((c & 0xE0) == 0xC0)
        char_len = 2;
      else if ((c & 0xF0) == 0xE0)
        char_len = 3;
      else if ((c & 0xF8) == 0xF0)
        char_len = 4;

      std::string ch = processed.substr(i, char_len);
      auto it = t.vocab_map.find(ch);
      if (it != t.vocab_map.end()) {
        ids.push_back(it->second);
      } else {
        for (int b = 0; b < char_len; b++) {
          unsigned char byte = (unsigned char)processed[i + b];
          char buf[8];
          snprintf(buf, sizeof(buf), "<0x%02X>", byte);
          auto bit = t.vocab_map.find(buf);
          if (bit != t.vocab_map.end()) ids.push_back(bit->second);
        }
      }
      i += char_len;
    }

    if (ids.empty()) continue;

    // BPE merge：优先队列 + 链表，O(n logn)
    int n = (int)ids.size();
    std::vector<int> nxt(n), prv(n);
    std::vector<bool> deleted(n, false);
    for (int j = 0; j < n; j++) {
      nxt[j] = j + 1;
      prv[j] = j - 1;
    }

    // MergeItem：(rank, pos, left_id, right_id)
    // lazy deletion：取出时检查 ids[pos]==left && ids[nxt[pos]]==right
    struct MergeItem {
      int rank, pos, left, right;
      bool operator>(const MergeItem& o) const { return rank > o.rank; }
    };
    std::priority_queue<MergeItem, std::vector<MergeItem>, std::greater<MergeItem>> pq;

    // 初始化所有相邻 pair
    auto try_push = [&](int pos) {
      if (pos < 0 || pos >= n || deleted[pos]) return;
      int r = nxt[pos];
      if (r >= n || deleted[r]) return;
      std::string pair = t.vocab[ids[pos]] + " " + t.vocab[ids[r]];
      auto it = t.merge_rank.find(pair);
      if (it != t.merge_rank.end()) {
        pq.push({it->second, pos, ids[pos], ids[r]});
      }
    };

    for (int j = 0; j < n - 1; j++) try_push(j);

    // BPE merge 主循环
    while (!pq.empty()) {
      auto [rank, pos, left, right] = pq.top();
      pq.pop();

      // lazy deletion 检查
      if (deleted[pos]) continue;
      int r = nxt[pos];
      if (r >= n || deleted[r]) continue;
      if (ids[pos] != left || ids[r] != right) continue;

      // 执行 merge：把 left+right 合并到 pos，删除 r
      std::string merged = t.vocab[left] + t.vocab[right];
      auto mit = t.vocab_map.find(merged);
      if (mit == t.vocab_map.end()) continue;
      ids[pos] = mit->second;

      // 删除 r，更新链表
      deleted[r] = true;
      int rr = nxt[r];
      nxt[pos] = rr;
      if (rr < n) prv[rr] = pos;

      // 检查左边新 pair
      try_push(prv[pos]);
      // 检查右边新 pair（pos 已经更新为 merged）
      try_push(pos);
    }

    // 收集结果
    for (int j = 0; j < n; j++) {
      if (!deleted[j]) tokens.push_back(ids[j]);
    }
  }

  return 0;
}

std::string apply_chat_template(const std::string& user_input) {
  return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
         "<|im_start|>user\n" +
         user_input +
         "<|im_end|>\n"
         "<|im_start|>assistant\n";
}

int argmax(const float* logits, int size) {
  int max_idx = 0;
  float max_val = logits[0];
  for (int i = 1; i < size; i++) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      max_idx = i;
    }
  }
  return max_idx;
}

int sample_topk(const float* logits, int size, int k, float temperature, std::mt19937& rng) {
  if (temperature == 0.0f) {
    return argmax(logits, size);
  }

  if (k <= 0 || k >= size) {
    // 退化成普通 temperature 采样
    std::vector<float> probs(size);
    float max_val = *std::max_element(logits, logits + size);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
      probs[i] = expf((logits[i] - max_val) / temperature);
      sum += probs[i];
    }
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float r = dis(rng);
    float cur = 0.0f;
    for (int i = 0; i < size; i++) {
      cur += probs[i] / sum;
      if (r <= cur) return i;
    }
    return size - 1;
  }

  // 1. 找第 k 大的阈值
  std::vector<float> tmp(logits, logits + size);
  std::nth_element(tmp.begin(), tmp.begin() + k - 1, tmp.end(), std::greater<float>());
  float threshold = tmp[k - 1];

  // 2. 只保留 top-k，其余归零
  std::vector<float> probs(size);
  float max_val = logits[0];
  for (int i = 1; i < size; i++) max_val = fmaxf(max_val, logits[i]);

  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    if (logits[i] >= threshold) {
      probs[i] = expf((logits[i] - max_val) / temperature);
    } else {
      probs[i] = 0.0f;
    }
    sum += probs[i];
  }

  // 3. 轮盘赌采样
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  float r = dis(rng);
  float cur = 0.0f;
  for (int i = 0; i < size; i++) {
    cur += probs[i] / sum;
    if (r <= cur) return i;
  }
  return size - 1;
}
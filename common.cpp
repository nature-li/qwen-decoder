#include "common.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

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
      fprintf(stderr, "key %s type=%d, expected UINT32\n", key.c_str(),
              it->second.type);
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
      fprintf(stderr, "key %s type=%d, expected FLOAT32\n", key.c_str(),
              it->second.type);
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

  printf("dim            = %d\n", config.dim);
  printf("hidden_dim     = %d\n", config.hidden_dim);
  printf("n_layers       = %d\n", config.n_layers);
  printf("n_heads        = %d\n", config.n_heads);
  printf("n_kv_heads     = %d\n", config.n_kv_heads);
  printf("vocab_size     = %d\n", config.vocab_size);
  printf("seq_len        = %d\n", config.seq_len);
  printf("rope_freq_base = %.1f\n", config.rope_freq_base);

  return 0;
}

// 根据张量名找到数据指针
static void* find_tensor(const GGUFFile& gguf, const ModelFile& mf,
                         const std::string& name) {
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
int load_weights(Weights& w, const Config& config, const GGUFFile& gguf,
                 const ModelFile& mf) {
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
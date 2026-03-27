#pragma once
#include "decoder.h"
#include "gguf_loader.h"

struct RunState {
  float* x;        // [dim] 当前 token 的隐藏状态
  float* xb;       // [dim] RMSNorm 输出缓冲区
  float* xb2;      // [dim] attention/FFN 输出投影缓冲区
  float* q;        // [dim] Query
  float* k;        // [kv_dim] Key
  float* v;        // [kv_dim] Value
  float* att;      // [n_heads, seq_len] attention scores
  float* hb;       // [hidden_dim] FFN 中间结果 SiLU(w1(x))
  float* hb2;      // [hidden_dim] FFN 中间结果 w3(x)
  float* logits;   // [vocab_size] 输出 logits
  float* k_cache;  // [n_layers, seq_len, kv_dim] Key Cache
  float* v_cache;  // [n_layers, seq_len, kv_dim] Value Cache
};

class CPUDecoder : public Decoder {
 public:
  CPUDecoder(const std::string& model_file);
  ~CPUDecoder();

  void forward(int token, int pos) override;
  float* get_logits() override { return state.logits; }

 private:
  ModelFile mf;
  GGUFFile gguf;
  Weights w;
  RunState state;
};
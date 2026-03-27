#include <cstdio>

#include "common.h"
#include "gguf_loader.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model_file>\n", argv[0]);
    return 1;
  }

  // 1. 加载 GGUF
  GGUFFile gguf;
  if (load_gguf(gguf, argv[1]) != 0) return 1;

  // 2. 加载 Config
  Config config;
  if (load_config(config, gguf) != 0) return 1;

  // 3. mmap 文件
  ModelFile mf;
  if (open_model(argv[1], mf) != 0) return 1;

  // 4. 加载权重
  Weights w;
  if (load_weights(w, config, gguf, mf) != 0) return 1;

  // 5. 验证几个权重值
  // token_embedding[0] 的前几个元素
  printf("\ntoken_embedding[0][0] = %f\n", fp16_to_float(w.token_embedding[0]));
  printf("token_embedding[0][1] = %f\n", fp16_to_float(w.token_embedding[1]));

  // rms_final 前几个元素（fp32 直接读）
  printf("rms_final[0] = %f\n", w.rms_final[0]);
  printf("rms_final[1] = %f\n", w.rms_final[1]);

  // 第0层 rms_att
  printf("rms_att[0][0] = %f\n", w.rms_att[0][0]);

  // 第0层 wq 第一个元素
  printf("wq[0][0] = %f\n", fp16_to_float(w.wq[0][0]));

  free_weights(w);
  close_model(mf);
  return 0;
}
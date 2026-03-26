#include <cstdio>

#include "gguf_loader.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
    return 1;
  }

  GGUFFile gguf;
  if (load_gguf(gguf, argv[1]) != 0) return 1;

  print_gguf_info(gguf);
  return 0;
}
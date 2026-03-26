#include "gguf_loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// 读取工具函数
// ============================================================================

static void read_bytes(FILE* f, void* dst, size_t size) {
  if (fread(dst, size, 1, f) != 1) {
    fprintf(stderr, "failed to read %zu bytes\n", size);
    exit(1);
  }
}

template <typename T>
static T read_val(FILE* f) {
  T val;
  read_bytes(f, &val, sizeof(T));
  return val;
}

static std::string read_string(FILE* f) {
  uint64_t len = read_val<uint64_t>(f);
  std::string str(len, '\0');
  read_bytes(f, str.data(), len);
  return str;
}

// ============================================================================
// 读取 GGUFValue
// ============================================================================

static GGUFValue read_value(FILE* f, GGUFValueType type);

static GGUFValue read_value(FILE* f, GGUFValueType type) {
  GGUFValue val;
  val.type = type;

  switch (type) {
    case GGUF_TYPE_UINT8:
      val.u8 = read_val<uint8_t>(f);
      break;
    case GGUF_TYPE_INT8:
      val.i8 = read_val<int8_t>(f);
      break;
    case GGUF_TYPE_UINT16:
      val.u16 = read_val<uint16_t>(f);
      break;
    case GGUF_TYPE_INT16:
      val.i16 = read_val<int16_t>(f);
      break;
    case GGUF_TYPE_UINT32:
      val.u32 = read_val<uint32_t>(f);
      break;
    case GGUF_TYPE_INT32:
      val.i32 = read_val<int32_t>(f);
      break;
    case GGUF_TYPE_FLOAT32:
      val.f32 = read_val<float>(f);
      break;
    case GGUF_TYPE_BOOL:
      val.bool_ = read_val<bool>(f);
      break;
    case GGUF_TYPE_UINT64:
      val.u64 = read_val<uint64_t>(f);
      break;
    case GGUF_TYPE_INT64:
      val.i64 = read_val<int64_t>(f);
      break;
    case GGUF_TYPE_FLOAT64:
      val.f64 = read_val<double>(f);
      break;
    case GGUF_TYPE_STRING:
      val.str = read_string(f);
      break;
    case GGUF_TYPE_ARRAY: {
      GGUFValueType elem_type = (GGUFValueType)read_val<uint32_t>(f);
      uint64_t count = read_val<uint64_t>(f);
      val.arr.resize(count);
      for (uint64_t i = 0; i < count; i++) {
        val.arr[i] = read_value(f, elem_type);
      }
      break;
    }
    default:
      fprintf(stderr, "unknown value type: %d\n", type);
      exit(1);
  }

  return val;
}

// ============================================================================
// load_gguf
// ============================================================================

int load_gguf(GGUFFile& gguf, const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "failed to open: %s\n", path.c_str());
    return -1;
  }

  // 1. 读文件头
  read_bytes(f, &gguf.header, sizeof(GGUFHeader));

  // 验证 magic number "GGUF"
  if (gguf.header.magic != 0x46554747) {
    fprintf(stderr, "invalid GGUF magic: 0x%08x\n", gguf.header.magic);
    fclose(f);
    return -1;
  }

  printf("GGUF version: %d\n", gguf.header.version);
  printf("n_tensors:    %lu\n", gguf.header.n_tensors);
  printf("n_kv:         %lu\n", gguf.header.n_kv);

  // 2. 读元数据 key-value
  for (uint64_t i = 0; i < gguf.header.n_kv; i++) {
    std::string key = read_string(f);
    GGUFValueType type = (GGUFValueType)read_val<uint32_t>(f);
    GGUFValue val = read_value(f, type);
    gguf.metadata[key] = val;
  }

  // 3. 读张量描述
  gguf.tensors.resize(gguf.header.n_tensors);
  for (uint64_t i = 0; i < gguf.header.n_tensors; i++) {
    TensorInfo& t = gguf.tensors[i];
    t.name = read_string(f);
    t.n_dims = read_val<uint32_t>(f);
    t.shape.resize(t.n_dims);
    for (uint32_t d = 0; d < t.n_dims; d++) {
      t.shape[d] = read_val<uint64_t>(f);
    }
    t.type = (GGMLType)read_val<uint32_t>(f);
    t.offset = read_val<uint64_t>(f);
  }

  // 4. 记录数据起始位置（对齐到 32 字节）
  size_t cur = ftell(f);
  gguf.data_offset = (cur + 31) & ~31;

  fclose(f);
  return 0;
}

// ============================================================================
// print_gguf_info
// ============================================================================

void print_gguf_info(const GGUFFile& gguf) {
  printf("\n=== Metadata ===\n");
  for (auto& [key, val] : gguf.metadata) {
    printf("  %s = ", key.c_str());
    switch (val.type) {
      case GGUF_TYPE_UINT32:
        printf("%u\n", val.u32);
        break;
      case GGUF_TYPE_INT32:
        printf("%d\n", val.i32);
        break;
      case GGUF_TYPE_UINT64:
        printf("%lu\n", val.u64);
        break;
      case GGUF_TYPE_FLOAT32:
        printf("%f\n", val.f32);
        break;
      case GGUF_TYPE_STRING:
        printf("%s\n", val.str.c_str());
        break;
      case GGUF_TYPE_BOOL:
        printf("%s\n", val.bool_ ? "true" : "false");
        break;
      case GGUF_TYPE_ARRAY:
        printf("[array, size=%zu]\n", val.arr.size());
        break;
      default:
        printf("...\n");
        break;
    }
  }

  printf("\n=== Tensors ===\n");
  for (auto& t : gguf.tensors) {
    printf("  %-50s [", t.name.c_str());
    for (uint32_t d = 0; d < t.n_dims; d++) {
      printf("%lu%s", t.shape[d], d + 1 < t.n_dims ? ", " : "");
    }
    printf("] type=%d offset=%lu\n", t.type, t.offset);
  }
}
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// GGUF 文件头
struct GGUFHeader {
  uint32_t magic;      // 0x46554747 = "GGUF"
  uint32_t version;    // 目前是 3
  uint64_t n_tensors;  // 张量数量
  uint64_t n_kv;       // 元数据 key-value 数量
};

// 元数据值类型
enum GGUFValueType : uint32_t {
  GGUF_TYPE_UINT8 = 0,
  GGUF_TYPE_INT8 = 1,
  GGUF_TYPE_UINT16 = 2,
  GGUF_TYPE_INT16 = 3,
  GGUF_TYPE_UINT32 = 4,
  GGUF_TYPE_INT32 = 5,
  GGUF_TYPE_FLOAT32 = 6,
  GGUF_TYPE_BOOL = 7,
  GGUF_TYPE_STRING = 8,
  GGUF_TYPE_ARRAY = 9,
  GGUF_TYPE_UINT64 = 10,
  GGUF_TYPE_INT64 = 11,
  GGUF_TYPE_FLOAT64 = 12,
};

// 张量数据类型
enum GGMLType : uint32_t {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q8_0 = 8,
};

// 张量描述
struct TensorInfo {
  std::string name;
  uint32_t n_dims;
  std::vector<uint64_t> shape;  // 维度
  GGMLType type;
  uint64_t offset;  // 数据在文件中的偏移
};

// 元数据值
struct GGUFValue {
  GGUFValueType type;
  union {
    uint8_t u8;
    int8_t i8;
    uint16_t u16;
    int16_t i16;
    uint32_t u32;
    int32_t i32;
    float f32;
    bool bool_;
    uint64_t u64;
    int64_t i64;
    double f64;
  };
  std::string str;             // type == STRING
  std::vector<GGUFValue> arr;  // type == ARRAY
};

struct GGUFFile {
  GGUFHeader header;
  std::unordered_map<std::string, GGUFValue> metadata;
  std::vector<TensorInfo> tensors;
  size_t data_offset;  // 权重数据起始位置
};

int load_gguf(GGUFFile& gguf, const std::string& path);
void print_gguf_info(const GGUFFile& gguf);
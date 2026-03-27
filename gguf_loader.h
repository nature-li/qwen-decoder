#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * GGUF 文件开头固定的 20 字节，描述文件的基本信息
 */
struct GGUFHeader {
  uint32_t magic;      // 对应 ASCII "GGUF"，用来验证文件格式
  uint32_t version;    // 格式版本号，目前最新是 3
  uint64_t n_tensors;  // 文件里有多少个权重张量
  uint64_t n_kv;       // 文件里有多少个元数据键值对
};

/**
 * GGUF 文件里的元数据是 key-value 格式
 * 读文件时先读类型，再按类型读值
 */
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

/**
 * 张量里每个元素用多少位存储
 */
enum GGMLType : uint32_t {
  GGML_TYPE_F32 = 0,    // 每个元素 32 位浮点，标准 float
  GGML_TYPE_F16 = 1,    // 每个元素 16 位浮点，半精度
  GGML_TYPE_Q4_0 = 2,   // 每个元素 4 位整数量化
  GGML_TYPE_Q4_K = 12,  // 每个元素 4 位，K-quant 格式（更精确的量化）
  GGML_TYPE_Q8_0 = 8,   // 每个元素 8 位整数量化
};

/**
 * 每个张量的元信息，描述这个张量是什么、在哪里
 */
struct TensorInfo {
  std::string name;             // 张量名字
  uint32_t n_dims;              // 维度数量
  std::vector<uint64_t> shape;  //  每维大小
  GGMLType type;                // 数据类型
  uint64_t offset;              // 数据偏移
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

/**
 * 整个 GGUF 文件解析后的结果，把文件内容组织成方便访问的结构
 */
struct GGUFFile {
  // 文件头，版本号、张量数量、kv 数量
  GGUFHeader header;
  // 所有元数据键值对，用 map 存方便按名字查找，如
  // metadata["qwen2.block_count"].u32 = 24
  // metadata["qwen2.rope.freq_base"].f32 = 1000000.0
  std::unordered_map<std::string, GGUFValue> metadata;
  // 所有张量的描述信息（名字、shape、类型、偏移）
  // tensors[0].name = "output.weight"
  // tensors[0].shape = [896, 151936]
  std::vector<TensorInfo> tensors;
  // 权重数据区的起始位置（字节偏移
  // 实际数据 = mf.data + data_offset + tensor.offset
  size_t data_offset;
};

int load_gguf(GGUFFile& gguf, const std::string& path);
void print_gguf_info(const GGUFFile& gguf);
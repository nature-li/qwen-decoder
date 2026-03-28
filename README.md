# qwen-decoder

从零手写的 Qwen2.5 推理引擎，支持加载 GGUF 格式模型，提供 CPU C++、CUDA GPU 多个版本实现，记录了完整的性能优化过程。

## 项目结构

| 文件/目录 | 说明 |
| :--- | :--- |
| `common.h` | 公共数据结构（Config、Weights、Tokenizer、ModelFile） |
| `common.cpp` | 公共函数（GGUF 加载、BPE tokenizer、采样） |
| `gguf_loader.h/cpp` | GGUF 文件格式解析 |
| `decoder.h/cpp` | Decoder 基类，generate 逻辑 |
| `cpu_decoder.h/cpp` | CPU 版本实现 |
| `gpu_v1/` | GPU 朴素版本，手写 fp16 matmul kernel |
| `gpu_v2/` | GPU v2，cublasHgemm 替换手写 kernel |
| `gpu_v3/` | GPU v3，全程 fp16 中间状态 |
| `gpu_v4/` | GPU v4，Flash Attention（decode 阶段） |
| `tests/` | softmax / attention 算法对比实现 |
| `doc/` | nsys profile 截图 |
| `CMakeLists.txt` | 构建配置 |

## 实现的核心模块

**模型加载：**
- GGUF 文件格式解析（header、metadata key-value、tensor 描述）
- mmap 零拷贝加载权重
- fp16 权重直接上传 GPU

**Tokenizer：**
- GPT2 BPE tokenizer（encode/decode）
- bytes_to_unicode 完整映射（256 字节全覆盖）
- 特殊 token 识别（`<|im_start|>`、`<|im_end|>` 等）
- Chat template 格式化

**Forward Pass：**
- Embedding lookup（fp16）
- RMSNorm（warp reduce 归约）
- QKV 投影（cublasHgemm fp16）
- Q/K/V bias
- RoPE 位置编码（动态计算，rope_freq_base=1000000）
- KV Cache（fp16）
- Multi-Head Attention（GQA，n_heads=16，n_kv_heads=2）
- SwiGLU FFN
- Temperature + Top-K 采样

## 与 llama2c-decoder 的主要差异

| 特性 | llama2c-decoder | qwen-decoder |
| :--- | :--- | :--- |
| 文件格式 | llama2.c 自定义格式 | GGUF 标准格式 |
| 权重精度 | float32 | fp16 |
| Tokenizer | BPE（简单） | GPT2 BPE（bytes_to_unicode） |
| Q/K/V bias | 无 | 有 |
| RoPE theta | 10000 | 1000000 |
| GQA | 可选 | 必须（n_kv_heads=2） |
| 对话格式 | 无 | Chat template |

## 环境依赖

- Linux
- NVIDIA GPU + CUDA 12.x（需要至少 10GB 显存）
- CMake 3.18+

## 编译

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

## 运行

下载模型：

```bash
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF',
    allow_patterns=['*fp16*'],
    local_dir='.'
)
"
```

3B fp16 模型分为两个分片，需要先用 llama.cpp 工具合并：

```bash
/path/to/llama.cpp/build/bin/llama-gguf-split \
    --merge \
    qwen2.5-3b-instruct-fp16-00001-of-00002.gguf \
    qwen2.5-3b-instruct-fp16.gguf
```

运行：

```bash
# CPU 版本
./cpu_decoder qwen2.5-3b-instruct-fp16.gguf

# GPU v3（推荐）
./gpu_decoder_v3 qwen2.5-3b-instruct-fp16.gguf
```

## 性能对比（Qwen2.5-3B，RTX 5060 Ti 16GB）

| 版本 | tokens/s | 显存 | 说明 |
| :--- | :--- | :--- | :--- |
| CPU C++ | ~2 | - | fp16 逐元素转换，朴素 matmul |
| GPU v1 | ~6 | ~8.4GB | 手写 fp16 matmul kernel |
| GPU v2 | ~34 | ~8.4GB | cublasHgemm，中间状态仍是 fp32 |
| GPU v3 | ~45 | ~8.4GB | 全程 fp16，消除类型转换开销 |
| GPU v4 | ~45 | ~8.4GB | Flash Attention（decode 阶段） |

## 优化分析

### GPU v1 → v2（5x 提升）

用 `cublasHgemm` 替换手写 `matmul_fp16_kernel`。

v1 手写 kernel 的 grid 只有几个 block，SM 大量闲置，SM 利用率不足 15%。cuBLAS 底层使用 cutlass，自动选择最优调度，SM 利用率大幅提升。

**v1 nsys profile：**

![nsys v1](doc/nsys-v1.png)

```
matmul_fp16_kernel: 99%  ← 几乎所有时间在 matmul
attention_kernel:    0.4%
rmsnorm_kernel:      0.3%
```

---

### GPU v2 → v3（30% 提升）

把 `GPURunState` 所有中间状态从 `float*` 改成 `__half*`，消除了每次 matmul 前后的两次类型转换 kernel，显存占用同时减半。

**v2 nsys profile：**

![nsys v2](doc/nsys-v2.png)

```
Memory 占比: 84.4%  ← 大量时间花在 HtoD memcpy（fp32↔fp16 转换）
fp16_to_fp32_kernel: 1.5%
fp32_to_fp16_kernel: 1.3%
```

**v3 nsys profile：**

![nsys v3](doc/nsys-v3.png)

```
Memory 占比: 38.3%  ← 内存传输大幅减少
fp16/fp32 转换 kernel: 0%  ← 完全消除
cuBLAS (cutlass): 93.8%
```

只有以下计算保留 fp32：
- RMSNorm 的平方和累加（防止溢出，dim=2048 个值累加容易超出 fp16 范围）
- Attention softmax（精度敏感）
- 最终 logits 输出（采样需要 fp32）

### Attention shared memory 限制

标准 attention kernel 用 shared memory 存 attention scores，大小为 `seq_len * sizeof(float)`：

```
3B: seq_len=32768 → shared_mem=128KB  ← 超出 GPU per-block 限制（通常 48-96KB）
```

kernel 启动失败时静默返回，导致 attention 输出错误但无报错，输出全是乱码。通过加 `CHECK_KERNEL()` 错误检查宏定位：

```cpp
#define CHECK_KERNEL() \
  { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "Kernel error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err)); \
      exit(1); \
    } \
  }
```

v2/v3 改为用全局内存（`gs.att`，预分配 `n_heads * seq_len`）存 attention scores，支持任意 seq_len，无 shared memory 限制。

---

### GPU v3 → v4（Flash Attention）

在 decode 阶段实现 Flash Attention，用 online softmax 分块处理 K、V，scores 不写 HBM：

```
standard attention:
  scores[seq_len] 写到全局内存 → softmax 读回 → weighted sum 读回
  HBM 读写: O(seq_len)

flash attention（decode）:
  分块处理 K、V，scores 留在 shared memory
  shared memory 大小: O(block_size + head_dim)，不随 seq_len 增长
  HBM 读写: O(seq_len)（K、V 各读一遍）
```

**v4 nsys profile：**

![nsys v4](doc/nsys-v4.png)

```
Kernels: 49.0%，Memory: 51.0%
attention_kernel: 1.8%  ← v3 的 2.6% 降到 1.8%
```

**decode 阶段 flash attention 提升有限的原因：**

decode 阶段 q 只有 1 行，scores 大小为 `1 × seq_len`，本身不大。flash attention 的分块反而导致 K、V 被多次读取，Memory 占比从 38.3% 升到 51.0%。

flash attention 真正有价值的是 **prefill 阶段**：

```
prefill: q 有 seq_len 行
scores 大小 = seq_len × seq_len × 4B
           = 32768 × 32768 × 4 = 4GB
```

prefill 阶段 flash attention 能把显存从 O(seq_len²) 降到 O(seq_len)，差距极大，这也是生产推理框架中 flash attention 的主要价值。

## tests/

`tests/` 目录包含算法验证代码，从基础到完整逐步实现：

```
tests/softmax/    softmax → online softmax，numpy / torch / C++ / CUDA 四种实现对比
tests/attention/  standard attention → flash attention，numpy / C++ / CUDA 三种实现对比
```

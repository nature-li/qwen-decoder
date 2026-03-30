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
| `gpu_v4/` | GPU v4，Flash Attention decode 阶段 |
| `gpu_v5/` | GPU v5，batch prefill + standard attention |
| `gpu_v6/` | GPU v6，batch prefill + flash attention prefill |
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

# GPU v6（推荐）
./gpu_decoder_v6 qwen2.5-3b-instruct-fp16.gguf
```

## 性能对比（Qwen2.5-3B，RTX 5060 Ti 16GB）

### decode 阶段（短 prompt，~20 tokens）

| 版本 | tokens/s | 说明 |
| :--- | :--- | :--- |
| CPU C++ | ~2 | fp16 逐元素转换，朴素 matmul |
| GPU v1 | ~6 | 手写 fp16 matmul kernel |
| GPU v2 | ~34 | cublasHgemm，中间状态仍是 fp32 |
| GPU v3 | ~45 | 全程 fp16，消除类型转换开销 |
| GPU v4 | ~45 | Flash Attention decode 阶段（无提升） |
| GPU v5 | ~46 | batch prefill + standard attention |
| GPU v6 | ~46 | batch prefill + flash attention prefill |

### prefill 阶段对比（prompt=3528 tokens）

| 版本 | prefill tokens/s | prefill 方式 | attention |
| :--- | :--- | :--- | :--- |
| GPU v4 | ~58 | 逐 token 循环 | flash attention decode kernel |
| GPU v5 | ~1059 | batch prefill | standard attention（scores 写 HBM，759MB） |
| GPU v6 | ~578 | batch prefill | flash attention（简化版，反而更慢） |

## 优化分析

### GPU v1 → v2（5x 提升）

用 `cublasHgemm` 替换手写 `matmul_fp16_kernel`。v1 手写 kernel 的 grid 只有几个 block，SM 大量闲置。cuBLAS 底层使用 cutlass，自动选择最优调度，SM 利用率大幅提升。

**v1 nsys profile：**

![nsys v1](doc/nsys-v1.png)

```
matmul_fp16_kernel: 99%  ← 几乎所有时间在 matmul
attention_kernel:    0.4%
rmsnorm_kernel:      0.3%
```

---

### GPU v2 → v3（30% 提升）

把 `GPURunState` 所有中间状态从 `float*` 改成 `__half*`，消除每次 matmul 前后的两次类型转换 kernel，显存占用同时减半。

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

只有以下计算保留 fp32：RMSNorm 平方和累加（防止溢出）、attention softmax（精度敏感）、最终 logits 输出。

---

### GPU v3 → v4（Flash Attention decode，无提升）

在 decode 阶段实现 Flash Attention，用 online softmax 分块处理 K、V，scores 不写 HBM。

**v4 nsys profile：**

![nsys v4](doc/nsys-v4.png)

```
Kernels: 49.0%，Memory: 51.0%
attention_kernel: 1.8%  ← 从 2.6% 降到 1.8%
```

decode 阶段 flash attention 提升有限：q 只有 1 行，scores 大小为 `1 × seq_len`，不大。分块反而导致 K、V 被多次读取，Memory 占比从 38.3% 升到 51.0%。

---

### GPU v4 → v5（batch prefill，18x prefill 提升）

把逐 token prefill 改成 batch prefill，一次处理整个 prompt：

```
逐 token: matmul [1, dim] × [dim, dim]  → gemv，GPU 利用率低
batch:    matmul [n, dim] × [dim, dim]  → gemm，GPU 利用率高
```

prefill 速度从 58 tokens/s 提升到 1059 tokens/s（**18x**）。

**v5 nsys profile：**

![nsys v5](doc/nsys-v5.png)

```
Kernels: 21.4%，Memory: 78.6%
attention_kernel: 2.1%
```

---

### GPU v5 → v6（Flash Attention prefill，反而更慢）

在 prefill 阶段用 flash attention，scores 不写 HBM。但实验结果显示反而更慢：

```
v5（standard）: prefill 1059 tokens/s，att_pre HBM 读写 759MB
v6（flash）:    prefill  578 tokens/s，反而慢了 45%
```

**v6 nsys profile：**

![nsys v6](doc/nsys-v6.png)

```
Kernels: 18.9%，Memory: 81.1%  ← Memory 比 v5 更高
attention_kernel: 2.0%
```

**根本原因：** 我们实现的是简化版 flash attention，只分了 K/V 块，没有分 Q 块：

```
简化版（我们的实现）：
  Grid = dim3(n_heads, n_tokens)
  每个 block 独立处理一个 query token
  K/V 被读取 n_tokens 次，HBM 读写 O(n²)

真正的 flash attention（论文 Algorithm 1）：
  Q/K/V 双重分块
  外层循环遍历 K/V 块，内层循环遍历 Q 块
  每个 K/V 块只读一次，被所有 Q 块复用
  HBM 读写降到 O(n)
```

这个实验验证了：**分块不等于 flash attention，IO 感知的正确实现（Q/K/V 双重分块）才是关键。**

## Attention shared memory 限制

标准 attention kernel 用 shared memory 存 attention scores，大小为 `seq_len * sizeof(float)`：

```
3B: seq_len=32768 → shared_mem=128KB  ← 超出 GPU per-block 限制（通常 48-96KB）
```

kernel 启动失败时静默返回，导致 attention 输出错误但无报错。通过 `CHECK_KERNEL()` 宏定位：

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

v2/v3 改为用全局内存（`gs.att`，预分配 `n_heads * seq_len`）存 attention scores。

## tests/

`tests/` 目录包含算法验证代码，从基础到完整逐步实现：

```
tests/softmax/    softmax → online softmax，numpy / torch / C++ / CUDA 四种实现对比
tests/attention/  standard attention → flash attention，numpy / C++ / CUDA 三种实现对比
```

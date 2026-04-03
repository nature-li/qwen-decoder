# qwen-decoder

从零手写的 Qwen2.5 推理引擎，支持加载 GGUF 格式模型，提供 CPU C++、CUDA GPU 多个版本实现，记录了完整的性能优化过程。

## 项目结构

| 文件/目录 | 说明 |
| :--- | :--- |
| `common.h` | 公共数据结构（Config、Weights、Tokenizer、ModelFile） |
| `common.cpp` | 公共函数（GGUF 加载、BPE tokenizer、采样） |
| `gguf_loader.h/cpp` | GGUF 文件格式解析 |
| `decoder.h/cpp` | Decoder 基类，generate 逻辑 |
| `scheduler.h/cpp` | 请求调度器，continuous batching 核心 |
| `kv_cache.h/cpp` | BlockPool + BlockTable，PagedAttention KV cache 管理 |
| `pd_comm.h/cpp` | P/D 分离通信，TCP 握手 + 多 NCCL stream 并发传输 |
| `cpu_decoder.h/cpp` | CPU 版本实现 |
| `gpu_v1/` | GPU 朴素版本，手写 fp16 matmul kernel |
| `gpu_v2/` | GPU v2，cublasHgemm 替换手写 kernel |
| `gpu_v3/` | GPU v3，全程 fp16 中间状态 |
| `gpu_v4/` | GPU v4，Flash Attention decode 阶段 |
| `gpu_v5/` | GPU v5，batch prefill + standard attention |
| `gpu_v6/` | GPU v6，batch prefill + flash attention prefill |
| `gpu_v7/` | GPU v7，fixed batch decode，多请求并发 |
| `gpu_v8/` | GPU v8，continuous batching，动态请求调度 |
| `gpu_v9/` | GPU v9，PagedAttention，KV cache 动态分页分配 |
| `gpu_v10/` | GPU v10，1-flat 推理，chunked prefill，decode attention 并行 |
| `gpu_v11/` | GPU v11，P/D 分离，多 NCCL stream 并发 KV cache 传输 |
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
- KV Cache（fp16，PagedAttention 动态分页）
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
- NCCL 2.x（v11 P/D 分离需要）

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
# 1-flat 推理（max_batch=64，max_prefill_len=4096，max_prefill_tokens_per_step=512）
./gpu_decoder_v11 qwen2.5-3b-instruct-fp16.gguf 64 4096 4096

# P/D 分离：P节点先启动，D节点后启动
# P节点（prefill）
NCCL_SOCKET_IFNAME=eth0 ./prefill_node qwen2.5-3b-instruct-fp16.gguf

# D节点（decode）
NCCL_SOCKET_IFNAME=eth0 ./decode_node qwen2.5-3b-instruct-fp16.gguf <p_ip>
```

## 性能对比（Qwen2.5-3B fp16，RTX 5060 Ti 16GB，temperature=0）

### 完整优化路径

| 版本 | tokens/s | 较 v1 提升 | 说明 |
| :--- | :--- | :--- | :--- |
| CPU C++ | 0.11 | - | fp16 逐元素转换，朴素 matmul |
| GPU v1 | 7.18 | 1x | 手写 fp16 matmul kernel |
| GPU v2 | 47.48 | 6.6x | cublasHgemm，中间状态仍是 fp32 |
| GPU v3 | 48.29 | 6.7x | 全程 fp16，消除类型转换开销 |
| GPU v4 | 48.99 | 6.8x | Flash Attention decode 阶段（无提升） |
| GPU v5 | 53.12 | 7.4x | batch prefill，prefill 速度提升 18x |
| GPU v6 | 52.60 | 7.3x | batch prefill + flash attention（反而更慢） |
| GPU v7 | 209.34 | 29x | fixed batch decode，4并发 |
| GPU v8 | 168.0 | 23x | continuous batching，4并发 |
| GPU v9 | 869.9 | 121x | PagedAttention，64并发 |
| GPU v10 | 1341.0 | 187x | 1-flat 推理，64并发 |

### prefill 阶段对比（prompt=3528 tokens）

| 版本 | prefill tokens/s | prefill 方式 | attention |
| :--- | :--- | :--- | :--- |
| GPU v4 | ~58 | 逐 token 循环 | flash attention decode kernel |
| GPU v5 | ~1059 | batch prefill | standard attention（scores 写 HBM，759MB） |
| GPU v6 | ~578 | batch prefill | flash attention（简化版，反而更慢） |

### GPU 利用率对比

| 版本 | 并发数 | GPU Kernels | GPU Memory | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| v6 | 1 | ~49% | ~51% | 单请求，decode memory-bound |
| v7 | 4 | ~27% | ~73% | fixed batch，请求完成后槽位浪费 |
| v8 | 4 | ~87% | ~13% | continuous batching，GPU 始终满载 |
| v9 | 64 | ~90.7% | ~9.3% | PagedAttention，并发数大幅提升 |
| v10 | 64 | ~68.5% | ~31.5% | 1-flat，prefill 阶段 memcpy 较多 |
| v11（D节点） | - | ~94.5% | ~5.5% | P/D 分离，D节点专注 decode |

## 优化分析

### GPU v1 → v2（6.6x 提升）

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

---

### GPU v6 → v7（fixed batch decode，总吞吐 4x）

实现 fixed batch decode，支持多请求同时推理。每个请求有独立的 KV cache 和 pos，attention 时只看自己的 KV cache 槽。

**v7 nsys profile：**

![nsys v7](doc/nsys-v7.png)

```
Kernels: 27.0%，Memory: 73.0%
```

**Memory 高达 73% 的原因：** fixed batch 里请求长度不齐，短的请求完成后槽位空着，实际 batch size 退化，matmul 退化成 gemv，变成 memory-bound。

---

### GPU v7 → v8（continuous batching，GPU 利用率 87%）

实现 Scheduler，完成的请求立即让出槽位给等待队列里的新请求，GPU 始终满载：

```
fixed batch（v7）：
  step1: [req0, req1, req2, req3]
  step2: [req0, req1, ----, req3]  ← req2 完成，槽位空着浪费
  step3: [req0, ----, ----, req3]  ← 继续浪费

continuous batching（v8）：
  step1: [req0, req1, req2, req3]
  step2: [req0, req1, req4, req3]  ← req2 完成，req4 立即填入
  step3: [req0, req5, req4, req3]  ← req1 完成，req5 立即填入
```

**v8 nsys profile：**

![nsys v8](doc/nsys-v8.png)

```
Kernels: 86.8%，Memory: 13.2%
attention_batch_kernel: 8.6%
```

---

### GPU v8 → v9（PagedAttention，GPU 利用率 90.7%）

实现 PagedAttention，KV cache 从静态预分配改为动态分页分配：

```
v8 静态 KV cache：
  k_cache[max_batch, n_layers, seq_len, kv_dim]
  每个请求固定占用 seq_len=32768 的空间，浪费严重

v9 PagedAttention：
  k_cache[total_blocks, block_size, n_layers, kv_dim]
  每个 block 存 block_size=16 个 token 的 KV
  请求用多少申请多少，释放后立即归还 BlockPool
```

核心数据结构：

```
BlockPool：管理所有物理块，空闲块队列，allocate()/free()
BlockTable：每个请求的逻辑块→物理块映射
  table[0]=9, table[1]=3, table[2]=7  ← 物理块不必连续

attention kernel 通过 BlockTable 查找：
  token t → 逻辑块 t/block_size → 物理块 id → 实际地址
```

**v9 nsys profile：**

![nsys v9](doc/nsys-v9.png)

```
Kernels: 90.7%，Memory: 9.3%
attention_paged_kernel: 14.2%
cuBLAS: 76.2%
```

---

### GPU v9 → v10（1-flat 推理，64并发，1341 tokens/s）

核心改动四处：

**1. 1-flat 推理**

把所有请求的 token 打包成一个 flat batch，一次 forward 处理：

```
v9：
  prefill：串行，一个请求 prefill 完再 prefill 下一个
  decode：batch，多个请求并发

v10：
  flat batch = [req0_prefill_tokens..., req1_decode_token, req2_decode_token, ...]
  matmul/FFN 一次处理所有 token
```

**2. chunked prefill**

每步 prefill token 总数不超过 `max_prefill_tokens_per_step`，多个请求共享预算：

```
没有 chunked prefill：
  新请求进来，prompt=2000 tokens，一次性全部 prefill
  total_tokens 突然变大，GPU 负载波动，decode 请求被饿死

有 chunked prefill（max_prefill_tokens_per_step=512）：
  每步最多 512 个 prefill token，和 decode 请求交替进行
  total_tokens 始终稳定，TTFT 降低
```

**3. decode attention 并行**

```
v9：for each decode request → attention kernel（串行 launch）
v10：gather_q → 一次 launch 处理所有 decode token → scatter_xb
```

**4. block_table 增量更新**

```
v9：每步上传 max_batch * max_blocks_per_seq 个 int
v10：只上传有变化的槽位的实际用到的 block 数
```

**v10 nsys profile：**

![nsys v10](doc/nsys-v10.png)

```
Kernels: 68.5%，Memory: 31.5%
cuBLAS: 74.4%
decode_attention_kernel: 16.3%
```

---

### GPU v10 → v11（P/D 分离，多 NCCL stream 并发）

将 prefill 和 decode 分离到两台机器，P节点专注 prefill，D节点专注 decode：

```
单机（v10）：
  prefill 和 decode 竞争同一块 GPU
  prefill 计算密集，会阻塞 decode，导致 decode 请求延迟抖动

P/D 分离（v11）：
  P节点（RTX 5060 Ti）：专注 prefill，算完 KV cache 传给 D节点
  D节点（RTX 5060 Ti）：专注 decode，GPU 几乎全部用于 decode kernel
```

**通信架构：**

```
控制信息（TCP）：
  D节点 → P节点：slot_id + req_id + prompt_tokens
  P节点 → D节点：slot_id + req_id + n_tokens_done（结束信号）

KV cache（多 NCCL stream 并发）：
  每个 slot 独立的 NCCL 通信域和 CUDA stream
  P节点 gather kernel → ncclSend(stream[slot_id])
  D节点 ncclRecv(stream[slot_id]) → scatter kernel
  多个请求的 KV cache 并发传输，互不阻塞
```

**KV cache 传输流程：**

```
P节点 prefill 完一个 chunk（max_prefill_len tokens）：
  gather_kernel：PagePool 分散数据 → 连续 kv_buffer（GPU）
  ncclSend：kv_buffer → D节点（GPU 到 GPU，1Gbps 网络）

D节点接收：
  ncclRecv：kv_buffer（GPU）
  scatter_kernel：kv_buffer → D节点 PagePool
  收到结束信号 → 请求加入 decode 队列
```

**v11 D节点 nsys profile：**

![nsys v11 decode](doc/nsys-v11-decode.png)

```
Kernels: 94.5%，Memory: 5.5%
Stream 13（decode）:  73.4% Kernels
Stream 14（NCCL）:    21.4% Kernels（接收 KV cache）
```

**单机 vs D节点对比：**

```
              单机（v10）    D节点（v11）
Kernels:      68.5%         94.5%  ← decode GPU 利用率大幅提升
Memory:       31.5%          5.5%
NCCL:         无            21.4%  ← KV cache 传输开销
```

D节点 GPU 利用率 94.5%，几乎全部用于 decode，是所有版本中最高的。21.4% 的 NCCL 开销来自 1Gbps 网络传输 KV cache，这也验证了 P/D 分离需要高带宽网络（RDMA）才能充分发挥优势。

**1Gbps 网络下的 KV cache 传输时间：**

```
prompt=512 tokens 的 KV cache 大小：
  512 * 36 * 256 * 2 * 2bytes = 18MB
  1Gbps 网络传输时间 ≈ 144ms

单机 prefill 512 tokens 时间：
  512 / 1059 tokens/s ≈ 484ms

结论：1Gbps 网络下，KV cache 传输时间远小于 prefill 时间
     P/D 分离在此场景下可以正常工作
     若需要真正的低延迟 P/D 分离，需要 RDMA（100Gbps+）
```

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

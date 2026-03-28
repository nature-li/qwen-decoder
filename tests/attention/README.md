# attention 实现对比

softmax → online softmax → standard attention → flash attention，四种实现逐步递进。

## 文件

| 文件 | 说明 |
| :--- | :--- |
| `attention_numpy.py` | NumPy 版本，标准 attention + flash attention |
| `attention_cpu.cpp` | C++ 版本，标准 attention + flash attention |
| `attention_gpu.cu` | CUDA 版本，softmax / online softmax / standard attention / flash attention |

## 运行

```bash
# NumPy
python attention_numpy.py

# C++
g++ -O2 -o attention_cpu attention_cpu.cpp && ./attention_cpu

# CUDA
nvcc -arch=sm_100 -o attention_gpu attention_gpu.cu && ./attention_gpu
```

## 算法演进

### 1. softmax

两遍扫描：

```
第一遍: max = max(x)
第二遍: exp(x - max) / sum(exp(x - max))
```

### 2. online softmax

一遍扫描，分块处理，不需要提前看所有元素：

```
初始: m = -inf, d = 0

for each block:
    m_new = max(m, max(block))
    d = d * exp(m - m_new) + sum(exp(block - m_new))
    m = m_new

probs = exp(x - m) / d
```

`exp(m - m_new)` 是修正项，把历史 d 从旧 base 换算到新 base，利用了 `exp(a-b) = exp(a)/exp(b)` 性质。

### 3. standard attention（decode 阶段）

```
q: [head_dim]          ← 当前 token
k: [seq_len, head_dim] ← KV cache
v: [seq_len, head_dim] ← KV cache

1. scores = q @ k^T * scale  → 写 HBM（seq_len 个值）
2. probs  = softmax(scores)  → 读写 HBM
3. out    = probs @ v        → 读 HBM
```

问题：scores 需要写到 HBM，softmax 再读回来，seq_len 越大 IO 开销越大。

### 4. flash attention

用 online softmax 同时累计输出，scores 不写 HBM：

```
初始: m = -inf, d = 0, o = 0

for each block of (K, V):
    s = q @ K_block^T * scale          ← 局部变量，不写 HBM
    m_new = max(m, max(s))
    correction = exp(m - m_new)
    d = d * correction + sum(exp(s - m_new))
    o = o * correction + exp(s - m_new) @ V_block
    m = m_new

out = o / d
```

和 online softmax 相比只多了一行：`o = o * correction + exp(s - m_new) @ V_block`，含义是修正历史输出 + 加新块贡献。

## 内存访问对比

```
standard attention:
  HBM 写: scores[seq_len]       ← Q@K^T 写入
  HBM 读: scores[seq_len]       ← softmax 读入
  HBM 写: scores[seq_len]       ← softmax 写回
  HBM 读: scores[seq_len] + V   ← weighted sum 读入
  总计: O(seq_len²)

flash attention:
  HBM 读: K, V（各一遍）
  HBM 写: out（一次）
  scores 全程在 shared memory / 寄存器里
  总计: O(seq_len)
```

seq_len 越大，flash attention 优势越明显。

## CUDA 版本说明

`attention_gpu.cu` 包含四个 kernel：

| kernel | 说明 |
| :--- | :--- |
| `softmax_kernel` | 标准 softmax，两级 warp reduce |
| `online_softmax_kernel` | online softmax，一遍扫描 |
| `standard_attention_kernel` | 标准 attention，scores 写 HBM |
| `flash_attention_kernel` | flash attention，scores 在 shared memory |

`flash_attention_kernel` 用 `extern __shared__` 动态分配 shared memory，在 kernel 启动时指定大小：

```cpp
size_t smem_size = (BLOCK_SIZE + head_dim + 8 + 8 + 2) * sizeof(float);
flash_attention_kernel<<<1, 256, smem_size>>>(...);
```

shared memory 大小是 O(block_size + head_dim)，不随 seq_len 增长。
# softmax 四种实现对比

同一个 softmax，四种实现方式，方便对比学习。

## 文件

| 文件 | 语言 | 说明 |
| :--- | :--- | :--- |
| `softmax_numpy.py` | Python + NumPy | 最简洁，向量化操作 |
| `softmax_torch.py` | Python + PyTorch | 和 NumPy 类似，支持 GPU |
| `softmax_cpu.cpp` | C++ | 显式循环，逻辑最清晰 |
| `softmax_gpu.cu` | CUDA | 两级 warp reduce，并行归约 |

## 运行

```bash
# NumPy
python softmax_numpy.py

# PyTorch
python softmax_torch.py

# C++
g++ -O2 -o softmax_cpu softmax_cpu.cpp && ./softmax_cpu

# CUDA
nvcc -arch=sm_100 -o softmax_gpu softmax_gpu.cu && ./softmax_gpu
```

## 核心逻辑对比

三步都一样，只是实现方式不同：

```
1. 找 max（数值稳定）
2. exp(x - max) 并求 sum
3. 归一化 / sum
```

| 实现 | 找 max | 求 sum | 归一化 |
| :--- | :--- | :--- | :--- |
| NumPy | `np.max(x)` | `e.sum()` | `e / sum` |
| PyTorch | `x.max()` | `e.sum()` | `e / sum` |
| C++ | `std::max_element` | 循环累加 | 循环除法 |
| CUDA | 两级 warp reduce | 两级 warp reduce | 跨步循环 |

## CUDA 版本说明

256 个线程 = 8 个 warp，归约分两级：

```
第一级：warp 内用 __shfl_down_sync 归约（无需 shared memory，极快）
        每个 warp 的 lane 0 得到该 warp 内的结果

第二级：8 个 warp 的结果写到 shared memory
        warp 0 再做一次归约得到全局结果
```

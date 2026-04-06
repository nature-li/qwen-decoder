import numpy as np

np.random.seed(42)
T = 2
dim = 4
hidden = 6
n_ranks = 2

def silu(x):
    return x / (1 + np.exp(-x))

def section(title):
    print()
    print("=" * 50)
    print(title)
    print("=" * 50)

def divider():
    print("-" * 50)

X = np.random.randn(T, dim).astype(np.float32)
Wq = np.random.randn(dim, dim).astype(np.float32)
Wk = np.random.randn(dim, dim).astype(np.float32)
Wv = np.random.randn(dim, dim).astype(np.float32)
Wo = np.random.randn(dim, dim).astype(np.float32)
W1 = np.random.randn(dim, hidden).astype(np.float32)
W3 = np.random.randn(dim, hidden).astype(np.float32)
W2 = np.random.randn(hidden, dim).astype(np.float32)

# 切分权重
Wq0 = Wq[:, :dim//2];   Wq1 = Wq[:, dim//2:]
Wk0 = Wk[:, :dim//2];   Wk1 = Wk[:, dim//2:]
Wv0 = Wv[:, :dim//2];   Wv1 = Wv[:, dim//2:]
Wo0 = Wo[:dim//2, :];   Wo1 = Wo[dim//2:, :]
W1_0 = W1[:, :hidden//2]; W1_1 = W1[:, hidden//2:]
W3_0 = W3[:, :hidden//2]; W3_1 = W3[:, hidden//2:]
W2_0 = W2[:hidden//2, :]; W2_1 = W2[hidden//2:, :]

# ============================================================
section("1. Q 投影：Wq 按列切")
Q_full = X @ Wq
print(f"X.shape{X.shape} @ Wq.shape{Wq.shape} -> Q_full.shape:{Q_full.shape}")
divider()

Q0 = X @ Wq0
print(f"X.shape{X.shape} @ Wq0.shape{Wq0.shape} -> Q0.shape:{Q0.shape}")
divider()

Q1 = X @ Wq1
print(f"X.shape{X.shape} @ Wq1.shape{Wq1.shape} -> Q1.shape:{Q1.shape}")
divider()

Q_tp = np.concatenate([Q0, Q1], axis=1)
print(f"[Q0.shape{Q0.shape} | Q1.shape{Q1.shape}] -> Q_tp.shape:{Q_tp.shape}")
divider()

print(f"Q_full:\n{Q_full}")
divider()
print(f"Q_tp:\n{Q_tp}")
divider()
print(f"误差: {np.max(np.abs(Q_full - Q_tp)):.6f}")
divider()

# ============================================================
section("2. K 投影：Wk 按列切")
K_full = X @ Wk
print(f"X.shape{X.shape} @ Wk.shape{Wk.shape} -> K_full.shape:{K_full.shape}")
divider()

K0 = X @ Wk0
print(f"X.shape{X.shape} @ Wk0.shape{Wk0.shape} -> K0.shape:{K0.shape}")
divider()

K1 = X @ Wk1
print(f"X.shape{X.shape} @ Wk1.shape{Wk1.shape} -> K1.shape:{K1.shape}")
divider()

K_tp = np.concatenate([K0, K1], axis=1)
print(f"[K0.shape{K0.shape} | K1.shape{K1.shape}] -> K_tp.shape:{K_tp.shape}")
divider()
print(f"误差: {np.max(np.abs(K_full - K_tp)):.6f}")
divider()

# ============================================================
section("3. V 投影：Wv 按列切")
V_full = X @ Wv
V0 = X @ Wv0
V1 = X @ Wv1
V_tp = np.concatenate([V0, V1], axis=1)
print(f"X.shape{X.shape} @ Wv.shape{Wv.shape} -> V_full.shape:{V_full.shape}")
divider()
print(f"[V0.shape{V0.shape} | V1.shape{V1.shape}] -> V_tp.shape:{V_tp.shape}")
divider()
print(f"误差: {np.max(np.abs(V_full - V_tp)):.6f}")
divider()

# ============================================================
section("4. Attention 计算（简化版，不做 softmax）")
scale_full = np.sqrt(dim)
Attn_full = (Q_full @ K_full.T / scale_full) @ V_full
print(f"Q_full{Q_full.shape} @ K_full.T{K_full.T.shape} / sqrt({dim}) @ V_full{V_full.shape} -> Attn_full{Attn_full.shape}")
divider()

scale_tp = np.sqrt(dim // 2)
Attn0 = (Q0 @ K0.T / scale_tp) @ V0
Attn1 = (Q1 @ K1.T / scale_tp) @ V1
print(f"rank0: Q0{Q0.shape} @ K0.T{K0.T.shape} / sqrt({dim//2}) @ V0{V0.shape} -> Attn0{Attn0.shape}")
divider()
print(f"rank1: Q1{Q1.shape} @ K1.T{K1.T.shape} / sqrt({dim//2}) @ V1{V1.shape} -> Attn1{Attn1.shape}")
divider()
print(f"注意：简化版 attention 各自独立计算，误差来自 scale 不同，这里仅演示 shape")
divider()

# ============================================================
section("5. O 投影：Wo 按行切 + AllReduce")
Out_attn_full = Attn_full @ Wo
print(f"Attn_full{Attn_full.shape} @ Wo{Wo.shape} -> Out_attn_full{Out_attn_full.shape}")
divider()

# Attn 已被切，rank0 有 Attn0，rank1 有 Attn1
Out0_attn = Attn0 @ Wo0
Out1_attn = Attn1 @ Wo1
print(f"rank0: Attn0{Attn0.shape} @ Wo0{Wo0.shape} -> Out0_attn{Out0_attn.shape}  (部分和)")
divider()
print(f"rank1: Attn1{Attn1.shape} @ Wo1{Wo1.shape} -> Out1_attn{Out1_attn.shape}  (部分和)")
divider()
Out_attn_tp = Out0_attn + Out1_attn
print(f"AllReduce: Out0_attn{Out0_attn.shape} + Out1_attn{Out1_attn.shape} -> Out_attn_tp{Out_attn_tp.shape}")
divider()

# ============================================================
section("6. FFN SwiGLU：W1/W3 按列切，W2 按行切 + AllReduce")
X2 = X  # 简化，直接用 X 演示 FFN

# 完整计算
H1_full = X2 @ W1
H3_full = X2 @ W3
H_full  = silu(H1_full) * H3_full
Out_ffn_full = H_full @ W2
print(f"完整 FFN:")
print(f"  X2{X2.shape} @ W1{W1.shape} -> H1{H1_full.shape}")
print(f"  X2{X2.shape} @ W3{W3.shape} -> H3{H3_full.shape}")
print(f"  silu(H1) * H3 -> H{H_full.shape}  (逐元素乘)")
print(f"  H{H_full.shape} @ W2{W2.shape} -> Out{Out_ffn_full.shape}")
divider()

# 切分计算
H0 = silu(X2 @ W1_0) * (X2 @ W3_0)
H1_tp = silu(X2 @ W1_1) * (X2 @ W3_1)
print(f"rank0: silu(X2{X2.shape} @ W1_0{W1_0.shape}) * (X2 @ W3_0{W3_0.shape}) -> H0{H0.shape}")
divider()
print(f"rank1: silu(X2{X2.shape} @ W1_1{W1_1.shape}) * (X2 @ W3_1{W3_1.shape}) -> H1{H1_tp.shape}")
divider()

Out0_ffn = H0 @ W2_0
Out1_ffn = H1_tp @ W2_1
print(f"rank0: H0{H0.shape} @ W2_0{W2_0.shape} -> Out0_ffn{Out0_ffn.shape}  (部分和)")
divider()
print(f"rank1: H1{H1_tp.shape} @ W2_1{W2_1.shape} -> Out1_ffn{Out1_ffn.shape}  (部分和)")
divider()

Out_ffn_tp = Out0_ffn + Out1_ffn
print(f"AllReduce: Out0_ffn + Out1_ffn -> Out_ffn_tp{Out_ffn_tp.shape}")
divider()
print(f"Out_ffn_full:\n{Out_ffn_full}")
divider()
print(f"Out_ffn_tp:\n{Out_ffn_tp}")
divider()
print(f"误差: {np.max(np.abs(Out_ffn_full - Out_ffn_tp)):.6f}")
divider()

# ============================================================
section("7. 完整一层总结")
print("每层通信点：")
print("  Attention 后：AllReduce（wo 输出部分和相加）")
print("  FFN 后：      AllReduce（w2 输出部分和相加）")
print(f"  共 2 次 AllReduce，数据量 = T×dim×4bytes = {T}×{dim}×4 = {T*dim*4} bytes")
divider()
print("权重切分方式：")
print(f"  Wq: [{dim},{dim}] -> rank0:[{dim},{dim//2}] rank1:[{dim},{dim//2}]  (按列切)")
print(f"  Wk: [{dim},{dim}] -> rank0:[{dim},{dim//2}] rank1:[{dim},{dim//2}]  (按列切)")
print(f"  Wv: [{dim},{dim}] -> rank0:[{dim},{dim//2}] rank1:[{dim},{dim//2}]  (按列切)")
print(f"  Wo: [{dim},{dim}] -> rank0:[{dim//2},{dim}] rank1:[{dim//2},{dim}]  (按行切)")
print(f"  W1: [{dim},{hidden}] -> rank0:[{dim},{hidden//2}] rank1:[{dim},{hidden//2}]  (按列切)")
print(f"  W3: [{dim},{hidden}] -> rank0:[{dim},{hidden//2}] rank1:[{dim},{hidden//2}]  (按列切)")
print(f"  W2: [{hidden},{dim}] -> rank0:[{hidden//2},{dim}] rank1:[{hidden//2},{dim}]  (按行切)")
divider()
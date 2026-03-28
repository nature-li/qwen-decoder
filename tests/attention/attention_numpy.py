import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    # 减去 max 保证数值稳定，防止 exp 溢出
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def online_softmax(x: np.ndarray, block_size: int) -> np.ndarray:
    m = -np.inf  # 当前全局最大值
    d = 0.0  # 当前分母 sum(exp)

    for start in range(0, len(x), block_size):
        block = x[start : start + block_size]

        # 更新全局最大值
        m_new = max(m, max(block))

        # 修正旧的 d, 换算到新的 base
        correction = np.exp(m - m_new)

        # 旧 d 修正后加上新块的贡献
        d = d * correction + sum(np.exp(v - m_new) for v in block)

        m = m_new

    # 最终归一化
    probs = np.exp(x - m) / d
    return probs


def standard_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float
) -> np.ndarray:
    """
    标准 attention
    q: [head_dim]
    k: [seq_len, head_dim]
    v: [seq_len, head_dim]

    三步, scores 需要完整存在内存里:
    - 1. scores = q @ K^T * scale   -> 写内存
    - 2. probs = softmax(scores)    -> 读写内存
    - 3. out = probs @ v            -> 读内存
    """
    scores = (q @ k.T) * scale  # [seq_len]
    probs = softmax(scores)  # [seq_len]
    out = probs @ v  # [head_dim]
    return out


def flash_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float, block_size: int
) -> np.ndarray:
    """
    flash attention (decode 阶段, q 只有一行)
    q: [head_dim]
    k: [seq_len, head_dim]
    v: [seq_len, head_dim]

    分块处理 K, V, scores 不写内存，用 online softmax 同时累计输出:
    m: 当前全局最大值
    d: 当前分母 sum(exp)
    o: 当前累计输出 (未归一化)
    """
    seq_len = k.shape[0]
    head_dim = q.shape[0]

    m = -np.inf
    d = 0.0
    o = np.zeros(head_dim)

    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)

        k_block = k[start:end]  # [block_size, head_dim]
        v_block = v[start:end]  # [block_size, head_dim]

        # 1. 计算块内 scores，大小是 block_size，不是 seq_len
        # 标准 attention 需要完整的 scores[seq_len] 才能做 softmax
        # flash attention 只需要当前块的 scores，用 online softmax 边算边更新
        s_block = (q @ k_block.T) * scale  # [block_size]

        # 2. online softmax 更新 m 和 d
        m_new = max(m, np.max(s_block))
        correction = np.exp(m - m_new)

        d = d * correction + np.sum(np.exp(s_block - m_new))

        # 3. 更新输出 (修正历史贡献 + 新块贡献)
        o = o * correction + np.exp(s_block - m_new) @ v_block

        m = m_new

    # 最终归一化
    return o / d


if __name__ == "__main__":
    np.random.seed(42)
    head_dim = 8
    seq_len = 4
    scale = 1.0 / np.sqrt(head_dim)

    q = np.random.randn(head_dim).astype(np.float32)
    k = np.random.randn(seq_len, head_dim).astype(np.float32)
    v = np.random.randn(seq_len, head_dim).astype(np.float32)

    out_std = standard_attention(q, k, v, scale)
    out_flash = flash_attention(q, k, v, scale, block_size=2)

    print("standard_attention:")
    print(f"  out[:4] = {out_std[:4]}")

    print("flash_attention:")
    print(f"  out[:4] = {out_flash[:4]}")

    print(f"match: {np.allclose(out_std, out_flash, atol=1e-5)}")

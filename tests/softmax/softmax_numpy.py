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
        block = x[start: start + block_size]
        
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


if __name__ == "__main__":
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = softmax(x)
    for i, v in enumerate(y):
        print(f"y[{i}] = {v:.6f}")
    print(f"sum  = {y.sum():.6f}")

    y = online_softmax(x, 3)
    for i, v in enumerate(y):
        print(f"y[{i}] = {v:.6f}")
    print(f"sum  = {y.sum():.6f}")

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    # 减去 max 保证数值稳定，防止 exp 溢出
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


if __name__ == "__main__":
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = softmax(x)
    for i, v in enumerate(y):
        print(f"y[{i}] = {v:.6f}")
    print(f"sum  = {y.sum():.6f}")

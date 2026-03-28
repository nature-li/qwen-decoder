import torch


def softmax(x: torch.Tensor) -> torch.Tensor:
    # 减去 max 保证数值稳定，防止 exp 溢出
    x = x - x.max()
    e = x.exp()
    return e / e.sum()


if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = softmax(x)
    for i, v in enumerate(y):
        print(f"y[{i}] = {v:.6f}")
    print(f"sum  = {y.sum():.6f}")

    # 验证和内置结果一致
    y_builtin = torch.softmax(x, dim=0)
    print(f"\ntorch.softmax: {y_builtin}")
    print(f"match: {torch.allclose(y, y_builtin)}")

import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()          # y = x₁² + x₂² + x₃² = 14
y.backward()                # 自动算梯度

print(y.item())             # 14.0
print(x.grad)               # tensor([2., 4., 6.])
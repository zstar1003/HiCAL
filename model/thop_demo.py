import torch
import torch.nn as nn
from thop import profile

# 假设你有一个模型
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3),
    nn.ReLU()
)

# 输入张量
input_tensor = torch.randn(2, 3, 640, 640)  # 例如 224x224 的 RGB 图像

# 计算模型的参数量和 FLOPS
macs, params = profile(model, inputs=(input_tensor,))

# 输出结果
print(f"Parameters: {params / 1e6:.3f} M")  # 转换为百万参数
print(f"FLOPS: {macs / 1e9:.3f} GFLOPS")   # 转换为十亿 FLOPS
'''
Parameters: 0.076 M
FLOPS: 3.654 GFLOPS
'''

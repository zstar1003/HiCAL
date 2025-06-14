import torch
import torch.nn as nn
from thop import profile
from YOLOv9.models.yolo import Model

weights = "weights/visdrone_init.pt"  # 示例权重路径
cfg = "models/detect/yolov9-c.yaml"  # 配置文件路径
nc = 20  # 类别数量
device = "cuda"

model = Model(cfg, ch=3, nc=nc, anchors=None).to(device)  # 创建模型实例

pretrained = weights.endswith(".pt")
if pretrained:
    ckpt = torch.load(weights, map_location="cpu")  # 加载权重
    model.load_state_dict(ckpt["model"].float().state_dict(), strict=False)

# 创建输入张量（例如，图像大小 640x640）
input_tensor = torch.randn(1, 3, 640, 640).to(device)  # Batch size 1, RGB 图像 640x640

# 使用 thop 计算 FLOPS 和 参数量
macs, params = profile(model, inputs=(input_tensor,))

# 输出结果
print(f"Parameters: {params / 1e6:.3f} M")  # 转换为百万参数
print(
    f"FLOPS: {macs * 2 / 1e9:.3f} GFLOPS"
)  # 将 MACs 转换为 FLOPS（1 MAC = 2 FLOP），并转换为 GFLOPS

import torch
import torch.nn as nn
from models.common import Conv, RepNCSPELAN4, ADown, Concat, SPPELAN
from models.yolo import DualDDetect


class DetectionHead(nn.Module):
    def __init__(self, num_classes=10):
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            SPPELAN(512, 256),  # 使用相应的模块
            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(1),
            RepNCSPELAN4(512, 512, 256, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(1),
            RepNCSPELAN4(256, 256, 128, 1),
            ADown(256, 256),
            Concat(1),
            RepNCSPELAN4(512, 512, 256, 1),
            ADown(512, 512),
            Concat(1),
            RepNCSPELAN4(512, 512, 256, 1),
            DualDDetect(num_classes)
        )

    def forward(self, x):
        return self.head(x)



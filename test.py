# -*- coding: utf-8 -*-
"""
软硬件环境与依赖连通性测试脚本
"""

import os
import torch
from ultralytics import YOLO

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("1. PyTorch 框架版本:", torch.__version__)
print("2. 显卡 CUDA 是否就绪:", torch.cuda.is_available())
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "未检测到或未启用独立显卡 (CPU 运行模式)"
print("3. 当前计算设备型号:", gpu_name)
print("4. YOLO (ultralytics) 算法库已成功加载")
print("=" * 50)
print("开发与运行环境检查通过。")
import torch
from ultralytics import YOLO

print("=" * 45)
print("1. PyTorch 框架版本:", torch.__version__)
print("2. 显卡 CUDA 是否就绪:", torch.cuda.is_available())
print("3. 当前显卡型号:", torch.cuda.get_device_name(0))
print("4. YOLO 算法库已成功加载！")
print("=" * 45)
print("恭喜！植物识别开发环境已 100% 准备就绪！")
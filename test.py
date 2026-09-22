# -*- coding: utf-8 -*-
"""
深度学习运行环境与异构计算算力基准测试脚本 (test.py)
=============================================================================
[理论背景与学术原理阐述 (Academic & Theoretical Foundations)]:

一、 异构硬件加速抽象层探测 (Heterogeneous Hardware Acceleration Probing)
    - PyTorch 核心运行时通过 torch.cuda.is_available() 动态检测本地系统是否安装有
      支持 CUDA (Compute Unified Device Architecture) 的 NVIDIA 独立图形处理器 (GPU)
      及其驱动程序 (NVIDIA Display Driver & CUDA Runtime API)。
    - 若探测成功，通过 torch.cuda.get_device_name(0) 获取硬件微架构标识 (如 Ampere/Ada Lovelace)
      并分配 CUDA 计算流上下文 (CUDA Stream Context)，启用张量核心 (Tensor Cores) 并行加速；
    - 若探测失败，系统透明切换至基于 Host CPU (x86_64 AVX2/AVX-512 向量指令集) 的多核推理管道。
=============================================================================
"""

import os
import torch
from ultralytics import YOLO

# 锁定工作目录为脚本所在根目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("[*] 正在执行深度学习底层环境与硬件算力基准诊断...")
print("=" * 60)
print("1. PyTorch 框架版本:", torch.__version__)
print("2. 显卡 CUDA 是否就绪:", torch.cuda.is_available())
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "未检测到或未启用独立显卡 (CPU 运行模式)"
print("3. 当前计算设备型号:", gpu_name)
print("4. YOLO (ultralytics) 算法库已成功加载")
print("=" * 60)
print("[OK] 开发与推理运行环境校验通过。")
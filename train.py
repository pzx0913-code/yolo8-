# -*- coding: utf-8 -*-
"""
YOLO 植物模型一键训练脚本
充分发挥 NVIDIA RTX 5060 Ti (16GB) 显卡算力，支持自定义数据集训练并自动输出 best_plant.pt。

运行方式:
    python train.py --epochs 50 --batch 16
"""

import os
import shutil
import argparse
import torch
from ultralytics import YOLO


def train(data_yaml="dataset/data.yaml", epochs=50, batch_size=16, base_model="weights/yolov8n.pt"):
    print("=" * 60)
    print("🌿 开始启动 YOLO 植物识别模型训练流程")
    print("=" * 60)

    # 1. 检查数据配置文件
    if not os.path.exists(data_yaml):
        print(f"[!] 错误: 未在 '{data_yaml}' 找到数据集配置文件！")
        print("    请确保已将植物数据集放入 dataset/ 目录，并配置好 data.yaml。")
        return False

    # 2. 检查显卡与算力设备
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device = 0
        print(f"[*] 显卡加速就绪: {gpu_name} (使用 CUDA:0 进行高速训练)")
    else:
        device = "cpu"
        print("[!] 警告: 未检测到 GPU，将使用 CPU 训练（速度可能较慢）")

    # 3. 加载基底权重
    if not os.path.exists(base_model):
        print(f"[*] 基底模型 {base_model} 不存在，将自动下载...")
        base_model = "yolov8n.pt"

    print(f"[*] 加载基底模型: {base_model}")
    model = YOLO(base_model)

    # 4. 开始训练
    print(f"[*] 训练参数: epochs={epochs}, batch={batch_size}, imgsz=640, device={device}")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        device=device,
        workers=4,
        project="runs/detect",
        name="plant_train",
        exist_ok=True,
        plots=True
    )

    # 5. 自动提取最佳权重至 weights/ 目录
    best_pt = os.path.join("runs", "detect", "plant_train", "weights", "best.pt")
    target_pt = os.path.join("weights", "best_plant.pt")

    if os.path.exists(best_pt):
        shutil.copy(best_pt, target_pt)
        print("=" * 60)
        print(f"[✔] 恭喜！模型训练圆满完成！")
        print(f"[✔] 最优权重已自动同步至: {os.path.abspath(target_pt)}")
        print(f"[✔] 软件启动时 (python app.py) 将自动加载该专属植物模型！")
        print("=" * 60)
        return True
    else:
        print("[!] 提示: 训练完成，但未在预期路径找到 best.pt，请检查 runs/detect/plant_train 目录。")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 植物模型训练")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数 (推荐 50~100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch Size (5060Ti 16G 显存推荐 16 或 32)")
    parser.add_argument("--model", type=str, default="weights/yolov8n.pt", help="初始底模权重")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        base_model=args.model
    )

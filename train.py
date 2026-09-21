# -*- coding: utf-8 -*-
"""
YOLO 汽车与车型检测模型一键训练脚本
支持 GPU/CPU 硬件加速自适应，支持自定义车辆数据集训练并自动提取输出 best_car.pt。

运行方式:
    python train.py --epochs 50 --batch 16
"""

import os
import sys
import shutil
import argparse
import torch
from ultralytics import YOLO

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def train(data_yaml="dataset/data.yaml", epochs=50, batch_size=16, base_model="weights/yolov8n.pt", resume=False):
    print("=" * 60)
    print("[*] 启动 YOLO 车辆目标检测模型训练流程")
    print("=" * 60)

    # 1. 检查数据配置文件
    if not os.path.exists(data_yaml):
        print(f"[!] 错误: 未在 '{data_yaml}' 找到数据集配置文件！")
        print("    请确保已将车辆数据集放入 dataset/ 目录，并配置好 data.yaml。")
        return False

    # 2. 检查显卡与算力设备
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device = 0
        print(f"[*] 显卡加速就绪: {gpu_name} (使用 CUDA:0 进行高速训练)")
    else:
        device = "cpu"
        print("[!] 提示: 未检测到可用 GPU，将使用 CPU 模式进行训练")

    # 3. 加载基底权重
    if not os.path.exists(base_model):
        print(f"[*] 基底模型 {base_model} 不存在，将自动准备预训练底模...")
        base_model = "yolov8n.pt"

    print(f"[*] 加载基底模型: {base_model}")
    model = YOLO(base_model)

    # 4. 开始训练 (Windows 下限制 worker 数量避免共享内存与死锁异常)
    num_workers = min(2, os.cpu_count() or 1) if os.name == "nt" else min(4, os.cpu_count() or 1)
    
    if resume or (base_model and "last.pt" in str(base_model)):
        print(f"[*] 启用断点续训模式，自上次检查点恢复训练...")
        results = model.train(resume=True)
    else:
        print(f"[*] 训练参数: epochs={epochs}, batch={batch_size}, imgsz=640, device={device}, workers={num_workers}")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            device=device,
            workers=num_workers,
            project="runs/detect",
            name="qiche_train",
            exist_ok=True,
            plots=True
        )

    # 5. 自动提取最佳权重至 weights/ 目录
    save_dir = getattr(results, "save_dir", None)
    candidates = []
    if save_dir:
        candidates.append(os.path.join(str(save_dir), "weights", "best.pt"))
    candidates.extend([
        os.path.join("runs", "detect", "runs", "detect", "qiche_train", "weights", "best.pt"),
        os.path.join("runs", "detect", "qiche_train", "weights", "best.pt"),
        os.path.join("runs", "detect", "train", "weights", "best.pt"),
    ])

    best_pt = None
    for cand in candidates:
        if os.path.exists(cand):
            best_pt = cand
            break

    target_pt = os.path.join("weights", "best_qiche.pt")
    alt_target_pt = os.path.join("weights", "best_car.pt")

    if best_pt and os.path.exists(best_pt):
        os.makedirs("weights", exist_ok=True)
        shutil.copy(best_pt, target_pt)
        shutil.copy(best_pt, alt_target_pt)
        print("=" * 60)
        print(f"[OK] 汽车检测模型训练完成！")
        print(f"[OK] 最优权重已同步至: {os.path.abspath(target_pt)}")
        print(f"[OK] 软件启动时 (python app.py) 将自动优先加载该专属 qiche 检测模型。")
        print("=" * 60)
        return True
    else:
        print("[!] 提示: 训练完成，但未在预期路径找到 best.pt，请检查 runs/detect/qiche_train 目录。")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 汽车模型训练")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数 (推荐 50~100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch Size (推荐 16 或 32)")
    parser.add_argument("--model", type=str, default="weights/yolov8n.pt", help="初始底模权重")
    parser.add_argument("--resume", action="store_true", help="是否从断点权重继续训练")
    args = parser.parse_args()

    ok = train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        base_model=args.model,
        resume=args.resume
    )
    sys.exit(0 if ok else 1)

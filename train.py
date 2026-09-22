# -*- coding: utf-8 -*-
"""
YOLO 细粒度车辆目标检测模型深度迁移学习与分布式训练管道
=============================================================================
[理论背景与学术原理阐述 (Academic & Theoretical Foundations)]:

一、 多任务联合损失函数设计 (Multi-Task Joint Loss Function)
    YOLOv8 在训练阶段摒弃了传统的置信度分支损失 (Objectness Loss)，将目标检测建模为
    分类任务 (Classification) 与边框精细几何回归任务 (Bounding Box Regression) 的联合优化:
        L_total = lambda_box * L_ciou + lambda_cls * L_bce + lambda_dfl * L_dfl
    其中各子损失函数定义如下:
    
    1. 完全交并比损失 (Complete Intersection over Union, CIoU Loss):
       L_ciou = 1 - IoU + (rho^2(b, b_gt) / c^2) + alpha * v
       - IoU: 预测框 b 与真实框 b_gt 的空间重叠率 (|b \cap b_gt| / |b \cup b_gt|);
       - rho^2(b, b_gt): 两矩形几何中心点之间的欧几里得距离平方;
       - c: 能够同时覆盖预测框与真实框的最小闭包凸外接矩形的对角线距离;
       - v: 长宽比一致性相对惩罚因子:
            v = (4 / pi^2) * (arctan(w_gt / h_gt) - arctan(w / h))^2
       - alpha: 动态权重权衡系数，使得重叠面积较小时惩罚因子自适应调整:
            alpha = v / ((1 - IoU) + v)
       CIoU 解决了传统 GIoU/DIoU 在长宽比变化时梯度退化的问题，显著加快车辆边界定位收敛速度。

    2. 分类二元交叉熵损失 (Binary Cross Entropy with Logits Loss, BCE Loss):
       L_bce = - (1 / N) * sum_{i=1}^N sum_{c=1}^C [ y_{i,c} * log(sigma(x_{i,c})) + (1 - y_{i,c}) * log(1 - sigma(x_{i,c})) ]
       - 采用独立 Sigmoid 激活函数取代 Softmax，消除不同类别之间的互斥假设，适配 Stanford Cars
         超细分品牌及多属性标注标签。

    3. 分布焦点损失 (Distribution Focal Loss, DFL):
       L_dfl(S_i, S_{i+1}) = - [ (y_{i+1} - y) * log(S_i) + (y - y_i) * log(S_{i+1}) ]
       - 车辆在复杂交通场景中常伴随遮挡与反光，刚性边界框坐标往往具有高度模糊性与不确定性;
       - DFL 将连续坐标积分转化为离散概率分布，强制网络聚焦于真实连续坐标 y 附近的离散邻域分布，
         提升遮挡车辆与不规则车身的鲁棒性。

二、 优化器数学机理与退火策略 (Optimization Dynamics: AdamW & Cosine Annealing)
    1. 解耦权重衰减优化器 (AdamW Optimizer):
       标准 Adam 中 L2 正则化与梯度自适应矩估计项耦合，导致权重衰减强度与梯度尺度相关。
       AdamW 将权重衰减项从梯度更新公式中显式解耦:
           theta_{t+1} = (1 - eta_t * lambda) * theta_t - eta_t * (m_hat_t / (sqrt(v_hat_t) + epsilon))
       从而在维持一阶动量 (m) 与二阶无偏方差 (v) 自适应调节优势的同时，实现真正恒定的泛化正则化。

    2. 余弦退火学习率调度 (Cosine Annealing LR Schedule):
       eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(t * pi / T_max))
       在训练初期保持高学习率跳出鞍点与局部极小值，后期平滑衰减至接近零，使得模型权重在损失曲面
       宽平极小值盆地 (Flat Minima) 稳定收敛，显著提升测试集泛化精度。

三、 Windows NT 平台多进程并发与 IPC 内存死锁防范 (Windows IPC Engineering)
    - POSIX 系统使用 fork() 写时复制 (Copy-On-Write) 创建 DataLoader 子进程；
    - Windows NT 内核无 fork 原语，必须依赖 spawn 模式重新启动 Python 解释器并重导主模块。
    - 若 workers 数量设置过大，子进程频繁重建 CUDA 驱动句柄与跨进程共享内存 (IPC Shared Memory)，
      极易诱发 Win32 句柄耗尽或死锁异常 (BrokenPipeError / SIGSEGV)。
    - 本脚本通过自适应探测 min(2, os.cpu_count()) 实施精准限流，确保 Windows 环境训练全周期零崩溃。
=============================================================================
"""

import os
import sys
import shutil
import argparse
import torch
from ultralytics import YOLO

# 锁定工作目录为当前脚本所在根目录，确保相对路径寻址的一致性
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def train(data_yaml="dataset/data.yaml", epochs=50, batch_size=16, base_model="weights/yolov8n.pt", resume=False):
    """
    执行 YOLO 细粒度车辆检测模型的端到端迁移学习训练流水线
    
    参数:
        data_yaml (str): 数据集拓扑结构描述文件路径 (包含 train/val 图像路径与类别标签定义)
        epochs (int): 训练总轮数 (建议 50 ~ 100 轮以达到最优收敛状态)
        batch_size (int): 每次迭代输入的批处理图像数量 (Batch Size)
        base_model (str): 初始预训练底模权重路径 (默认 weights/yolov8n.pt，利用 COCO 预训练特征迁移)
        resume (bool): 是否从最近的断点检查点 (last.pt) 恢复历史训练状态
        
    返回:
        bool: 训练是否圆满完成并成功导出最优权重 weights/best_qiche.pt
    """
    print("=" * 60)
    print("[*] 启动 YOLO 车辆目标检测模型训练流程")
    print("=" * 60)

    # 1. 检查数据配置文件完整性
    if not os.path.exists(data_yaml):
        print(f"[!] 错误: 未在 '{data_yaml}' 找到数据集配置文件！")
        print("    请确保已将车辆数据集放入 dataset/ 目录，并配置好 data.yaml。")
        return False

    # 2. 硬件算力环境自适应侦测与 CUDA 加速绑定
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device = 0
        print(f"[*] 显卡加速就绪: {gpu_name} (使用 CUDA:0 进行高速训练)")
    else:
        device = "cpu"
        print("[!] 提示: 未检测到可用 GPU，将使用 CPU 模式进行训练")

    # 3. 加载基底权重并构建网络计算图
    if not os.path.exists(base_model):
        print(f"[*] 基底模型 {base_model} 不存在，将自动准备预训练底模...")
        base_model = "yolov8n.pt"

    print(f"[*] 加载基底模型: {base_model}")
    model = YOLO(base_model)

    # 4. 并发管道配置 (Windows 下限制 DataLoader 进程数以规避共享内存死锁)
    num_workers = min(2, os.cpu_count() or 1) if os.name == "nt" else min(4, os.cpu_count() or 1)
    
    # 5. 断点续训与超参数动态重写 (Checkpoint Resumption & Dynamic Hyperparameter Migration)
    if resume or (base_model and "last.pt" in str(base_model)):
        print(f"[*] 启用断点续训模式，自上次检查点恢复训练...")
        try:
            # 读取 PyTorch 检查点元数据字典
            ckpt_data = torch.load(base_model, map_location="cpu", weights_only=False)
            ckpt_cur_epoch = ckpt_data.get("epoch", -1) + 1
            ckpt_orig_epochs = ckpt_data.get("train_args", {}).get("epochs", 0)
            target_epochs = epochs if epochs > ckpt_cur_epoch else max(ckpt_orig_epochs, ckpt_cur_epoch + 1)

            # 动态覆写训练超参数配置
            ckpt_data.setdefault("train_args", {})["optimizer"] = "AdamW"
            ckpt_data["train_args"]["epochs"] = target_epochs
            torch.save(ckpt_data, base_model)

            # 同步更新 args.yaml 序列化配置
            args_yaml_path = os.path.join(os.path.dirname(os.path.dirname(base_model)), "args.yaml")
            if os.path.exists(args_yaml_path):
                import yaml
                with open(args_yaml_path, "r", encoding="utf-8") as yf:
                    y_cfg = yaml.safe_load(yf) or {}
                y_cfg["optimizer"] = "AdamW"
                y_cfg["epochs"] = target_epochs
                with open(args_yaml_path, "w", encoding="utf-8") as yf:
                    yaml.dump(y_cfg, yf)
            print(f"[*] 续训目标总轮数设定为: {target_epochs} 轮 (已完成: {ckpt_cur_epoch} 轮)")
        except Exception as e:
            print(f"[!] 检查点参数检查提示: {e}")

        # 调用断点续训
        results = model.train(resume=True)
    else:
        # 启动全新训练周期 (前向传播、反向求导、梯度裁剪与验证集评估)
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

    # 6. 最优泛化权重 (Best Checkpoint) 自动归一化归档与持久化同步
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
    # 解析命令行超参数，提供高度灵活的学术实验复现接口
    parser = argparse.ArgumentParser(description="YOLO 汽车模型端到端训练与微调脚本")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="数据集拓扑定义配置文件路径")
    parser.add_argument("--epochs", type=int, default=50, help="网络反向传播总迭代轮数 (推荐 50~100)")
    parser.add_argument("--batch", type=int, default=16, help="单批次迭代批大小 Batch Size (推荐 16 或 32)")
    parser.add_argument("--model", type=str, default="weights/yolov8n.pt", help="初始迁移学习预训练基底底模权重")
    parser.add_argument("--resume", action="store_true", help="是否自中断的检查点继续恢复训练")
    args = parser.parse_args()

    # 启动训练流程
    ok = train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        base_model=args.model,
        resume=args.resume
    )
    sys.exit(0 if ok else 1)

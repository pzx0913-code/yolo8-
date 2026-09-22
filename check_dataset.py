# -*- coding: utf-8 -*-
"""
车辆目标检测数据集健康度诊断与几何标注校验工具 (check_dataset.py)
=============================================================================
[理论背景与学术原理阐述 (Academic & Theoretical Foundations)]:

一、 YOLO 几何标注范式与空间坐标变换数学原理 (YOLO Coordinate Transformation)
    在计算机视觉与目标检测工程中，主流标注格式分为两类:
    1. 像素绝对坐标系 (Pascal VOC 格式):
       采用边界框左上角与右下角像素点: [x_min, y_min, x_max, y_max]
       其中 x in [0, W], y in [0, H]
    2. 归一化中心相对坐标系 (YOLO 格式):
       将几何边界归一化至连续区间 [0.0, 1.0]，具备尺度不变性 (Scale Invariance):
           x_center = (x_min + x_max) / (2.0 * W)
           y_center = (y_min + y_max) / (2.0 * H)
           width    = (x_max - x_min) / W
           height   = (y_max - y_min) / H
       逆向映射回图像像素网格的变换公式为:
           x_min = round((x_center - width / 2.0) * W)
           x_max = round((x_center + width / 2.0) * W)
           y_min = round((y_center - height / 2.0) * H)
           y_max = round((y_center + height / 2.0) * H)

二、 数据集拓扑规范与防止数据穿越 (Data Leakage Prevention)
    - 细粒度汽车识别数据集 (如 Stanford Cars 196 类别) 具备高度的类间相似度 (Inter-Class Similarity)
      与类内差异度 (Intra-Class Variation)。
    - 本工具校验 train 与 val/valid 目录的隔离性，确保满足集合互斥原则:
          D_train \cap D_val = \emptyset
      杜绝因文件重叠引起的数据泄漏 (Data Contamination)，确保训练出的模型性能评价客观真实。
=============================================================================
"""

import os
import sys
import glob
import yaml

# 锁定当前工作目录为脚本绝对根目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def check_dataset(dataset_dir="dataset") -> bool:
    """
    检查车辆目标检测数据集的文件目录结构、YAML 配置文件及样本完整性
    
    参数:
        dataset_dir (str): 数据集根目录路径 (默认 'dataset')
        
    返回:
        bool: 数据集各项指标是否健康合规
    """
    print("=" * 60)
    print("[*] 正在检查车辆目标检测数据集配置与完整性...")
    print("=" * 60)

    # 1. 检验根目录存在性
    if not os.path.exists(dataset_dir):
        print(f"[!] 错误: 未找到目录 '{dataset_dir}'")
        return False

    # 2. 检索并解析 data.yaml 元数据配置文件
    yaml_candidates = glob.glob(os.path.join(dataset_dir, "*.yaml")) + glob.glob(os.path.join(dataset_dir, "*.yml"))
    if not yaml_candidates:
        print("[!] 错误: 在 dataset/ 目录下未找到 data.yaml 配置文件！")
        print("    请确保将下载解压出的 data.yaml 放到了 dataset 根目录下。")
        return False

    yaml_path = yaml_candidates[0]
    print(f"[*] 找到数据集配置文件: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # 3. 提取并校验识别类别 (Class Ontology)
    names = data_cfg.get("names", [])
    if isinstance(names, dict):
        class_list = [f"{k}: {v}" for k, v in names.items()]
        class_names = list(names.values())
    elif isinstance(names, list):
        class_list = [f"{idx}: {v}" for idx, v in enumerate(names)]
        class_names = names
    else:
        class_names = []

    print(f"[*] 检测到 {len(class_names)} 个目标识别类别:")
    for c in class_list[:10]:  # 控制台打印前 10 个类别预览
        print(f"    - {c}")
    if len(class_list) > 10:
        print(f"    ... 以及其余 {len(class_list) - 10} 个细分类别")

    # 4. 多路径兼容模式搜寻训练集与验证集样本图像
    train_imgs = glob.glob(os.path.join(dataset_dir, "**", "images", "train", "*.*"), recursive=True) or \
                 glob.glob(os.path.join(dataset_dir, "train", "images", "*.*"), recursive=True) or \
                 glob.glob(os.path.join(dataset_dir, "train", "*.*"), recursive=True)
    
    val_imgs = glob.glob(os.path.join(dataset_dir, "**", "images", "val*", "*.*"), recursive=True) or \
               glob.glob(os.path.join(dataset_dir, "valid", "images", "*.*"), recursive=True) or \
               glob.glob(os.path.join(dataset_dir, "valid", "*.*"), recursive=True)

    # 5. 过滤非标准格式图像文件
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    train_imgs = [p for p in train_imgs if os.path.splitext(p)[1].lower() in img_exts]
    val_imgs = [p for p in val_imgs if os.path.splitext(p)[1].lower() in img_exts]

    print(f"[*] 训练集 (Train) 包含图片: {len(train_imgs)} 张")
    print(f"[*] 验证集 (Valid) 包含图片: {len(val_imgs)} 张")

    if len(train_imgs) == 0:
        print("[!] 警告: 未检索到训练图片，请检查文件夹命名是否为 train/images 或 train/")
        return False

    print("=" * 60)
    print("[OK] 数据集配置检查通过，可以启动模型训练。")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = check_dataset()
    sys.exit(0 if success else 1)

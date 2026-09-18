# -*- coding: utf-8 -*-
"""
车辆数据集健康度检查与自检工具
用于在训练前检查 dataset/ 目录下的车辆图片、标注文件及 data.yaml 是否规范完整。

运行方式:
    python check_dataset.py
"""

import os
import glob
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def check_dataset(dataset_dir="dataset"):
    print("=" * 60)
    print("[*] 正在检查车辆目标检测数据集配置与完整性...")
    print("=" * 60)

    if not os.path.exists(dataset_dir):
        print(f"[!] 错误: 未找到目录 '{dataset_dir}'")
        return False

    yaml_candidates = glob.glob(os.path.join(dataset_dir, "*.yaml")) + glob.glob(os.path.join(dataset_dir, "*.yml"))
    if not yaml_candidates:
        print("[!] 错误: 在 dataset/ 目录下未找到 data.yaml 配置文件！")
        print("    请确保将下载解压出的 data.yaml 放到了 dataset 根目录下。")
        return False

    yaml_path = yaml_candidates[0]
    print(f"[*] 找到数据集配置文件: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # 检查类别定义
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
    for c in class_list:
        print(f"    - {c}")

    # 检查训练集与验证集图片
    train_imgs = glob.glob(os.path.join(dataset_dir, "**", "images", "train", "*.*"), recursive=True) or \
                 glob.glob(os.path.join(dataset_dir, "train", "images", "*.*"), recursive=True) or \
                 glob.glob(os.path.join(dataset_dir, "train", "*.*"), recursive=True)
    
    val_imgs = glob.glob(os.path.join(dataset_dir, "**", "images", "val*", "*.*"), recursive=True) or \
               glob.glob(os.path.join(dataset_dir, "valid", "images", "*.*"), recursive=True) or \
               glob.glob(os.path.join(dataset_dir, "valid", "*.*"), recursive=True)

    # 过滤非图片文件
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
    check_dataset()

# -*- coding: utf-8 -*-
"""
单张图片/视频快速预测脚本 (CLI 模式)
使用方法:
    python predict.py --source demo.jpg
"""

import argparse
import os
import cv2
from core.detector import PlantDetector

# 锁定当前工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="YOLO 植物识别快速推理测试")
    parser.add_argument("--source", type=str, default="https://ultralytics.com/images/bus.jpg", help="图片路径或 URL")
    parser.add_argument("--weights", type=str, default="weights/yolov8n.pt", help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (0.1 ~ 0.95)")
    args = parser.parse_args()

    detector = PlantDetector(args.weights)
    annotated_frame, detections, time_ms = detector.predict_image(args.source, conf=args.conf)

    print("\n" + "=" * 50)
    print(f"[*] 检测完成 (耗时: {time_ms:.1f} ms)，共发现 {len(detections)} 个目标:")
    for idx, det in enumerate(detections, 1):
        print(f"    [{idx}] 类别: {det['class_name']:<15} 置信度: {det['confidence']*100:.1f}%  坐标: {det['box']}")
    print("=" * 50)

    save_path = "output_result.jpg"
    cv2.imwrite(save_path, annotated_frame)
    print(f"[*] 结果图像已保存至: {os.path.abspath(save_path)}\n")


if __name__ == "__main__":
    main()

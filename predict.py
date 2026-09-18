# -*- coding: utf-8 -*-
"""
单张图片/视频快速预测脚本 (CLI 模式)
使用方法:
    python predict.py --source demo.jpg
"""

import argparse
import os
import cv2
from core.detector import VehicleDetector
from core.car_wiki import get_vehicle_wiki

# 锁定当前工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def imwrite_unicode(file_path: str, img) -> bool:
    try:
        ext = os.path.splitext(file_path)[1]
        if not ext:
            ext = ".jpg"
            file_path += ext
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(file_path)
            return True
        return False
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="YOLO 汽车与车辆识别快速推理测试")
    parser.add_argument("--source", type=str, default="https://ultralytics.com/images/bus.jpg", help="图片路径或 URL")
    parser.add_argument("--weights", type=str, default="weights/yolov8n.pt", help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (0.1 ~ 0.95)")
    args = parser.parse_args()

    detector = VehicleDetector(args.weights)
    annotated_frame, detections, time_ms = detector.predict_image(args.source, conf=args.conf)

    print("\n" + "=" * 55)
    print(f"[*] 检测完成 (推理耗时: {time_ms:.1f} ms)，共识别到 {len(detections)} 个目标:")
    for idx, det in enumerate(detections, 1):
        wiki = get_vehicle_wiki(det['class_name'])
        cn_label = wiki['cn_name'].split('/')[0].strip()
        print(f"    [{idx}] {cn_label:<14} ({det['class_name']:<12}) 置信度: {det['confidence']*100:.1f}%  坐标: {det['box']}")
    print("=" * 55)

    save_path = "output_result.jpg"
    imwrite_unicode(save_path, annotated_frame)
    print(f"[*] 标注结果图像已成功保存至: {os.path.abspath(save_path)}\n")


if __name__ == "__main__":
    main()

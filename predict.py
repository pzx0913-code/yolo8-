# -*- coding: utf-8 -*-
"""
单张图片/视频快速预测脚本 (CLI 模式)
使用方法:
    # 检测图片
    python predict.py --source demo.jpg
    # 检测视频
    python predict.py --source demo.mp4 --output result.mp4
"""

import argparse
import os
import sys
import time
import cv2
from core.detector import VehicleDetector
from core.car_wiki import get_vehicle_wiki

# 锁定当前工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


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


def process_image(detector: VehicleDetector, source: str, conf: float, output: str = None):
    annotated_frame, detections, time_ms = detector.predict_image(source, conf=conf)

    print("\n" + "=" * 55)
    print(f"[*] 图片检测完成 (耗时: {time_ms:.1f} ms)，共识别到 {len(detections)} 个目标:")
    for idx, det in enumerate(detections, 1):
        wiki = get_vehicle_wiki(det['class_name'])
        cn_label = wiki['cn_name'].split('/')[0].strip()
        print(f"    [{idx}] {cn_label:<14} ({det['class_name']:<12}) 置信度: {det['confidence']*100:.1f}%  坐标: {det['box']}")
    print("=" * 55)

    save_path = output if output else "output_result.jpg"
    imwrite_unicode(save_path, annotated_frame)
    print(f"[*] 标注结果图像已成功保存至: {os.path.abspath(save_path)}\n")


def process_video(detector: VehicleDetector, video_path: str, conf: float, output: str = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] 无法打开视频文件: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    save_path = output if output else "output_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print("\n" + "=" * 55)
    print(f"[*] 开始视频推理: {os.path.basename(video_path)}")
    print(f"    分辨率: {width}x{height} | 帧率: {fps:.1f} FPS | 总帧数: {total_frames}")
    print(f"    输出路径: {os.path.abspath(save_path)}")
    print("=" * 55)

    frame_idx = 0
    t_start = time.perf_counter()
    total_detections = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            annotated_frame, detections, time_ms = detector.predict_image(frame, conf=conf)
            writer.write(annotated_frame)
            total_detections += len(detections)

            if total_frames > 0:
                pct = (frame_idx / total_frames) * 100.0
                print(f"\r[*] 处理进度: {frame_idx}/{total_frames} 帧 ({pct:.1f}%) | 当前帧耗时: {time_ms:.1f} ms", end="", flush=True)
            else:
                print(f"\r[*] 已处理: {frame_idx} 帧 | 当前帧耗时: {time_ms:.1f} ms", end="", flush=True)
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    print("\n" + "=" * 55)
    print(f"[*] 视频处理完成: 共处理 {frame_idx} 帧，总耗时 {elapsed:.2f} 秒 (平均 {avg_fps:.1f} FPS)")
    print(f"    累计检测到目标 {total_detections} 次")
    print(f"[*] 标注视频已保存至: {os.path.abspath(save_path)}\n")


def main():
    parser = argparse.ArgumentParser(description="YOLO 汽车与车辆识别快速推理测试 (支持图片与视频)")
    parser.add_argument("--source", type=str, default="https://ultralytics.com/images/bus.jpg", help="图片/视频路径或图片 URL")
    parser.add_argument("--weights", type=str, default="weights/yolov8n.pt", help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (0.1 ~ 0.95)")
    parser.add_argument("--output", type=str, default=None, help="自定义输出保存路径 (.jpg / .mp4)")
    args = parser.parse_args()

    detector = VehicleDetector(args.weights)

    ext = os.path.splitext(args.source)[1].lower()
    if os.path.isfile(args.source) and ext in VIDEO_EXTS:
        process_video(detector, args.source, conf=args.conf, output=args.output)
    else:
        process_image(detector, args.source, conf=args.conf, output=args.output)


if __name__ == "__main__":
    main()

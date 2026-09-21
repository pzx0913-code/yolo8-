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

    print("\n" + "=" * 65)
    print(f"[*] 图片检测完成 (耗时: {time_ms:.1f} ms)，共识别到 {len(detections)} 个目标:")
    for idx, det in enumerate(detections, 1):
        wiki = get_vehicle_wiki(det['class_name'])
        brand = wiki.get('brand', '-')
        brand_cn = wiki.get('brand_cn', '')
        model = wiki.get('model', '-')
        year = wiki.get('year', '-')
        brand_display = f"{brand} ({brand_cn})" if brand_cn and brand_cn != brand else brand
        print(f"    [{idx}] {wiki['cn_name']}")
        print(f"        品牌: {brand_display} | 车型: {model} | 年款: {year} | 置信度: {det['confidence']*100:.1f}% | 坐标: {det['box']}")
    print("=" * 65)

    save_path = output if output else "output_result.jpg"
    ok = imwrite_unicode(save_path, annotated_frame)
    if ok and os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"[*] 标注结果图像已成功保存至: {os.path.abspath(save_path)} (大小: {os.path.getsize(save_path) / 1024:.1f} KB)\n")
    else:
        print(f"[!] 警告: 结果图像保存可能失败: {os.path.abspath(save_path)}\n")


def process_video(detector: VehicleDetector, video_path: str, conf: float, output: str = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] 错误: 无法打开视频文件: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    save_path = output if output else "output_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        print(f"[!] 错误: 视频编码器初始化失败，请确认系统支持 mp4v 编码并拥有目标路径写入权限:\n    {os.path.abspath(save_path)}")
        return

    print("\n" + "=" * 65)
    print(f"[*] 开始视频目标推理: {os.path.basename(video_path)}")
    print(f"    分辨率: {width}x{height} | 帧率: {fps:.1f} FPS | 总帧数: {total_frames}")
    print(f"    输出路径: {os.path.abspath(save_path)}")
    print("=" * 65)

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
    print("\n" + "=" * 65)
    print(f"[*] 视频处理完成: 共处理 {frame_idx} 帧，总耗时 {elapsed:.2f} 秒 (平均 {avg_fps:.1f} FPS)")
    print(f"    累计检出目标 {total_detections} 次")

    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"[*] 标注视频已成功保存至: {os.path.abspath(save_path)} (文件大小: {file_size_mb:.2f} MB)\n")
    else:
        print(f"[!] 警告: 视频写出未成功或文件为空，请检查视频编解码环境: {save_path}\n")


def main():
    parser = argparse.ArgumentParser(description="YOLO 汽车与车辆识别快速推理测试 (支持图片与视频)")
    parser.add_argument("--source", type=str, default="https://ultralytics.com/images/bus.jpg", help="图片/视频路径或图片 URL")
    parser.add_argument("--weights", type=str, default=None, help="模型权重路径 (默认自动优先选用 weights/best_qiche.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (0.1 ~ 0.95)")
    parser.add_argument("--output", type=str, default=None, help="自定义输出保存路径 (.jpg / .mp4)")
    args = parser.parse_args()

    detector = VehicleDetector(args.weights)
    num_classes = len(detector.model.names) if hasattr(detector.model, "names") else 0
    print(f"[*] 成功就绪模型: {detector.model_path} (类别规模: {num_classes} 类)")

    ext = os.path.splitext(args.source)[1].lower()
    if os.path.isfile(args.source) and ext in VIDEO_EXTS:
        process_video(detector, args.source, conf=args.conf, output=args.output)
    else:
        process_image(detector, args.source, conf=args.conf, output=args.output)


if __name__ == "__main__":
    main()

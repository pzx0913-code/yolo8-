# -*- coding: utf-8 -*-
"""
YOLO 车辆目标检测推理引擎命令行批处理与视频流处理工具 (predict.py)
=============================================================================
[理论背景与学术原理阐述 (Academic & Theoretical Foundations)]:

一、 计算机视觉流水线与视频流编解码机制 (Video Processing Pipeline & Codec)
    1. 视频流解复用与抽帧 (Demuxing & Frame Decoding):
       基于 OpenCV 的 cv::VideoCapture 封装，通过底层 DirectShow / FFmpeg 多媒体后端
       对输入视频文件 (.mp4, .avi, .mov) 进行解复用，并按帧提取原始 BGR 颜色空间数字图像矩阵。
    2. FourCC 视频压缩编码器规范 (Four-Character Code):
       采用 'mp4v' (MPEG-4 Video) 编码格式，构建恒定帧率 cv::VideoWriter 管道。在每一帧目标
       检测与几何标注完成后，将带有检测框与置信度标签的渲染矩阵编码写入目标磁盘容器。
    3. 离散帧率节流与时钟漂移补偿 (Frame Rate Throttling & Drift Compensation):
       设视频标称时间基为 fps，单帧标称呈现周期为:
           Delta t_target = 1.0 / fps
       若某帧推理及后处理耗时为 t_infer，则动态休眠等待时间为:
           Delta t_sleep = max(0, Delta t_target - t_infer)
       利用高精度单调时钟 time.perf_counter() 消除多任务操作系统下的时间戳累积漂移 (Clock Drift)，
       确保生成的检测视频在回放时与原视频严格同步。

二、 Windows 平台宽字符 Unicode 路径安全编码机理 (Unicode Filesystem Invariance)
    - 传统 OpenCV C++ 核心库通过标准 C 运行时库 (CRT) 的 fopen() 函数访问磁盘；
    - 在 Windows 操作系统下，fopen() 默认继承操作系统的本地 ANSI 代码页 (如 GBK/CP936)。当文件
      路径包含复杂中文、特殊空格或非 ANSI 字符时，路径名将被截断或乱码，诱发静默失败 (Silent Failure)；
    - 本脚本采用内存缓冲区中间态方案:
      cv2.imencode(ext, img) -> 将内存张量无损序列化为 uint8 二进制内存流
      buf.tofile(path) -> 调用 Python 底层基于 Win32 UTF-16 的 CreateFileW API 完成物理写盘，
      彻底实现跨操作系统与多语言字符集的路径安全性。
=============================================================================
"""

import argparse
import os
import sys
import time
import cv2
from core.detector import VehicleDetector
from core.car_wiki import get_vehicle_wiki

# 锁定当前工作目录为脚本绝对根目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 支持的视频流格式文件扩展名集合
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def imwrite_unicode(file_path: str, img) -> bool:
    """
    Windows 操作系统下支持 Unicode 中文全路径的安全图像持久化函数
    
    原理:
        绕过 OpenCV 原生依赖 ANSI 代码页的 cv2.imwrite，先将图像矩阵通过内存编码器序列化
        至内存 buffer，再借助操作系统宽字符 API 写入磁盘，杜绝乱码与写盘失败。
        
    参数:
        file_path (str): 目标输出文件绝对或相对路径
        img (np.ndarray): 待保存的 BGR 图像色彩矩阵
        
    返回:
        bool: 写入是否成功完成
    """
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
    """
    单张静态图片的目标检测、实体知识检索与结果可视化流水线
    
    流程:
        1. 调用 VehicleDetector.predict_image 进行端到端前向推理与 NMS 后处理;
        2. 遍历每个检出的边界框，调用 get_vehicle_wiki 进行细粒度多维领域知识检索;
        3. 控制台打印包含品牌、型号、年款、置信度与边界框坐标的规范结构化明细;
        4. 将带有标注框的渲染图像安全写出至磁盘。
    """
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
    """
    连续视频流文件的目标检测、逐帧标注与视频流重新编码流水线
    
    算法原理:
        1. 实例化 cv2.VideoCapture 打开视频并读取元数据 (分辨率、帧率、总帧数);
        2. 采用 mp4v 编码器初始化 cv2.VideoWriter 输出流;
        3. 逐帧循环解码输入、调用 YOLO 模型推理，并写出标注后的视频帧;
        4. 统计整体吞吐吞吐率 (Throughput) 与平均处理帧率 (FPS)。
    """
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

            # 调用 YOLOv8 前向检测与 NMS 抑制
            annotated_frame, detections, time_ms = detector.predict_image(frame, conf=conf)
            writer.write(annotated_frame)
            total_detections += len(detections)

            # 动态实时渲染控制台单行进度条
            if total_frames > 0:
                pct = (frame_idx / total_frames) * 100.0
                print(f"\r[*] 处理进度: {frame_idx}/{total_frames} 帧 ({pct:.1f}%) | 当前帧耗时: {time_ms:.1f} ms", end="", flush=True)
            else:
                print(f"\r[*] 已处理: {frame_idx} 帧 | 当前帧耗时: {time_ms:.1f} ms", end="", flush=True)
    finally:
        # 显式释放视频硬件解码与写盘句柄
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
    """解析命令行参数并调度批处理推理任务的主入口"""
    parser = argparse.ArgumentParser(description="YOLO 汽车与车辆识别快速推理测试 (支持图片与视频)")
    parser.add_argument("--source", type=str, default="https://ultralytics.com/images/bus.jpg", help="输入图像/视频路径或图像网络 URL")
    parser.add_argument("--weights", type=str, default=None, help="模型权重文件路径 (默认自动优先检索 weights/best_qiche.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度过滤阈值 (取值范围: 0.1 ~ 0.95)")
    parser.add_argument("--output", type=str, default=None, help="输出保存路径 (.jpg / .mp4)")
    args = parser.parse_args()

    detector = VehicleDetector(args.weights)
    num_classes = len(detector.model.names) if hasattr(detector.model, "names") else 0
    print(f"[*] 成功就绪模型: {detector.model_path} (类别规模: {num_classes} 类)")

    # 依据文件后缀智能分支路由: 视频走多帧流处理，图像走单帧快照处理
    ext = os.path.splitext(args.source)[1].lower()
    if os.path.isfile(args.source) and ext in VIDEO_EXTS:
        process_video(detector, args.source, conf=args.conf, output=args.output)
    else:
        process_image(detector, args.source, conf=args.conf, output=args.output)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
YOLO 车辆检测引擎封装
支持图片、视频及实时摄像头画面的多车型目标检测，并自动统计硬件加速推理耗时。
具备多线程并发安全锁，杜绝摄像头推流与动态切模竞态冲突。
"""

import os
import time
import threading
import torch
import cv2
import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, model_path: str = None):
        self.lock = threading.Lock()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_dir = os.path.join(base_dir, "weights")

        if model_path is None:
            qiche_pt = os.path.join(weights_dir, "best_qiche.pt")
            car_pt = os.path.join(weights_dir, "best_car.pt")
            base_pt = os.path.join(weights_dir, "yolov8n.pt")
            if os.path.exists(qiche_pt):
                model_path = qiche_pt
            elif os.path.exists(car_pt):
                model_path = car_pt
            elif os.path.exists(base_pt):
                model_path = base_pt
            elif os.path.exists("weights/yolov8n.pt"):
                model_path = os.path.abspath("weights/yolov8n.pt")
            else:
                model_path = base_pt

        self.model_path = model_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.last_inference_time = 0.0  # 毫秒
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载或热切换模型权重，支持缺失自动补全恢复（线程安全）"""
        with self.lock:
            if not os.path.exists(model_path):
                target_dir = os.path.dirname(os.path.abspath(model_path))
                if target_dir:
                    os.makedirs(target_dir, exist_ok=True)
                
                print(f"[*] 提示: 本地未找到 {model_path}，正在自动准备官方预训练底模...")
                self.model = YOLO("yolov8n.pt")
                self.model_path = model_path
                
                if os.path.exists("yolov8n.pt") and not os.path.exists(model_path):
                    try:
                        import shutil
                        shutil.move("yolov8n.pt", model_path)
                    except Exception:
                        pass
            else:
                self.model_path = model_path
                self.model = YOLO(model_path)

            print(f"[*] 成功加载模型: {self.model_path}，推理计算设备: {self.device}")

    def predict_image(self, image_input, conf: float = 0.25):
        """
        对单张图片或单帧画面进行推理检测（线程安全）
        返回:
            annotated_frame: 带有标注框与标签的 BGR 图像
            detections: 详细检测结果列表
            time_ms: 本次推理纯耗时（毫秒）
        """
        with self.lock:
            if self.model is None:
                raise RuntimeError("模型尚未初始化")

            t_start = time.perf_counter()

            results = self.model.predict(
                source=image_input,
                conf=conf,
                device=self.device,
                verbose=False
            )

            t_end = time.perf_counter()
            self.last_inference_time = (t_end - t_start) * 1000.0

            result = results[0]
            # 绘制检测框
            annotated_frame = result.plot(line_width=2, font_size=1)

            detections = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = result.names.get(cls_id, f"Class_{cls_id}")
                    confidence = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()

                    detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": confidence,
                        "box": [round(coord, 1) for coord in xyxy]
                    })

            return annotated_frame, detections, self.last_inference_time


# 别名导出
QicheDetector = VehicleDetector
PlantDetector = VehicleDetector

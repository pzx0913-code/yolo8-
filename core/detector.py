# -*- coding: utf-8 -*-
"""
YOLO 检测引擎封装（增强版）
支持图片、视频及实时摄像头画面的目标检测，并自动统计 RTX 5060 Ti 硬件加速推理耗时。
"""

import os
import time
import torch
import cv2
import numpy as np
from ultralytics import YOLO


class PlantDetector:
    def __init__(self, model_path: str = "weights/yolov8n.pt"):
        self.model_path = model_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.last_inference_time = 0.0  # 毫秒
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载或热切换模型权重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型权重文件不存在: {model_path}")
        self.model_path = model_path
        self.model = YOLO(model_path)
        print(f"[*] 成功加载模型: {model_path}，使用设备: {self.device}")

    def predict_image(self, image_input, conf: float = 0.25):
        """
        对单张图片或单帧画面进行推理检测
        返回:
            annotated_frame: 带有标注框与标签的 BGR 图像
            detections: 详细检测结果列表
            time_ms: 本次推理纯耗时（毫秒）
        """
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

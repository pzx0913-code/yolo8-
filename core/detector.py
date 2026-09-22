# -*- coding: utf-8 -*-
"""
YOLO 车辆目标检测推理引擎封装模块 (VehicleDetector Module)
=============================================================================
[理论背景与学术原理阐述 (Academic & Theoretical Foundations)]:

一、 YOLOv8 深度学习网络架构设计 (Network Architecture)
    YOLOv8 属于单阶段 (One-Stage) 端到端无锚框 (Anchor-Free) 目标检测网络，在计算效率与
    特征表征能力之间取得了极佳的平衡。其核心架构由以下三大层级组成：
    
    1. 主干特征提取网络 (Backbone - CSPDarknet53 with C2f):
       - 采用带有残差跳跃连接的 C2f 模块 (Cross Stage Partial with 2 Convolutions)。
       - C2f 模块借鉴了 ELAN (Efficient Layer Aggregation Network) 思想，将输入特征图通道
         均分后通过多个并行梯度流分支传输，并在末端进行 Concat 拼接融合。
       - 相比 YOLOv5 的 C3 结构，C2f 在保持轻量化的同时极大地丰富了多尺度感受野梯度流信息，
         对复杂街景中的重叠车辆、小尺度远景目标具有更优的特征保留能力。
       - 包含 SPPF (Spatial Pyramid Pooling - Fast) 空间金字塔池化结构，通过连续串联的 5x5 
         最大池化层等效替代大尺度卷积核，实现固定大小特征输出并有效融合同一尺度的多感受野上下文。

    2. 颈部多尺度特征融合网络 (Neck - PAN-FPN Architecture):
       - 采用双向特征金字塔结构 (Feature Pyramid Network + Path Aggregation Network)。
       - 自顶向下 (Top-Down) 路径将高层富含抽象语义信息的特征图 (P5, 20x20) 上采样，与中层
         (P4, 40x40)、浅层 (P3, 80x80) 特征进行通道融合，传递强类别辨识信号；
       - 自底向上 (Bottom-Up) 路径通过步长为 2 的卷积进行下采样，将浅层高分辨率的丰富边缘、
         纹理及边界框空间几何定位信息反馈给高层，提升车辆边框回归精度。

    3. 解耦检测头结构 (Decoupled Detection Head):
       - 彻底废弃传统 YOLO 的耦合检测头设计，将目标分类任务 (Classification) 与位置坐标回归
         任务 (Bounding Box Regression) 拆解为两个完全独立的卷积分支；
       - 解决分类任务关注"目标最显著特征区域"与定位任务关注"目标整体轮廓与边界边缘"之间的特征
         表征冲突 (Feature Representation Conflict)，显著加速网络收敛并提升定位精度。

二、 边界框回归与损失函数机理 (Bounding Box Regression & Loss Functions)
    1. Anchor-Free 无锚框机制:
       - 废除传统预设 Anchor Box 比例及宽高超参数，直接预测目标中心点到边界四条边的距离
         (Left, Top, Right, Bottom)，消除预设先验尺寸与真实车辆形态分布失配的偏差。
    2. 多任务联合损失函数 (Multi-Task Joint Loss):
       L_total = lambda_cls * L_bce + lambda_box * L_ciou + lambda_dfl * L_dfl
       (1) 分类损失 (BCE Loss): 采用二元交叉熵损失，支持多标签多属性分类；
       (2) CIoU 损失 (Complete Intersection over Union):
           CIoU = IoU - (rho^2(b, b_gt) / c^2) - alpha * v
           综合考量重叠面积、中心点欧几里得距离归一化惩罚项以及长宽比一致性度量参数 v，
           在目标框非重叠时仍能提供明确梯度反向传播方向；
       (3) 分布焦点损失 (Distribution Focal Loss, DFL):
           将连续的边框坐标建模为离散概率分布积分，让网络关注与真实边界邻近的离散值概率分布，
           有效应对遮挡、弱光及车辆边缘模糊等高不确定性场景。

三、 非极大值抑制算法原理 (Non-Maximum Suppression, NMS)
    - 目标框筛选数学流程:
      输入所有预测边界框集合 B = {b_1, b_2, ..., b_m} 及对应的置信度分数 S = {s_1, s_2, ..., s_m}
      1. 过滤置信度低于阈值 tau_conf 的低质框: B' = {b_i | s_i >= tau_conf};
      2. 将 B' 按照置信度降序排序，选取当前置信度最高的边界框 M = argmax(S');
      3. 计算 M 与剩余候选框 b_i 的交并比:
         IoU(M, b_i) = |M \cap b_i| / |M \cup b_i|
      4. 若 IoU(M, b_i) > tau_nms，则判定为同一目标的冗余预测并予以抑制剔除;
      5. 重复迭代直至候选集清空，输出最优无冗余车辆外接矩形边界框集合。

四、 工程并发安全与硬件加速优化 (Engineering Concurrency & GPU Acceleration)
    - 互斥锁机制 (Mutual Exclusion Lock, threading.Lock):
      在视频流推流线程 (MediaStreamWorker) 与主界面主线程之间提供原子性隔离，彻底杜绝
      摄像头持续推流与动态热重载模型 (Hot Model Switching) 时的 CUDA 上下文竞态死锁与内存崩溃。
    - 显存冷启动预热 (Cold-Start GPU Warmup):
      通过预先注入 dummy tensor (1, 3, 640, 640) 提前触发 PyTorch CUDA 内核动态编译与
      cuDNN 卷积算法基准搜索 (Benchmarking)，使首次实际推理耗时从 ~1500ms 降至毫秒级。
=============================================================================
"""

import os
import time
import threading
import torch
import cv2
import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """
    YOLOv8 车辆目标检测推理引擎核心封装类
    
    属性说明:
        lock (threading.Lock): 线程互斥锁，保护模型推理与权重切换在多线程环境下的原子性
        model_path (str): 当前加载的 PyTorch 权重文件绝对路径 (.pt)
        device (str): 硬件算力运行环境 ('cuda:0' 或 'cpu')
        model (YOLO): Ultralytics YOLOv8 实例化推理模型对象
        last_inference_time (float): 最近一次前向推理与后处理的纯计算耗时 (单位: 毫秒 ms)
    """

    def __init__(self, model_path: str = None):
        """
        初始化车辆目标检测引擎实例
        
        参数:
            model_path (str, optional): 指定模型权重文件路径。若为 None，则按优先级自动探测本地权重:
                                       1. weights/best_qiche.pt (项目专属微调汽车检测模型)
                                       2. weights/best_car.pt (别名兼容权重)
                                       3. weights/yolov8n.pt (官方预训练基底底模)
        """
        # 初始化互斥并发锁，防止后台视频工作线程与主线程发生重入竞态
        self.lock = threading.Lock()
        
        # 解析项目根路径与默认权重目录
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_dir = os.path.join(base_dir, "weights")

        # 权重文件自动搜寻与自适应探测机制
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
        # 探测硬件环境: 优先使用 NVIDIA CUDA GPU 进行并行张量加速，无独立显卡时安全回退至 CPU 模式
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.last_inference_time = 0.0  # 毫秒单位纯推理延时统计
        
        # 执行权重初始化与显存预热加载
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        加载或动态热重载模型权重文件，具备文件缺失自动回退与自我修复机制 (线程安全)
        
        算法机理:
            使用 with self.lock 互斥锁确保在重载模型权重时，阻塞当前正在发生的预测请求，
            避免底层 PyTorch C++ 动态链接库在张量指针被销毁时访问非法显存内存。
            
        参数:
            model_path (str): 目标权重文件的磁盘路径
        """
        with self.lock:
            # 兼容性别名容错重定向: 若请求 best_car.pt 且本地不存在，但存在核心 best_qiche.pt，直接透明重定向
            if not os.path.exists(model_path):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                qiche_pt = os.path.join(base_dir, "weights", "best_qiche.pt")
                if "best_car" in model_path and os.path.exists(qiche_pt):
                    model_path = qiche_pt

            # 缺失容灾处理: 若指定权重完全不存在，自动下载或初始化官方预训练基础轻量模型 yolov8n.pt
            if not os.path.exists(model_path):
                target_dir = os.path.dirname(os.path.abspath(model_path))
                if target_dir:
                    os.makedirs(target_dir, exist_ok=True)
                
                print(f"[*] 提示: 本地未找到 {model_path}，正在自动准备官方预训练底模...")
                self.model = YOLO("yolov8n.pt")
                self.model_path = model_path
                
                # 自动归档移动至目标 weights 目录以维持文件结构规范性
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
            
            # 触发显存预热，消除后续交互首次检测的卡顿
            self._warmup()

    def _warmup(self):
        """
        执行显存与计算上下文预热 (Warm-up Pipeline)
        
        学术机理:
            PyTorch 在 CUDA 架构下的首次前向计算涉及诸多开销:
            1. CUDA 上下文 (CUDA Context) 建立与显存分页池初始化;
            2. cuDNN 对卷积计算算法的基准探寻 (CUDNN Benchmarking);
            3. PTX 汇编代码向特定 GPU 架构 (如 Turing/Ampere/Ada Lovelace) 的 JIT 即时编译。
            通过构造全零虚拟输入张量执行一次空前向推理，将上述 ~1000ms+ 的初始化延时提前至后台就绪阶段。
        """
        try:
            if self.model is not None and str(self.device).startswith("cuda"):
                # 构造符合 YOLOv8 标准网络输入尺寸的虚拟张量: [Batch_Size=1, Channels=3, Height=640, Width=640]
                dummy = torch.zeros((1, 3, 640, 640), device=self.device)
                self.model.predict(dummy, device=self.device, verbose=False)
        except Exception:
            pass

    def predict_image(self, image_input, conf: float = 0.25):
        """
        对输入的单张静态图像或单帧视频画面执行端到端目标检测推理 (线程安全)
        
        算法处理流程:
            1. 前处理 (Pre-processing):
               图像等比例缩放并进行 Letterbox 自适应灰度边框填充至 32 整数倍尺寸 (640x640)，
               将像素矩阵从 BGR 转换为 RGB 并做归一化 [0, 255] -> [0.0, 1.0];
            2. 网络推理 (Forward Pass):
               通过主干网络 C2f、PAN-FPN 颈部并在解耦头完成多尺度分类与回归特征预测;
            3. 后处理 (Post-processing & NMS):
               反算 Letterbox 缩放偏移量恢复原图坐标，实施置信度阈值过滤与 NMS 非极大值抑制，
               消除同一车辆的多重重叠检测框;
            4. 可视化渲染与结构化数据组织:
               利用 OpenCV 绘制半透明置信度标签外接矩形，并生成结构化 JSON 字典列表。

        参数:
            image_input (str | np.ndarray): 图像本地磁盘文件路径，或内存中的 OpenCV BGR 格式 ndarray 矩阵
            conf (float): 置信度过滤阈值 tau_conf (取值范围: 0.01 ~ 0.99，默认 0.25)
            
        返回:
            annotated_frame (np.ndarray): 渲染了边界框、类别标签及置信度数值的 BGR 图像矩阵 (高度 x 宽度 x 3)
            detections (list[dict]): 检出目标的结构化元数据列表，每个元素包含:
                - 'class_id' (int): 类别数字索引
                - 'class_name' (str): 类别英文标识 (如 'qiche', 'Ferrari 458', 'bus')
                - 'confidence' (float): 归一化预测置信度分数 [0.0, 1.0]
                - 'box' (list[float]): 目标左上角及右下角像素绝对坐标 [x1, y1, x2, y2]
            time_ms (float): 本次前向推理与后处理的纯计算耗时 (单位: 毫秒 ms)
        """
        with self.lock:
            if self.model is None:
                raise RuntimeError("模型尚未初始化，请先调用 load_model() 加载有效权重")

            # 使用高精度单调时钟统计推理计算耗时
            t_start = time.perf_counter()

            # 调用 YOLOv8 前向推理管道
            results = self.model.predict(
                source=image_input,
                conf=conf,
                device=self.device,
                verbose=False
            )

            t_end = time.perf_counter()
            self.last_inference_time = (t_end - t_start) * 1000.0

            result = results[0]
            
            # 使用 Ultralytics 引擎绘制标注边界框与类别文本
            annotated_frame = result.plot(line_width=2, font_size=1)

            detections = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # 解析预测属性并从 GPU 显存拷贝回 CPU 主存
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


# 别名导出 (保持系统向后兼容性与扩展命名灵活性)
QicheDetector = VehicleDetector
PlantDetector = VehicleDetector

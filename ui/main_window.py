# -*- coding: utf-8 -*-
"""
YOLO 智能车辆检测与车型分析系统 - 响应式主界面
具备原生深色标题栏、平滑图像自适应缩放、多平台显卡动态适配与排版优化。
"""

import sys
import os
import ctypes
from ctypes import c_int, byref, sizeof
import torch
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QSlider, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextBrowser, QGroupBox, QMessageBox, QSplitter,
    QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent

from core.detector import VehicleDetector
from core.car_wiki import get_vehicle_wiki


def enable_windows_dark_title_bar(window: QMainWindow):
    """在 Windows 10/11 上启用原生深色沉浸式标题栏"""
    if sys.platform == "win32":
        try:
            hwnd = int(window.winId())
            value = c_int(1)
            ret = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 20, byref(value), sizeof(value)
            )
            if ret != 0:
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, 19, byref(value), sizeof(value)
                )
        except Exception:
            pass


def imwrite_unicode(file_path: str, img: np.ndarray) -> bool:
    """Windows 下安全保存带有中文路径的图像，杜绝 OpenCV 写入乱码或静默失败"""
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


class SmoothImageDisplay(QLabel):
    """自适应平滑图像视窗组件：窗口拉伸放大或缩小时，图片按原比例动态重绘适配"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(280, 240)
        self._raw_pixmap = None

    def set_display_pixmap(self, pixmap: QPixmap):
        self._raw_pixmap = pixmap
        self._refresh()

    def clear_display(self):
        self._raw_pixmap = None
        self.clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._raw_pixmap and not self._raw_pixmap.isNull():
            target_size = self.size()
            avail_w = max(10, target_size.width() - 8)
            avail_h = max(10, target_size.height() - 8)
            scaled = self._raw_pixmap.scaled(
                avail_w, avail_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled)


class MediaStreamWorker(QThread):
    """后台视频流 / 摄像头连续推流与推理线程"""
    frame_ready = Signal(QImage, list, float, object)
    error_occurred = Signal(str)

    def __init__(self, detector: VehicleDetector, source=0, conf: float = 0.25):
        super().__init__()
        self.detector = detector
        self.source = source
        self.conf = conf
        self.running = False

    def set_conf(self, conf: float):
        self.conf = conf

    def run(self):
        cap = cv2.VideoCapture(self.source)
        try:
            if not cap.isOpened():
                src_name = f"摄像头 (索引 {self.source})" if isinstance(self.source, int) else f"视频文件 ({os.path.basename(str(self.source))})"
                self.error_occurred.emit(f"无法启动 {src_name}，请确认设备未被占用或文件格式正确。")
                return

            self.running = True
            is_file = isinstance(self.source, str)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_delay = max(0.001, 1.0 / fps)

            while self.running:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, detections, time_ms = self.detector.predict_image(frame, conf=self.conf)

                h, w, ch = annotated_frame.shape
                bytes_per_line = ch * w
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

                self.frame_ready.emit(q_img, detections, time_ms, annotated_frame)

                if is_file:
                    elapsed = time.perf_counter() - t0
                    rem = frame_delay - elapsed
                    if rem > 0.005:
                        time.sleep(rem)
        finally:
            cap.release()

    def stop(self):
        self.running = False
        self.wait()


CameraWorker = MediaStreamWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 智能汽车与车型识别系统")
        self.resize(1260, 780)
        self.setMinimumSize(980, 620)
        self.setAcceptDrops(True)

        self.detector = VehicleDetector()

        self.current_raw_img_path = None
        self.current_annotated_bgr = None
        self.current_detections = []
        self.current_stream_type = None  # None | "camera" | "video"
        self.stream_worker = None
        self._current_wiki_cls = None
        self._user_selected_cls = None

        # 置信度滑块防抖定时器 (150ms，避免拖动滑块时频繁阻塞主线程)
        self.slider_timer = QTimer(self)
        self.slider_timer.setSingleShot(True)
        self.slider_timer.setInterval(150)
        self.slider_timer.timeout.connect(self._on_debounced_conf_apply)

        self._init_ui()
        self._load_stylesheet()

    @property
    def cam_worker(self):
        return self.stream_worker

    @cam_worker.setter
    def cam_worker(self, worker):
        self.stream_worker = worker

    def showEvent(self, event):
        super().showEvent(event)
        enable_windows_dark_title_bar(self)

    def _init_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(14, 10, 14, 14)
        main_layout.setSpacing(10)

        # 1. 顶部 Header / 导航条
        header_frame = QFrame()
        header_frame.setObjectName("headerBar")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(10, 6, 10, 6)
        header_layout.setSpacing(10)

        title_box = QHBoxLayout()
        title_box.setSpacing(8)
        self.lbl_title = QLabel("YOLO 智能车辆检测系统")
        self.lbl_title.setObjectName("appTitle")
        lbl_version = QLabel("v2.0")
        lbl_version.setObjectName("badgeVersion")
        title_box.addWidget(self.lbl_title)
        title_box.addWidget(lbl_version)
        header_layout.addLayout(title_box)

        header_layout.addStretch(1)

        self.lbl_model_badge = QLabel(f"模型: {os.path.basename(self.detector.model_path)}")
        self.lbl_model_badge.setObjectName("badgeModel")

        self.btn_switch_model = QPushButton("切换权重...")
        self.btn_switch_model.setObjectName("btnSecondary")
        self.btn_switch_model.clicked.connect(self._on_switch_model)

        # 动态检测当前机器的显卡型号
        if self.detector.device.startswith("cuda") and torch.cuda.is_available():
            try:
                gpu_name = f"算力: {torch.cuda.get_device_name(0)} (CUDA)"
            except Exception:
                gpu_name = "算力: GPU 加速"
        else:
            gpu_name = "算力: CPU 模式"

        self.lbl_gpu_badge = QLabel(gpu_name)
        self.lbl_gpu_badge.setObjectName("badgeGpu")

        self.lbl_perf_badge = QLabel("推理耗时: -- ms")
        self.lbl_perf_badge.setObjectName("badgePerf")

        header_layout.addWidget(self.lbl_model_badge)
        header_layout.addWidget(self.btn_switch_model)
        header_layout.addWidget(self.lbl_gpu_badge)
        header_layout.addWidget(self.lbl_perf_badge)
        main_layout.addWidget(header_frame)

        # 2. 核心分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("mainSplitter")
        splitter.setChildrenCollapsible(False)

        # === 左侧视窗展示区域 ===
        left_box = QWidget()
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.canvas_container = QFrame()
        self.canvas_container.setObjectName("canvasContainer")
        canvas_layout = QVBoxLayout(self.canvas_container)
        canvas_layout.setContentsMargins(6, 6, 6, 6)

        self.lbl_display = SmoothImageDisplay()
        self._set_placeholder_text()
        canvas_layout.addWidget(self.lbl_display)

        left_layout.addWidget(self.canvas_container, stretch=1)

        # 视窗底部工具栏
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(6)
        self.btn_open_img = QPushButton("打开图片")
        self.btn_open_img.setObjectName("btnPrimary")
        self.btn_open_img.clicked.connect(self._on_open_image)

        self.btn_open_video = QPushButton("打开视频")
        self.btn_open_video.setObjectName("btnSecondary")
        self.btn_open_video.clicked.connect(self._on_open_video)

        self.btn_toggle_cam = QPushButton("开启摄像头")
        self.btn_toggle_cam.setObjectName("btnSecondary")
        self.btn_toggle_cam.clicked.connect(self._on_toggle_camera)

        self.btn_save = QPushButton("保存结果")
        self.btn_save.setObjectName("btnSecondary")
        self.btn_save.clicked.connect(self._on_save_result)

        self.btn_clear = QPushButton("清空画面")
        self.btn_clear.setObjectName("btnSecondary")
        self.btn_clear.clicked.connect(self._on_clear_canvas)

        self.lbl_status_summary = QLabel("就绪")
        self.lbl_status_summary.setObjectName("lblStatus")

        toolbar_layout.addWidget(self.btn_open_img)
        toolbar_layout.addWidget(self.btn_open_video)
        toolbar_layout.addWidget(self.btn_toggle_cam)
        toolbar_layout.addWidget(self.btn_save)
        toolbar_layout.addWidget(self.btn_clear)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self.lbl_status_summary)
        left_layout.addLayout(toolbar_layout)

        splitter.addWidget(left_box)

        # === 右侧参数控制与知识库信息展示 ===
        right_box = QWidget()
        right_box.setMinimumWidth(460)
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 卡片 1: 置信度调节
        grp_conf = QGroupBox("置信度阈值调节 (Confidence)")
        conf_layout = QVBoxLayout(grp_conf)
        conf_layout.setContentsMargins(10, 8, 10, 8)
        slider_row = QHBoxLayout()
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 95)
        self.slider_conf.setValue(25)
        self.slider_conf.valueChanged.connect(self._on_conf_changed)

        self.lbl_conf_val = QLabel("25%")
        self.lbl_conf_val.setObjectName("lblConfVal")

        slider_row.addWidget(self.slider_conf)
        slider_row.addWidget(self.lbl_conf_val)
        conf_layout.addLayout(slider_row)
        right_layout.addWidget(grp_conf)

        # 卡片 2: 检测目标列表
        grp_results = QGroupBox("车辆检测清单")
        res_layout = QVBoxLayout(grp_results)
        res_layout.setContentsMargins(8, 8, 8, 8)

        self.table_res = QTableWidget(0, 3)
        self.table_res.setHorizontalHeaderLabels(["车型 / 类别", "置信度", "目标定位坐标 (XYXY)"])
        self.table_res.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_res.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_res.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_res.setMinimumHeight(120)
        self.table_res.setMaximumHeight(220)
        self.table_res.cellClicked.connect(self._on_table_cell_clicked)
        res_layout.addWidget(self.table_res)
        right_layout.addWidget(grp_results, stretch=1)

        # 卡片 3: 车辆知识与规格特征卡片
        grp_wiki = QGroupBox("车辆规格与车型特征卡片")
        wiki_layout = QVBoxLayout(grp_wiki)
        wiki_layout.setContentsMargins(10, 10, 10, 10)

        self.txt_wiki = QTextBrowser()
        self.txt_wiki.setObjectName("txtWiki")
        self.txt_wiki.setOpenExternalLinks(True)
        self._reset_wiki_placeholder()
        wiki_layout.addWidget(self.txt_wiki)
        right_layout.addWidget(grp_wiki, stretch=4)

        splitter.addWidget(right_box)
        splitter.setSizes([620, 560])
        main_layout.addWidget(splitter, stretch=1)

    def _set_placeholder_text(self):
        self.lbl_display.setText(
            "<div style='text-align: center; color: #475569; padding: 24px;'>"
            "<p style='font-size: 16px; font-weight: 600; color: #94A3B8; margin-bottom: 6px;'>点击【打开图片】或将车辆照片/交通路况图像拖放至此处</p>"
            "<p style='font-size: 13px; color: #64748B;'>支持 JPG / PNG / WEBP 等常见格式 · 支持车载及路口摄像头实时推流分析</p>"
            "</div>"
        )

    def _reset_wiki_placeholder(self):
        self._current_wiki_cls = None
        self.txt_wiki.setHtml(
            "<div style='color: #64748B; padding: 24px; text-align: center; line-height: 1.8;'>"
            "<p style='font-size: 16px; color: #94A3B8; margin-bottom: 8px;'>暂未选中车辆目标</p>"
            "<p style='font-size: 13px; color: #64748B;'>导入图片或开启摄像头后，点击上方列表中的车辆条目，<br>此处将以卡片呈现对应的<b>车型划分、动力构型、车身结构及安全行车参数</b>。</p>"
            "</div>"
        )

    def _load_stylesheet(self):
        qss_path = os.path.join(os.path.dirname(__file__), "style.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    # --- 交互事件 ---
    def _on_switch_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重", "weights", "YOLO Weights (*.pt)"
        )
        if file_path:
            was_stream_running = bool(self.stream_worker and self.stream_worker.isRunning())
            saved_source = self.stream_worker.source if was_stream_running else None
            saved_type = self.current_stream_type if was_stream_running else None
            if was_stream_running:
                self._stop_stream_if_running()

            try:
                self.detector.load_model(file_path)
                self.lbl_model_badge.setText(f"模型: {os.path.basename(file_path)}")
                QMessageBox.information(self, "切换成功", f"成功载入新权重:\n{os.path.basename(file_path)}")

                # 恢复之前的工作状态
                if was_stream_running and saved_source is not None:
                    self._start_stream(saved_source, saved_type)
                elif self.current_raw_img_path:
                    self._process_single_image(self.current_raw_img_path)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"载入权重失败: {str(e)}")
                if was_stream_running and saved_source is not None:
                    self._start_stream(saved_source, saved_type)

    def _on_conf_changed(self, value):
        self.lbl_conf_val.setText(f"{value}%")
        conf = value / 100.0

        if self.cam_worker and self.cam_worker.isRunning():
            self.cam_worker.set_conf(conf)
        elif self.current_raw_img_path:
            # 重启防抖计时器，避免连续拖动阻塞主线程
            self.slider_timer.start()

    def _on_debounced_conf_apply(self):
        if self.current_raw_img_path and not (self.cam_worker and self.cam_worker.isRunning()):
            conf = self.slider_conf.value() / 100.0
            self._process_single_image(self.current_raw_img_path, conf=conf)

    def _on_open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择车辆图片", "", "图片文件 (*.jpg *.png *.jpeg *.bmp *.webp)"
        )
        if file_path:
            self._process_single_image(file_path)

    def _on_clear_canvas(self):
        self._stop_stream_if_running()
        self.current_raw_img_path = None
        self.current_annotated_bgr = None
        self.current_detections = []
        self._user_selected_cls = None
        self.lbl_display.clear_display()
        self._set_placeholder_text()
        self.table_res.setRowCount(0)
        self._reset_wiki_placeholder()
        self.lbl_status_summary.setText("已清空")
        self.lbl_perf_badge.setText("推理耗时: -- ms")

    def _process_single_image(self, file_path: str, conf: float = None):
        self._stop_stream_if_running()
        self.current_raw_img_path = file_path

        if conf is None:
            conf = self.slider_conf.value() / 100.0

        try:
            annotated_frame, detections, time_ms = self.detector.predict_image(file_path, conf=conf)
            self.current_annotated_bgr = annotated_frame
            self.current_detections = detections

            self._display_bgr_frame(annotated_frame)
            self._update_results_table(detections)

            fps = 1000.0 / time_ms if time_ms > 0 else 0
            self.lbl_perf_badge.setText(f"{time_ms:.1f} ms · {fps:.0f} FPS")
            self.lbl_status_summary.setText(f"目标: {len(detections)} 辆/处 | {annotated_frame.shape[1]}x{annotated_frame.shape[0]}")
        except Exception as e:
            QMessageBox.critical(self, "检测失败", f"推理过程出错: {str(e)}")

    def _display_bgr_frame(self, bgr_frame: np.ndarray):
        h, w, ch = bgr_frame.shape
        bytes_per_line = ch * w
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.lbl_display.set_display_pixmap(pixmap)

    def _update_results_table(self, detections: list):
        self.table_res.setRowCount(0)
        vehicle_row_to_focus = None
        user_selected_row = None

        vehicle_priority_classes = [
            "qiche", "car", "suv", "bus", "truck", "motorcycle", "van", "mpv",
            "sports car", "pickup", "bicycle", "traffic light", "stop sign"
        ]

        for row, det in enumerate(detections):
            self.table_res.insertRow(row)

            cls_name = det["class_name"]
            wiki = get_vehicle_wiki(cls_name)
            display_name = f"{wiki['cn_name']} ({cls_name})"
            item_name = QTableWidgetItem(display_name)
            item_name.setData(Qt.UserRole, cls_name)

            conf_val = det["confidence"] * 100
            item_conf = QTableWidgetItem(f"{conf_val:.1f}%")
            item_conf.setTextAlignment(Qt.AlignCenter)
            if conf_val >= 60:
                item_conf.setForeground(Qt.cyan)

            box_str = f"[{det['box'][0]}, {det['box'][1]}, {det['box'][2]}, {det['box'][3]}]"
            item_box = QTableWidgetItem(box_str)
            item_box.setTextAlignment(Qt.AlignCenter)

            self.table_res.setItem(row, 0, item_name)
            self.table_res.setItem(row, 1, item_conf)
            self.table_res.setItem(row, 2, item_box)

            if self._user_selected_cls and cls_name.lower() == self._user_selected_cls.lower():
                user_selected_row = row

            if vehicle_row_to_focus is None and cls_name.lower() in vehicle_priority_classes:
                vehicle_row_to_focus = row

        if len(detections) > 0:
            if user_selected_row is not None:
                target_row = user_selected_row
            elif vehicle_row_to_focus is not None:
                target_row = vehicle_row_to_focus
            else:
                target_row = 0
            self.table_res.selectRow(target_row)
            selected_cls = self.table_res.item(target_row, 0).data(Qt.UserRole)
            self._show_vehicle_wiki(selected_cls)
        else:
            self._reset_wiki_placeholder()

    def _on_table_cell_clicked(self, row, col):
        item = self.table_res.item(row, 0)
        if item:
            raw_cls = item.data(Qt.UserRole)
            self._user_selected_cls = raw_cls
            self._show_vehicle_wiki(raw_cls, force=True)

    def _show_vehicle_wiki(self, class_name: str, force: bool = False):
        if not force and class_name == self._current_wiki_cls:
            return
        self._current_wiki_cls = class_name
        wiki = get_vehicle_wiki(class_name)
        html = f"""
        <div style='line-height: 1.7; font-size: 14px;'>
            <div style='margin-bottom: 8px;'>
                <span style='font-size: 20px; font-weight: bold; color: #38BDF8;'>{wiki['cn_name']}</span>
                <span style='color: #64748B; font-size: 13px; margin-left: 10px;'>标识: <i>{wiki['en_name']}</i></span>
            </div>
            <div style='color: #94A3B8; font-size: 13px; margin-bottom: 10px;'>
                <b>分类归属：</b><span style='color: #CBD5E1;'>{wiki['category']}</span>
            </div>
            <div style='color: #94A3B8; font-size: 13px; margin-bottom: 10px;'>
                <b>牌照准驾：</b><span style='color: #60A5FA;'>{wiki['license_plate']}</span>
            </div>
            
            <hr style='border: none; border-top: 1px solid #1E293B; margin: 8px 0;'>
            
            <div style='background-color: #1E293B; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;'>
                <p style='margin: 4px 0; font-size: 13px;'>
                    <b style='color: #38BDF8;'>动力驱动：</b>
                    <span style='color: #F8FAFC;'>{wiki['powertrain']}</span>
                </p>
                <p style='margin: 4px 0; font-size: 13px;'>
                    <b style='color: #F59E0B;'>尺寸规格：</b>
                    <span style='color: #F8FAFC;'>{wiki['dimensions']}</span>
                </p>
            </div>

            <div style='margin-bottom: 10px;'>
                <p style='margin: 3px 0; font-size: 14px; font-weight: bold; color: #94A3B8;'>典型场景与路况：</p>
                <p style='margin: 2px 0; color: #E2E8F0; font-size: 13px; line-height: 1.6;'>{wiki['scenario']}</p>
            </div>

            <div style='margin-bottom: 10px;'>
                <p style='margin: 3px 0; font-size: 14px; font-weight: bold; color: #60A5FA;'>车身构造与技术特性：</p>
                <p style='margin: 2px 0; color: #DBEAFE; font-size: 13px; line-height: 1.6;'>{wiki['tech_features']}</p>
            </div>

            <div>
                <p style='margin: 3px 0; font-size: 14px; font-weight: bold; color: #34D399;'>安全驾驶与通行规程：</p>
                <p style='margin: 2px 0; color: #A7F3D0; font-size: 13px; line-height: 1.6;'>{wiki['safety_tips']}</p>
            </div>
        </div>
        """
        self.txt_wiki.setHtml(html)

    # 别名与向后兼容方法
    def _show_qiche_wiki(self, class_name: str, force: bool = False):
        self._show_vehicle_wiki(class_name, force=force)

    _show_plant_wiki = _show_qiche_wiki

    # --- 视频流与摄像头连续检测控制 ---
    def _on_toggle_camera(self):
        if self.stream_worker and self.stream_worker.isRunning() and self.current_stream_type == "camera":
            self._stop_stream_if_running()
        else:
            self._stop_stream_if_running()
            self._start_stream(source=0, stream_type="camera")

    def _on_open_video(self):
        if self.stream_worker and self.stream_worker.isRunning() and self.current_stream_type == "video":
            self._stop_stream_if_running()
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择车辆视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        if file_path:
            self._stop_stream_if_running()
            self._start_stream(source=file_path, stream_type="video")

    def _start_stream(self, source, stream_type: str = "camera"):
        conf = self.slider_conf.value() / 100.0
        self.current_stream_type = stream_type
        self.stream_worker = MediaStreamWorker(self.detector, source=source, conf=conf)
        self.stream_worker.frame_ready.connect(self._on_stream_frame_ready)
        self.stream_worker.error_occurred.connect(self._on_stream_error)
        self.stream_worker.finished.connect(self._on_stream_finished)
        self.stream_worker.start()

        if stream_type == "camera":
            self.btn_toggle_cam.setText("停止摄像头")
            self.btn_toggle_cam.setObjectName("btnDanger")
            self.btn_toggle_cam.setStyle(self.btn_toggle_cam.style())
            self.lbl_status_summary.setText("摄像头推流中")
        else:
            self.btn_open_video.setText("停止视频")
            self.btn_open_video.setObjectName("btnDanger")
            self.btn_open_video.setStyle(self.btn_open_video.style())
            self.lbl_status_summary.setText(f"视频分析中: {os.path.basename(str(source))}")

    def _start_camera(self):
        self._start_stream(source=0, stream_type="camera")

    def _on_stream_frame_ready(self, q_img: QImage, detections: list, time_ms: float, bgr_frame: np.ndarray = None):
        if bgr_frame is not None:
            self.current_annotated_bgr = bgr_frame
        self.current_detections = detections
        pixmap = QPixmap.fromImage(q_img)
        self.lbl_display.set_display_pixmap(pixmap)
        self._update_results_table(detections)

        fps = 1000.0 / time_ms if time_ms > 0 else 0
        self.lbl_perf_badge.setText(f"{time_ms:.1f} ms · {fps:.0f} FPS")

    _on_cam_frame_ready = _on_stream_frame_ready

    def _reset_stream_ui_state(self, status_text: str = None):
        self.btn_toggle_cam.setText("开启摄像头")
        self.btn_toggle_cam.setObjectName("btnSecondary")
        self.btn_toggle_cam.setStyle(self.btn_toggle_cam.style())

        self.btn_open_video.setText("打开视频")
        self.btn_open_video.setObjectName("btnSecondary")
        self.btn_open_video.setStyle(self.btn_open_video.style())

        if status_text:
            self.lbl_status_summary.setText(status_text)
        elif self.current_stream_type == "camera":
            self.lbl_status_summary.setText("摄像头已关闭")
        elif self.current_stream_type == "video":
            self.lbl_status_summary.setText("视频处理已完成")
        else:
            self.lbl_status_summary.setText("就绪")

        self.current_stream_type = None

    _reset_camera_ui_state = _reset_stream_ui_state

    def _on_stream_finished(self):
        self._reset_stream_ui_state()
        self.stream_worker = None

    _on_cam_finished = _on_stream_finished

    def _on_stream_error(self, err_msg: str):
        self._reset_stream_ui_state("推流异常中断")
        if self.stream_worker:
            if self.stream_worker.isRunning():
                self.stream_worker.stop()
            self.stream_worker = None
        QMessageBox.warning(self, "设备/视频错误", err_msg)

    _on_cam_error = _on_stream_error

    def _stop_stream_if_running(self):
        if self.stream_worker:
            if self.stream_worker.isRunning():
                self.stream_worker.stop()
            self.stream_worker = None
        self._reset_stream_ui_state()

    _stop_camera_if_running = _stop_stream_if_running

    def _on_save_result(self):
        if self.current_annotated_bgr is None and not self.current_detections:
            QMessageBox.information(self, "提示", "暂无检测结果图像或数据可供保存")
            return

        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存检测结果",
            "car_detection_result.jpg",
            "JPEG 图像 (*.jpg);;PNG 图像 (*.png);;CSV 数据报表 (*.csv)"
        )
        if not save_path:
            return

        # 检查是否缺失扩展名，若未输入则根据用户在下拉框选中的格式过滤器智能补齐
        ext = os.path.splitext(save_path)[1].lower()
        if not ext:
            if "png" in selected_filter.lower():
                save_path += ".png"
            elif "csv" in selected_filter.lower():
                save_path += ".csv"
            else:
                save_path += ".jpg"

        if save_path.lower().endswith(".csv") or "csv" in selected_filter.lower():
            if not save_path.lower().endswith(".csv"):
                save_path += ".csv"
            try:
                import csv
                with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "序号", "类别英文", "中文名称", "置信度",
                        "X1", "Y1", "X2", "Y2",
                        "分类归属", "动力构型", "尺寸规格", "牌照准驾", "安全规程"
                    ])
                    for idx, det in enumerate(self.current_detections, 1):
                        wiki = get_vehicle_wiki(det["class_name"])
                        box = det.get("box", [0, 0, 0, 0])
                        writer.writerow([
                            idx,
                            det["class_name"],
                            wiki.get("cn_name", ""),
                            f"{det['confidence']*100:.1f}%",
                            box[0], box[1], box[2], box[3],
                            wiki.get("category", ""),
                            wiki.get("powertrain", ""),
                            wiki.get("dimensions", ""),
                            wiki.get("license_plate", ""),
                            wiki.get("safety_tips", "")
                        ])
                QMessageBox.information(self, "导出成功", f"检测数据清单已成功导出至:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"写入 CSV 文件失败: {str(e)}")
        else:
            if self.current_annotated_bgr is None:
                QMessageBox.warning(self, "提示", "当前无有效画面帧图像可保存")
                return
            ok = imwrite_unicode(save_path, self.current_annotated_bgr)
            if ok:
                QMessageBox.information(self, "保存成功", f"结果图像已成功保存至:\n{save_path}")
            else:
                QMessageBox.critical(self, "保存失败", f"写入图像文件失败，请检查文件路径:\n{save_path}")

    # --- 拖拽支持 ---
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".jpg", ".png", ".jpeg", ".bmp", ".webp"]:
                self._process_single_image(file_path)
            elif ext in [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]:
                self._stop_stream_if_running()
                self._start_stream(source=file_path, stream_type="video")

    def closeEvent(self, event):
        self._stop_stream_if_running()
        event.accept()

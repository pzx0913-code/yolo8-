# -*- coding: utf-8 -*-
"""
YOLO 植物识别软件 - 应用程序启动入口
运行命令:
    python app.py
"""

import sys
import os

# 将当前根目录添加到模块搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from ui.main_window import MainWindow


def main():
    # 启用高分屏支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO 植物识别监测系统")
    app.setOrganizationName("PlantAI")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

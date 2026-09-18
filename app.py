# -*- coding: utf-8 -*-
"""
YOLO 汽车与车辆识别软件 - 桌面客户端启动入口
内置全自动环境诊断、崩溃捕获与原生弹窗引导机制。
"""

import sys
import os
import traceback

# 锁定当前工作目录与模块导入路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def show_fatal_error_dialog(title: str, message: str):
    """即使在 GUI 依赖完全缺失的情况下，也能弹出 Windows 原生错误弹窗"""
    print(f"\n[FATAL ERROR] {title}\n{message}\n", file=sys.stderr)
    try:
        with open(os.path.join(CURRENT_DIR, "crash_log.txt"), "w", encoding="utf-8") as f:
            f.write(f"Title: {title}\n")
            f.write(f"Message:\n{message}\n\n")
            traceback.print_exc(file=f)
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0,
                f"{message}\n\n详细排查日志已生成至:\ncrash_log.txt",
                title,
                0x10 | 0x0  # MB_ICONERROR | MB_OK
            )
        except Exception:
            pass


def main():
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
    except ImportError as e:
        show_fatal_error_dialog(
            "缺少图形界面依赖库 (PySide6)",
            f"未检测到 PySide6 库或环境不匹配。\n\n"
            f"系统提示: {str(e)}\n\n"
            f"解决办法:\n"
            f"1. 请在项目根目录下双击运行 'install_laptop.bat' 自动配置环境；\n"
            f"2. 或在命令行执行: pip install PySide6 -i https://pypi.tuna.tsinghua.edu.cn/simple"
        )
        sys.exit(1)

    try:
        import torch
        import ultralytics
        import cv2
    except ImportError as e:
        show_fatal_error_dialog(
            "缺少核心视觉算法依赖库",
            f"未检测到必要的算法依赖: {str(e)}\n\n"
            f"解决办法:\n"
            f"请在项目目录下双击运行 'install_laptop.bat' 一键自动安装 PyTorch 与 YOLO 依赖。"
        )
        sys.exit(1)

    try:
        from ui.main_window import MainWindow

        # 启用高分屏清晰渲染
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

        app = QApplication(sys.argv)
        app.setApplicationName("YOLO 智能车辆检测分析系统")
        app.setOrganizationName("VehicleAI")

        window = MainWindow()
        window.show()

        sys.exit(app.exec())

    except Exception as e:
        show_fatal_error_dialog(
            "软件运行时异常退出",
            f"软件在启动或渲染主窗口时捕获到未处理异常:\n\n{str(e)}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

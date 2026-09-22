# -*- coding: utf-8 -*-
"""
YOLO 汽车与车辆识别软件 - 桌面客户端启动入口 (Application Bootstrap Entry)

【答辩与学术论文架构设计原理说明】：
1. 软件工程生命周期设计模式 (Bootstrap Pattern):
   本模块作为整个系统的启动门面（Facade），将复杂的环境诊断、高分屏渲染策略配置、
   AI 引擎预热和主界面实例化进行了高度解耦与顺序流水线化管理。

2. 容错性与异常安全防御机制 (Fault Tolerance & Fail-safe Strategy):
   - 传统 Python GUI 软件在缺少依赖库（如 PySide6 或 PyTorch）时，控制台直接闪退且无任何交互反馈，
     导致用户或评审人员产生“程序无法运行”的误解。
   - 本模块采用双层保护机制：外层通过纯 Python 内置的 ctypes 直接调用 Windows 底层 Win32 API
     (user32.dll: MessageBoxW)，即使在整个 Qt 图形框架与 PyTorch 深度学习库彻底崩溃或未安装的极端场景下，
     依然能强行拉起 Windows 操作系统原生错误交互弹窗，并自动生成结构化崩溃日志 (crash_log.txt)，
     符合软件系统可用性（Availability）与可维护性（Maintainability）的工业级标准。

3. 消除感知延迟的启动加载器 (SplashScreen Anti-Perception-Lag):
   深度神经网络在首次加载权重至 GPU/CPU 时，需要进行 CUDA 运行时上下文初始化、权重反序列化以及
   计算图（Computation Graph）的静态构建，通常产生 1~2 秒的“启动真空期”。
   本系统引入 QSplashScreen 异步占位技术，在毫秒级拉起科技风启动画面，有效消除用户界面冻结感。
"""

import sys
import os
import traceback

# 锁定当前工作目录与模块导入路径 (防止相对路径寻找失效及模块导入作用域污染)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def show_fatal_error_dialog(title: str, message: str, exc: BaseException = None):
    """
    基于底层操作系统 API 的崩溃诊断与原生弹窗处理器
    
    【学术与答辩原理解析】：
    - 运行原理：通过 ctypes 模块实现 C 语言级别的动态库链接，直接装载 'user32.dll' 并调用 'MessageBoxW'。
    - 优势：完全绕过 Python 虚拟环境中的任何第三方库，具备最高等级的调用优先级与独立性。
    - 参数说明：
      * title: 弹窗标题 (Unicode 宽字符)
      * message: 面向用户的引导式排错文案
      * exc: 异常基类对象，通过 traceback.print_exception 提取完整栈帧并落盘
    """
    print(f"\n[FATAL ERROR] {title}\n{message}\n", file=sys.stderr)
    try:
        # 将结构化的调用栈写入本地日志，用于事故回溯与排障审计
        with open(os.path.join(CURRENT_DIR, "crash_log.txt"), "w", encoding="utf-8") as f:
            f.write(f"Title: {title}\n")
            f.write(f"Message:\n{message}\n\n")
            f.write("Traceback Details:\n")
            if exc is not None:
                traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
            else:
                traceback.print_exc(file=f)
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            import ctypes
            # 0x10 代表 MB_ICONERROR (错误红叉图标)，0x0 代表 MB_OK (单确定按钮)
            ctypes.windll.user32.MessageBoxW(
                0,
                f"{message}\n\n详细排查日志已生成至:\ncrash_log.txt",
                title,
                0x10 | 0x0  # MB_ICONERROR | MB_OK
            )
        except Exception:
            pass


def main():
    """
    应用程序主函数：环境巡检 -> 高分屏配置 -> 启动动画 -> 核心视觉预热 -> 事件循环
    """
    # 阶段一：GUI 基础框架依赖自检
    try:
        from PySide6.QtWidgets import QApplication, QSplashScreen
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap, QPainter, QColor, QFont
    except ImportError as e:
        show_fatal_error_dialog(
            "缺少图形界面依赖库 (PySide6)",
            f"未检测到 PySide6 库或环境不匹配。\n\n"
            f"系统提示: {str(e)}\n\n"
            f"解决办法:\n"
            f"1. 请在项目根目录下双击运行 'install_laptop.bat' 自动配置环境；\n"
            f"2. 或在命令行执行: pip install PySide6 -i https://pypi.tuna.tsinghua.edu.cn/simple",
            exc=e
        )
        sys.exit(1)

    # 阶段二：高分辨率屏幕渲染策略配置 (High-DPI Scaling Policy)
    # 【原理说明】：在 2K/4K 笔记本或缩放比例为 125%/150%/175% 的屏幕上，默认整数缩放会导致字体发虚或控件变形。
    # 设置 PassThrough 原生透传策略，交由 Qt 底层矢量引擎依据设备像素比（DPR, Device Pixel Ratio）精确渲染。
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO 智能车辆检测分析系统")
    app.setOrganizationName("VehicleAI")

    # 阶段三：异步启动加载器 (QSplashScreen)
    # 【原理说明】：利用双缓冲绘图机制 (QPainter) 在内存离屏画布 (QPixmap) 绘制深色科技风加载窗，
    # 并置顶展示 (WindowStaysOnTopHint)。
    splash_pix = QPixmap(440, 200)
    splash_pix.fill(QColor('#0F172A'))
    p = QPainter(splash_pix)
    p.setPen(QColor('#1E293B'))
    p.drawRect(0, 0, 439, 199)

    p.setFont(QFont('Microsoft YaHei', 14, QFont.Bold))
    p.setPen(QColor('#38BDF8'))
    p.drawText(24, 62, 'YOLO 智能车辆检测分析系统')

    p.setFont(QFont('Microsoft YaHei', 10))
    p.setPen(QColor('#94A3B8'))
    p.drawText(24, 112, '正在初始化 AI 视觉内核与硬件算力...')

    p.setPen(QColor('#64748B'))
    p.drawText(24, 152, 'NVIDIA RTX 独显与模型引擎快速预热中...')
    p.end()

    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.show()
    # 强制分发并处理当前消息队列中的重绘事件，确保启动窗口瞬间呈现在桌面
    app.processEvents()

    # 阶段四：AI 深度学习核心依赖与硬件运行时探测
    try:
        import torch
        import ultralytics
        import cv2
    except ImportError as e:
        splash.close()
        show_fatal_error_dialog(
            "缺少核心视觉算法依赖库",
            f"未检测到必要的算法依赖: {str(e)}\n\n"
            f"解决办法:\n"
            f"请在项目目录下双击运行 'install_laptop.bat' 一键自动安装 PyTorch 与 YOLO 依赖。",
            exc=e
        )
        sys.exit(1)

    # 阶段五：实例化主窗口并交付事件循环
    try:
        from ui.main_window import MainWindow

        # 主窗体实例化（包含 UI 布局、信号槽绑定与模型就绪检测）
        window = MainWindow()
        window.show()
        # 平滑关闭启动页并将焦点转交主窗体
        splash.finish(window)

        # 【核心原理】：进入 Qt 主事件循环 (Event Loop)
        # 底层基于 Windows 消息泵 (GetMessage/DispatchMessage)，以非阻塞事件驱动机制监听鼠标、键盘及后台线程信号
        sys.exit(app.exec())

    except Exception as e:
        splash.close()
        show_fatal_error_dialog(
            "软件运行时异常退出",
            f"软件在启动或渲染主窗口时捕获到未处理异常:\n\n{str(e)}",
            exc=e
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

@echo off
chcp 65001 >nul
cd /d "%~dp0"
title YOLO Plant Recognition System - Installer

echo ==========================================================
echo       YOLO Plant Recognition System - Environment Setup
echo ==========================================================
echo.

:: 1. 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] 未检测到可用 Python 环境。
    echo.
    echo 请先安装 Python 3.10 或 3.11 (安装时勾选 Add python.exe to PATH):
    echo 官方下载直链: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    echo.
    pause
    exit /b 1
)

echo [OK] Python 已检测就绪:
python --version
echo.

:: 2. 选择显卡模式
echo ----------------------------------------------------------
echo 请选择硬件运行模式：
echo   [1] 普通轻薄本 / 集成显卡 (纯 CPU 模式，下载仅约 150MB，快速安装)
echo   [2] 游戏本 / 拥有 NVIDIA 独立显卡 (GPU 加速模式，需下载约 2.5GB 核心)
echo ----------------------------------------------------------
set /p choice="请输入选项 [1 或 2，直接回车默认选 1]: "

if "%choice%"=="2" (
    echo.
    echo [*] 正在为 NVIDIA 显卡安装 PyTorch GPU 版本...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
) else (
    echo.
    echo [*] 正在为电脑安装轻量 CPU 版 PyTorch...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo [*] 正在安装算法核心、视觉处理库及图形界面依赖 (清华镜像加速)...
python -m pip install ultralytics opencv-python PySide6 matplotlib pandas pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo ==========================================================
echo [OK] 依赖环境配置完成！
echo.
echo 启动软件请直接双击运行目录下的 "run.bat"。
echo ==========================================================
echo.
pause

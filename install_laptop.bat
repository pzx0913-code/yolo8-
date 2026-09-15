@echo off
chcp 65001 >nul
cd /d "%~dp0"
title YOLO 植物识别系统 - 笔记本环境一键安装向导

echo ==========================================================
echo       🌿 YOLO 植物识别系统 - 笔记本环境一键自动化安装
echo ==========================================================
echo.

:: 1. 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] 错误: 未检测到 Python！
    echo.
    echo 请先下载安装 Python 3.11 (记得勾选 Add python.exe to PATH):
    echo 官方下载直链: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    echo.
    pause
    exit /b 1
)

echo [✔] Python 已就绪:
python --version
echo.

:: 2. 询问笔记本显卡类型
echo ----------------------------------------------------------
echo 请选择你笔记本的显卡配置：
echo   [1] 普通轻薄本 / 集成显卡 / 没有 NVIDIA 显卡 (推荐纯 CPU 模式，下载只需100MB，极快)
echo   [2] 游戏本 / 拥有 NVIDIA 独立显卡 (开启 GPU 硬件加速，需下载约 2.5GB 核心)
echo ----------------------------------------------------------
set /p choice="请输入数字 (1 或 2，默认按回车选 1): "

if "%choice%"=="2" (
    echo.
    echo [*] 正在为 NVIDIA 显卡安装 PyTorch GPU 版...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
) else (
    echo.
    echo [*] 正在为笔记本安装轻量纯净版 PyTorch (CPU 高速运行版)...
    python -m pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo.
echo [*] 正在安装 YOLO 算法核心、OpenCV 视觉库及 PySide6 桌面界面库 (清华镜像加速)...
python -m pip install ultralytics opencv-python PySide6 matplotlib pandas pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo ==========================================================
echo [✔] 恭喜！笔记本环境全部配置完毕！
echo.
echo 现在你可以直接双击运行本目录下的 "run.bat" 启动植物识别软件！
echo ==========================================================
echo.
pause

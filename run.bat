@echo off
chcp 65001 >nul
cd /d "%~dp0"
title YOLO 植物识别系统 - 便携启动器

:: 智能寻找 Python 环境：优先本地环境 -> 独立目录环境 -> 系统全局 Python
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "..\plant_yolo_env\Scripts\python.exe" (
    set "PY=..\plant_yolo_env\Scripts\python.exe"
) else if exist "D:\AI\plant_yolo_env\Scripts\python.exe" (
    set "PY=D:\AI\plant_yolo_env\Scripts\python.exe"
) else (
    set "PY=python"
)

%PY% app.py
if errorlevel 1 (
    echo.
    echo 提示: 如果未安装环境，请先双击运行 install_laptop.bat
    echo.
    pause
)

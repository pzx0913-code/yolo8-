@echo off
cd /d "%~dp0"
title YOLO Plant Recognition System

:: Detect Python: .venv -> ..\plant_yolo_env -> D:\AI\plant_yolo_env -> system python
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
    echo ==========================================================
    echo [ERROR] Application failed to launch or exited with error.
    echo If dependencies are missing, please run install_laptop.bat
    echo ==========================================================
    echo.
    pause
)

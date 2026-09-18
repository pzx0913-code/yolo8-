@echo off
cd /d "%~dp0"
title YOLO Vehicle Detection System

:: Scan for working Python interpreters that have PySide6 installed
set "PY="

:: Priority 1: Local project virtual environment (.venv)
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -c "import PySide6" >nul 2>&1 && set "PY=%~dp0.venv\Scripts\python.exe"
)

:: Priority 2: Sibling environment (..\plant_yolo_env)
if not defined PY if exist "..\plant_yolo_env\Scripts\python.exe" (
    "..\plant_yolo_env\Scripts\python.exe" -c "import PySide6" >nul 2>&1 && set "PY=..\plant_yolo_env\Scripts\python.exe"
)

:: Priority 3: Fixed desktop path (D:\AI\plant_yolo_env)
if not defined PY if exist "D:\AI\plant_yolo_env\Scripts\python.exe" (
    "D:\AI\plant_yolo_env\Scripts\python.exe" -c "import PySide6" >nul 2>&1 && set "PY=D:\AI\plant_yolo_env\Scripts\python.exe"
)

:: Priority 4: System PATH python
if not defined PY (
    python -c "import PySide6" >nul 2>&1 && set "PY=python"
)

:: Fallback if no Python with PySide6 is ready: choose candidate to display error
if not defined PY (
    if exist "%~dp0.venv\Scripts\python.exe" (
        set "PY=%~dp0.venv\Scripts\python.exe"
    ) else if exist "D:\AI\plant_yolo_env\Scripts\python.exe" (
        set "PY=D:\AI\plant_yolo_env\Scripts\python.exe"
    ) else (
        set "PY=python"
    )
    echo.
    echo ==========================================================
    echo [WARNING] No complete environment with PySide6 detected!
    echo Attempting to launch with: %PY%
    echo If launch fails, please run: install_laptop.bat
    echo ==========================================================
    echo.
)

echo [*] Starting YOLO Vehicle Detection System...
echo [*] Python interpreter: %PY%
echo.

"%PY%" app.py
if errorlevel 1 (
    echo.
    echo ==========================================================
    echo [ERROR] Application failed to launch or exited with error.
    echo.
    echo Troubleshooting tips:
    echo   1. If dependencies are missing, please double-click:
    echo      install_laptop.bat
    echo   2. Check crash_log.txt for detailed diagnostic information.
    echo ==========================================================
    echo.
    pause
)

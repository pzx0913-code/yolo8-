@echo off
cd /d "%~dp0"
title YOLO Plant Recognition System - Installer

set "PYTHON_EXE=python"
python --version >nul 2>&1
if %errorlevel% neq 0 (
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=py"
    ) else (
        echo.
        echo ==========================================================
        echo [ERROR] Python was not found on this computer.
        echo Please install Python 3.10 or 3.11 from:
        echo   https://www.python.org/downloads/
        echo.
        echo IMPORTANT: Make sure to check the box:
        echo   [x] Add python.exe to PATH
        echo during the installation!
        echo ==========================================================
        echo.
        pause
        exit /b 1
    )
)

%PYTHON_EXE% setup_laptop.py
if errorlevel 1 (
    echo.
    echo [ERROR] Setup script encountered an error.
    pause
    exit /b 1
)

pause

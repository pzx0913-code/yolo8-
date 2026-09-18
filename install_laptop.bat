@echo off
chcp 65001 >nul
cd /d "%~dp0"
title YOLO Qiche Detection System - Installer

:: Check if setup_laptop.py exists
if not exist "setup_laptop.py" (
    echo.
    echo ==========================================================
    echo [ERROR] setup_laptop.py not found in the current folder!
    echo Please make sure all project files are downloaded together.
    echo ==========================================================
    echo.
    pause
    exit /b 1
)

:: Detect Python interpreter (avoid batch %errorlevel% pre-expansion traps)
set "PYTHON_EXE="
python --version >nul 2>&1 && set "PYTHON_EXE=python"
if not defined PYTHON_EXE (
    py --version >nul 2>&1 && set "PYTHON_EXE=py"
)

:: If neither python nor py is found
if not defined PYTHON_EXE (
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

:: Run setup script
%PYTHON_EXE% setup_laptop.py
if errorlevel 1 (
    echo.
    echo [ERROR] Setup script encountered an error.
    pause
    exit /b 1
)

pause

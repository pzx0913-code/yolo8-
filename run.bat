@echo off
cd /d "%~dp0"
title YOLO Plant Recognition System
"D:\AI\plant_yolo_env\Scripts\python.exe" app.py
if errorlevel 1 pause

# -*- coding: utf-8 -*-
"""
深度学习工程虚拟环境自动化隔离构建与依赖分发向导 (setup_laptop.py)
=============================================================================
[理论背景与学术原理阐述 (Academic & Theoretical Foundations)]:

一、 PEP 405 运行时虚拟环境隔离机制 (Runtime Virtual Environment Isolation)
    - 深度学习项目对底层 C/C++ 动态链接库 (如 PyTorch 的 libtorch_cuda.dll、CUDA Runtime、
      cuDNN) 与特定 Python ABI (Application Binary Interface) 具有严格的版本绑定要求；
    - 直接在全局系统 Python 环境中安装依赖极易诱发 "依赖地狱 (Dependency Hell)" 与符号版本冲突；
    - 本脚本基于 Python 官方标准 PEP 405 实现轻量化沙盒隔离，通过在项目根目录下构建独立
      的 sys.prefix、独立 site-packages 目录及专属 Scripts/bin 启动器，达成项目运行环境的
      自包含性 (Self-Containment) 与可移植性 (Portability)。

二、 PyPI 预编译轮子分发 (Pre-built Wheel Distributions) 与镜像容灾流
    - 深度学习依赖包 (torch, torchvision, opencv-python) 包含大量 C++ 与 CUDA 扩展源码，
      在缺少本地 MSVC C++ 构建工具链的机器上无法通过源码包 (sdist) 编译；
    - 本脚本设计了自动化探测机制: 优先自清华/阿里云国内镜像源加速下载预编译二进制 Wheel 包，
      当网络受阻或不可达时自动降级容灾，确保新设备开箱即用、一键就绪。
=============================================================================
"""
import os
import sys
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

def print_header(title: str):
    print("\n" + "=" * 62, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 62 + "\n", flush=True)

def run_cmd(cmd_list, desc: str = ""):
    if desc:
        print(f"[*] {desc} ...", flush=True)
    ret = subprocess.run(cmd_list)
    if ret.returncode != 0:
        print(f"\n[!] 命令执行失败 (退出代码: {ret.returncode}): {' '.join(cmd_list)}", flush=True)
        return False
    return True

def main():
    print_header("YOLO 智能车辆识别系统 - 环境一键安装向导")
    print(f"项目目录: {PROJECT_DIR}", flush=True)
    print(f"当前 Python 版本: {sys.version.split()[0]} ({sys.executable})\n", flush=True)

    # 1. 确定或创建虚拟环境 (.venv)
    venv_dir = os.path.join(PROJECT_DIR, ".venv")
    if sys.platform == "win32":
        venv_py = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_py = os.path.join(venv_dir, "bin", "python")

    if not os.path.exists(venv_py):
        print("[*] 正在为本项目创建独立的 Python 虚拟环境 (.venv) ...", flush=True)
        res = subprocess.run([sys.executable, "-m", "venv", ".venv"])
        if res.returncode != 0 or not os.path.exists(venv_py):
            print("[!] 虚拟环境创建失败，将直接使用当前系统 Python 环境进行安装。", flush=True)
            venv_py = sys.executable
        else:
            print("[OK] 独立虚拟环境 (.venv) 创建成功。", flush=True)
    else:
        print(f"[OK] 检测到已存在的虚拟环境: {venv_py}", flush=True)

    # 2. 升级 pip
    print("\n[*] 检查并更新 pip 包管理器...", flush=True)
    subprocess.run([venv_py, "-m", "pip", "install", "--upgrade", "pip", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])

    # 3. 硬件选择
    print("\n" + "-" * 62, flush=True)
    print("请选择笔记本硬件运行模式：", flush=True)
    print("  [1] 普通轻薄本 / 核显 (CPU 模式，下载仅约 150MB，快速完成) [默认推荐]", flush=True)
    print("  [2] 游戏本 / NVIDIA 独立显卡 (GPU 加速模式，需下载约 2.5GB CUDA 核心)", flush=True)
    print("-" * 62, flush=True)

    try:
        choice = input("请输入选项 [1 或 2，直接回车默认选 1]: ").strip()
    except EOFError:
        choice = "1"

    if choice == "2":
        print("\n[*] 正在下载并安装 PyTorch GPU 加速版 (CUDA 12.4) ...", flush=True)
        ok = run_cmd([
            venv_py, "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ], "安装 PyTorch GPU 版")
        if not ok:
            print("\n[!] 官方源下载超时或失败，尝试切换国内镜像重试...", flush=True)
            run_cmd([
                venv_py, "-m", "pip", "install",
                "torch", "torchvision",
                "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
            ], "使用清华镜像安装 PyTorch")
    else:
        print("\n[*] 正在下载并安装 PyTorch CPU 轻量版 (约 150MB) ...", flush=True)
        ok = run_cmd([
            venv_py, "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], "安装 PyTorch CPU 版")
        if not ok:
            print("\n[!] 官方 CPU 源超时，尝试使用国内镜像源...", flush=True)
            run_cmd([
                venv_py, "-m", "pip", "install",
                "torch", "torchvision",
                "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
            ], "使用清华镜像安装 PyTorch")

    # 4. 安装计算机视觉、界面及算法核心依赖 (国内高速镜像)
    print("\n[*] 正在安装视觉计算与 UI 界面库 (清华大学开源镜像站)...", flush=True)
    deps = ["ultralytics", "opencv-python", "PySide6", "matplotlib", "pandas", "pyyaml"]
    ok = run_cmd([
        venv_py, "-m", "pip", "install"
    ] + deps + [
        "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
    ], "安装依赖库")

    if not ok:
        print("\n[!] 部分依赖包安装可能出现异常，请检查上方日志。", flush=True)

    # 5. 依赖就绪自检
    print_header("正在进行环境完整性验证")
    check_code = (
        "import torch; "
        "import ultralytics; "
        "import cv2; "
        "import PySide6; "
        "cuda = torch.cuda.is_available(); "
        "dev = torch.cuda.get_device_name(0) if cuda else 'CPU'; "
        "print(f'[OK] PyTorch: {torch.__version__} | 计算设备: {dev}'); "
        "print(f'[OK] Ultralytics: {ultralytics.__version__}'); "
        "print(f'[OK] OpenCV: {cv2.__version__}'); "
        "print(f'[OK] PySide6: {PySide6.__version__}')"
    )
    verify = subprocess.run([venv_py, "-c", check_code])

    if verify.returncode == 0:
        print("\n" + "=" * 62, flush=True)
        print("  [SUCCESS] 恭喜！开发与运行环境已全部配置就绪！", flush=True)
        print("=" * 62, flush=True)
        print("\n后续使用方法：", flush=True)
        print("  1. 启动桌面客户端：直接双击项目根目录下的 'run.bat'", flush=True)
        print("  2. 运行单图测试：直接运行 python predict.py", flush=True)
        print("  3. 若在 VS Code 中打开项目，请将解释器选择为 .venv 环境。\n", flush=True)
    else:
        print("\n[!] 依赖自检未完全通过，请检查是否有安装步骤被中断。", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] 安装已被用户取消。", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] 安装过程发生异常: {e}", flush=True)
        sys.exit(1)

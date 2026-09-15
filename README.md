# YOLO 植物识别与养护监测系统 (Plant Detection & Care Assistant)

本项目基于 YOLO 目标检测模型与 PySide6 桌面框架开发，旨在实现对常见绿植与花卉的多目标定位识别、实时视频流检测及植物养护知识的结构化呈现。

---

## 1. 系统特性

- **多目标检测与定位**：基于 YOLO 神经网络架构，单次前向推理即可同时完成图像中多个植物目标的边界框回归与分类。支持通过 NVIDIA CUDA 进行张量运算加速。
- **双输入模式（图像 / 视频流）**：
  - 静态图像：支持文件对话框选择及鼠标拖拽直接载入。
  - 实时视频流：支持接入本地及外置 USB 摄像头进行动态连续推理，采用 QThread 异步多线程架构避免界面阻塞。
- **知识库联动检索**：检测到植物目标后，系统自动提取类别信息并检索内置植物百科，展示对应学名、分类、光照偏好、水分需求及养护要点。
- **交互式置信度过滤**：提供 10% ~ 95% 置信度阈值滑块，支持在当前图像上即时重测与渲染，支持检测结果图像的一键导出保存。
- **现代化桌面客户端**：采用 PySide6 构建暗色主题界面，支持窗口拉伸自适应缩放与 Windows 沉浸式标题栏。

---

## 2. 目录结构

```text
yolozhiwu/
├── core/
│   ├── detector.py           # YOLO 目标检测引擎封装与性能统计
│   └── plant_wiki.py         # 植物百科与养护知识库
├── ui/
│   ├── main_window.py        # PySide6 响应式主界面
│   └── style.qss             # 客户端样式表
├── weights/
│   └── yolov8n.pt            # 预训练模型权重
├── dataset/
│   └── data.yaml.template    # 自定义数据集配置模板
├── app.py                    # 桌面客户端主启动入口
├── train.py                  # 本地模型训练脚本
├── predict.py                # 命令行单图测试脚本
├── check_dataset.py          # 数据集格式自检脚本
├── run.bat                   # 跨平台自适应快捷启动脚本
├── install_laptop.bat        # Windows 依赖自动化安装脚本
└── requirements.txt          # Python 依赖清单
```

---

## 3. 环境要求与安装

### 环境要求
- 操作系统：Windows 10 / Windows 11 (x64)
- 运行环境：Python 3.10 或 Python 3.11 (推荐 64 位)
- 硬件支持：CPU 模式（任意 PC 均可运行）；GPU 模式（支持 NVIDIA 独立显卡与 CUDA）

### 方式一：使用自动化脚本安装（推荐）
在项目根目录下双击运行 `install_laptop.bat`，脚本将自动检测环境、配置国内镜像源并完成依赖包安装。

### 方式二：手动配置安装
通过命令行进入项目根目录并执行：
```bash
pip install -r requirements.txt
```

若需启用 NVIDIA GPU 显卡加速，请根据本地显卡驱动版本安装匹配 CUDA 版本的 PyTorch：
```bash
# 示例：安装 CUDA 12.4 版本的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## 4. 运行方式

- **方式一（快捷启动）**：直接双击运行 `run.bat`。
- **方式二（命令行启动）**：
  ```bash
  python app.py
  ```

---

## 5. 自定义数据集模型训练

如需针对特定植物品类进行重新训练，请按以下步骤操作：

1. 准备标注格式为 YOLO 格式的数据集（包含 `train/`、`valid/` 及 `data.yaml`），放入 `dataset/` 目录；
2. 运行数据集格式检查工具：
   ```bash
   python check_dataset.py
   ```
3. 启动本地训练（默认调用 GPU 0 进行加速）：
   ```bash
   python train.py --epochs 50 --batch 16
   ```
4. 训练完成后，最优权重将自动同步至 `weights/best_plant.pt`，客户端再次启动时将优先加载该模型。

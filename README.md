# YOLO 智能汽车与车型识别系统 (Vehicle Detection & Analytics System)

本项目基于 YOLO 目标检测模型与 PySide6 桌面客户端框架开发，专为机动车辆目标检测、车型类别分类、车流视频流实时分析及汽车规格特征知识联动而设计。

---

## 1. 系统特性

- **多目标车辆检测与定位**：基于 YOLO 神经网络架构，单次前向推理即可同时完成图像中多个车辆目标（轿车、SUV、客车、卡车、摩托车、非机动车及交通标志设施）的边界框回归与分类。支持通过 NVIDIA CUDA 进行张量运算加速。
- **双输入模式（图像 / 视频流）**：
  - 静态图像：支持点击选择或直接将道路交通照片鼠标拖拽至视窗载入。
  - 实时视频流：支持接入本地及外置 USB 摄像头/车载监控推流进行连续动态推理，采用 QThread 异步多线程架构，保障高帧率下界面丝滑无卡顿。
- **车型规格与特征知识库联动**：检测到目标车辆后，系统自动提取类别信息并检索内置车辆知识库，呈现对应的车型分类定位、动力构型与驱动形式、车身尺寸、号牌准驾要求、典型路况用途、技术特征及安全行车规程。
- **交互式置信度过滤**：提供 10% ~ 95% 置信度阈值滑块，支持在当前图像上即时重测与高亮渲染，支持带目标标注框的结果图像一键无损导出。
- **现代化科技暗色客户端**：采用 PySide6 构建高科技 Slate & Cyber Blue 汽车智能视觉主题界面，支持自适应动态缩放与 Windows 原生深色标题栏。

---

## 2. 目录结构

```text
yolo_vehicle/
├── core/
│   ├── detector.py           # YOLO 车辆检测引擎封装与硬件加速推断
│   └── car_wiki.py           # 车辆规格与车型特征知识库
├── ui/
│   ├── main_window.py        # PySide6 响应式汽车识别主界面
│   └── style.qss             # 智能交通深色科技蓝样式表
├── weights/
│   └── yolov8n.pt            # 预训练模型权重 (原生支持车辆识别)
├── dataset/
│   └── data.yaml.template    # 自定义车辆数据集配置模板
├── app.py                    # 桌面客户端主启动入口
├── train.py                  # 本地模型训练脚本 (输出 best_car.pt)
├── predict.py                # 命令行单图/视频快速测试脚本
├── check_dataset.py          # 车辆数据集格式自检工具
├── run.bat                   # 跨平台自适应快捷启动脚本
├── install_laptop.bat        # Windows 依赖自动化安装引导脚本
├── setup_laptop.py           # 跨平台环境与依赖自动配置向导
└── requirements.txt          # Python 依赖清单
```

---

## 3. 环境要求与安装

### 环境要求
- 操作系统：Windows 10 / Windows 11 (x64)
- 运行环境：Python 3.10 或 Python 3.11 (推荐 64 位)
- 硬件支持：CPU 模式（任意 PC 均可运行）；GPU 模式（支持 NVIDIA 独立显卡与 CUDA）

### 方式一：使用自动化脚本安装（推荐）
在项目根目录下双击运行 `install_laptop.bat`，脚本将自动检测环境、配置虚拟环境并调用国内高速镜像源完成依赖包安装。

### 方式二：手动配置安装
通过命令行进入项目根目录并执行：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
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

## 5. 自定义车辆数据集训练

系统自带的底模权重已原生支持常见车辆目标（轿车、公交车、卡车、摩托车等）。如需针对特定特种车辆或细分车型进行定制化重训，步骤如下：

1. 准备标注格式为 YOLO 格式的数据集（包含 `train/`、`val/` 及 `data.yaml`），放入 `dataset/` 目录；
2. 运行数据集格式检查工具：
   ```bash
   python check_dataset.py
   ```
3. 启动本地训练：
   ```bash
   python train.py --epochs 50 --batch 16
   ```
4. 训练完成后，最优权重将自动同步至 `weights/best_car.pt`，客户端再次启动时将自动优先加载该专属车辆模型。

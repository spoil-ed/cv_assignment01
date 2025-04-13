# CIFAR-10 卷积神经网络项目

本项目实现了一个基于 CuPy（GPU 加速）的三层卷积神经网络（CNN），用于 CIFAR-10 数据集的图像分类。网络包括卷积层、批归一化、ReLU 激活、最大池化和全连接层，支持完整的训练、预测和可视化功能。

## 项目目录结构
```
├── data/
│   └── cifar-10-batches-py/
├── experiments/
│   ├── results/
│   │   └── best_model_weights.npz
│   ├── logs/
│   └── plots/
├── src/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── visualize.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```
- `data/`: 用于存放 CIFAR-10 数据集（默认路径：`data/cifar-10-batches-py/`）。
- `experiments/`: 存储训练结果，包括权重（`best_model_weights.npz`）、日志（`logs/`）、可视化图表（`plots/`）和超参数调优数据。
- `src/`: 包含核心代码文件：
  - `model.py`: 定义 `Conv3LayerNN` 类，包含卷积、批归一化、池化和全连接层的前向和反向传播实现。
  - `train.py`: 实现模型训练、超参数调优（学习率、正则化系数、卷积核数量）和 CIFAR-10 数据集评估。
  - `predict.py`: 支持对测试集或单张图像进行预测，输出类别概率和准确率。
  - `visualize.py`: 生成可视化结果，包括训练曲线、卷积核、权重分布、批归一化参数、类别准确率和超参数调优热图。
  - `utils.py`: 提供工具函数，包括 CIFAR-10 数据加载和日志设置。
- `.gitignore`: Git 忽略文件。
- `README.md`: 项目说明文档。
- `requirements.txt`: 项目依赖库列表。

## 安装与依赖

1. 确保已安装 CUDA 和 cuDNN（若使用 GPU）。
2. 安装 Python 3.8+。
3. 通过以下命令安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

依赖文件（`requirements.txt`）内容如下：
```
cupy-cuda11x>=10.0.0
numpy>=1.20.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.60.0
pillow>=8.0.0
scipy>=1.7.0
```

## 使用方法

### 1. 数据准备
- 下载 CIFAR-10 数据集（Python 版本）并解压到 `data/cifar-10-batches-py/`。
- 或者在运行脚本时指定数据路径。

### 2. 训练模型
运行 `train.py` 进行模型训练和超参数调优：
```bash
python src/train.py
```
- 默认数据路径：`data/cifar-10-batches-py/`。
- 训练结果（权重、日志、数据）保存至 `experiments/`。
- 可通过修改 `src/train.py` 中的超参数（如 `lrs`, `regs`, `conv1_filters_list`）自定义调优范围。

### 3. 预测
使用训练好的模型进行预测：
```bash
python src/predict.py
```
- 默认权重路径：`experiments/best_model_weights.npz`。
- 支持测试集批量预测或单张图像预测（需修改代码指定图像路径）。
- 输出预测类别、概率和测试集准确率。

### 4. 可视化
生成训练过程和模型参数的可视化结果：
```bash
python src/visualize.py
```
- 默认保存路径：`experiments/plots/`。
- 包括损失曲线、卷积核、权重热图、参数分布、批归一化参数趋势和类别准确率图。

### 5. 日志
- 日志文件保存在 `experiments/logs/`，记录训练、预测和可视化过程。
- 可通过 `src/utils.py` 中的 `setup_logging` 函数自定义日志设置。

## 示例输出

- **训练日志**（`experiments/logs/`）：
  ```
  2025-04-13 10:00:00 [INFO] Starting CIFAR-10 training...
  2025-04-13 10:01:00 [INFO] Epoch 1/20, Train Loss: 2.3456, Train Acc: 0.3456, Val Loss: 2.1234, Val Acc: 0.4000
  ```
- **可视化结果**（`experiments/plots/`）：
  - `loss_curves.png`: 训练和验证损失曲线。
  - `conv1_kernels.png`: 第一层卷积核可视化。
  - `class_accuracy.png`: 每个类别的预测准确率柱状图。
- **预测结果**：
  ```
  样本 1：预测类别 = airplane (概率：0.7523)
  测试集准确率：0.6789
  ```

## 注意事项

- **硬件要求**: 需要 NVIDIA GPU 和 CUDA 支持以使用 CuPy。若无 GPU，可尝试将 CuPy 替换为 NumPy（参考 `src_numpy/` 中的代码）。
- **权重文件**: 确保 `experiments/best_model_weights.npz` 存在，否则预测和可视化可能使用随机权重。
- **路径配置**: 默认路径基于 `/home/spoil/cv/assignment01/`，请根据实际环境调整。
- **批归一化参数历史**: `src/visualize.py` 中批归一化参数历史为模拟数据，实际使用需在训练时记录。
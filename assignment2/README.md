# 数字图像处理作业 2 - DIP with PyTorch 实验报告

本实验包含两个核心任务：基于 PyTorch 优化实现的传统图像处理方法（泊松图像编辑）以及基于全卷积网络（FCN）的端到端图像翻译（Pix2Pix）。

---

## 1. 实验环境

- **操作系统**: Windows 11
- **编程环境**: Python 3.9 / PyTorch 2.0+ / CUDA (可选)
- **主要依赖库**: 
    - `torch`, `torchvision`: 核心深度学习框架
    - `gradio`: 交互式前端界面
    - `opencv-python`, `pillow`: 图像处理与读取
    - `numpy`: 矩阵计算

---

## 2. 任务一：泊松图像编辑 (Poisson Image Editing)

### 2.1 实验原理
泊松图像融合（Poisson Blending）不再是简单的像素加权平均，而是通过保持图像的**梯度（Gradient）**不变来实现无缝衔接。
本实验利用 PyTorch 的自动求导功能，将融合过程转化为一个优化问题。目标函数是使融合区域的拉普拉斯算子（二阶梯度）与源图像的拉普拉斯算子之间的均方误差（MSE）最小。

### 2.2 核心实现
- **`create_mask_from_points`**: 利用多边形顶点坐标，调用 `PIL.ImageDraw` 在内存中绘制填充多边形，生成二值化掩码。
- **`cal_laplacian_loss`**: 
    - 定义拉普拉斯卷积核 `[[0,1,0],[1,-4,1],[0,1,0]]`。
    - 使用 `torch.nn.functional.conv2d` 对前景图和混合图分别进行卷积提取纹理特征。
    - 计算两者的 L2 损失。

### 2.3 运行指令
在终端执行以下脚本启动 Gradio 界面：
```bash
python run_blending_gradio.py
```

### 2.4 实验结果记录
<img width="1263" height="669" alt="Desktop Screenshot 2026 04 13 - 18 14 18 74(1)" src="https://github.com/user-attachments/assets/14f7d007-2715-4608-808c-9b325f0d8249" />



---

## 3. 任务二：Pix2Pix 图像翻译 (FCN 实现)

### 3.1 实验原理
Pix2Pix 属于图像到图像的转换任务（Image-to-Image Translation）。本实验使用全卷积网络（FCN）作为生成模型，学习从语义标注图（Semantic Map）到真实建筑照片（Real Photo）的非线性映射。

### 3.2 算法实现
- **数据加载 (`facades_dataset.py`)**: 实现 `Dataset` 类，读取 `512x256` 的拼接图并将其切分为输入和目标，归一化至 `[-1, 1]`。
- **网络结构 (`FCN_network.py`)**: 
    - 构建了 6 层 Encoder（下采样）和 6 层 Decoder（上采样）。
    - 编码器使用 `LeakyReLU` 以保留更多梯度信息。
    - 解码器使用 `ReLU` 并通过转置卷积 `ConvTranspose2d` 恢复尺寸。
    - 最后一层使用 `Tanh` 激活函数。
- **损失函数**: 采用像素级的 L1 Loss。

### 3.3 训练过程
1. **生成路径索引**: 需要下载facades数据集后（链接：http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz），将facades文件夹移动至Pix2Pix文件夹下，然后运行 `gen_list.py`。
2. **启动训练**:
```bash
python train.py
```

### 3.4 实验结果分析

#### 3.4.1 损失函数收敛情况
> 📋 **记录不同阶段的 Loss 数值**
> 这里只展示了第297epoch之后的结果：
> <img width="483" height="600" alt="QQ_1776080446775" src="https://github.com/user-attachments/assets/0f34d05c-c5ef-486d-acf6-42b435bb138d" />




---

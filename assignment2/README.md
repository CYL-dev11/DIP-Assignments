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
> 📋 **此处请填入你的实验截图路径或直接粘贴图片**

| 场景描述 | 原始背景图像 | 前景选择 (Mask) | 无缝融合结果 |
| :--- | :--- | :--- | :--- |
| 示例  | ![bg1](
![OIP](https://github.com/user-attachments/assets/da6c6937-d4fa-4ca8-85c6-adf720744bc7)
) | ![mask1](![OIP (1)](https://github.com/user-attachments/assets/254c7c2b-2e84-4aa0-9a5d-85d065f0f03c)) | ![res1](<img width="1263" height="669" alt="Desktop Screenshot 2026 04 13 - 18 14 18 74(1)" src="https://github.com/user-attachments/assets/642bf5ab-2262-43f9-9468-ef2e5dfc1222" />
) |


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
1. **生成路径索引**: 需要下载facades数据集后，将facades文件夹移动至Pix2Pix文件夹下，然后运行 `gen_list.py`。
2. **启动训练**:
```bash
python train.py
```

### 3.4 实验结果分析

#### 3.4.1 损失函数收敛情况
> 📋 **记录不同阶段的 Loss 数值**

| 训练轮次 (Epoch) | 训练集 Loss (L1) | 验证集 Loss (L1) |
| :--- | :--- | :--- |
| 10 | | |
| 150 | | |
| 300 | | |

#### 3.4.2 验证集生成效果对比
> 📋 **引用 val_results 文件夹中的 epoch_300 结果对比图**

| 样本编号 | 输入 (Input) | 真值 (Ground Truth) | 模型预测 (Output) |
| :--- | :--- | :--- | :--- |
| 样本 1 | ![in1]() | ![gt1]() | ![out1]() |
| 样本 2 | ![in2]() | ![gt2]() | ![out2]() |

---

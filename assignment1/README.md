# Assignment 1: Image Warping 实验报告

作业来源：[DIP-Teaching Assignments/01_ImageWarping](https://github.com/YudongGuo/DIP-Teaching/blob/main/Assignments/01_ImageWarping/README.md)

学号：SA25001037

## 1. 实验目的

本实验实现图像几何变换和基于控制点的图像形变，主要包括两个任务：

- Task 1：实现基础图像全局几何变换，包括缩放、旋转、平移和水平翻转。
- Task 2：实现点导向图像形变，通过用户指定的 source/target 控制点对图像进行局部变形。

实验使用 `Gradio` 搭建交互界面，便于上传图像、调整参数和观察变换结果。

## 2. 文件结构

```text
assignment1/
├── README.md
├── run_global_transform.py
└── run_point_transform.py
```

## 3. 实验环境

主要依赖：

```text
opencv-python
numpy
gradio
```

安装依赖：

```bash
pip install opencv-python numpy gradio
```

## 4. Task 1: Basic Image Geometric Transformation

### 4.1 实现目标

`run_global_transform.py` 实现了一个交互式全局图像变换工具。用户可以通过滑块和复选框控制：

- Scale：图像缩放比例
- Rotation：旋转角度
- Translation X/Y：水平和竖直方向平移
- Flip Horizontal：水平翻转

### 4.2 实现方法

代码中先对输入图像进行白色 padding，减少旋转和平移时图像内容被裁剪的问题。随后构造多个齐次坐标变换矩阵，并进行组合：

```text
M = T * R_scale * F
```

其中：

- `T` 是平移矩阵。
- `R_scale` 是绕图像中心的旋转与缩放矩阵。
- `F` 是绕图像中心的水平翻转矩阵。

最终取组合矩阵的前两行作为 OpenCV 所需的 `2 x 3` 仿射矩阵，并调用：

```python
cv2.warpAffine(image, M_affine, (w, h), borderValue=(255, 255, 255))
```

完成图像变换。边界颜色设置为白色，保证输出图像背景与 padding 区域一致。

### 4.3 运行方式

```bash
python run_global_transform.py
```

运行后在 Gradio 页面中上传图片，并调整各个变换参数即可得到结果。

### 4.4 实验结果

下图展示了基础几何变换的交互界面和不同参数下的变换效果。

<img width="1280" height="720" alt="Desktop Screenshot 2026 03 23 - 15 25 10 77" src="https://github.com/user-attachments/assets/3c74a754-9cc8-461f-93af-3233031ba05d" />

<img width="1280" height="720" alt="Desktop Screenshot 2026 03 23 - 15 29 33 13" src="https://github.com/user-attachments/assets/7f160214-0988-461a-9a56-389f24e05b2e" />

<img width="1280" height="720" alt="Desktop Screenshot 2026 03 23 - 15 29 46 54" src="https://github.com/user-attachments/assets/aa7c0081-a148-4288-bd79-a6a5e33314cf" />

<img width="1280" height="720" alt="Desktop Screenshot 2026 03 23 - 15 30 15 78" src="https://github.com/user-attachments/assets/4bba41e0-9125-418b-8228-9db662472430" />

### 4.5 结果分析

从实验结果可以看到，图像能够按照用户设置完成缩放、旋转、平移和水平翻转。由于旋转与缩放围绕图像中心进行，图像主体在变换后保持了较自然的位置关系。加入 padding 后，图像旋转时边界裁剪问题得到缓解；同时，使用白色边界填充避免了默认黑色边框对结果的干扰。

## 5. Task 2: Point Based Image Deformation

### 5.1 实现目标

`run_point_transform.py` 实现了基于控制点的图像局部形变。用户在图像上交替点击 source point 和 target point，程序会记录控制点对，并用箭头显示每一组控制点的移动方向。点击 `Run Warping` 后，程序根据控制点对生成形变后的图像。

### 5.2 实现方法

本实验采用 Moving Least Squares 中的 rigid MLS 思路进行点导向形变。为了避免 forward mapping 产生空洞，代码采用 backward mapping：

- 输出图像中的每个像素记为 `v`。
- 使用目标控制点作为输出空间中的点 `p`。
- 使用源控制点作为输入空间中的点 `q`。
- 对每个输出像素计算其对应的输入图像采样位置 `u`。

权重计算为：

```text
w_i = 1 / (||p_i - v||^2 + eps)
```

然后计算加权中心：

```text
p* = sum(w_i p_i) / sum(w_i)
q* = sum(w_i q_i) / sum(w_i)
```

在得到 backward mapping 的源坐标后，程序使用双线性插值从原图采样像素值。这样可以减少输出图像中的空洞，并得到较平滑的形变结果。

### 5.3 运行方式

```bash
python run_point_transform.py
```

操作流程：

1. 上传输入图像。
2. 在图像上依次点击 source point 和 target point。
3. 至少选择 3 组控制点。
4. 点击 `Run Warping` 得到点导向形变结果。
5. 点击 `Clear Points` 可清除当前控制点并重新选择。

### 5.4 实验结果

下图展示了点导向形变的结果。蓝色点表示 source point，红色点表示 target point，绿色箭头表示控制点移动方向。

<img width="1280" height="720" alt="Desktop Screenshot 2026 03 23 - 15 42 00 95" src="https://github.com/user-attachments/assets/aa257f67-0a60-45bc-8428-cbefa108cb6b" />

### 5.5 结果分析

从实验结果可以看到，控制点附近的图像区域会按照用户指定的方向发生明显形变，而远离控制点的区域变化较小。这符合 MLS 方法的局部加权特性：距离当前像素越近的控制点权重越大，对该像素的形变影响也越强。

本实现采用逐像素 Python 循环，逻辑直观，便于理解 MLS 的计算过程；但当输入图像分辨率较高时，运行速度会变慢。若进一步优化，可以使用 NumPy 向量化或 GPU 加速来提升形变计算效率。

## 6. 实验总结

本实验完成了图像全局几何变换和点导向局部形变两个任务。Task 1 中，通过组合仿射矩阵实现了缩放、旋转、平移和翻转；Task 2 中，通过 rigid MLS 和 backward mapping 实现了基于控制点的图像形变。

实验结果表明，全局变换适合处理整幅图像的一致性几何操作，而点导向变形可以根据用户指定的控制点产生局部、非刚性的图像变形。两者共同展示了 image warping 在数字图像处理中的基本思想和典型应用方式。

## 7. 参考资料

- [DIP-Teaching Assignment 1: Image Warping](https://github.com/YudongGuo/DIP-Teaching/blob/main/Assignments/01_ImageWarping/README.md)
- [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf)
- [OpenCV Geometric Transformations](https://docs.opencv.org/)

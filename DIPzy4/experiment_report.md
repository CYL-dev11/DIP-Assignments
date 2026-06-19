# Assignment 4: Simplified 3D Gaussian Splatting 实验报告

作业来源：[DIP-Teaching Assignments/04_3DGS](https://github.com/YudongGuo/DIP-Teaching/tree/main/Assignments/04_3DGS)

实验目录：`C:\Users\ASUS\Desktop\DIPzy4`

本报告记录的是 2026-06-17 在本机实际运行 Task 1 到 Task 3 后得到的结果。

## 1. 实验目标

本实验实现一个纯 PyTorch 的简化版 3D Gaussian Splatting pipeline，并与官方 3DGS 实现进行对比：

- Task 1：使用 COLMAP 从多视角图像恢复相机参数与稀疏 3D 点云，并通过重投影检查结果。
- Task 2：实现简化 3DGS 的核心数学模块，包括 3D Gaussian 协方差、3D 到 2D 投影、2D Gaussian 取值和 alpha blending 渲染，并完成训练。
- Task 3：在同一 `chair` 序列上运行官方 3DGS，对比渲染质量、训练速度和显存/计算开销。

默认数据集为 `data/chair/images`，共 100 张多视角图像。

## 2. 文件结构

```text
DIPzy4/
├── data/chair/images/                 # chair 输入图像
├── data/chair/projections/            # Task 1 重投影调试图
├── data/chair/checkpoints/            # Task 2 简化版训练输出
├── data/chair/official_3dgs_output/   # 官方 3DGS 非 eval 运行输出
├── data/chair/official_3dgs_eval_output/ # 官方 3DGS eval 输出
├── data_utils.py
├── debug_mvs_by_projecting_pts.py
├── gaussian_model.py
├── gaussian_renderer.py
├── mvs_with_colmap.py
├── render_3dgs_mv.py
├── train.py
├── verify_implementation.py
├── run_task1_colmap.ps1
├── run_task2_train_simplified.ps1
├── run_task3_official_3dgs.ps1
├── requirements.txt
└── experiment_report.md
```

## 3. Task 1: COLMAP 相机恢复与重投影检查

运行命令：

```powershell
cd C:\Users\ASUS\Desktop\DIPzy4
.\run_task1_colmap.ps1
```

脚本执行了：

```powershell
python mvs_with_colmap.py --data_dir data/chair
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

本次运行输出：

| 项目 | 结果 |
|---|---:|
| 输入图像数量 | 100 |
| COLMAP 注册图像数量 | 100 |
| 稀疏 3D 点数量 | 13,729 |
| 重投影检查图数量 | 100 |

生成的主要文件包括：

- `data/chair/database.db`
- `data/chair/sparse/0/`
- `data/chair/sparse/0_text/cameras.txt`
- `data/chair/sparse/0_text/images.txt`
- `data/chair/sparse/0_text/points3D.txt`
- `data/chair/projections/*.png`

`projections` 目录中的图像用于检查稀疏点云投影回各视角图像后的位置是否合理。本次 COLMAP 成功注册全部 100 张图像，说明相机位姿和稀疏点云可作为 3DGS 初始化输入。

## 4. Task 2: 简化版 3D Gaussian Splatting

### 4.1 3D Gaussian 协方差

在 `gaussian_model.py` 中，每个 Gaussian 的尺度参数使用 log-space 优化，旋转参数使用四元数。实现时先将四元数归一化并转为旋转矩阵 `R`，再将尺度指数化构造对角矩阵 `S`。

实现公式：

```text
Sigma = R S S^T R^T
```

核心代码：

```python
Covs3d = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
```

### 4.2 3D Gaussian 投影到 2D

在 `gaussian_renderer.py` 中，先将世界坐标变换到相机坐标：

```text
x_cam = R x_world + t
```

针孔相机投影为：

```text
u = fx * x / z + cx
v = fy * y / z + cy
```

对应 Jacobian：

```text
[ fx/z,   0, -fx*x/z^2 ]
[   0, fy/z, -fy*y/z^2 ]
```

世界坐标系下的 3D 协方差先旋转到相机坐标系：

```text
Sigma_cam = R Sigma_world R^T
```

再投影到 2D：

```text
Sigma_2D = J Sigma_cam J^T
```

### 4.3 2D Gaussian 取值

对每个像素 `x`，根据 2D Gaussian 的均值和协方差计算：

```text
f(x) = 1 / (2 pi sqrt(|Sigma|)) * exp(-0.5 (x - mu)^T Sigma^-1 (x - mu))
```

实现中加入了对角线 epsilon、determinant clamp 和 exponent clamp，避免协方差接近奇异时出现 NaN 或 Inf。

### 4.4 Alpha Blending 渲染

深度从近到远排序后，使用前向 alpha compositing：

```text
alpha_i = opacity_i * gaussian_i
T_i = product_{j<i}(1 - alpha_j)
weight_i = alpha_i * T_i
color = sum_i weight_i * color_i
```

实现中将 `alpha` clamp 到 `[0, 0.999]`，避免透射率乘积出现数值不稳定。

### 4.5 训练设置与结果

运行命令：

```powershell
cd C:\Users\ASUS\Desktop\DIPzy4
.\run_task2_train_simplified.ps1 -Epochs 200 -DebugEvery 10
```

由于纯 PyTorch renderer 会显式构造形状接近 `(N, H, W)` 的中间张量，当 COLMAP 点数过大时会非常慢且占用大量显存。因此 `data_utils.py` 中对稀疏点云进行 farthest-point sampling，将训练点数控制为 3000。

本次训练设置：

| 项目 | 设置 |
|---|---:|
| 输入图像数量 | 100 |
| 训练 Gaussian 数量 | 3000 |
| 训练图像分辨率 | 100 x 100 |
| 训练 epoch 数 | 200 |
| debug 渲染间隔 | 10 epoch |

本次运行结果：

| 项目 | 结果 |
|---|---|
| 训练状态 | 完成 epoch 0 到 epoch 199 |
| 最后一个周期 checkpoint | `checkpoint_000180.pt` |
| debug 图像 | 生成到 `debug_images/epoch_0190.png` |
| 最终视频 | `data/chair/checkpoints/debug_rendering.mp4` |
| 最后一轮进度条显示的平均 loss | 约 0.0422 |

说明：训练程序默认每 20 个 epoch 保存一次 checkpoint，因此最后一个周期性保存的模型是 epoch 180；训练结束后使用 epoch 199 的内存中模型生成了 `debug_rendering.mp4`。

## 5. 代码正确性检查

运行命令：

```powershell
cd C:\Users\ASUS\Desktop\DIPzy4
python verify_implementation.py
```

本次快速检查输出：

```text
init_scales tensor(1.3333) tensor(1.6095)
GaussianModel covariance: (4, 3, 3) positive definite
GaussianRenderer output: (8, 8, 3) range [0.000000, 0.999000]
All implementation checks passed.
```

数据集读取检查：

```text
images: 100
sampled points3D: (3000, 3)
downsampled image shape: (100, 100, 3)
```

检查结论：

- 3D 协方差矩阵 shape 正确，且特征值为正，说明构造出的 Gaussian 协方差是正定的。
- renderer 输出 shape 为 `(H, W, 3)`，数值有限且落在 `[0, 1]` 范围内。
- 主要计算均由 PyTorch Tensor 运算构成，可以反向传播。

## 6. Task 3: 与官方 3DGS 对比

官方 3DGS repo 路径：

```text
C:\Users\ASUS\Desktop\Stage 1_Pipeline\gaussian-splatting
```

有效对比实验使用 eval split 运行：

```powershell
cd C:\Users\ASUS\Desktop\DIPzy4
.\run_task3_official_3dgs.ps1 -Iterations 7000 -Eval
```

输出目录：

```text
data/chair/official_3dgs_eval_output
```

官方 3DGS 本次 eval 运行设置与输出：

| 项目 | 结果 |
|---|---:|
| 初始 COLMAP 点数量 | 13,729 |
| 训练视角数量 | 87 |
| 测试视角数量 | 13 |
| 训练迭代数 | 7000 |
| 官方训练结束时 test PSNR | 12.1601 |
| 官方 metrics PSNR | 12.1591 |
| 官方 metrics SSIM | 0.4371 |
| 官方 metrics LPIPS | 0.1371 |

`results.json` 中记录的指标：

```json
{
  "ours_7000": {
    "SSIM": 0.4370569586753845,
    "PSNR": 12.159066200256348,
    "LPIPS": 0.13711874186992645
  }
}
```

同时生成：

- `data/chair/official_3dgs_eval_output/chkpnt7000.pth`
- `data/chair/official_3dgs_eval_output/point_cloud/iteration_7000/point_cloud.ply`
- `data/chair/official_3dgs_eval_output/train/ours_7000/renders/`
- `data/chair/official_3dgs_eval_output/test/ours_7000/renders/`
- `data/chair/official_3dgs_eval_output/results.json`
- `data/chair/official_3dgs_eval_output/per_view.json`

补充说明：曾先运行一次不带 `-Eval` 的官方 3DGS。该次运行能够训练和渲染，但由于没有测试集划分，`metrics.py` 对 test set 计算得到 NaN。因此正式对比采用上面带 `-Eval` 的运行结果。

### 6.1 渲染质量对比

简化版实现主要用于教学验证，颜色模型为每个 Gaussian 的 RGB 常量颜色，没有 spherical harmonics 视角相关颜色，也没有 adaptive densification 和 pruning。它可以学习到整体形状和颜色趋势，但在边界锐度、高频纹理、遮挡细节和视角相关外观方面较弱。

官方 3DGS 使用 CUDA rasterizer、spherical harmonics、densification/pruning 等完整工程组件。本次 7000 iteration 的官方 eval 结果为 PSNR 12.1591、SSIM 0.4371、LPIPS 0.1371。由于简化版训练使用 100 x 100 下采样图像和 3000 个固定 Gaussian，没有直接生成与官方 test split 完全一致的 PSNR/SSIM/LPIPS，因此两者的量化指标不做直接等价比较。

### 6.2 训练速度对比

简化版 renderer 的核心复杂度接近：

```text
O(N * H * W)
```

其中 `N` 是 Gaussian 数量，`H, W` 是图像分辨率。它直接在所有 Gaussian 和所有像素之间计算密度，公式透明但速度较慢。本次简化版在 3000 个 Gaussian 和 100 x 100 分辨率下完成 200 epoch，后半段从 epoch 141 到 199 的恢复训练耗时约 5148 秒，折合约 87 秒/epoch。

官方 3DGS 使用 CUDA tile-based rasterizer，只处理与 tile/pixel 有覆盖关系的 Gaussian，并且有高度优化的排序和混合流程。本次官方 eval 从训练、渲染到 metrics 完整运行约 395 秒，明显快于简化版。

### 6.3 显存与计算开销对比

简化版会显式构造类似 `(N, H, W)` 的 Gaussian value、alpha 和 weight 张量。以本次设置 `N=3000, H=W=100` 为例，仅单个 float32 中间张量就约为：

```text
3000 * 100 * 100 * 4 bytes ≈ 120 MB
```

实际训练中还需要保存多个中间张量和梯度，因此显存会随 Gaussian 数量和分辨率快速增长。这也是本实验将 COLMAP 点云采样到 3000 点、并把图像下采样到 100 x 100 的原因。

官方 3DGS 使用 CUDA kernel、tile lists 和更紧凑的中间状态，避免为所有 Gaussian 和所有像素构造全局稠密张量，因此更适合高分辨率和大规模 Gaussian 场景。

### 6.4 对比总结

| 项目 | 简化 PyTorch 3DGS | 官方 3DGS |
|---|---|---|
| 实现目标 | 教学清晰，公式直观 | 高质量、高效率重建 |
| 输入点数 | COLMAP 点云采样到 3000 | 使用 13,729 个初始点并可 densify |
| 训练分辨率 | 100 x 100 | 原始 800 x 800 图像 |
| Rasterizer | 纯 PyTorch 全局 per-pixel 计算 | CUDA tile-based rasterizer |
| 颜色模型 | RGB 常量颜色 | Spherical harmonics |
| Gaussian 数量控制 | 固定数量，无 densification | 支持 densification/pruning |
| 本次运行速度 | 后半段约 87 秒/epoch | 7000 iter 完整流程约 395 秒 |
| 定量指标 | 使用 loss 和 debug render 验证 | PSNR 12.1591, SSIM 0.4371, LPIPS 0.1371 |
| 显存效率 | 较低，显式 `(N,H,W)` 中间张量 | 较高，CUDA tile 化实现 |
| 适用场景 | 理解 3DGS 原理、小规模实验 | 完整重建、真实实验和高质量渲染 |

## 7. 结论

本实验完成了 Task 1 到 Task 3 的完整流程。Task 1 中 COLMAP 成功注册 100 张图像并恢复 13,729 个稀疏 3D 点；Task 2 中补全了简化 3DGS 的协方差、投影、Gaussian 取值和 alpha blending，并完成 200 epoch 训练，生成 checkpoint、debug 图像和最终视频；Task 3 中使用官方 3DGS 在同一序列上完成 7000 iteration eval 运行，得到 PSNR 12.1591、SSIM 0.4371、LPIPS 0.1371。

简化版的主要价值在于公式透明、代码短、便于理解 3DGS 的核心思想；官方 3DGS 的主要优势在于工程优化完整、训练速度快、显存效率高，并能通过 densification 和 spherical harmonics 得到更好的细节质量。

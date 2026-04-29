# Assignment 1 - 3D Reconstruction: Bundle Adjustment vs. COLMAP

作业实现了从多视角 2D 观测恢复 3D 头部模型的两种方案：一是基于 PyTorch 的手动 Bundle Adjustment (BA) 优化实现，二是基于工业级软件 COLMAP 的自动化三维重建流水线。

> 📋 实验目标：对比底层参数优化（Task 1）与端到端视觉重建系统（Task 2）的差异与局限性。

## Requirements

要运行本项目，请确保安装以下环境：

```bash
# 基础 Python 环境
pip install torch numpy tqdm matplotlib open3d

# Task 1 推荐安装 (用于旋转矩阵转换)
pip install pytorch3d

# Task 2 需要安装 COLMAP 并将其 bin 目录加入系统路径 (PATH)
```

## Task 1: Implement Bundle Adjustment with PyTorch

### 实现方案
1. **参数化**：
   - 焦距 $f$：全相机共享的单参数优化。
   - 相机外参：每组视角使用 Euler 角 ($XYZ$ 顺序) 和平移向量 $T$ 进行参数化。
   - 3D 点云：20,000 个 3D 坐标 $(X, Y, Z)$。
2. **投影模型**：
   采用透视投影公式：$u = -f \cdot \frac{X_c}{Z_c} + c_x, \quad v = f \cdot \frac{Y_c}{Z_c} + c_y$。
3. **优化目标**：
   最小化可见点（Visibility = 1.0）的重投影误差：
   $$\mathcal{L} = \sum_{i,j} \text{vis}_{i,j} \cdot \| \text{proj}(P_j, K_i, R_i, T_i) - x_{i,j} \|^2$$

### 运行命令
```bash
# 运行 PyTorch 优化脚本
python task1_ba.py
```

## Task 2: 3D Reconstruction with COLMAP

### 流程说明
使用 COLMAP 命令行工具进行完整的重建，包括以下步骤：
1. **特征提取与匹配**：使用 SIFT 算子提取图像特征，并通过穷举匹配（Exhaustive Matching）建立关联。
2. **稀疏重建**：通过增量式 Mapper 恢复相机位姿和稀疏点云。
3. **稠密重建**：通过 Patch Match Stereo 计算深度图，并进行立体融合（Stereo Fusion）生成最终的 `.ply` 模型。

### 运行命令
```bash
# 运行自动化重建脚本
bash run_colmap.sh
```

## Results

### 1. Loss 变化曲线 (Task 1)
> 📋 运行结束后生成的 Loss 曲线截图。

| 初始 Loss | 最终 Loss | 优化时长 | 恢复焦距 $f$ |
| :--- | :--- | :--- | :--- |
| ~4.5e5 | < 0.5 | ~5 mins | ~912.4 |

### 2. 重建结果对比分析

我们将 Task 1 的参数优化结果与 Task 2 的稠密重建结果进行可视化对比：

| 方案 | 稀疏/稠密结果图 | 结果特点 |
| :--- | :--- | :--- |
| **Task 1: PyTorch BA** | ![BA_Result](https://via.placeholder.com/300) | 模型极其完整，覆盖了衣服等所有区域。 |
| **Task 2: COLMAP** | ![COLMAP_Result](https://via.placeholder.com/300) | 皮肤细节丰富，但黑色衣服区域存在明显空洞。 |

### 3. 对比讨论：为什么 COLMAP 结果存在“空洞”？
实验发现 COLMAP 在处理黑色衣服区域时效果不如 Task 1 完整，原因如下：
- **特征点缺失**：黑色衣服属于“弱纹理区域”，COLMAP 依赖的 SIFT 算子难以在无明显纹理的暗色区域提取到稳定的特征点，导致无法计算匹配。
- **先验信息差异**：Task 1 使用了预设的 20,000 个点坐标作为先验，优化过程仅需对齐坐标；而 Task 2 必须根据像素内容从零计算深度。
- **鲁棒性过滤**：COLMAP 在稠密融合阶段会剔除光度一致性较低的区域，以保证生成的 3D 模型精度，而这在低对比度的衣服区域会导致点云丢失。

## Pre-trained / Reconstructed Models

你可以通过以下方式查看重建的 3D 模型：
- **Task 1**: [reconstructed_head.obj](./reconstructed_head.obj) (带顶点的颜色)
- **Task 2**: [fused.ply](./data/colmap/dense/fused.ply) (高精度点云)

> 📋 建议使用 [Online 3D Viewer](https://3dviewer.net/) 或 MeshLab 查看以上文件。

## Contributing

- 算法实现：[你的名字]
- 框架支持：PyTorch, COLMAP

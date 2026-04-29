# Assignment 3 - 3D Reconstruction: Bundle Adjustment vs. COLMAP

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
- **参数化**：焦距 $f$、相机外参 $(R, T)$、3D 点云坐标 $P$。
- **投影模型**：
  采用透视投影公式，将相机坐标系下的点 $[X_c, Y_c, Z_c]$ 投影至像素坐标 $[u, v]$：
  
  $$u = -f \cdot \frac{X_c}{Z_c} + c_x, \quad v = f \cdot \frac{Y_c}{Z_c} + c_y$$

- **优化目标**：
  最小化所有视角下可见点的重投影误差（Reprojection Error）：

  
  $$
  \mathcal{L} = \sum_{i,j} \text{vis}_{i,j} \cdot \| \text{proj}(P_j, K_i, R_i, T_i) - x_{i,j} \|^2
  $$
  

### 运行命令
```bash
# 运行 PyTorch 优化脚本
python reconstructed_head.py
```

## Task 2: 3D Reconstruction with COLMAP

### 流程说明
使用 COLMAP 命令行工具进行完整的重建，包括以下步骤：
1. **特征提取与匹配**：使用 SIFT 算子提取图像特征，并通过穷举匹配（Exhaustive Matching）建立关联。
2. **稀疏重建**：通过增量式 Mapper 恢复相机位姿和稀疏点云。
3. **稠密重建**：通过 Patch Match Stereo 计算深度图，并进行立体融合（Stereo Fusion）生成最终的 `.ply` 模型。

### 运行命令
```bash
mkdir data/colmap/sparse -Force
mkdir data/colmap/dense -Force
colmap feature_extractor --database_path data/colmap/database.db --image_path data/images --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1
colmap exhaustive_matcher --database_path data/colmap/database.db
colmap mapper --database_path data/colmap/database.db --image_path data/images --output_path data/colmap/sparse
colmap image_undistorter --image_path data/images --input_path data/colmap/sparse/0 --output_path data/colmap/dense
colmap patch_match_stereo --workspace_path data/colmap/dense
colmap stereo_fusion --workspace_path data/colmap/dense --output_path data/colmap/dense/fused.ply
```

## Results

### 1. Loss 变化曲线 (Task 1)
> 📋 运行结束后生成的 Loss 曲线截图。

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/4f69ed8e-6c4b-4a51-b427-499966590a97" />


### 2. 重建结果对比分析

我们将 Task 1 的参数优化结果与 Task 2 的稠密重建结果进行可视化对比：



**Task 1: PyTorch BA**
<img width="1593" height="1152" alt="b611dc877c7cbe70c9b12ff1f93920e7" src="https://github.com/user-attachments/assets/19a3d68b-2244-4dac-a038-0a3109825fd4" />
<img width="1902" height="1185" alt="e1e1ce222284458401d2b2c13cbf2eb1" src="https://github.com/user-attachments/assets/81c39a30-5e5a-463a-a0c0-64e2cf5f4e6c" />
<img width="1887" height="1185" alt="8d5e090192e50b4e520264ceddcde3a9" src="https://github.com/user-attachments/assets/258e4293-8224-4af5-ad4d-8446568a9e3f" />


模型极其完整，覆盖了衣服等所有区域。 

**Task 2: COLMAP** 
<img width="1917" height="1293" alt="cac3cb65b75c48c1c5a29ae8c3bf524d" src="https://github.com/user-attachments/assets/e1bc6ad7-bd7e-4b18-8c74-abee597de491" />
<img width="1914" height="1290" alt="226d7f418509b62b952f47e4a7258290" src="https://github.com/user-attachments/assets/b8eb13be-acef-43bf-a673-2c9fa6724c55" />
<img width="1914" height="1287" alt="500ca5776808e40381465f42684437cf" src="https://github.com/user-attachments/assets/eb58f5cc-7b76-4b0a-9e94-edd5b2945f4e" />

皮肤细节丰富，但黑色衣服区域存在明显空洞。

### 3. 对比讨论：为什么 COLMAP 结果存在“空洞”？
实验发现 COLMAP 在处理黑色衣服区域时效果不如 Task 1 完整，原因如下：
- **特征点缺失**：黑色衣服属于“弱纹理区域”，COLMAP 依赖的 SIFT 算子难以在无明显纹理的暗色区域提取到稳定的特征点，导致无法计算匹配。
- **先验信息差异**：Task 1 使用了预设的 20,000 个点坐标作为先验，优化过程仅需对齐坐标；而 Task 2 必须根据像素内容从零计算深度。
- **鲁棒性过滤**：COLMAP 在稠密融合阶段会剔除光度一致性较低的区域，以保证生成的 3D 模型精度，而这在低对比度的衣服区域会导致点云丢失。



## Contributing
- 框架支持：PyTorch, COLMAP

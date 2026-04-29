import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from pytorch3d.transforms import euler_angles_to_matrix
except ImportError:
    def euler_angles_to_matrix(euler_angles, convention="XYZ"):
        """手动实现简单的 XYZ Euler 转 旋转矩阵"""
        x, y, z = euler_angles.unbind(-1)
        cx, sx = x.cos(), x.sin()
        cy, sy = y.cos(), y.sin()
        cz, sz = z.cos(), z.sin()
        
        # Rx, Ry, Rz
        zeros = torch.zeros_like(cx)
        ones = torch.ones_like(cx)
        
        Rx = torch.stack([ones,  zeros, zeros,
                          zeros, cx,   -sx,
                          zeros, sx,    cx], dim=-1).reshape(-1, 3, 3)
        Ry = torch.stack([cy,    zeros, sy,
                          zeros, ones,  zeros,
                          -sy,   zeros, cy], dim=-1).reshape(-1, 3, 3)
        Rz = torch.stack([cz,   -sz,    zeros,
                          sz,    cz,    zeros,
                          zeros, zeros, ones], dim=-1).reshape(-1, 3, 3)
        return torch.bmm(Rx, torch.bmm(Ry, Rz))

# 1. 加载数据
def load_data(data_path='data/'):
    points2d_data = np.load(os.path.join(data_path, 'points2d.npz'))
    # 加载 50 个视角的数据，形状 (50, 20000, 3)
    obs_2d = np.stack([points2d_data[f'view_{i:03d}'] for i in range(50)])
    colors = np.load(os.path.join(data_path, 'points3d_colors.npy')) # (20000, 3)
    return torch.from_numpy(obs_2d).float(), torch.from_numpy(colors).float()

# 2. 定义投影函数
def project_points(points_3d, focal_length, euler_angles, translations, img_size=(1024, 1024)):
    """
    points_3d: (N, 3)
    focal_length: (1,)
    euler_angles: (V, 3)
    translations: (V, 3)
    """
    # 转旋转矩阵 (V, 3, 3)
    R = euler_angles_to_matrix(euler_angles, convention="XYZ")
    
    # 变换到相机坐标系: [Xc, Yc, Zc] = R @ P + T
    # 使用广播机制处理 50 个相机和 20000 个点
    # (V, 1, 3, 3) @ (1, N, 3, 1) -> (V, N, 3)
    pc = torch.matmul(R.unsqueeze(1), points_3d.unsqueeze(0).unsqueeze(-1)).squeeze(-1) + translations.unsqueeze(1)
    
    xc, yc, zc = pc[..., 0], pc[..., 1], pc[..., 2]
    
    cx, cy = img_size[0] / 2, img_size[1] / 2
    
    # 根据提示公式投影
    u = -focal_length * xc / zc + cx
    v = focal_length * yc / zc + cy
    
    return torch.stack([u, v], dim=-1)

# 3. 优化与评估
def run_bundle_adjustment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_2d_full, colors = load_data()
    obs_2d_full = obs_2d_full.to(device)
    
    num_views = 50
    num_points = 20000
    
    # --- 参数初始化 ---
    # 焦距初始化: 假设 FoV=60度, f = 1024 / (2 * tan(30)) 约为 886
    f = nn.Parameter(torch.tensor([900.0], device=device))
    # 相机外参: Euler角初始化为0，平移初始化为 [0, 0, -2.5]
    eulers = nn.Parameter(torch.zeros((num_views, 3), device=device))
    trans = nn.Parameter(torch.tensor([[0.0, 0.0, -2.5]], device=device).repeat(num_views, 1))
    # 3D点初始化: 原点附近的随机位置
    points_3d = nn.Parameter(torch.randn((num_points, 3), device=device) * 0.1)
    
    optimizer = optim.Adam([f, eulers, trans, points_3d], lr=0.01)
    
    # 分解观测数据
    obs_xy = obs_2d_full[..., :2]       # (50, 20000, 2)
    visibility = obs_2d_full[..., 2:3]  # (50, 20000, 1)
    
    loss_history = []
    
    print("Starting optimization...")
    for epoch in tqdm(range(2000)):
        optimizer.zero_grad()
        
        # 前向投影
        pred_xy = project_points(points_3d, f, eulers, trans)
        
        # 计算重投影误差，仅考虑可见点 (visibility == 1.0)
        error = torch.sum(((pred_xy - obs_xy) ** 2) * visibility)
        loss = error / (torch.sum(visibility) + 1e-6)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, f: {f.item():.2f}")

    # 4. 可视化 Loss 曲线
    plt.figure()
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title('Bundle Adjustment Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Reprojection MSE')
    plt.show()

    # 5. 保存重建结果为带颜色的 OBJ 文件
    save_obj("reconstructed_head.obj", points_3d.detach().cpu().numpy(), colors.numpy())

def save_obj(filename, points, colors):
    """保存为带颜色的 OBJ 格式: v x y z r g b"""
    with open(filename, 'w') as f:
        for p, c in zip(points, colors):
            # OBJ 颜色通常在 [0, 1] 范围，按提示写入
            f.write(f"v {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
    print(f"Result saved to {filename}")

if __name__ == "__main__":
    run_bundle_adjustment()

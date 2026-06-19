"""Quick checks for the simplified 3D Gaussian Splatting implementation."""

import torch

from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer


def check_gaussian_model():
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    rgb = torch.tensor(
        [[255.0, 0.0, 0.0],
         [0.0, 255.0, 0.0],
         [0.0, 0.0, 255.0],
         [255.0, 255.0, 255.0]],
        dtype=torch.float32,
    )
    model = GaussianModel(xyz, rgb)
    cov = model.compute_covariance()
    eigvals = torch.linalg.eigvalsh(cov)
    assert cov.shape == (4, 3, 3)
    assert torch.isfinite(cov).all()
    assert (eigvals > 0).all()
    return cov


def check_renderer():
    renderer = GaussianRenderer(image_height=8, image_width=8)
    means3d = torch.tensor([[0.0, 0.0, 5.0], [0.5, 0.2, 6.0]], dtype=torch.float32)
    covs3d = torch.eye(3).unsqueeze(0).repeat(2, 1, 1) * 0.05
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    opacities = torch.ones(2, 1) * 0.5
    K = torch.tensor([[6.0, 0.0, 4.0], [0.0, 6.0, 4.0], [0.0, 0.0, 1.0]])

    rendered = renderer(
        means3D=means3d,
        covs3d=covs3d,
        colors=colors,
        opacities=opacities,
        K=K,
        R=torch.eye(3),
        t=torch.zeros(3),
    )
    assert rendered.shape == (8, 8, 3)
    assert torch.isfinite(rendered).all()
    assert rendered.min() >= 0.0
    assert rendered.max() <= 1.0
    return rendered


def main():
    cov = check_gaussian_model()
    rendered = check_renderer()
    print("GaussianModel covariance:", tuple(cov.shape), "positive definite")
    print(
        "GaussianRenderer output:",
        tuple(rendered.shape),
        "range",
        f"[{float(rendered.min()):.6f}, {float(rendered.max()):.6f}]",
    )
    print("All implementation checks passed.")


if __name__ == "__main__":
    main()

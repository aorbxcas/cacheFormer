from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def perspective_projection(
    fov_y_deg: torch.Tensor | float,
    aspect: float = 1.0,
    near: float = 0.05,
    far: float = 50.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """OpenGL 风格透视投影（垂直 FOV，角度制）。"""
    if isinstance(fov_y_deg, torch.Tensor):
        fov_y = fov_y_deg.float() * (math.pi / 180.0)
    else:
        fov_y = torch.tensor(float(fov_y_deg) * math.pi / 180.0, device=device, dtype=dtype)
    f = 1.0 / torch.tan(fov_y * 0.5)
    nf = near - far
    P = torch.zeros((4, 4), device=device, dtype=dtype)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (far + near) / nf
    P[2, 3] = (2.0 * far * near) / nf
    P[3, 2] = -1.0
    return P


def look_at_view(
    eye: torch.Tensor,
    target: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """世界 → 相机 4×4（OpenGL：相机 -Z 为观察方向）。"""
    f = F.normalize(target - eye, dim=-1)
    up_n = F.normalize(up, dim=-1)
    r = F.normalize(torch.cross(f, up_n, dim=-1), dim=-1)
    u = torch.cross(r, f, dim=-1)

    V = torch.eye(4, device=eye.device, dtype=eye.dtype)
    V[0, :3] = r
    V[1, :3] = u
    V[2, :3] = -f
    T = torch.eye(4, device=eye.device, dtype=eye.dtype)
    T[:3, 3] = -eye
    return V @ T


def orthographic_projection(
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    P = torch.zeros((4, 4), device=device, dtype=dtype)
    P[0, 0] = 2.0 / (right - left)
    P[1, 1] = 2.0 / (top - bottom)
    P[2, 2] = -2.0 / (far - near)
    P[0, 3] = -(right + left) / (right - left)
    P[1, 3] = -(top + bottom) / (top - bottom)
    P[2, 3] = -(far + near) / (far - near)
    P[3, 3] = 1.0
    return P


def world_to_clip(
    pos_world: torch.Tensor,
    c2w: torch.Tensor,
    fov_y_deg: torch.Tensor | float,
    near: float = 0.05,
    far: float = 50.0,
) -> torch.Tensor:
    device = pos_world.device
    dtype = pos_world.dtype
    w2c = torch.linalg.inv(c2w)
    P = perspective_projection(fov_y_deg, aspect=1.0, near=near, far=far, device=device, dtype=dtype)
    mvp = P @ w2c
    pos_h = torch.cat([pos_world, torch.ones((pos_world.shape[0], 1), device=device, dtype=dtype)], dim=-1)
    return pos_h @ mvp.T


def light_mvp(
    light_eye: torch.Tensor,
    light_target: torch.Tensor,
    half_extent: float,
    near: float = 0.05,
    far: float = 20.0,
) -> torch.Tensor:
    device = light_eye.device
    dtype = light_eye.dtype
    up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    V = look_at_view(light_eye, light_target, up)
    P = orthographic_projection(
        -half_extent,
        half_extent,
        -half_extent,
        half_extent,
        near,
        far,
        device,
        dtype,
    )
    return P @ V


def project_world_to_light_ndc(
    pos_world: torch.Tensor,
    mvp_light: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns clip [N,4], uv [N,2], depth z/w [N,1]."""
    pos_h = torch.cat(
        [pos_world, torch.ones((pos_world.shape[0], 1), device=pos_world.device, dtype=pos_world.dtype)],
        dim=-1,
    )
    clip = pos_h @ mvp_light.T
    w = clip[:, 3:4].clamp(min=1e-8)
    ndc = clip[:, :3] / w
    uv = ndc[:, :2] * 0.5 + 0.5
    return clip, uv, ndc[:, 2:3]

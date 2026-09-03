# -*- coding: utf-8 -*-
"""全分辨率深度重投影：相机运动时跟视图，不降分辨率。"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from renderformer.utils.ray_generator import RayGenerator

_ray_gen = RayGenerator()


def _ensure_fov(fov: torch.Tensor) -> torch.Tensor:
    if fov.dim() == 2:
        return fov.unsqueeze(-1)
    return fov


def analytic_plane_depth(
    c2w: torch.Tensor,
    fov: torch.Tensor,
    resolution: int,
    plane_point: torch.Tensor | None = None,
    plane_normal: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    全分辨率解析平面深度（视线距离），供重投影，无半分辨率、无光线求交。
    默认：过原点、法线朝向相机的平面。
    """
    fov = _ensure_fov(fov)
    b, v = c2w.shape[0], c2w.shape[1]
    device = c2w.device
    rays_o, rays_d = _ray_gen(c2w.float(), torch.deg2rad(fov.float()), resolution)
    if plane_point is None:
        plane_point = torch.zeros(3, device=device, dtype=torch.float32)
    else:
        plane_point = plane_point.to(device=device, dtype=torch.float32)
    if plane_normal is None:
        # 朝向相机：用相机位置作为外侧
        eye = c2w[..., :3, 3]
        plane_normal = F.normalize(eye.mean(dim=(0, 1)) - plane_point, dim=0)
    else:
        plane_normal = F.normalize(plane_normal.to(device=device, dtype=torch.float32), dim=0)

    n = plane_normal.view(1, 1, 1, 1, 3)
    p0 = plane_point.view(1, 1, 1, 1, 3)
    denom = (rays_d * n).sum(-1, keepdim=True)
    denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
    t = ((p0 - rays_o[:, :, None, None, :]) * n).sum(-1, keepdim=True) / denom
    t = t.clamp(0.05, 50.0)
    return t.expand(b, v, resolution, resolution, 1)


def camera_rotation_deg(c2w_a: torch.Tensor, c2w_b: torch.Tensor) -> float:
    """两相机朝向夹角（度），取 batch/view 最大。"""
    ra = c2w_a[..., :3, :3]
    rb = c2w_b[..., :3, :3]
    fa = F.normalize((-ra[..., :, 2]).reshape(-1, 3), dim=-1)
    fb = F.normalize((-rb[..., :, 2]).reshape(-1, 3), dim=-1)
    cos = (fa * fb).sum(-1).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cos)).max().item())


@torch.no_grad()
def warp_depth_forward(
    depth_src: torch.Tensor,
    c2w_src: torch.Tensor,
    fov_src: torch.Tensor,
    c2w_dst: torch.Tensor,
    fov_dst: torch.Tensor,
    *,
    z_eps: float = 1e-4,
) -> torch.Tensor:
    """将 src 深度前向溅射到 dst（nearest，保边缘），空洞用解析深度填。"""
    assert depth_src.dim() == 5
    b, v, h, w, _ = depth_src.shape
    fov_src = _ensure_fov(fov_src)
    fov_dst = _ensure_fov(fov_dst)
    device = depth_src.device
    dtype = torch.float32

    depth_f = depth_src.float().clamp_min(0.0)
    rays_o, rays_d = _ray_gen(c2w_src.float(), torch.deg2rad(fov_src.float()), h)
    pts = rays_o[:, :, None, None, :] + rays_d * depth_f

    w2c = torch.linalg.inv(c2w_dst.float())
    R = w2c[..., :3, :3]
    t = w2c[..., :3, 3]
    pts_cam = torch.einsum("bvij,bvhwj->bvhwi", R, pts) + t[:, :, None, None, :]
    z = (-pts_cam[..., 2:3]).clamp_min(z_eps)

    fov_y = fov_dst.float()
    if fov_y.dim() == 3:
        fov_y = fov_y[..., 0]
    fx = (h / 2.0) / torch.tan(0.5 * torch.deg2rad(fov_y))
    fx = fx[:, :, None, None, None]
    u = pts_cam[..., 0:1] / z * fx + (h / 2.0)
    vv = -pts_cam[..., 1:2] / z * fx + (h / 2.0)
    u_i = torch.floor(u[..., 0]).long()
    v_i = torch.floor(vv[..., 0]).long()

    # dst 视线距离：世界点到 dst 相机原点
    rays_o_d, _ = _ray_gen(c2w_dst.float(), torch.deg2rad(fov_dst.float()), h)
    t_dst = (pts - rays_o_d[:, :, None, None, :]).norm(dim=-1, keepdim=True)

    valid = (
        (depth_f[..., 0] > z_eps)
        & (pts_cam[..., 2] < 0)
        & (u_i >= 0)
        & (u_i < w)
        & (v_i >= 0)
        & (v_i < h)
    )

    out = torch.zeros((b, v, h, w, 1), device=device, dtype=dtype)
    zbuf = torch.full((b, v, h, w), float("inf"), device=device, dtype=dtype)
    n_pix = h * w
    for bi in range(b):
        for vi in range(v):
            m = valid[bi, vi].reshape(-1)
            if not bool(m.any()):
                continue
            ui = u_i[bi, vi].reshape(-1)[m]
            vj = v_i[bi, vi].reshape(-1)[m]
            zz = z[bi, vi, ..., 0].reshape(-1)[m]
            td = t_dst[bi, vi, ..., 0].reshape(-1)[m]
            flat = (vj * w + ui).clamp(0, n_pix - 1)
            zimg = torch.full((n_pix,), float("inf"), device=device, dtype=dtype)
            zimg.scatter_reduce_(0, flat, zz, reduce="amin", include_self=True)
            keep = zz <= zimg[flat] + 1e-5
            flat2, td2 = flat[keep], td[keep]
            # 同像素取更近 cam-z 对应的视线距
            acc = torch.full((n_pix,), float("inf"), device=device, dtype=dtype)
            acc.scatter_reduce_(0, flat2, td2, reduce="amin", include_self=True)
            hit = torch.isfinite(acc) & (acc < 1e20)
            pix = torch.zeros((n_pix,), device=device, dtype=dtype)
            pix[hit] = acc[hit]
            out[bi, vi, ..., 0] = pix.view(h, w)
            zbuf[bi, vi] = torch.where(hit.view(h, w), zimg.view(h, w), zbuf[bi, vi])

    hole = out[..., 0] <= z_eps
    if bool(hole.any()):
        fallback = analytic_plane_depth(c2w_dst, fov_dst, h)
        out = torch.where(hole.unsqueeze(-1), fallback, out)
    return out


@torch.no_grad()
def reproject_inverse_bilinear(
    hdr_src: torch.Tensor,
    depth_src: torch.Tensor,
    c2w_src: torch.Tensor,
    fov_src: torch.Tensor,
    c2w_dst: torch.Tensor,
    fov_dst: torch.Tensor,
    depth_dst: torch.Tensor | None = None,
    *,
    z_eps: float = 1e-4,
    z_rel_tol: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    目标视图反向重投影 + 双线性采样（全分辨率）。

    对每个 dst 像素：用 dst 深度反投影到世界 → 投到 src → grid_sample。
    避免前向溅射的空洞/锯齿随相机拉开逐帧加重。
    """
    assert hdr_src.dim() == 5 and depth_src.dim() == 5
    b, v, h, w, c = hdr_src.shape
    fov_src = _ensure_fov(fov_src)
    fov_dst = _ensure_fov(fov_dst)
    device = hdr_src.device

    if depth_dst is None:
        depth_dst = analytic_plane_depth(c2w_dst, fov_dst, h)
    depth_dst = depth_dst.float().clamp_min(0.0)
    depth_src_f = depth_src.float().clamp_min(0.0)
    hdr_f = hdr_src.float()

    rays_o_d, rays_d_d = _ray_gen(
        c2w_dst.float(), torch.deg2rad(fov_dst.float()), h
    )
    pts = rays_o_d[:, :, None, None, :] + rays_d_d * depth_dst

    w2c_s = torch.linalg.inv(c2w_src.float())
    R = w2c_s[..., :3, :3]
    t = w2c_s[..., :3, 3]
    pts_cam = torch.einsum("bvij,bvhwj->bvhwi", R, pts) + t[:, :, None, None, :]

    z = (-pts_cam[..., 2:3]).clamp_min(z_eps)
    fov_y = fov_src.float()
    if fov_y.dim() == 3:
        fov_y = fov_y[..., 0]
    fx = (h / 2.0) / torch.tan(0.5 * torch.deg2rad(fov_y))
    fx = fx[:, :, None, None, None]
    cx = cy = h / 2.0

    u = pts_cam[..., 0:1] / z * fx + cx
    vv = -pts_cam[..., 1:2] / z * fx + cy

    # align_corners=True: 像素坐标 0..W-1 ↔ -1..1
    gx = (2.0 * u / max(w - 1, 1)) - 1.0
    gy = (2.0 * vv / max(h - 1, 1)) - 1.0
    grid = torch.cat([gx, gy], dim=-1)  # [B,V,H,W,2]

    in_front = pts_cam[..., 2:3] < 0
    in_bound = (u >= 0) & (u <= w - 1) & (vv >= 0) & (vv <= h - 1) & (z > z_eps)
    base_valid = in_front & in_bound & (depth_dst > z_eps)

    hdr_nchw = hdr_f.reshape(b * v, h, w, c).permute(0, 3, 1, 2).contiguous()
    dep_nchw = depth_src_f.reshape(b * v, h, w, 1).permute(0, 3, 1, 2).contiguous()
    grid_n = grid.reshape(b * v, h, w, 2)

    sampled = F.grid_sample(
        hdr_nchw, grid_n, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    depth_s = F.grid_sample(
        dep_nchw, grid_n, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    # src 相机原点到世界点的距离 vs 采样到的 src 深度（遮挡/错切拒绝）
    rays_o_s, _ = _ray_gen(c2w_src.float(), torch.deg2rad(fov_src.float()), h)
    t_exp = (pts - rays_o_s[:, :, None, None, :]).norm(dim=-1, keepdim=True).clamp_min(z_eps)
    depth_s = depth_s.permute(0, 2, 3, 1).reshape(b, v, h, w, 1)
    if z_rel_tol < 0:
        # 关闭遮挡测试（解析平面深度）
        depth_ok = torch.ones_like(base_valid)
    else:
        depth_ok = (depth_s > z_eps) & (
            (t_exp - depth_s).abs() <= z_rel_tol * torch.maximum(t_exp, depth_s)
        )

    valid = base_valid & depth_ok
    out = sampled.permute(0, 2, 3, 1).reshape(b, v, h, w, c)
    out = torch.where(valid.expand_as(out), out, torch.zeros_like(out))

    if bool((~valid).any()):
        out = _fill_holes_edge_aware(out, valid)

    hole_ratio = float((~valid).float().mean().item())
    return out.to(hdr_src.dtype), valid, hole_ratio


def _fill_holes_edge_aware(hdr: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """小范围填洞；次数少，减轻锯齿糊边。"""
    b, v, h, w, c = hdr.shape
    x = hdr.reshape(b * v, h, w, c).permute(0, 3, 1, 2).contiguous()
    m = valid.reshape(b * v, h, w, 1).permute(0, 3, 1, 2).float()
    filled = x * m
    kernel = torch.ones((c, 1, 3, 3), device=hdr.device, dtype=x.dtype)
    k1 = torch.ones((1, 1, 3, 3), device=hdr.device, dtype=x.dtype)
    for _ in range(3):
        num = F.conv2d(filled, kernel, padding=1, groups=c)
        den = F.conv2d(m, k1, padding=1).clamp_min(1e-6)
        avg = num / den
        need = m < 0.5
        filled = torch.where(need.expand_as(filled), avg, filled)
        m = torch.clamp(m + need.float(), 0, 1)
    return filled.permute(0, 2, 3, 1).reshape(b, v, h, w, c)


@torch.no_grad()
def reproject_by_depth(
    hdr: torch.Tensor,
    depth: torch.Tensor,
    c2w_src: torch.Tensor,
    fov_src: torch.Tensor,
    c2w_dst: torch.Tensor,
    fov_dst: torch.Tensor,
    *,
    z_eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    src → dst 全分辨率前向溅射（近深度优先）。遗留接口；跟视图请用 reproject_inverse_bilinear。
    """
    assert hdr.dim() == 5 and depth.dim() == 5
    b, v, h, w, c = hdr.shape
    fov_src = _ensure_fov(fov_src)
    fov_dst = _ensure_fov(fov_dst)
    device = hdr.device
    dtype = torch.float32

    hdr_f = hdr.float()
    depth_f = depth.float().clamp_min(0.0)

    rays_o, rays_d = _ray_gen(
        c2w_src.float(),
        torch.deg2rad(fov_src.float()),
        h,
    )
    pts = rays_o[:, :, None, None, :] + rays_d * depth_f

    w2c = torch.linalg.inv(c2w_dst.float())
    R = w2c[..., :3, :3]
    t = w2c[..., :3, 3]
    pts_cam = torch.einsum("bvij,bvhwj->bvhwi", R, pts) + t[:, :, None, None, :]

    z = (-pts_cam[..., 2:3]).clamp_min(z_eps)
    fov_y = fov_dst.float()
    if fov_y.dim() == 3:
        fov_y = fov_y[..., 0]
    fx = (h / 2.0) / torch.tan(0.5 * torch.deg2rad(fov_y))
    fx = fx[:, :, None, None, None]
    fy = fx
    cx = cy = h / 2.0

    u = pts_cam[..., 0:1] / z * fx + cx
    vv = -pts_cam[..., 1:2] / z * fy + cy
    u_i = torch.floor(u[..., 0]).long()
    v_i = torch.floor(vv[..., 0]).long()

    in_frustum = (
        (depth_f[..., 0] > z_eps)
        & (z[..., 0] > z_eps)
        & (u_i >= 0)
        & (u_i < w)
        & (v_i >= 0)
        & (v_i < h)
        & (pts_cam[..., 2] < 0)
    )

    out = torch.zeros_like(hdr_f)
    valid = torch.zeros((b, v, h, w, 1), device=device, dtype=torch.bool)
    n_pix = h * w

    for bi in range(b):
        for vi in range(v):
            m = in_frustum[bi, vi].reshape(-1)
            if not bool(m.any()):
                continue
            ui = u_i[bi, vi].reshape(-1)[m]
            vj = v_i[bi, vi].reshape(-1)[m]
            zz = z[bi, vi, ..., 0].reshape(-1)[m]
            col = hdr_f[bi, vi].reshape(-1, c)[m]
            flat = (vj * w + ui).clamp(0, n_pix - 1)

            zimg = torch.full((n_pix,), float("inf"), device=device, dtype=dtype)
            zimg.scatter_reduce_(0, flat, zz, reduce="amin", include_self=True)

            near_ok = zz <= zimg[flat] + 1e-5
            flat2 = flat[near_ok]
            col2 = col[near_ok]
            acc = torch.zeros((n_pix, c), device=device, dtype=dtype)
            cnt = torch.zeros((n_pix, 1), device=device, dtype=dtype)
            acc.index_add_(0, flat2, col2)
            cnt.index_add_(0, flat2, torch.ones((flat2.shape[0], 1), device=device, dtype=dtype))
            hit = cnt[:, 0] > 0
            pix = torch.zeros((n_pix, c), device=device, dtype=dtype)
            pix[hit] = acc[hit] / cnt[hit]
            out[bi, vi] = pix.view(h, w, c)
            valid[bi, vi, ..., 0] = hit.view(h, w)

    if bool((~valid).any()):
        out = _fill_holes(out, valid)

    hole_ratio = float((~valid).float().mean().item())
    return out.to(hdr.dtype), valid, hole_ratio


def _fill_holes(hdr: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return _fill_holes_edge_aware(hdr, valid)

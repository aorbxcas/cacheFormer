from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from renderformer.hybrid.runtime_direct.lights import EmissiveLight, extract_emissive_lights
from renderformer.hybrid.runtime_direct.mesh_buffer import MeshBuffer
from renderformer.utils.ray_generator import RayGenerator


class RuntimeDirectRendererLite:
    """
    R0-lite：PyTorch 光线-三角求交 + 0-bounce direct + shadow ray。

    无 nvdiffrast 时的回退；大场景较慢。
    """

    def __init__(
        self,
        ray_chunk: int = 8192,
        shadow_bias: float = 2e-3,
        min_emissive: float = 1.0,
        ambient: float = 0.0,
        scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.ray_chunk = ray_chunk
        self.shadow_bias = shadow_bias
        # 场景尺度 ~O(1)：忽略极近命中，避免同面/邻接三角自阴影
        self.shadow_ray_t_min = max(shadow_bias * 5.0, 0.012)
        self.min_emissive = min_emissive
        self.ambient = ambient
        self.scene_center = torch.tensor(scene_center, dtype=torch.float32)
        self.ray_generator = RayGenerator()

    def render(
        self,
        triangles: torch.Tensor,
        texture: torch.Tensor,
        vn: torch.Tensor,
        mask: torch.Tensor,
        c2w: torch.Tensor,
        fov: torch.Tensor,
        resolution: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            hdr_direct: [B, nv, H, W, 3] linear HDR
            depth: [B, nv, H, W, 1] 沿视线到命中点的距离
        """
        mesh = MeshBuffer.from_scene_tensors(triangles, vn, texture, mask)
        lights = extract_emissive_lights(
            mesh, min_emissive=self.min_emissive, scene_center=self.scene_center.to(triangles.device)
        )

        bs, nv = c2w.shape[0], c2w.shape[1]
        if fov.dim() == 2:
            fov = fov.unsqueeze(-1)

        shadow_tris = mesh.triangles[mesh.shadow_caster_mask]
        primary_tris = mesh.triangles[mesh.mask]

        hdr_views = []
        depth_views = []
        for b in range(bs):
            for v in range(nv):
                hdr, depth = self._render_single_view(
                    mesh=mesh,
                    lights=lights,
                    shadow_tris=shadow_tris,
                    primary_tris=primary_tris,
                    c2w=c2w[b, v],
                    fov=fov[b, v],
                    resolution=resolution,
                )
                hdr_views.append(hdr)
                depth_views.append(depth)

        hdr = torch.stack(hdr_views, dim=0).reshape(bs, nv, resolution, resolution, 3)
        depth = torch.stack(depth_views, dim=0).reshape(bs, nv, resolution, resolution, 1)
        return hdr, depth

    def _render_single_view(
        self,
        mesh: MeshBuffer,
        lights: list[EmissiveLight],
        shadow_tris: torch.Tensor,
        primary_tris: torch.Tensor,
        c2w: torch.Tensor,
        fov: torch.Tensor,
        resolution: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = c2w.device
        dtype = c2w.dtype

        # 与 RenderFormerRenderingPipeline 一致：H5 中 fov 为角度制
        fov_rad = fov.reshape(1, 1) / 180.0 * math.pi
        c2w_b = c2w.unsqueeze(0)
        rays_o, rays_d = self.ray_generator(c2w_b, fov_rad, resolution)
        rays_o = rays_o[0]
        rays_d = rays_d[0]

        num_pixels = resolution * resolution
        rays_o_flat = rays_o.reshape(1, 3).expand(num_pixels, 3)
        rays_d_flat = rays_d.reshape(-1, 3)

        hit_t = torch.full((num_pixels,), float("inf"), device=device, dtype=dtype)
        hit_tri = torch.full((num_pixels,), -1, device=device, dtype=torch.long)

        visible_idx = mesh.mask.nonzero(as_tuple=False).squeeze(-1)
        for start in range(0, num_pixels, self.ray_chunk):
            end = min(start + self.ray_chunk, num_pixels)
            t_chunk, tri_chunk = _intersect_rays_triangles(
                rays_o_flat[start:end],
                rays_d_flat[start:end],
                primary_tris,
            )
            hit_t[start:end] = t_chunk
            hit_tri[start:end] = torch.where(
                tri_chunk >= 0,
                visible_idx[tri_chunk],
                torch.full_like(tri_chunk, -1),
            )

        hdr = torch.zeros(num_pixels, 3, device=device, dtype=dtype)
        valid = hit_tri >= 0
        if valid.any():
            hdr[valid] = self._shade_hits(
                mesh=mesh,
                lights=lights,
                shadow_tris=shadow_tris,
                hit_tri=hit_tri[valid],
                hit_t=hit_t[valid],
                rays_o=rays_o_flat[valid],
                rays_d=rays_d_flat[valid],
            )

        depth = hit_t.reshape(resolution, resolution, 1)
        depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
        hdr = hdr.reshape(resolution, resolution, 3)
        return hdr, depth

    def _shade_hits(
        self,
        mesh: MeshBuffer,
        lights: list[EmissiveLight],
        shadow_tris: torch.Tensor,
        hit_tri: torch.Tensor,
        hit_t: torch.Tensor,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> torch.Tensor:
        tris = mesh.triangles[hit_tri]
        hit_p = rays_o + rays_d * hit_t.unsqueeze(-1)

        face_n = mesh.face_normals()[hit_tri]
        n = _face_normal_toward_ray(face_n, rays_d)

        emissive_hit = mesh.emissive_mask[hit_tri]
        color = torch.zeros((hit_tri.shape[0], 3), device=hit_p.device, dtype=hit_p.dtype)
        if emissive_hit.any():
            color[emissive_hit] = mesh.irradiance[hit_tri[emissive_hit]].clamp(min=0.0)

        shade_mask = ~emissive_hit
        if not shade_mask.any() or not lights:
            return color

        idx = shade_mask.nonzero(as_tuple=False).squeeze(-1)
        hit_p_s = hit_p[idx]
        n_s = n[idx]
        rays_d_s = rays_d[idx]
        hit_tri_s = hit_tri[idx]
        skip_shadow_tri = mesh.shadow_local_indices(hit_tri_s)

        kd = mesh.diffuse[hit_tri_s].clamp(min=0.0)
        ks = mesh.specular[hit_tri_s].clamp(min=0.0)
        rough = mesh.roughness[hit_tri_s].squeeze(-1).clamp(min=0.02, max=1.0)
        alpha = rough * rough
        view_dir = F.normalize(-rays_d_s, dim=-1)

        lit = torch.zeros_like(kd)
        if self.ambient > 0:
            lit = lit + kd * self.ambient

        for light in lights:
            to_light = light.position.unsqueeze(0) - hit_p_s
            dist2 = (to_light * to_light).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            dist = torch.sqrt(dist2)
            l_dir = to_light / dist

            ndotl = (n_s * l_dir).sum(dim=-1, keepdim=True).clamp(min=0.0)
            if not (ndotl > 0).any():
                continue

            visible = _shadow_visibility(
                origin=hit_p_s + n_s * self.shadow_bias,
                direction=l_dir,
                max_dist=dist.squeeze(-1) - self.shadow_bias,
                triangles=shadow_tris,
                skip_tri_local=skip_shadow_tri,
                t_min=self.shadow_ray_t_min,
            )
            visible = visible.unsqueeze(-1)

            h = F.normalize(l_dir + view_dir, dim=-1)
            ndoth = (n_s * h).sum(dim=-1, keepdim=True).clamp(min=0.0)
            spec = _ggx_specular(ndoth, ndotl, alpha)

            # 三角面光源：Le * (A * cos(theta_light)) / r^2
            cos_light = (-l_dir * light.normal.unsqueeze(0)).sum(dim=-1, keepdim=True).clamp(min=0.0)
            radiance = light.radiance.unsqueeze(0) * (light.area * cos_light / dist2)

            diffuse_term = kd * radiance * ndotl
            spec_term = ks * radiance * spec * ndotl
            lit = lit + (diffuse_term + spec_term) * visible

        color[idx] = lit
        return color


def _face_normal_toward_ray(normals: torch.Tensor, ray_dirs: torch.Tensor) -> torch.Tensor:
    """将法线翻转到朝向相机（与入射射线相反）的一侧。"""
    flip = (normals * (-ray_dirs)).sum(dim=-1, keepdim=True) < 0.0
    return torch.where(flip, -normals, normals)


def _intersect_rays_triangles(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    triangles: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        ray_o: [R, 3]
        ray_d: [R, 3] 单位方向
        triangles: [T, 3, 3]
    Returns:
        t: [R] 最近命中距离，未命中为 inf
        tri_idx: [R] 命中三角索引，未命中为 -1
    """
    R = ray_o.shape[0]
    T = triangles.shape[0]
    if T == 0:
        return (
            torch.full((R,), float("inf"), device=ray_o.device, dtype=ray_o.dtype),
            torch.full((R,), -1, device=ray_o.device, dtype=torch.long),
        )

    t_best = torch.full((R,), float("inf"), device=ray_o.device, dtype=ray_o.dtype)
    tri_best = torch.full((R,), -1, device=ray_o.device, dtype=torch.long)

    tri_chunk = max(256, min(2048, 65536 // max(R, 1)))
    for t0 in range(0, T, tri_chunk):
        t1 = min(t0 + tri_chunk, T)
        tri_sl = triangles[t0:t1]
        v0c = tri_sl[:, 0]
        e1c = tri_sl[:, 1] - v0c
        e2c = tri_sl[:, 2] - v0c

        pvec = torch.cross(ray_d[:, None, :], e2c[None, :, :], dim=-1)
        det = (e1c[None, :, :] * pvec).sum(dim=-1)
        mask = det.abs() > eps
        inv_det = torch.where(mask, 1.0 / det, torch.zeros_like(det))

        tvec = ray_o[:, None, :] - v0c[None, :, :]
        u = (tvec * pvec).sum(dim=-1) * inv_det
        qvec = torch.cross(tvec, e1c[None, :, :], dim=-1)
        v = (ray_d[:, None, :] * qvec).sum(dim=-1) * inv_det
        t_hit = (e2c[None, :, :] * qvec).sum(dim=-1) * inv_det

        valid = mask & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t_hit > eps)
        t_hit = torch.where(valid, t_hit, torch.full_like(t_hit, float("inf")))

        t_local, local_idx = t_hit.min(dim=1)
        better = t_local < t_best
        t_best = torch.where(better, t_local, t_best)
        global_idx = local_idx + t0
        tri_best = torch.where(better, global_idx, tri_best)

    miss = ~torch.isfinite(t_best)
    tri_best = torch.where(miss, torch.full_like(tri_best, -1), tri_best)
    t_best = torch.where(miss, torch.full_like(t_best, float("inf")), t_best)
    return t_best, tri_best


def _shadow_visibility(
    origin: torch.Tensor,
    direction: torch.Tensor,
    max_dist: torch.Tensor,
    triangles: torch.Tensor,
    skip_tri_local: torch.Tensor | None = None,
    t_min: float = 5e-3,
) -> torch.Tensor:
    R = origin.shape[0]
    visible = torch.ones(R, device=origin.device, dtype=torch.bool)
    if triangles.shape[0] == 0:
        return visible

    chunk = max(512, min(4096, 65536 // max(R, 1)))
    for t0 in range(0, triangles.shape[0], chunk):
        t1 = min(t0 + chunk, triangles.shape[0])
        t_hit, tri_idx = _intersect_rays_triangles(origin, direction, triangles[t0:t1])
        local_idx = tri_idx + t0
        hit = (tri_idx >= 0) & (t_hit > t_min) & (t_hit < max_dist)
        if skip_tri_local is not None:
            hit = hit & (local_idx != skip_tri_local)
        visible = visible & ~hit
        if not visible.any():
            break
    return visible


def _barycentric(p: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0
    d00 = (v0v1 * v0v1).sum(dim=-1)
    d01 = (v0v1 * v0v2).sum(dim=-1)
    d11 = (v0v2 * v0v2).sum(dim=-1)
    d20 = (v0p * v0v1).sum(dim=-1)
    d21 = (v0p * v0v2).sum(dim=-1)
    denom = d00 * d11 - d01 * d01
    denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return torch.stack([u, v, w], dim=-1).clamp(min=0.0)


def _ggx_specular(ndoth: torch.Tensor, ndotl: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.reshape(-1, 1)
    alpha2 = alpha * alpha
    denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0
    d = alpha2 / (math.pi * denom * denom + 1e-7)
    return d * ndotl

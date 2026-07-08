from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from renderformer.hybrid.runtime_direct.lights import EmissiveLight
    from renderformer.hybrid.runtime_direct.mesh_buffer import MeshBuffer


def face_normal_toward_vector(normals: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """将法线翻到与 direction 同侧（direction 通常为 -ray 或 toward camera）。"""
    flip = (normals * direction).sum(dim=-1, keepdim=True) < 0.0
    return torch.where(flip, -normals, normals)


def ggx_specular(ndoth: torch.Tensor, ndotl: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(-1)
    elif alpha.dim() == ndotl.dim() - 1:
        alpha = alpha.unsqueeze(-1)
    alpha2 = alpha * alpha
    denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0
    d = alpha2 / (math.pi * denom * denom + 1e-7)
    return d * ndotl


def sample_shadow_map(
    shadow_depth: torch.Tensor,
    light_uv: torch.Tensor,
    compare_z: torch.Tensor,
    bias: float = 5e-4,
) -> torch.Tensor:
    """
    比较 light-space 深度与 shadow map。

    Args:
        shadow_depth: [1, S, S, 1] 光源视角 raster depth (nvdiffrast z/w)
        light_uv: [H, W, 2] 在 [0,1] 内为有效
        compare_z: [H, W, 1] 同空间深度
    Returns:
        visible [H, W, 1] float 0/1
    """
    import nvdiffrast.torch as dr

    # nvdiffrast texture 采样：uv 为 [1,H,W,2]，范围 [0,1]
    uv = light_uv.unsqueeze(0).clamp(0.0, 1.0).contiguous()
    tex = shadow_depth.contiguous()
    ref = dr.texture(tex, uv, filter_mode="linear", boundary_mode="clamp")
    ref_z = ref[0, ..., 0:1]
    lit_z = compare_z
    valid = (light_uv[..., 0:1] >= 0) & (light_uv[..., 0:1] <= 1) & (light_uv[..., 1:2] >= 0) & (light_uv[..., 1:2] <= 1)
    visible = (lit_z <= ref_z + bias) & valid
    # 无 shadow map 命中（背景）视为受光
    no_hit = ref_z <= 0.0
    visible = visible | (no_hit & valid)
    return visible.float()


def shade_surface_direct(
    mesh: "MeshBuffer",
    global_tri: torch.Tensor,
    world_pos: torch.Tensor,
    view_dir: torch.Tensor,
    lights: list["EmissiveLight"],
    shadow_visible: Optional[torch.Tensor] = None,
    ambient: float = 0.0,
) -> torch.Tensor:
    """
    对命中表面做 0-bounce direct shading。

    Args:
        global_tri: [...] 全局三角索引，-1 表示背景
        world_pos: [..., 3]
        view_dir: [..., 3] 指向相机
        shadow_visible: [..., 1] 可选，来自 shadow map
    """
    device = world_pos.device
    dtype = world_pos.dtype
    out_shape = world_pos.shape[:-1] + (3,)
    color = torch.zeros(out_shape, device=device, dtype=dtype)

    valid = global_tri >= 0
    if not valid.any():
        return color

    flat_tri = global_tri.clamp(min=0)
    face_n = mesh.face_normals()[flat_tri]
    n = face_normal_toward_vector(face_n, view_dir)

    emissive = mesh.emissive_mask[flat_tri]
    if emissive.any():
        color[valid & emissive] = mesh.irradiance[flat_tri[valid & emissive]].clamp(min=0.0)

    shade = valid & (~emissive)
    if not shade.any() or not lights:
        return color

    kd = mesh.diffuse[flat_tri].clamp(min=0.0)
    ks = mesh.specular[flat_tri].clamp(min=0.0)
    rough = mesh.roughness[flat_tri].squeeze(-1).clamp(min=0.02, max=1.0)
    alpha = rough * rough

    lit = torch.zeros_like(kd)
    if ambient > 0:
        lit = lit + kd * ambient

    for i, light in enumerate(lights):
        to_light = light.position - world_pos
        dist2 = (to_light * to_light).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        dist = torch.sqrt(dist2)
        l_dir = to_light / dist

        ndotl = (n * l_dir).sum(dim=-1, keepdim=True).clamp(min=0.0)

        if shadow_visible is None:
            vis = torch.ones_like(ndotl)
        else:
            vis = shadow_visible
            if shadow_visible.dim() == len(out_shape) - 1:
                vis = shadow_visible.unsqueeze(-1)

        h = F.normalize(l_dir + view_dir, dim=-1)
        ndoth = (n * h).sum(dim=-1, keepdim=True).clamp(min=0.0)
        spec = ggx_specular(ndoth, ndotl, alpha)

        cos_light = (-l_dir * light.normal).sum(dim=-1, keepdim=True).clamp(min=0.0)
        radiance = light.radiance * (light.area * cos_light / dist2)

        lit = lit + (kd * radiance * ndotl + ks * radiance * spec * ndotl) * vis

    color[shade] = lit[shade]
    return color

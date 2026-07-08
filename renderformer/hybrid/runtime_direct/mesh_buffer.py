from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# 与 to_h5.py / RenderFormerConfig.texture_channels 一致
TEXTURE_DIFFUSE = slice(0, 3)
TEXTURE_SPECULAR = slice(3, 6)
TEXTURE_ROUGHNESS = slice(6, 7)
TEXTURE_NORMAL = slice(7, 10)
TEXTURE_IRRADIANCE = slice(10, 13)


@dataclass
class MeshBuffer:
    """H5 几何与 per-triangle 材质（世界坐标）。"""

    triangles: torch.Tensor
    vn: torch.Tensor
    diffuse: torch.Tensor
    specular: torch.Tensor
    roughness: torch.Tensor
    irradiance: torch.Tensor
    mask: torch.Tensor

    @classmethod
    def from_scene_tensors(
        cls,
        triangles: torch.Tensor,
        vn: torch.Tensor,
        texture: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> "MeshBuffer":
        """
        Args:
            triangles: [N, 3, 3] 或 [B, N, 3, 3]（仅支持 B=1）
            vn: 同形状顶点法线
            texture: [N, C, H, W] 或 [B, N, C, H, W]
            mask: [N] 或 [B, N]
        """
        if triangles.dim() == 4:
            if triangles.shape[0] != 1:
                raise ValueError("MeshBuffer 当前仅支持 batch_size=1")
            triangles = triangles[0]
            vn = vn[0]
            texture = texture[0]
            if mask is not None and mask.dim() == 2:
                mask = mask[0]

        if mask is None:
            mask = torch.ones(triangles.shape[0], dtype=torch.bool, device=triangles.device)
        else:
            mask = mask.to(dtype=torch.bool, device=triangles.device)

        mat = _material_from_texture(texture)
        return cls(
            triangles=triangles,
            vn=vn,
            diffuse=mat["diffuse"],
            specular=mat["specular"],
            roughness=mat["roughness"],
            irradiance=mat["irradiance"],
            mask=mask,
        )

    @property
    def num_triangles(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def emissive_mask(self) -> torch.Tensor:
        strength = self.irradiance.max(dim=-1).values
        return (strength > 1.0) & self.mask

    @property
    def shadow_caster_mask(self) -> torch.Tensor:
        """阴影投射体：排除 emissive 三角，避免阴影射线命中灯面自身。"""
        return self.mask & ~self.emissive_mask

    def shadow_local_indices(self, global_tri: torch.Tensor) -> torch.Tensor:
        """将全局三角索引映射到 shadow_caster 列表中的局部索引；无则 -1。"""
        device = global_tri.device
        shadow_global = self.shadow_caster_mask.nonzero(as_tuple=False).squeeze(-1)
        lut = torch.full((self.num_triangles,), -1, dtype=torch.long, device=device)
        lut[shadow_global] = torch.arange(shadow_global.shape[0], device=device, dtype=torch.long)
        return lut[global_tri]

    def face_normals(self) -> torch.Tensor:
        v0, v1, v2 = self.triangles.unbind(dim=1)
        return F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)

    def shading_normals(self) -> torch.Tensor:
        """平滑法线：三顶点法线平均后归一化。"""
        n = self.vn.mean(dim=1)
        return F.normalize(n, dim=-1)

    def triangle_areas(self) -> torch.Tensor:
        v0, v1, v2 = self.triangles.unbind(dim=1)
        return 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)

    def centroids(self) -> torch.Tensor:
        return self.triangles.mean(dim=1)


def _material_from_texture(texture: torch.Tensor) -> dict[str, torch.Tensor]:
    """对 32×32 patch 取均值得到 per-triangle 材质。"""
    if texture.dim() == 5:
        texture = texture.mean(dim=(-1, -2))
    elif texture.dim() == 4:
        texture = texture.mean(dim=(-1, -2))
    else:
        raise ValueError(f"Unexpected texture shape: {tuple(texture.shape)}")

    return {
        "diffuse": texture[:, TEXTURE_DIFFUSE],
        "specular": texture[:, TEXTURE_SPECULAR],
        "roughness": texture[:, TEXTURE_ROUGHNESS].clamp(min=1e-3, max=1.0),
        "irradiance": texture[:, TEXTURE_IRRADIANCE],
    }

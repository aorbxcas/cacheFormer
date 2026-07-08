from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EmissiveLight:
    position: torch.Tensor
    radiance: torch.Tensor
    area: torch.Tensor
    normal: torch.Tensor


def extract_emissive_lights(
    mesh,
    min_emissive: float = 1.0,
    scene_center: torch.Tensor | None = None,
) -> list[EmissiveLight]:
    """
    从 per-triangle irradiance 通道提取 emissive 三角面光源。

    Args:
        mesh: MeshBuffer 实例
        scene_center: 场景中心，用于将光源法线朝向场景内部
    """
    strength = mesh.irradiance.max(dim=-1).values
    emissive_mask = (strength > min_emissive) & mesh.mask
    if not emissive_mask.any():
        return []

    idx = emissive_mask.nonzero(as_tuple=False).squeeze(-1)
    tris = mesh.triangles[idx]
    centroids = tris.mean(dim=1)
    normals = mesh.face_normals()[idx]
    areas = mesh.triangle_areas()[idx]
    radiance = mesh.irradiance[idx]

    if scene_center is None:
        scene_center = mesh.triangles.mean(dim=(0, 1))
    else:
        scene_center = scene_center.to(device=centroids.device, dtype=centroids.dtype)

    # 法线指向场景内部，便于计算 cos(theta_light)
    to_interior = scene_center.unsqueeze(0) - centroids
    flip = (normals * to_interior).sum(dim=-1, keepdim=True) < 0.0
    normals = torch.where(flip, -normals, normals)

    lights: list[EmissiveLight] = []
    for i in range(idx.shape[0]):
        lights.append(
            EmissiveLight(
                position=centroids[i],
                radiance=radiance[i],
                area=areas[i],
                normal=normals[i],
            )
        )
    return lights


def parse_lights_from_json(json_path: str | None) -> list[dict]:
    """
    可选：从场景 JSON 解析额外光源（首期以 H5 emissive 为主，此函数预留 R1 扩展）。
    """
    if json_path is None:
        return []
    import json
    from pathlib import Path

    path = Path(json_path)
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lights = []
    for key, obj in data.get("objects", {}).items():
        emissive = obj.get("material", {}).get("emissive", [0, 0, 0])
        if max(emissive) <= 0:
            continue
        lights.append({"name": key, "emissive": emissive})
    return lights

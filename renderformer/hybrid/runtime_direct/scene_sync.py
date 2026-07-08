from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CameraBatch:
    """与 RenderFormer 推理一致的相机参数。"""

    c2w: torch.Tensor
    fov: torch.Tensor
    resolution: int

    @property
    def batch_size(self) -> int:
        return self.c2w.shape[0]

    @property
    def num_views(self) -> int:
        return self.c2w.shape[1]


class SceneLightSync:
    """
    R1：保证 Runtime Direct 与 RenderFormer 使用同一套 c2w / fov / resolution。
    """

    @staticmethod
    def build_camera_batch(
        c2w: torch.Tensor,
        fov: torch.Tensor,
        resolution: int,
    ) -> CameraBatch:
        if fov.dim() == 2:
            fov = fov.unsqueeze(-1)
        return CameraBatch(c2w=c2w, fov=fov, resolution=resolution)

    @staticmethod
    def validate_shared_inputs(
        triangles: torch.Tensor,
        c2w: torch.Tensor,
        fov: torch.Tensor,
        resolution: int,
    ) -> None:
        if triangles.dim() != 4 or triangles.shape[0] != 1:
            raise ValueError("Hybrid 路径 B 当前要求 triangles 为 [1, N, 3, 3]")
        if c2w.shape[0] != 1:
            raise ValueError("Hybrid 路径 B 当前要求 batch_size=1")
        if fov.shape[0] != c2w.shape[0] or fov.shape[1] != c2w.shape[1]:
            raise ValueError("fov 与 c2w 的 batch/view 维不一致")

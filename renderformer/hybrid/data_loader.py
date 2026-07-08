from __future__ import annotations

from typing import Any, Dict

import h5py
import numpy as np
import torch


def load_h5_scene(file_path: str) -> Dict[str, Any]:
    """加载 H5 场景为 CPU 张量（与 infer.py 一致，无 batch 维）。"""
    with h5py.File(file_path, "r") as f:
        triangles = torch.from_numpy(np.array(f["triangles"]).astype(np.float32))
        num_tris = triangles.shape[0]
        texture = torch.from_numpy(np.array(f["texture"]).astype(np.float32))
        mask = torch.ones(num_tris, dtype=torch.bool)
        vn = torch.from_numpy(np.array(f["vn"]).astype(np.float32))
        c2w = torch.from_numpy(np.array(f["c2w"]).astype(np.float32))
        fov = torch.from_numpy(np.array(f["fov"]).astype(np.float32))
    return {
        "triangles": triangles,
        "texture": texture,
        "mask": mask,
        "vn": vn,
        "c2w": c2w,
        "fov": fov,
    }


def add_batch_dim(data: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """为推理添加 batch 维并搬到 device。"""
    out = {}
    out["triangles"] = data["triangles"].unsqueeze(0).to(device)
    out["texture"] = data["texture"].unsqueeze(0).to(device)
    out["mask"] = data["mask"].unsqueeze(0).to(device)
    out["vn"] = data["vn"].unsqueeze(0).to(device)
    out["c2w"] = data["c2w"].unsqueeze(0).to(device)
    fov = data["fov"]
    if fov.dim() == 1:
        fov = fov.unsqueeze(-1)
    out["fov"] = fov.unsqueeze(0).to(device)
    return out

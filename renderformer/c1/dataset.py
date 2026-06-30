# -*- coding: utf-8 -*-
"""C1 训练样本 Dataset（读取 prepare 脚本导出的 npz）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


SAMPLE_KEYS = (
    "hdr_neural",
    "hdr_direct",
    "depth",
    "indirect_target",
    "hdr_gt",
)


class C1ResidualDataset(Dataset):
    """
    每个样本一个 .npz，字段均为 float32：
      hdr_neural      [H,W,3]
      hdr_direct      [H,W,3]
      depth           [H,W,1]
      indirect_target [H,W,3]
      hdr_gt          [H,W,3]
    """

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"缺少 manifest.json: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        entries = manifest.get(split) or manifest.get("samples") or []
        if split == "train" and not entries and "all" in manifest:
            entries = manifest["all"]
        self.entries = [self.root / e if not Path(e).is_absolute() else Path(e) for e in entries]
        if not self.entries:
            # 回退：扫描 samples/
            sample_dir = self.root / "samples"
            self.entries = sorted(sample_dir.glob("*.npz"))
        if not self.entries:
            raise RuntimeError(f"数据集为空: {self.root} split={split}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.entries[idx]
        data = np.load(path)
        out: dict[str, torch.Tensor] = {}
        for k in SAMPLE_KEYS:
            if k not in data:
                raise KeyError(f"{path} 缺少字段 {k}")
            arr = data[k].astype(np.float32)
            t = torch.from_numpy(arr)
            # -> NCHW for training convenience at batch level handled in collate
            out[k] = t
        out["meta_path"] = str(path)  # type: ignore
        return out


def c1_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """堆叠为 [B,3,H,W] / [B,1,H,W]。"""
    keys = [k for k in SAMPLE_KEYS if k in batch[0]]
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = []
        for item in batch:
            t = item[k]
            if t.dim() == 3 and t.shape[-1] in (1, 3):
                t = t.permute(2, 0, 1).contiguous()
            tensors.append(t)
        out[k] = torch.stack(tensors, dim=0)
    return out

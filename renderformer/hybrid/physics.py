from __future__ import annotations

from typing import Dict

import torch


def physics_correct(hdr: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    模块 5 轻量推理侧修正：非负 clamp + 非有限值修复。
    """
    before_neg = (hdr < 0).float().mean().item()
    before_nonfinite = (~torch.isfinite(hdr)).float().mean().item()

    out = torch.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0)
    out = out.clamp(min=0.0)

    after_neg = (out < 0).float().mean().item()
    return out, {
        "viol_nonneg_before": before_neg,
        "viol_nonneg_after": after_neg,
        "viol_nonfinite_before": before_nonfinite,
    }

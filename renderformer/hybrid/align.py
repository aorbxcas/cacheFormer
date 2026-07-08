from __future__ import annotations

from typing import Optional, Tuple

import torch


class HdrAligner:
    """Runtime direct 与 neural HDR 的线性 scale/bias 对齐。"""

    def __init__(self, scale: float = 1.0, bias: float = 0.0):
        self.scale = scale
        self.bias = bias

    def apply(self, hdr_direct: torch.Tensor) -> torch.Tensor:
        return hdr_direct * self.scale + self.bias

    @classmethod
    def fit_robust(
        cls,
        hdr_neural: torch.Tensor,
        hdr_direct: torch.Tensor,
        threshold: float = 1e-3,
    ) -> "HdrAligner":
        """
        在 direct 足够亮的像素上拟合 scale（median ratio），bias 默认 0。
        """
        mask = hdr_direct.mean(dim=-1) > threshold
        if mask.sum() < 16:
            return cls(scale=1.0, bias=0.0)

        ratio = hdr_neural[mask] / hdr_direct[mask].clamp(min=1e-6)
        ratio = ratio[torch.isfinite(ratio) & (ratio > 0)]
        if ratio.numel() == 0:
            return cls(scale=1.0, bias=0.0)
        scale = ratio.median().item()
        return cls(scale=scale, bias=0.0)

    def fit_and_apply(
        self,
        hdr_neural: torch.Tensor,
        hdr_direct: torch.Tensor,
        auto_fit: bool = False,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        if auto_fit:
            fitted = self.fit_robust(hdr_neural, hdr_direct)
            self.scale = fitted.scale
            self.bias = fitted.bias
        return self.apply(hdr_direct), self.scale if auto_fit else None

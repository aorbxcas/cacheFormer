# -*- coding: utf-8 -*-
"""C1 损失：间接光回归 + 可选合成图重建。"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualIndirectLoss(nn.Module):
    def __init__(
        self,
        w_indirect: float = 1.0,
        w_compose: float = 0.5,
        relative: bool = False,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.w_indirect = w_indirect
        self.w_compose = w_compose
        self.relative = relative
        self.eps = eps

    def _err(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 默认绝对 L1：I* 稀疏时相对误差会把头压成全零
        abs_err = (pred - target).abs()
        if self.relative:
            return (abs_err / (target.abs() + self.eps)).mean()
        return abs_err.mean()

    def forward(
        self,
        i_pred: torch.Tensor,
        i_target: torch.Tensor,
        hdr_direct: torch.Tensor | None = None,
        hdr_gt: torch.Tensor | None = None,
        alpha: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_i = self._err(i_pred, i_target)
        total = self.w_indirect * loss_i
        stats = {"loss_indirect": float(loss_i.detach())}

        if self.w_compose > 0 and hdr_direct is not None and hdr_gt is not None:
            # 合成项对异常大间接光做抑制，避免炸梯度
            i_safe = i_pred.clamp(max=float(i_target.detach().amax().clamp(min=1.0) * 4.0))
            composed = hdr_direct + alpha * i_safe
            loss_c = self._err(composed, hdr_gt)
            total = total + self.w_compose * loss_c
            stats["loss_compose"] = float(loss_c.detach())

        stats["loss_total"] = float(total.detach())
        return total, stats

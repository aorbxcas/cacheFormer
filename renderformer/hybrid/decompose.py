from __future__ import annotations

import torch


class DirectIndirectDecomposer:
    """H2：线性 HDR 分解 indirect = clamp(neural - direct, 0)。"""

    @staticmethod
    def decompose(
        hdr_neural: torch.Tensor,
        hdr_direct: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indirect = (hdr_neural - hdr_direct).clamp(min=0.0)
        return indirect, hdr_direct

    @staticmethod
    def fuse(
        hdr_direct: torch.Tensor,
        indirect: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if alpha.dim() == hdr_direct.dim() - 1:
            alpha = alpha.unsqueeze(-1)
        return hdr_direct + alpha * indirect

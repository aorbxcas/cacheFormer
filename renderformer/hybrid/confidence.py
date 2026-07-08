from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ConfidenceWeights:
    w0: float = 2.0
    w1: float = 4.0
    w2: float = 1.0
    w3: float = 2.0
    w4: float = 0.0


class ConfidenceMap:
    """H3：基于 violation + 深度边缘的 α 图。"""

    def __init__(self, weights: ConfidenceWeights | None = None):
        self.weights = weights or ConfidenceWeights()

    def compute(
        self,
        hdr_neural: torch.Tensor,
        hdr_direct: torch.Tensor,
        depth: torch.Tensor,
        prev_alpha: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        violations = compute_violations(hdr_neural, hdr_direct)
        v_grad = _image_gradient_magnitude(hdr_neural.mean(dim=-1))
        v_edge = _image_gradient_magnitude(depth.squeeze(-1))

        v_grad_n = _normalize_violation(v_grad)
        v_edge_n = _normalize_violation(v_edge)
        v_energy_n = torch.full_like(v_grad_n, violations["viol_energy"])
        v_nonneg_n = torch.full_like(v_grad_n, violations["viol_nonneg"])

        w = self.weights
        score = (
            w.w0
            - w.w1 * v_energy_n
            - w.w2 * v_grad_n
            - w.w3 * v_edge_n
            - w.w4 * (prev_alpha if prev_alpha is not None else 0.0)
        )
        alpha = torch.sigmoid(score)
        if alpha.dim() == hdr_neural.dim() - 1:
            alpha = alpha.unsqueeze(-1)
        return alpha, violations


def compute_violations(
    hdr_neural: torch.Tensor,
    hdr_direct: torch.Tensor,
) -> Dict[str, float]:
    """标量违反率，用于日志与 α 的全局项。"""
    neg = (hdr_neural < 0).float().mean().item()
    indirect_raw = hdr_neural - hdr_direct
    energy = (indirect_raw < 0).float().mean().item()
    nonfinite = (~torch.isfinite(hdr_neural)).float().mean().item()
    return {
        "viol_nonneg": neg,
        "viol_energy": energy,
        "viol_nonfinite": nonfinite,
    }


def _image_gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:
        b, nv, h, w, _ = x.shape
        x = x.reshape(b * nv, h, w)
        out = _image_gradient_magnitude(x)
        return out.reshape(b, nv, h, w)
    if x.dim() == 4:
        b, nv, h, w = x.shape
        x = x.reshape(b * nv, h, w)
        out = _image_gradient_magnitude(x)
        return out.reshape(b, nv, h, w)
    x = x.unsqueeze(1)
    gx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(gx * gx + gy * gy + 1e-8).squeeze(1)


def _normalize_violation(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(-1).float()
    lo = torch.quantile(flat, 0.05)
    hi = torch.quantile(flat, 0.95)
    return ((x.float() - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0).to(dtype=x.dtype)

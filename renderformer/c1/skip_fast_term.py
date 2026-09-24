# -*- coding: utf-8 -*-
"""L2 跳过帧快项：NIRC 式 cheap term，不跑 RF。

默认 closed-form 强度为 0，输出等于 warp(I)，保证相对 CacheFormer 几乎无色差。
有 checkpoint 时可用残差头做小混合；未训练头禁止默认开启。
"""

from __future__ import annotations

from typing import Optional

import torch

from renderformer.c1.residual_head import ResidualIndirectHead, fuse_direct_indirect


def closed_form_lighting_gate(
    indirect: torch.Tensor,
    direct_now: Optional[torch.Tensor],
    direct_warp: Optional[torch.Tensor],
    strength: float = 0.0,
) -> torch.Tensor:
    """
    用当前 Direct 相对 warp Direct 的亮度比缩放间接光（NRC 动态灯脏区思想的屏幕空间弱版）。

    strength=0 → 恒等。strength 建议 ≤0.15，否则相对 RF 锚点会漂。
    """
    if strength <= 1e-8 or direct_now is None or direct_warp is None:
        return indirect
    lum_n = direct_now.mean(dim=-1, keepdim=True).clamp_min(1e-4)
    lum_w = direct_warp.mean(dim=-1, keepdim=True).clamp_min(1e-4)
    lo, hi = 1.0 - float(strength), 1.0 + float(strength)
    gate = (lum_n / lum_w).clamp(lo, hi)
    return indirect * gate


def mix_residual_head(
    indirect: torch.Tensor,
    head: Optional[ResidualIndirectHead],
    hdr_direct: torch.Tensor,
    hdr_neural_warp: torch.Tensor,
    depth: torch.Tensor,
    mix: float = 0.0,
) -> torch.Tensor:
    """I' = (1-mix)*I_warp + mix*Head(D, N_warp, z)。mix=0 跳过头。"""
    if head is None or mix <= 1e-8:
        return indirect
    pred = head(hdr_direct, hdr_neural_warp, depth)
    m = float(mix)
    return (1.0 - m) * indirect + m * pred


def compose_skip_frame(
    *,
    quality_anchor: str,
    fused_warp: torch.Tensor,
    indirect_warp: torch.Tensor,
    direct_warp: torch.Tensor,
    direct_now: Optional[torch.Tensor],
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    quality_anchor=rf：输出保持 warp 后的 RF 融合图（对标 CacheFormer）。
    quality_anchor=hybrid：L = D_now + α I_warp（产品合同，相对 CF 会有 Direct 差）。
    """
    if quality_anchor == "hybrid" and direct_now is not None:
        fused = fuse_direct_indirect(direct_now, indirect_warp, alpha)
        return fused, direct_now, indirect_warp
    return fused_warp, direct_warp, indirect_warp

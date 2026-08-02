# -*- coding: utf-8 -*-
"""C1 推理：Direct + 冻结 RF + 残差头，可选 Hybrid confidence α。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from renderformer.c1.residual_head import ResidualIndirectHead, fuse_direct_indirect
from renderformer.hybrid.align import HdrAligner
from renderformer.hybrid.confidence import ConfidenceMap, ConfidenceWeights
from renderformer.hybrid.runtime_direct.factory import create_runtime_direct_renderer

if False:  # TYPE_CHECKING
    from renderformer.cache.vi_cache import ViewIndependentCache
    from renderformer.pipelines.rendering_pipeline import RenderFormerRenderingPipeline


@dataclass
class C1RenderContext:
    hdr_direct: torch.Tensor
    hdr_neural: torch.Tensor
    depth: torch.Tensor
    indirect_pred: torch.Tensor
    hdr_fused: torch.Tensor
    alpha: torch.Tensor
    meta: Dict[str, Any]


class C1ResidualPipeline:
    """
    L_out = L_direct + alpha * Head(direct, neural, depth)

    alpha 模式：
      - fixed: 标量 self.alpha
      - confidence: Hybrid ConfidenceMap（与方案四一致）
    """

    def __init__(
        self,
        rf_pipeline: "RenderFormerRenderingPipeline",
        head: ResidualIndirectHead,
        direct_renderer: Optional[object] = None,
        alpha: float = 1.0,
        run_neural: bool = True,
        use_confidence: bool = False,
        auto_align: bool = False,
        confidence_weights: Optional[dict] = None,
    ):
        self.rf_pipeline = rf_pipeline
        self.head = head
        self.direct_renderer = direct_renderer or create_runtime_direct_renderer(backend="auto")
        self.alpha = float(alpha)
        self.run_neural = run_neural
        self.use_confidence = use_confidence
        self.auto_align = auto_align
        self.aligner = HdrAligner()
        w = confidence_weights or {}
        self.confidence = ConfidenceMap(
            ConfidenceWeights(
                w0=float(w.get("w0", 2.0)),
                w1=float(w.get("w1", 4.0)),
                w2=float(w.get("w2", 1.0)),
                w3=float(w.get("w3", 2.0)),
                w4=float(w.get("w4", 0.0)),
            )
        )

    @property
    def device(self) -> torch.device:
        return self.rf_pipeline.device

    def to(self, device: torch.device) -> "C1ResidualPipeline":
        self.rf_pipeline.to(device)
        self.head.to(device)
        return self

    @torch.no_grad()
    def render(
        self,
        triangles: torch.Tensor,
        texture: torch.Tensor,
        mask: torch.Tensor,
        vn: torch.Tensor,
        c2w: torch.Tensor,
        fov: torch.Tensor,
        resolution: int = 256,
        torch_dtype: torch.dtype = torch.float16,
        vi_cache: Optional["ViewIndependentCache"] = None,
        hdr_neural: Optional[torch.Tensor] = None,
    ) -> C1RenderContext:
        meta: Dict[str, Any] = {}
        t0 = time.perf_counter()

        t_d = time.perf_counter()
        hdr_direct, depth = self.direct_renderer.render(
            triangles, texture, vn, mask, c2w, fov, resolution
        )
        meta["direct_ms"] = (time.perf_counter() - t_d) * 1000.0

        if hdr_neural is None and self.run_neural:
            t_n = time.perf_counter()
            hdr_neural, rf_info = self.rf_pipeline.render(
                triangles=triangles,
                texture=texture,
                mask=mask,
                vn=vn,
                c2w=c2w,
                fov=fov,
                resolution=resolution,
                torch_dtype=torch_dtype,
                vi_cache=vi_cache,
                return_vi_cache_info=True,
            )
            meta["rf_ms"] = (time.perf_counter() - t_n) * 1000.0
            meta.update(rf_info)
        elif hdr_neural is None:
            hdr_neural = torch.zeros_like(hdr_direct)

        hdr_direct = hdr_direct.to(dtype=torch.float32)
        hdr_neural = hdr_neural.to(dtype=torch.float32)
        depth = depth.to(dtype=torch.float32)

        if self.auto_align:
            hdr_direct, scale = self.aligner.fit_and_apply(
                hdr_neural, hdr_direct, auto_fit=True
            )
            meta["align_scale"] = scale
        else:
            hdr_direct = self.aligner.apply(hdr_direct)

        t_h = time.perf_counter()
        self.head.eval()
        with torch.autocast(device_type=self.device.type, enabled=False):
            i_pred = self.head(hdr_direct, hdr_neural, depth)
        meta["head_ms"] = (time.perf_counter() - t_h) * 1000.0

        violations: Dict[str, float] = {}
        if self.use_confidence:
            alpha, violations = self.confidence.compute(hdr_neural, hdr_direct, depth)
            meta["alpha_mode"] = "confidence"
            meta["alpha_mean"] = float(alpha.mean().item())
        else:
            alpha = torch.full(
                (*hdr_direct.shape[:-1], 1),
                self.alpha,
                device=hdr_direct.device,
                dtype=hdr_direct.dtype,
            )
            meta["alpha_mode"] = "fixed"
            meta["alpha_mean"] = self.alpha

        hdr_fused = fuse_direct_indirect(hdr_direct, i_pred, alpha)
        meta["total_ms"] = (time.perf_counter() - t0) * 1000.0
        meta["violations"] = violations

        return C1RenderContext(
            hdr_direct=hdr_direct,
            hdr_neural=hdr_neural,
            depth=depth,
            indirect_pred=i_pred,
            hdr_fused=hdr_fused,
            alpha=alpha,
            meta=meta,
        )

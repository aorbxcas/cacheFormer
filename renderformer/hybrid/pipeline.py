from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from renderformer.hybrid.align import HdrAligner
from renderformer.hybrid.confidence import ConfidenceMap, ConfidenceWeights
from renderformer.hybrid.context import HybridRenderContext
from renderformer.hybrid.decompose import DirectIndirectDecomposer
from renderformer.hybrid.physics import physics_correct
from renderformer.hybrid.profile import HybridProfile
from renderformer.hybrid.runtime_direct.factory import create_runtime_direct_renderer
from renderformer.hybrid.runtime_direct.scene_sync import SceneLightSync

if TYPE_CHECKING:
    from renderformer.cache.vi_cache import ViewIndependentCache
    from renderformer.pipelines.rendering_pipeline import RenderFormerRenderingPipeline


class HybridFusionPipeline:
    """
    H4：路径 B 端到端混合管线。

    Runtime Direct（R0）与 RenderFormer 并行或串行，再分解 + α 融合。
    """

    def __init__(
        self,
        rf_pipeline: "RenderFormerRenderingPipeline",
        profile: Optional[HybridProfile] = None,
        direct_renderer: Optional[object] = None,
    ):
        self.rf_pipeline = rf_pipeline
        profile = profile or HybridProfile()
        self.profile = profile
        self.direct_renderer = direct_renderer or create_runtime_direct_renderer(
            backend=profile.runtime_backend,
            shadow_map_size=profile.shadow_map_size,
        )
        self.aligner = HdrAligner(profile.direct_scale_s, profile.direct_bias_b)
        w = profile.confidence
        self.confidence = ConfidenceMap(
            ConfidenceWeights(
                w0=float(w.get("w0", 2.0)),
                w1=float(w.get("w1", 4.0)),
                w2=float(w.get("w2", 1.0)),
                w3=float(w.get("w3", 2.0)),
                w4=float(w.get("w4", 0.0)),
            )
        )
        self.decomposer = DirectIndirectDecomposer()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        profile: Optional[HybridProfile] = None,
    ) -> "HybridFusionPipeline":
        from renderformer.pipelines.rendering_pipeline import RenderFormerRenderingPipeline

        rf = RenderFormerRenderingPipeline.from_pretrained(model_id)
        return cls(rf_pipeline=rf, profile=profile)

    @property
    def device(self) -> torch.device:
        return self.rf_pipeline.device

    def to(self, device: torch.device) -> "HybridFusionPipeline":
        self.rf_pipeline.to(device)
        return self

    def render(
        self,
        triangles: torch.Tensor,
        texture: torch.Tensor,
        mask: torch.Tensor,
        vn: torch.Tensor,
        c2w: torch.Tensor,
        fov: torch.Tensor,
        resolution: int = 512,
        torch_dtype: torch.dtype = torch.float16,
        vi_cache: Optional["ViewIndependentCache"] = None,
        auto_align: bool = False,
        use_physics_correct: Optional[bool] = None,
        parallel_direct: bool = False,
    ) -> HybridRenderContext:
        SceneLightSync.validate_shared_inputs(triangles, c2w, fov, resolution)
        use_physics = (
            self.profile.use_physics_correct
            if use_physics_correct is None
            else use_physics_correct
        )

        meta: Dict[str, Any] = {}
        device = self.device

        t0 = time.perf_counter()
        if parallel_direct and device.type == "cuda":
            direct_stream = torch.cuda.Stream(device=device)
            rf_stream = torch.cuda.current_stream(device)
            with torch.cuda.stream(direct_stream):
                hdr_direct, depth = self.direct_renderer.render(
                    triangles, texture, vn, mask, c2w, fov, resolution
                )
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
            rf_stream.wait_stream(direct_stream)
        else:
            t_direct = time.perf_counter()
            hdr_direct, depth = self.direct_renderer.render(
                triangles, texture, vn, mask, c2w, fov, resolution
            )
            meta["direct_ms"] = (time.perf_counter() - t_direct) * 1000.0

            t_rf = time.perf_counter()
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
            meta["rf_ms"] = (time.perf_counter() - t_rf) * 1000.0

        meta.update(rf_info)
        meta["total_ms"] = (time.perf_counter() - t0) * 1000.0

        hdr_direct = hdr_direct.to(dtype=hdr_neural.dtype)
        depth = depth.to(dtype=hdr_neural.dtype)

        if auto_align:
            self.aligner.fit_and_apply(hdr_neural, hdr_direct, auto_fit=True)
        hdr_direct = self.aligner.apply(hdr_direct)

        indirect, _ = self.decomposer.decompose(hdr_neural, hdr_direct)

        if self.profile.fixed_alpha is not None:
            alpha = torch.full_like(indirect, float(self.profile.fixed_alpha))
            violations = {}
        else:
            alpha, violations = self.confidence.compute(hdr_neural, hdr_direct, depth)

        hdr_fused = self.decomposer.fuse(hdr_direct, indirect, alpha)

        if use_physics:
            hdr_fused, phys_viol = physics_correct(hdr_fused)
            violations.update(phys_viol)

        return HybridRenderContext(
            hdr_neural=hdr_neural,
            hdr_direct=hdr_direct,
            depth=depth,
            alpha=alpha,
            indirect_neural=indirect,
            hdr_fused=hdr_fused,
            violations=violations,
            meta=meta,
        )

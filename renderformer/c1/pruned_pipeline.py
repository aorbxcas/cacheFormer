# -*- coding: utf-8 -*-
"""
L0/L1 计算剪枝管线：间接缓冲 + 低频神经刷新。

L0（调度剪枝）:
  - 换场景 / 每 N 帧 → refresh：跑神经写 I_buffer
  - 其余帧：不调用 RF，L = Direct + I_buffer

L1（刷新降本）:
  - neural_res_scale / direct_res_scale：低分辨率 RF / Direct
  - `parallel_refresh`：刷新帧 Direct∥RF（墙钟 ≈ max）；**默认关**——lite Direct + RF 多流在本机实测会严重退化（刷新帧升至数秒）

direct_mode: always | refresh_only | stub
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from renderformer.c1.residual_head import ResidualIndirectHead, fuse_direct_indirect
from renderformer.cache.vi_cache import ViewIndependentCache, scene_fingerprint
from renderformer.hybrid.runtime_direct.factory import create_runtime_direct_renderer

if False:  # TYPE_CHECKING
    from renderformer.pipelines.rendering_pipeline import RenderFormerRenderingPipeline


@dataclass
class PrunedFrameResult:
    hdr_fused: torch.Tensor
    hdr_direct: torch.Tensor
    indirect: torch.Tensor
    refreshed: bool
    meta: Dict[str, Any] = field(default_factory=dict)


class PrunedIndirectPipeline:
    """L = Direct + α * I_buffer；神经只在 refresh 进入计算图。"""

    def __init__(
        self,
        rf_pipeline: "RenderFormerRenderingPipeline",
        *,
        refresh_every: int = 3,
        neural_res_scale: float = 0.5,
        direct_res_scale: float = 0.25,
        parallel_refresh: bool = False,
        alpha: float = 1.0,
        direct_mode: str = "always",
        direct_renderer: Optional[object] = None,
        head: Optional[ResidualIndirectHead] = None,
        vi_cache: Optional[ViewIndependentCache] = None,
        auto_align_direct: bool = True,
    ):
        if direct_mode not in ("always", "refresh_only", "stub"):
            raise ValueError("direct_mode must be always|refresh_only|stub")
        if not (0.125 <= neural_res_scale <= 1.0):
            raise ValueError("neural_res_scale should be in [0.125, 1]")
        if not (0.125 <= direct_res_scale <= 1.0):
            raise ValueError("direct_res_scale should be in [0.125, 1]")

        self.rf_pipeline = rf_pipeline
        self.refresh_every = max(1, int(refresh_every))
        self.neural_res_scale = float(neural_res_scale)
        self.direct_res_scale = float(direct_res_scale)
        self.parallel_refresh = bool(parallel_refresh)
        self.alpha = float(alpha)
        self.direct_mode = direct_mode
        self.direct_renderer = direct_renderer or create_runtime_direct_renderer(
            backend="lite", ray_chunk=2048
        )
        self.head = head
        self.vi_cache = vi_cache if vi_cache is not None else ViewIndependentCache(16)
        self.auto_align_direct = auto_align_direct

        self._frame_idx = 0
        self._last_scene_key: Optional[str] = None
        self._i_buffer: Optional[torch.Tensor] = None
        self._d_buffer: Optional[torch.Tensor] = None
        self._depth_buffer: Optional[torch.Tensor] = None

        self.stats = {
            "frames": 0,
            "refreshes": 0,
            "skips": 0,
            "rf_calls": 0,
            "direct_calls": 0,
        }

    @property
    def device(self) -> torch.device:
        return self.rf_pipeline.device

    def to(self, device: torch.device) -> "PrunedIndirectPipeline":
        self.rf_pipeline.to(device)
        if self.head is not None:
            self.head.to(device)
        return self

    def reset(self) -> None:
        self._frame_idx = 0
        self._last_scene_key = None
        self._i_buffer = None
        self._d_buffer = None
        self._depth_buffer = None
        self.vi_cache.clear()
        self.stats = {k: 0 for k in self.stats}

    def _scaled_res(self, resolution: int, scale: float) -> int:
        r = max(32, int(round(resolution * scale)))
        return r - (r % 2)

    def _upsample_to(self, x: torch.Tensor, resolution: int) -> torch.Tensor:
        if x.shape[2] == resolution and x.shape[3] == resolution:
            return x
        b, nv, h, w, c = x.shape
        flat = x.reshape(b * nv, h, w, c).permute(0, 3, 1, 2)
        flat = F.interpolate(
            flat, size=(resolution, resolution), mode="bilinear", align_corners=False
        )
        return flat.permute(0, 2, 3, 1).reshape(b, nv, resolution, resolution, c)

    def _texture_for_fingerprint(self, texture: torch.Tensor) -> torch.Tensor:
        tex = texture
        cfg = self.rf_pipeline.config
        if cfg.texture_encode_patch_size == 1 and tex.dim() == 5:
            tex = tex[:, :, :, 0, 0]
        if not cfg.use_ldr:
            tex = tex.clone()
            tex[:, :, -3:] = torch.log10(tex[:, :, -3:] + 1.0)
        return tex

    def _scene_key(
        self,
        triangles: torch.Tensor,
        texture: torch.Tensor,
        vn: torch.Tensor,
        mask: torch.Tensor,
        scene_key_override: Optional[str],
    ) -> str:
        if scene_key_override is not None:
            return scene_key_override
        tex = self._texture_for_fingerprint(texture)
        return scene_fingerprint(triangles, tex, vn, mask)

    def _need_refresh(self, scene_key: str, force: bool) -> Tuple[bool, str]:
        if force or self._i_buffer is None:
            return True, "force_or_empty"
        if scene_key != self._last_scene_key:
            return True, "scene_change"
        if self._frame_idx % self.refresh_every == 0:
            return True, "interval"
        return False, "skip"

    def _run_direct(
        self,
        triangles,
        texture,
        vn,
        mask,
        c2w,
        fov,
        resolution: int,
        out_resolution: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        d_res = self._scaled_res(resolution, self.direct_res_scale)
        t0 = time.perf_counter()
        hdr_d, depth = self.direct_renderer.render(
            triangles, texture, vn, mask, c2w, fov, d_res
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        self.stats["direct_calls"] += 1
        hdr_d = self._upsample_to(hdr_d.float(), out_resolution)
        depth = self._upsample_to(depth.float(), out_resolution)
        return hdr_d, depth, ms

    def _run_neural(
        self,
        triangles,
        texture,
        mask,
        vn,
        c2w,
        fov,
        resolution: int,
        torch_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Dict[str, Any], float]:
        neural_res = self._scaled_res(resolution, self.neural_res_scale)
        t0 = time.perf_counter()
        hdr, info = self.rf_pipeline.render(
            triangles=triangles,
            texture=texture,
            mask=mask,
            vn=vn,
            c2w=c2w,
            fov=fov,
            resolution=neural_res,
            torch_dtype=torch_dtype,
            vi_cache=self.vi_cache,
            return_vi_cache_info=True,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        self.stats["rf_calls"] += 1
        info = dict(info)
        info["neural_resolution"] = neural_res
        return hdr.float(), info, ms

    @staticmethod
    def _match_res(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-3:-1] == ref.shape[-3:-1]:
            return x
        b, nv, h, w, c = x.shape
        th, tw = ref.shape[2], ref.shape[3]
        flat = x.reshape(b * nv, h, w, c).permute(0, 3, 1, 2)
        flat = F.interpolate(flat, size=(th, tw), mode="bilinear", align_corners=False)
        return flat.permute(0, 2, 3, 1).reshape(b, nv, th, tw, c)

    def _estimate_indirect(
        self,
        hdr_direct: torch.Tensor,
        hdr_neural: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        hdr_neural = self._match_res(hdr_neural, hdr_direct)
        depth = self._match_res(depth, hdr_direct)
        if self.head is not None:
            self.head.eval()
            return self.head(hdr_direct, hdr_neural, depth)
        return torch.relu(hdr_neural - hdr_direct)

    def _refresh_direct_and_neural(
        self,
        triangles,
        texture,
        mask,
        vn,
        c2w,
        fov,
        resolution: int,
        torch_dtype: torch.dtype,
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.direct_mode == "stub":
            hdr_n, rf_info, rf_ms = self._run_neural(
                triangles, texture, mask, vn, c2w, fov, resolution, torch_dtype
            )
            meta["rf_ms"] = rf_ms
            meta["direct_ms"] = 0.0
            meta["vi_cache_hit"] = rf_info.get("vi_cache_hit", False)
            meta["neural_resolution"] = rf_info.get("neural_resolution")
            meta["parallel_refresh"] = False
            hdr_n = self._upsample_to(hdr_n, resolution)
            hdr_d = torch.zeros_like(hdr_n)
            depth = torch.zeros(
                *hdr_n.shape[:-1], 1, device=hdr_n.device, dtype=hdr_n.dtype
            )
            return hdr_d, depth, hdr_n

        use_parallel = self.parallel_refresh and self.device.type == "cuda"
        if not use_parallel:
            hdr_d, depth, d_ms = self._run_direct(
                triangles, texture, vn, mask, c2w, fov, resolution, resolution
            )
            hdr_n, rf_info, rf_ms = self._run_neural(
                triangles, texture, mask, vn, c2w, fov, resolution, torch_dtype
            )
            meta["direct_ms"] = d_ms
            meta["rf_ms"] = rf_ms
            meta["vi_cache_hit"] = rf_info.get("vi_cache_hit", False)
            meta["neural_resolution"] = rf_info.get("neural_resolution")
            meta["parallel_refresh"] = False
            meta["direct_resolution"] = self._scaled_res(resolution, self.direct_res_scale)
            return hdr_d, depth, hdr_n

        s_direct = torch.cuda.Stream()
        s_rf = torch.cuda.Stream()
        holder: Dict[str, Any] = {}
        t0 = time.perf_counter()

        with torch.cuda.stream(s_direct):
            d_res = self._scaled_res(resolution, self.direct_res_scale)
            hdr_d_raw, depth_raw = self.direct_renderer.render(
                triangles, texture, vn, mask, c2w, fov, d_res
            )
            holder["d"] = hdr_d_raw
            holder["z"] = depth_raw
            self.stats["direct_calls"] += 1

        with torch.cuda.stream(s_rf):
            neural_res = self._scaled_res(resolution, self.neural_res_scale)
            hdr_n_raw, rf_info = self.rf_pipeline.render(
                triangles=triangles,
                texture=texture,
                mask=mask,
                vn=vn,
                c2w=c2w,
                fov=fov,
                resolution=neural_res,
                torch_dtype=torch_dtype,
                vi_cache=self.vi_cache,
                return_vi_cache_info=True,
            )
            holder["n"] = hdr_n_raw
            holder["info"] = rf_info
            self.stats["rf_calls"] += 1

        torch.cuda.current_stream().wait_stream(s_direct)
        torch.cuda.current_stream().wait_stream(s_rf)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000.0

        hdr_d = self._upsample_to(holder["d"].float(), resolution)
        depth = self._upsample_to(holder["z"].float(), resolution)
        hdr_n = holder["n"].float()
        meta["direct_ms"] = wall
        meta["rf_ms"] = wall
        meta["wall_ms_parallel"] = wall
        meta["parallel_refresh"] = True
        meta["vi_cache_hit"] = bool(holder["info"].get("vi_cache_hit", False))
        meta["neural_resolution"] = self._scaled_res(resolution, self.neural_res_scale)
        meta["direct_resolution"] = self._scaled_res(resolution, self.direct_res_scale)
        return hdr_d, depth, hdr_n

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
        scene_key: Optional[str] = None,
        force_refresh: bool = False,
    ) -> PrunedFrameResult:
        meta: Dict[str, Any] = {}
        t_all = time.perf_counter()
        self.stats["frames"] += 1

        key = self._scene_key(triangles, texture, vn, mask, scene_key)
        refreshed, reason = self._need_refresh(key, force_refresh)
        meta["refresh_reason"] = reason
        meta["scene_key"] = key[:16]
        meta["frame_idx"] = self._frame_idx

        if refreshed:
            self.stats["refreshes"] += 1
            hdr_d, depth, hdr_n = self._refresh_direct_and_neural(
                triangles, texture, mask, vn, c2w, fov, resolution, torch_dtype, meta
            )

            if self.direct_mode == "stub":
                indirect = hdr_n
                meta["indirect_ms"] = 0.0
                meta["stub_direct"] = True
            else:
                if self.auto_align_direct:
                    n = self._match_res(hdr_n, hdr_d)
                    mask_pos = (n > 1e-4) & (hdr_d > 1e-4)
                    if mask_pos.any():
                        scale = (n[mask_pos] / hdr_d[mask_pos]).median().clamp(0.1, 100.0)
                        hdr_d = hdr_d * scale
                        meta["align_scale"] = float(scale.item())
                t_i = time.perf_counter()
                indirect = self._estimate_indirect(hdr_d, hdr_n, depth)
                meta["indirect_ms"] = (time.perf_counter() - t_i) * 1000.0

            self._i_buffer = indirect
            self._d_buffer = hdr_d
            self._depth_buffer = depth
            self._last_scene_key = key
            meta["rf_invoked"] = True
        else:
            self.stats["skips"] += 1
            meta["rf_invoked"] = False
            meta["vi_cache_hit"] = None

            if self.direct_mode == "always":
                hdr_d, depth, d_ms = self._run_direct(
                    triangles, texture, vn, mask, c2w, fov, resolution, resolution
                )
                meta["direct_ms"] = d_ms
                self._d_buffer = hdr_d
                self._depth_buffer = depth
            else:
                assert self._d_buffer is not None and self._i_buffer is not None
                hdr_d = self._d_buffer
                meta["direct_ms"] = 0.0
                meta["direct_reused"] = True

            indirect = self._i_buffer
            assert indirect is not None
            if indirect.shape[2:4] != hdr_d.shape[2:4]:
                indirect = self._match_res(indirect, hdr_d)

        hdr_fused = fuse_direct_indirect(hdr_d, indirect, self.alpha)
        meta["total_ms"] = (time.perf_counter() - t_all) * 1000.0
        meta["refreshed"] = refreshed
        meta["refresh_every"] = self.refresh_every
        meta["neural_res_scale"] = self.neural_res_scale
        meta["direct_res_scale"] = self.direct_res_scale
        meta["direct_mode"] = self.direct_mode

        self._frame_idx += 1
        return PrunedFrameResult(
            hdr_fused=hdr_fused,
            hdr_direct=hdr_d,
            indirect=indirect,
            refreshed=refreshed,
            meta=meta,
        )

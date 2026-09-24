# -*- coding: utf-8 -*-
"""三层间接光性能栈：L1 Direct/I 分频 + L2 跳过帧快项 + L3 VI/自适应刷新。"""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple

import torch

from renderformer.c1.pruned_pipeline import PrunedFrameResult, PrunedIndirectPipeline
from renderformer.c1.reproject import camera_rotation_deg
from renderformer.c1.residual_head import ResidualIndirectHead, fuse_direct_indirect
from renderformer.c1.skip_fast_term import (
    closed_form_lighting_gate,
    compose_skip_frame,
    mix_residual_head,
)


class LayeredIndirectPipeline(PrunedIndirectPipeline):
    """
    L1  每帧维护 I：跳过帧 inverse-reproject；可选 nvd Direct 跟帧（hybrid 锚点）。
    L2  快项：warp 为默认恒等；可选 closed-form 亮度门 / 残差头小 mix。
    L3  慢项预算：VI Cache + 小运动推迟 RF interval（max_skip 兜底）。

    quality_anchor=rf 时刷新帧存 RF HDR、跳过帧 warp 该图，相对 CacheFormer 色差最小。
    """

    def __init__(
        self,
        rf_pipeline,
        *,
        quality_anchor: str = "rf",
        l1_direct_follow: bool = False,
        l2_gate_strength: float = 0.0,
        l2_head_mix: float = 0.0,
        l3_adaptive: bool = True,
        soft_rot_deg: float = 20.0,
        max_skip_run: int = 5,
        **kwargs,
    ):
        if quality_anchor not in ("rf", "hybrid"):
            raise ValueError("quality_anchor must be rf|hybrid")
        kwargs.setdefault("quality_lock", True)
        kwargs.setdefault("neural_res_scale", 1.0)
        kwargs.setdefault("direct_res_scale", 1.0)
        kwargs.setdefault("view_follow", "reproject")
        if quality_anchor == "rf":
            kwargs.setdefault("direct_mode", "stub")
        kwargs.setdefault("depth_mode", "analytic")
        super().__init__(rf_pipeline, **kwargs)
        self.quality_anchor = quality_anchor
        self.l1_direct_follow = bool(l1_direct_follow)
        self.l2_gate_strength = float(l2_gate_strength)
        self.l2_head_mix = float(l2_head_mix)
        self.l3_adaptive = bool(l3_adaptive)
        self.soft_rot_deg = float(soft_rot_deg)
        self.max_skip_run = max(self.refresh_every, int(max_skip_run))
        self._last_refresh_idx = -10**9
        self._n_buffer: Optional[torch.Tensor] = None
        self.stats.update({"l2_applied": 0, "l1_direct_skips": 0, "adaptive_defers": 0})

    def reset(self) -> None:
        super().reset()
        self._last_refresh_idx = -10**9
        self._n_buffer = None
        self.stats["l2_applied"] = 0
        self.stats["l1_direct_skips"] = 0
        self.stats["adaptive_defers"] = 0

    def _elapsed_since_refresh(self) -> int:
        return int(self._frame_idx - self._last_refresh_idx)

    def _need_refresh(
        self,
        scene_key: str,
        force: bool,
        c2w: torch.Tensor,
    ) -> Tuple[bool, str]:
        if force or self._i_buffer is None or self._fused_buffer is None:
            return True, "force_or_empty"
        if scene_key != self._last_scene_key:
            return True, "scene_change"
        if self._guard_left > 0:
            return True, "post_change_guard"

        rot = 0.0
        if self._c2w_buffer is not None:
            rot = camera_rotation_deg(self._c2w_buffer, c2w)
            if rot >= self.max_camera_rot_deg:
                return True, "camera_motion"

        elapsed = self._elapsed_since_refresh()
        if elapsed >= self.max_skip_run:
            return True, "max_skip"

        if not self.l3_adaptive:
            if elapsed >= self.refresh_every:
                return True, "interval"
            return False, "skip"

        if elapsed >= self.refresh_every and rot >= self.soft_rot_deg:
            return True, "interval_motion"
        if elapsed >= self.refresh_every:
            self.stats["adaptive_defers"] += 1
        return False, "skip"

    def _apply_l2(
        self,
        indirect: torch.Tensor,
        fused: torch.Tensor,
        hdr_d: torch.Tensor,
        depth: torch.Tensor,
        meta: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        i2 = closed_form_lighting_gate(
            indirect, hdr_d, self._d_buffer, self.l2_gate_strength
        )
        n_warp = self._n_buffer
        if n_warp is not None and n_warp.shape[2:4] == fused.shape[2:4]:
            i2 = mix_residual_head(
                i2, self.head, hdr_d, n_warp, depth, self.l2_head_mix
            )
        changed = (self.l2_gate_strength > 0 and self._d_buffer is not None) or (
            self.l2_head_mix > 0 and self.head is not None
        )
        if changed:
            self.stats["l2_applied"] += 1
            if self.quality_anchor == "hybrid":
                fused = fuse_direct_indirect(hdr_d, i2, self.alpha)
            else:
                # RF 锚点：间接微调后仍以 warp 融合图为主，避免换 Direct 配方
                fused = fused
            meta["l2_applied"] = True
        else:
            meta["l2_applied"] = False
        return fused, i2

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
        meta: dict[str, Any] = {
            "quality_lock": self.quality_lock,
            "view_follow": self.view_follow,
            "quality_anchor": self.quality_anchor,
            "l3_adaptive": self.l3_adaptive,
            "elapsed_since_refresh": self._elapsed_since_refresh(),
        }
        t_all = time.perf_counter()
        self.stats["frames"] += 1

        key = self._scene_key(triangles, texture, vn, mask, scene_key)
        refreshed, reason = self._need_refresh(key, force_refresh, c2w)
        meta["refresh_reason"] = reason
        meta["scene_key"] = key[:16]
        meta["frame_idx"] = self._frame_idx

        if not refreshed:
            fused, hdr_d, indirect, retry = self._skip_reproject(c2w, fov, meta)
            if retry:
                refreshed, reason = True, "hole_ratio"
                meta["refresh_reason"] = reason
            else:
                depth = self._depth_buffer
                if depth is None:
                    depth = torch.zeros(
                        (*fused.shape[:-1], 1), device=fused.device, dtype=fused.dtype
                    )
                if self.l1_direct_follow and self.quality_anchor == "hybrid":
                    hdr_now, depth_now, d_ms = self._run_direct(
                        triangles, texture, vn, mask, c2w, fov, resolution, resolution
                    )
                    meta["direct_ms"] = d_ms
                    self.stats["l1_direct_skips"] += 1
                    fused, hdr_d, indirect = compose_skip_frame(
                        quality_anchor="hybrid",
                        fused_warp=fused,
                        indirect_warp=indirect,
                        direct_warp=hdr_d,
                        direct_now=hdr_now,
                        alpha=self.alpha,
                    )
                    depth = depth_now
                fused, indirect = self._apply_l2(indirect, fused, hdr_d, depth, meta)
                self.stats["skips"] += 1
                meta["rf_invoked"] = False
                meta["direct_ms"] = meta.get("direct_ms", meta.get("reproject_ms", 0.0))
                meta["total_ms"] = (time.perf_counter() - t_all) * 1000.0
                meta["refreshed"] = False
                self._frame_idx += 1
                return PrunedFrameResult(
                    hdr_fused=fused,
                    hdr_direct=hdr_d,
                    indirect=indirect,
                    refreshed=False,
                    meta=meta,
                )

        self.stats["refreshes"] += 1
        hdr_d, depth, hdr_n = self._refresh_direct_and_neural(
            triangles, texture, mask, vn, c2w, fov, resolution, torch_dtype, meta
        )
        self._n_buffer = hdr_n.detach()

        if self.quality_anchor == "rf" or self.direct_mode == "stub":
            indirect = hdr_n
            meta["indirect_ms"] = 0.0
            meta["stub_direct"] = True
            hdr_fused = hdr_n
            if self.direct_mode == "stub":
                hdr_d = torch.zeros_like(hdr_n)
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
            hdr_fused = fuse_direct_indirect(hdr_d, indirect, self.alpha)

        self._i_buffer = indirect
        self._d_buffer = hdr_d
        self._fused_buffer = hdr_fused
        self._depth_buffer = depth
        self._c2w_buffer = c2w.detach().clone()
        self._fov_buffer = fov.detach().clone()
        self._last_refresh_idx = self._frame_idx
        if reason == "scene_change":
            self._guard_left = self.guard_refreshes_after_scene_change
        elif reason == "post_change_guard" and self._guard_left > 0:
            self._guard_left -= 1
        self._last_scene_key = key
        meta["rf_invoked"] = True
        meta["guard_left"] = self._guard_left
        meta["total_ms"] = (time.perf_counter() - t_all) * 1000.0
        meta["refreshed"] = True
        meta["refresh_every"] = self.refresh_every
        meta["max_skip_run"] = self.max_skip_run
        self._frame_idx += 1
        return PrunedFrameResult(
            hdr_fused=hdr_fused,
            hdr_direct=hdr_d,
            indirect=indirect,
            refreshed=True,
            meta=meta,
        )

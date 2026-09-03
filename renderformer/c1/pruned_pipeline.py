# -*- coding: utf-8 -*-
"""
L0/L1 计算剪枝管线：间接/融合缓冲 + 低频神经刷新。

质量约束（quality_lock=True，P0–P2 默认）：
  - neural_res_scale / direct_res_scale 强制为 1.0（禁止半分辨率）
  - 跳过帧用全分辨率深度重投影跟相机（非降采样）
  - 刷新帧全分辨率 RF +（可选）深度/Direct；VI 缓存可叠

view_follow:
  - reproject: 跳过帧重投影上一帧融合结果（推荐，无 nvd 也能超 CF）
  - direct_plus_i: 跳过帧跑全分 Direct + 重投影 I（需快 Direct）
  - freeze: 跳过帧不跟相机（仅消融）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from renderformer.c1.reproject import (
    analytic_plane_depth,
    camera_rotation_deg,
    reproject_by_depth,
    reproject_inverse_bilinear,
    warp_depth_forward,
)
from renderformer.c1.residual_head import ResidualIndirectHead, fuse_direct_indirect
from renderformer.cache.vi_cache import ViewIndependentCache, scene_fingerprint

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
    """L = Direct + α * I；或质量模式下融合缓冲重投影。"""

    def __init__(
        self,
        rf_pipeline: "RenderFormerRenderingPipeline",
        *,
        refresh_every: int = 3,
        neural_res_scale: float = 1.0,
        direct_res_scale: float = 1.0,
        parallel_refresh: bool = False,
        alpha: float = 1.0,
        direct_mode: str = "stub",
        view_follow: str = "reproject",
        quality_lock: bool = True,
        max_camera_rot_deg: float = 25.0,
        max_hole_ratio: float = 0.35,
        depth_mode: str = "analytic",
        depth_aux_scale: float = 0.5,
        reproject_mode: str = "inverse",
        guard_refreshes_after_scene_change: int = 0,
        direct_renderer: Optional[object] = None,
        head: Optional[ResidualIndirectHead] = None,
        vi_cache: Optional[ViewIndependentCache] = None,
        auto_align_direct: bool = True,
    ):
        if direct_mode not in ("always", "refresh_only", "stub"):
            raise ValueError("direct_mode must be always|refresh_only|stub")
        if view_follow not in ("reproject", "direct_plus_i", "freeze"):
            raise ValueError("view_follow must be reproject|direct_plus_i|freeze")

        self.quality_lock = bool(quality_lock)
        if self.quality_lock:
            if neural_res_scale < 1.0 - 1e-6 or direct_res_scale < 1.0 - 1e-6:
                raise ValueError(
                    "quality_lock=True 禁止半分辨率：neural/direct_res_scale 必须为 1.0"
                )
            neural_res_scale = 1.0
            direct_res_scale = 1.0

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
        self.view_follow = view_follow
        self.max_camera_rot_deg = float(max_camera_rot_deg)
        self.max_hole_ratio = float(max_hole_ratio)
        if depth_mode not in ("analytic", "raycast"):
            raise ValueError("depth_mode must be analytic|raycast")
        self.depth_mode = depth_mode
        self.depth_aux_scale = float(depth_aux_scale)
        if not (0.25 <= self.depth_aux_scale <= 1.0):
            raise ValueError("depth_aux_scale should be in [0.25, 1]")
        if reproject_mode not in ("inverse", "forward"):
            raise ValueError("reproject_mode must be inverse|forward")
        self.reproject_mode = reproject_mode
        self.guard_refreshes_after_scene_change = max(
            0, int(guard_refreshes_after_scene_change)
        )
        self._guard_left = 0
        from renderformer.hybrid.runtime_direct.factory import (
            create_runtime_direct_renderer,
            nvdiffrast_available,
        )

        if direct_renderer is not None:
            self.direct_renderer = direct_renderer
        elif nvdiffrast_available():
            self.direct_renderer = create_runtime_direct_renderer(backend="nvdiffrast")
        else:
            self.direct_renderer = create_runtime_direct_renderer(
                backend="lite", ray_chunk=8192
            )
        self.head = head
        self.vi_cache = vi_cache if vi_cache is not None else ViewIndependentCache(16)
        self.auto_align_direct = auto_align_direct

        self._frame_idx = 0
        self._last_scene_key: Optional[str] = None
        self._i_buffer: Optional[torch.Tensor] = None
        self._d_buffer: Optional[torch.Tensor] = None
        self._fused_buffer: Optional[torch.Tensor] = None
        self._depth_buffer: Optional[torch.Tensor] = None
        self._c2w_buffer: Optional[torch.Tensor] = None
        self._fov_buffer: Optional[torch.Tensor] = None

        self.stats = {
            "frames": 0,
            "refreshes": 0,
            "skips": 0,
            "rf_calls": 0,
            "direct_calls": 0,
            "reprojects": 0,
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
        self._fused_buffer = None
        self._depth_buffer = None
        self._c2w_buffer = None
        self._fov_buffer = None
        self._guard_left = 0
        self.vi_cache.clear()
        self.stats = {k: 0 for k in self.stats}

    def _scaled_res(self, resolution: int, scale: float) -> int:
        if self.quality_lock:
            return int(resolution)
        r = max(32, int(round(resolution * scale)))
        return r - (r % 2)

    def _upsample_to(self, x: torch.Tensor, resolution: int) -> torch.Tensor:
        if x.shape[2] == resolution and x.shape[3] == resolution:
            return x
        if self.quality_lock and (x.shape[2] != resolution or x.shape[3] != resolution):
            raise RuntimeError("quality_lock: 禁止上采样补分辨率")
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
        if self._c2w_buffer is not None:
            rot = camera_rotation_deg(self._c2w_buffer, c2w)
            if rot >= self.max_camera_rot_deg:
                return True, "camera_motion"
        if self._frame_idx % self.refresh_every == 0:
            return True, "interval"
        return False, "skip"

    def _run_depth(
        self,
        triangles,
        texture,
        vn,
        mask,
        c2w,
        fov,
        resolution: int,
    ) -> Tuple[torch.Tensor, float]:
        t0 = time.perf_counter()
        if self.depth_mode == "analytic":
            depth = analytic_plane_depth(c2w, fov, resolution)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            return depth.float(), ms

        # 辅助深度可低于输出分辨率（只服务重投影，不降彩色输出分辨率）
        d_res = max(64, int(round(resolution * self.depth_aux_scale)))
        d_res = d_res - (d_res % 2)
        _hdr, depth = self.direct_renderer.render(
            triangles, texture, vn, mask, c2w, fov, d_res, depth_only=True
        )
        depth = depth.float()
        if d_res != resolution:
            # 深度用 nearest，保边缘，避免锯齿被双线性抹糊后再错切
            b, nv, dh, dw, _ = depth.shape
            flat = depth.reshape(b * nv, 1, dh, dw)
            flat = F.interpolate(flat, size=(resolution, resolution), mode="nearest")
            depth = flat.permute(0, 2, 3, 1).reshape(b, nv, resolution, resolution, 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        self.stats["direct_calls"] += 1
        return depth, ms

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
        # 质量路径：全分辨率 RF；Direct 按 mode；stub 另采 depth_only 供重投影
        hdr_n, rf_info, rf_ms = self._run_neural(
            triangles, texture, mask, vn, c2w, fov, resolution, torch_dtype
        )
        meta["rf_ms"] = rf_ms
        meta["vi_cache_hit"] = rf_info.get("vi_cache_hit", False)
        meta["neural_resolution"] = rf_info.get("neural_resolution")
        meta["parallel_refresh"] = False
        hdr_n = self._upsample_to(hdr_n, resolution)

        if self.direct_mode == "stub":
            depth, z_ms = self._run_depth(
                triangles, texture, vn, mask, c2w, fov, resolution
            )
            meta["direct_ms"] = 0.0
            meta["depth_ms"] = z_ms
            meta["stub_direct"] = True
            hdr_d = torch.zeros_like(hdr_n)
            return hdr_d, depth, hdr_n

        hdr_d, depth, d_ms = self._run_direct(
            triangles, texture, vn, mask, c2w, fov, resolution, resolution
        )
        meta["direct_ms"] = d_ms
        meta["direct_resolution"] = self._scaled_res(resolution, self.direct_res_scale)
        return hdr_d, depth, hdr_n

    def _skip_reproject(
        self,
        c2w: torch.Tensor,
        fov: torch.Tensor,
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """返回 fused, direct, indirect, need_refresh_retry。"""
        assert self._fused_buffer is not None
        assert self._depth_buffer is not None
        assert self._c2w_buffer is not None
        assert self._fov_buffer is not None
        assert self._i_buffer is not None
        assert self._d_buffer is not None

        t0 = time.perf_counter()
        # 用刷新帧几何深度 warp 到当前相机，再反向双线性采色 → 避免平面近似错切锯齿
        if self.depth_mode == "raycast":
            depth_dst = warp_depth_forward(
                self._depth_buffer,
                self._c2w_buffer,
                self._fov_buffer,
                c2w,
                fov,
            )
            # 深度已由同一几何 warp，关闭苛刻遮挡测试以免大片误判成洞 → 填洞锯齿
            z_tol = -1.0
        else:
            depth_dst = analytic_plane_depth(c2w, fov, self._fused_buffer.shape[2])
            z_tol = -1.0
        if self.reproject_mode == "inverse":
            fused, _valid, hole = reproject_inverse_bilinear(
                self._fused_buffer,
                self._depth_buffer,
                self._c2w_buffer,
                self._fov_buffer,
                c2w,
                fov,
                depth_dst,
                z_rel_tol=z_tol,
            )
        else:
            fused, _valid, hole = reproject_by_depth(
                self._fused_buffer,
                self._depth_buffer,
                self._c2w_buffer,
                self._fov_buffer,
                c2w,
                fov,
            )
        if self.direct_mode == "stub":
            hdr_d = torch.zeros_like(fused)
            indirect = fused
        else:
            if self.reproject_mode == "inverse":
                indirect, _, _ = reproject_inverse_bilinear(
                    self._i_buffer,
                    self._depth_buffer,
                    self._c2w_buffer,
                    self._fov_buffer,
                    c2w,
                    fov,
                    depth_dst,
                    z_rel_tol=z_tol,
                )
                hdr_d, _, _ = reproject_inverse_bilinear(
                    self._d_buffer,
                    self._depth_buffer,
                    self._c2w_buffer,
                    self._fov_buffer,
                    c2w,
                    fov,
                    depth_dst,
                    z_rel_tol=z_tol,
                )
            else:
                indirect, _, _ = reproject_by_depth(
                    self._i_buffer,
                    self._depth_buffer,
                    self._c2w_buffer,
                    self._fov_buffer,
                    c2w,
                    fov,
                )
                hdr_d, _, _ = reproject_by_depth(
                    self._d_buffer,
                    self._depth_buffer,
                    self._c2w_buffer,
                    self._fov_buffer,
                    c2w,
                    fov,
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        meta["reproject_ms"] = (time.perf_counter() - t0) * 1000.0
        meta["hole_ratio"] = hole
        meta["reproject_mode"] = self.reproject_mode
        self.stats["reprojects"] += 1
        need_retry = hole > self.max_hole_ratio
        return fused, hdr_d, indirect, need_retry

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
        meta: Dict[str, Any] = {
            "quality_lock": self.quality_lock,
            "view_follow": self.view_follow,
        }
        t_all = time.perf_counter()
        self.stats["frames"] += 1

        key = self._scene_key(triangles, texture, vn, mask, scene_key)
        refreshed, reason = self._need_refresh(key, force_refresh, c2w)
        meta["refresh_reason"] = reason
        meta["scene_key"] = key[:16]
        meta["frame_idx"] = self._frame_idx

        if not refreshed and self.view_follow == "reproject":
            fused, hdr_d, indirect, retry = self._skip_reproject(c2w, fov, meta)
            if retry:
                refreshed, reason = True, "hole_ratio"
                meta["refresh_reason"] = reason
            else:
                self.stats["skips"] += 1
                meta["rf_invoked"] = False
                meta["direct_ms"] = meta.get("reproject_ms", 0.0)
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

        if refreshed:
            self.stats["refreshes"] += 1
            hdr_d, depth, hdr_n = self._refresh_direct_and_neural(
                triangles, texture, mask, vn, c2w, fov, resolution, torch_dtype, meta
            )

            if self.direct_mode == "stub":
                indirect = hdr_n
                meta["indirect_ms"] = 0.0
                meta["stub_direct"] = True
                hdr_fused = hdr_n
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
            if reason == "scene_change":
                self._guard_left = self.guard_refreshes_after_scene_change
            elif reason == "post_change_guard" and self._guard_left > 0:
                self._guard_left -= 1
            self._last_scene_key = key
            meta["rf_invoked"] = True
            meta["guard_left"] = self._guard_left
        else:
            self.stats["skips"] += 1
            meta["rf_invoked"] = False
            meta["vi_cache_hit"] = None

            if self.view_follow == "direct_plus_i" or self.direct_mode == "always":
                hdr_d, depth, d_ms = self._run_direct(
                    triangles, texture, vn, mask, c2w, fov, resolution, resolution
                )
                meta["direct_ms"] = d_ms
                self._d_buffer = hdr_d
                self._depth_buffer = depth
                if self.view_follow == "direct_plus_i" and self._i_buffer is not None and self._c2w_buffer is not None:
                    indirect, _, hole = reproject_by_depth(
                        self._i_buffer,
                        self._depth_buffer if self._depth_buffer is not None else depth,
                        self._c2w_buffer,
                        self._fov_buffer,
                        c2w,
                        fov,
                    )
                    meta["hole_ratio"] = hole
                    self.stats["reprojects"] += 1
                else:
                    indirect = self._i_buffer
            else:
                assert self._d_buffer is not None and self._i_buffer is not None
                hdr_d = self._d_buffer
                meta["direct_ms"] = 0.0
                meta["direct_reused"] = True
                indirect = self._i_buffer

            assert indirect is not None
            if indirect.shape[2:4] != hdr_d.shape[2:4]:
                if self.quality_lock:
                    raise RuntimeError("quality_lock: I/D 分辨率不一致")
                indirect = self._match_res(indirect, hdr_d)
            hdr_fused = fuse_direct_indirect(hdr_d, indirect, self.alpha)
            self._fused_buffer = hdr_fused
            self._c2w_buffer = c2w.detach().clone()
            self._fov_buffer = fov.detach().clone()

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

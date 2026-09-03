#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0–P2 质量锁定基准：全分辨率 pruned（重投影跟视图）vs CacheFormer。

约束：neural/direct_res_scale=1.0，禁止半分辨率；验收速度 > CF 且 absL1 可控。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio.v3 as iio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_vi_cache import _apply_camera_variant, load_h5
from renderformer import RenderFormerRenderingPipeline, ViewIndependentCache
from renderformer.c1.pruned_pipeline import PrunedIndirectPipeline
from renderformer.c1.residual_head import ResidualIndirectHead

ROUGHNESS_CHANNEL = 6
SCENE_ROUGHNESS = [0.15, 0.45, 0.75, 0.95]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _orbit_variant(frame_idx: int, num_frames: int) -> dict:
    if num_frames <= 1:
        deg = 0.0
    else:
        deg = -30.0 + 60.0 * frame_idx / (num_frames - 1)
    return {"name": f"f{frame_idx + 1:02d}_orbit_{deg:+.0f}deg", "orbit_y_deg": deg}


def _scene_texture(base_texture: torch.Tensor, scene_id: int) -> torch.Tensor:
    tex = base_texture.clone()
    r = SCENE_ROUGHNESS[scene_id % len(SCENE_ROUGHNESS)]
    tex[:, :, ROUGHNESS_CHANNEL : ROUGHNESS_CHANNEL + 1, :, :] = r
    return tex


def _tonemap(hdr: np.ndarray) -> np.ndarray:
    x = np.clip(hdr.astype(np.float32), 0, None)
    scale = float(np.percentile(x[x > 0], 95)) if (x > 0).any() else 1.0
    ldr = 1.0 - np.exp(-x / (scale * 0.6 + 1e-6))
    return (np.clip(ldr, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def _to_hwc(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().cpu()
    while x.dim() > 3 and x.shape[0] == 1:
        x = x[0]
    if x.dim() == 4 and x.shape[-1] == 3:
        x = x[0]
    elif x.dim() == 3 and x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    return x.numpy().astype(np.float32)


def _psnr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse < eps:
        return 99.0
    peak = max(float(np.percentile(b, 99)), eps)
    return float(10.0 * np.log10((peak * peak) / mse))


def _ensure_c2w_fov(data: dict):
    c2w, fov = data["c2w"], data["fov"]
    if c2w.dim() == 3:
        c2w = c2w.unsqueeze(1)
    if fov.dim() == 2:
        fov = fov.unsqueeze(-1)
    return c2w, fov


def _load_head(path: Optional[str], device: torch.device) -> Optional[ResidualIndirectHead]:
    if not path or not Path(path).is_file():
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("head_cfg") or {}
    head = ResidualIndirectHead(
        use_neural=bool(cfg.get("use_neural", True)),
        use_depth=bool(cfg.get("use_depth", True)),
        base_channels=int(cfg.get("base_channels", 32)),
        num_blocks=int(cfg.get("num_blocks", 3)),
    )
    head.load_state_dict(ckpt["model"])
    return head.to(device).eval()


@torch.no_grad()
def run_cf(rf, data, num_frames, scene_change_every, device, dtype, resolution, out_dir):
    cache = ViewIndependentCache(16)
    base_c2w, base_fov = data["c2w"], data["fov"]
    frames_dir = out_dir / "cacheformer"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records, times, hdrs = [], [], []

    tex0 = _scene_texture(data["texture"], 0)
    c2w, fov = _apply_camera_variant(base_c2w, base_fov, _orbit_variant(0, num_frames), device, dtype)
    _sync(device)
    rf.render(
        data["triangles"], tex0, data["mask"], data["vn"], c2w, fov,
        resolution=resolution, torch_dtype=dtype, vi_cache=cache, return_vi_cache_info=True,
    )
    cache.clear()

    for i in range(num_frames):
        sid = i // scene_change_every
        tex = _scene_texture(data["texture"], sid)
        var = _orbit_variant(i, num_frames)
        c2w, fov = _apply_camera_variant(base_c2w, base_fov, var, device, dtype)
        _sync(device)
        t0 = time.perf_counter()
        hdr, info = rf.render(
            data["triangles"], tex, data["mask"], data["vn"], c2w, fov,
            resolution=resolution, torch_dtype=dtype, vi_cache=cache, return_vi_cache_info=True,
        )
        _sync(device)
        elapsed = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed)
        arr = _to_hwc(hdr)
        hdrs.append(arr)
        iio.imwrite(frames_dir / f"frame_{i + 1:02d}_s{sid}.png", _tonemap(arr))
        records.append(
            {
                "frame": i + 1,
                "scene_id": sid,
                "elapsed_ms": elapsed,
                "vi_cache_hit": bool(info.get("vi_cache_hit")),
            }
        )
        print(f"  [CF] f{i + 1:02d} s{sid} {elapsed:.1f} ms hit={info.get('vi_cache_hit')}")

    total = sum(times) / 1000.0
    return {
        "mean_ms": float(np.mean(times)),
        "fps": len(times) / total,
        "rf_calls": num_frames,
        "frames": records,
        "hdrs": hdrs,
    }


@torch.no_grad()
def run_quality(pipe, data, num_frames, scene_change_every, device, dtype, resolution, out_dir, tag):
    pipe.reset()
    base_c2w, base_fov = data["c2w"], data["fov"]
    frames_dir = out_dir / tag
    frames_dir.mkdir(parents=True, exist_ok=True)
    records, times, hdrs = [], [], []

    for i in range(num_frames):
        sid = i // scene_change_every
        tex = _scene_texture(data["texture"], sid)
        var = _orbit_variant(i, num_frames)
        c2w, fov = _apply_camera_variant(base_c2w, base_fov, var, device, dtype)
        _sync(device)
        t0 = time.perf_counter()
        out = pipe.render(
            data["triangles"], tex, data["mask"], data["vn"], c2w, fov,
            resolution=resolution, torch_dtype=dtype, scene_key=f"scene_{sid}",
        )
        _sync(device)
        elapsed = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed)
        arr = _to_hwc(out.hdr_fused)
        hdrs.append(arr)
        iio.imwrite(frames_dir / f"frame_{i + 1:02d}_s{sid}.png", _tonemap(arr))
        records.append(
            {
                "frame": i + 1,
                "scene_id": sid,
                "elapsed_ms": elapsed,
                "refreshed": out.refreshed,
                "reason": out.meta.get("refresh_reason"),
                "hole_ratio": out.meta.get("hole_ratio"),
                "rf_ms": out.meta.get("rf_ms"),
                "depth_ms": out.meta.get("depth_ms"),
                "reproject_ms": out.meta.get("reproject_ms"),
            }
        )
        print(
            f"  [{tag}] f{i + 1:02d} s{sid} {elapsed:.1f} ms "
            f"refresh={out.refreshed} reason={out.meta.get('refresh_reason')} "
            f"hole={out.meta.get('hole_ratio')}"
        )

    total = sum(times) / 1000.0
    return {
        "mean_ms": float(np.mean(times)),
        "fps": len(times) / total,
        "rf_calls": pipe.stats["rf_calls"],
        "refreshes": pipe.stats["refreshes"],
        "skips": pipe.stats["skips"],
        "reprojects": pipe.stats["reprojects"],
        "pipe_stats": dict(pipe.stats),
        "frames": records,
        "hdrs": hdrs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", default="tmp/c1_scenes/cbox.h5")
    parser.add_argument("--checkpoint", default="checkpoints/c1_cycles/best.pt")
    parser.add_argument("--model_id", default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--scene_change_every", type=int, default=3)
    parser.add_argument("--refresh_every", type=int, default=3)
    parser.add_argument("--direct_mode", default="stub", choices=["stub", "refresh_only", "always"])
    parser.add_argument("--view_follow", default="reproject", choices=["reproject", "direct_plus_i", "freeze"])
    parser.add_argument("--max_camera_rot_deg", type=float, default=18.0)
    parser.add_argument("--max_hole_ratio", type=float, default=0.99)
    parser.add_argument("--depth_mode", default="analytic", choices=["analytic", "raycast"])
    parser.add_argument("--depth_aux_scale", type=float, default=0.5)
    parser.add_argument("--guard_after_change", type=int, default=0)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--output_dir", default="out/compare_quality_p0_p2")
    parser.add_argument("--use_c1_head", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading...")
    data = load_h5(args.h5_file, device)
    data["c2w"], data["fov"] = _ensure_c2w_fov(data)
    rf = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    rf.to(device)
    head = _load_head(args.checkpoint, device) if args.use_c1_head else None

    print("\n=== CacheFormer ===")
    cf = run_cf(
        rf, data, args.num_frames, args.scene_change_every, device, dtype, args.resolution, out_dir
    )

    print("\n=== Quality pruned (full-res) ===")
    pipe = PrunedIndirectPipeline(
        rf,
        refresh_every=args.refresh_every,
        neural_res_scale=1.0,
        direct_res_scale=1.0,
        quality_lock=True,
        direct_mode=args.direct_mode,
        view_follow=args.view_follow,
        max_camera_rot_deg=args.max_camera_rot_deg,
        max_hole_ratio=args.max_hole_ratio,
        depth_mode=args.depth_mode,
        depth_aux_scale=args.depth_aux_scale,
        guard_refreshes_after_scene_change=args.guard_after_change,
        head=head,
        auto_align_direct=True,
    ).to(device)
    tag = f"quality_{args.direct_mode}_{args.view_follow}"
    pr = run_quality(
        pipe, data, args.num_frames, args.scene_change_every, device, dtype, args.resolution, out_dir, tag
    )

    # quality vs CF
    abs_l1, psnrs = [], []
    for a, b in zip(pr["hdrs"], cf["hdrs"]):
        abs_l1.append(float(np.mean(np.abs(a - b))))
        psnrs.append(_psnr(a, b))
    mean_abs = float(np.mean(abs_l1))
    mean_psnr = float(np.mean(psnrs))
    speedup = cf["mean_ms"] / pr["mean_ms"] if pr["mean_ms"] > 0 else 0.0

    # strip hdrs from json
    cf_out = {k: v for k, v in cf.items() if k != "hdrs"}
    pr_out = {k: v for k, v in pr.items() if k != "hdrs"}
    report = {
        "config": vars(args),
        "constraints": {
            "quality_lock": True,
            "neural_res_scale": 1.0,
            "direct_res_scale": 1.0,
            "no_half_res": True,
        },
        "cacheformer": cf_out,
        "quality_pruned": pr_out,
        "speedup_vs_cacheformer": speedup,
        "mean_abs_l1_vs_cf": mean_abs,
        "mean_psnr_vs_cf": mean_psnr,
        "per_frame_abs_l1": abs_l1,
        "per_frame_psnr": psnrs,
        "pass_speed": bool(speedup > 1.0),
        "pass_quality_soft": bool(mean_abs < 0.08),
    }
    path = out_dir / "report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # contact sheet
    try:
        from PIL import Image, ImageDraw, ImageFont

        rows = []
        for i in range(args.num_frames):
            sid = i // args.scene_change_every
            a = iio.imread(out_dir / "cacheformer" / f"frame_{i + 1:02d}_s{sid}.png")
            b = iio.imread(out_dir / tag / f"frame_{i + 1:02d}_s{sid}.png")
            rows.append(np.concatenate([a, b], axis=1))
        sheet = np.concatenate(rows[:4], axis=0)  # first 4 frames stacked
        # wider sheet: all frames in 2 rows
        top = np.concatenate(
            [iio.imread(out_dir / "cacheformer" / f"frame_{i + 1:02d}_s{i // args.scene_change_every}.png") for i in range(args.num_frames)],
            axis=1,
        )
        bot = np.concatenate(
            [iio.imread(out_dir / tag / f"frame_{i + 1:02d}_s{i // args.scene_change_every}.png") for i in range(args.num_frames)],
            axis=1,
        )
        sheet = np.concatenate([top, bot], axis=0)
        img = Image.fromarray(sheet)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        draw.text(
            (8, 8),
            f"CF {cf['mean_ms']:.0f}ms | quality {pr['mean_ms']:.0f}ms ({speedup:.2f}x) | "
            f"absL1={mean_abs:.4f} PSNR={mean_psnr:.1f}dB",
            fill=(255, 255, 0),
            font=font,
        )
        sheet_path = out_dir / "contact_sheet_quality.png"
        img.save(sheet_path)
        print(f"Contact sheet -> {sheet_path}")
    except Exception as e:
        print(f"[WARN] sheet: {e}")

    print("\n========== SUMMARY ==========")
    print(f"CacheFormer:     {cf['mean_ms']:.1f} ms")
    print(f"Quality pruned:  {pr['mean_ms']:.1f} ms  ({speedup:.2f}x)  rf_calls={pr['rf_calls']}")
    print(f"absL1 vs CF:     {mean_abs:.4f}")
    print(f"PSNR vs CF:      {mean_psnr:.2f} dB")
    print(f"pass_speed:      {report['pass_speed']}")
    print(f"pass_quality:    {report['pass_quality_soft']} (absL1<0.08)")
    print(f"Report -> {path}")


if __name__ == "__main__":
    main()

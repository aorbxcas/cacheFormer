#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L0/L1 剪枝管线 vs CacheFormer：动态场景连续帧 benchmark。

每 scene_change_every 帧切换 roughness；段内 orbit。
目标：剪枝路径平均帧时 < CacheFormer，且 RF 调用次数显著下降。

用法:
  python tools/benchmark_pruned_l0_l1.py --h5_file tmp/c1_scenes/cbox.h5
  python tools/benchmark_pruned_l0_l1.py --h5_file tmp/c1_scenes/cbox.h5 \\
      --direct_mode refresh_only --neural_res_scale 0.5 --refresh_every 3
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
def run_cacheformer(
    rf: RenderFormerRenderingPipeline,
    data: dict,
    num_frames: int,
    scene_change_every: int,
    device: torch.device,
    dtype: torch.dtype,
    resolution: int,
    out_dir: Path,
) -> Dict[str, Any]:
    cache = ViewIndependentCache(16)
    base_c2w, base_fov = data["c2w"], data["fov"]
    records: List[dict] = []
    times: List[float] = []

    # warmup
    tex0 = _scene_texture(data["texture"], 0)
    c2w, fov = _apply_camera_variant(base_c2w, base_fov, _orbit_variant(0, num_frames), device, dtype)
    _sync(device)
    rf.render(
        data["triangles"], tex0, data["mask"], data["vn"], c2w, fov,
        resolution=resolution, torch_dtype=dtype, vi_cache=cache, return_vi_cache_info=True,
    )
    cache.clear()

    frames_dir = out_dir / "cacheformer"
    frames_dir.mkdir(parents=True, exist_ok=True)

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
        iio.imwrite(frames_dir / f"frame_{i + 1:02d}_s{sid}.png", _tonemap(arr))
        records.append(
            {
                "frame": i + 1,
                "scene_id": sid,
                "roughness": SCENE_ROUGHNESS[sid % len(SCENE_ROUGHNESS)],
                "elapsed_ms": elapsed,
                "vi_cache_hit": bool(info.get("vi_cache_hit")),
                "rf_invoked": True,
            }
        )
        print(
            f"  [CF] f{i + 1:02d} s{sid} {elapsed:.1f} ms hit={info.get('vi_cache_hit')}"
        )

    total = sum(times) / 1000.0
    return {
        "label": "CacheFormer (RF+VI every frame)",
        "mean_ms": float(np.mean(times)),
        "fps": len(times) / total if total > 0 else 0.0,
        "total_sec": total,
        "rf_calls": num_frames,
        "cache_hits": sum(1 for r in records if r["vi_cache_hit"]),
        "cache_misses": sum(1 for r in records if not r["vi_cache_hit"]),
        "frames": records,
    }


@torch.no_grad()
def run_pruned(
    pipe: PrunedIndirectPipeline,
    data: dict,
    num_frames: int,
    scene_change_every: int,
    device: torch.device,
    dtype: torch.dtype,
    resolution: int,
    out_dir: Path,
    tag: str,
) -> Dict[str, Any]:
    pipe.reset()
    base_c2w, base_fov = data["c2w"], data["fov"]
    records: List[dict] = []
    times: List[float] = []

    # warmup refresh
    tex0 = _scene_texture(data["texture"], 0)
    c2w, fov = _apply_camera_variant(base_c2w, base_fov, _orbit_variant(0, num_frames), device, dtype)
    _sync(device)
    pipe.render(
        data["triangles"], tex0, data["mask"], data["vn"], c2w, fov,
        resolution=resolution, torch_dtype=dtype, scene_key="warm", force_refresh=True,
    )
    pipe.reset()

    frames_dir = out_dir / tag
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_frames):
        sid = i // scene_change_every
        tex = _scene_texture(data["texture"], sid)
        var = _orbit_variant(i, num_frames)
        c2w, fov = _apply_camera_variant(base_c2w, base_fov, var, device, dtype)
        _sync(device)
        t0 = time.perf_counter()
        out = pipe.render(
            data["triangles"],
            tex,
            data["mask"],
            data["vn"],
            c2w,
            fov,
            resolution=resolution,
            torch_dtype=dtype,
            scene_key=f"scene_{sid}",
        )
        _sync(device)
        # use wall clock including our outer timer (meta total_ms is internal)
        elapsed = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed)
        arr = _to_hwc(out.hdr_fused)
        iio.imwrite(frames_dir / f"frame_{i + 1:02d}_s{sid}.png", _tonemap(arr))
        rec = {
            "frame": i + 1,
            "scene_id": sid,
            "roughness": SCENE_ROUGHNESS[sid % len(SCENE_ROUGHNESS)],
            "elapsed_ms": elapsed,
            "refreshed": out.refreshed,
            "rf_invoked": out.meta.get("rf_invoked"),
            "vi_cache_hit": out.meta.get("vi_cache_hit"),
            "direct_ms": out.meta.get("direct_ms"),
            "rf_ms": out.meta.get("rf_ms"),
            "refresh_reason": out.meta.get("refresh_reason"),
        }
        records.append(rec)
        print(
            f"  [{tag}] f{i + 1:02d} s{sid} {elapsed:.1f} ms "
            f"refresh={out.refreshed} reason={out.meta.get('refresh_reason')} "
            f"rf={out.meta.get('rf_ms')} d={out.meta.get('direct_ms')}"
        )

    total = sum(times) / 1000.0
    return {
        "label": tag,
        "mean_ms": float(np.mean(times)),
        "fps": len(times) / total if total > 0 else 0.0,
        "total_sec": total,
        "rf_calls": pipe.stats["rf_calls"],
        "direct_calls": pipe.stats["direct_calls"],
        "refreshes": pipe.stats["refreshes"],
        "skips": pipe.stats["skips"],
        "frames": records,
        "pipe_stats": dict(pipe.stats),
    }


def _write_contact_sheet(
    out_dir: Path,
    cf_dir: Path,
    pruned_dir: Path,
    num_frames: int,
    scene_change_every: int,
    cf_mean_ms: float,
    pruned_mean_ms: float,
    speedup: float,
    pruned_label: str,
) -> Path:
    """Rows = [CacheFormer, pruned]; columns = frames."""
    from PIL import Image, ImageDraw, ImageFont

    frames = []
    for i in range(num_frames):
        sid = i // scene_change_every
        name = f"frame_{i + 1:02d}_s{sid}.png"
        a = iio.imread(cf_dir / name)
        b = iio.imread(pruned_dir / name)
        frames.append((a, b))

    h, w = frames[0][0].shape[:2]
    label_w = 168
    gap = 4
    sheet_w = label_w + num_frames * (w + gap) + gap
    sheet_h = 2 * (h + gap) + gap + 36
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    try:
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font_sm = ImageFont.load_default()

    titles = [
        f"CacheFormer  {cf_mean_ms:.0f}ms",
        f"{pruned_label}  {pruned_mean_ms:.0f}ms ({speedup:.2f}x)",
    ]
    for row, title in enumerate(titles):
        y0 = gap + row * (h + gap)
        draw.text((6, y0 + h // 2 - 8), title, fill=(220, 220, 230), font=font_sm)
        for col, (a, b) in enumerate(frames):
            img = a if row == 0 else b
            x = label_w + gap + col * (w + gap)
            sheet.paste(Image.fromarray(img), (x, y0))
            if row == 0:
                sid = col // scene_change_every
                draw.text(
                    (x + 4, sheet_h - 28),
                    f"f{col + 1} s{sid}",
                    fill=(180, 180, 190),
                    font=font_sm,
                )

    draw.text(
        (8, sheet_h - 28),
        "dynamic: roughness every N + orbit",
        fill=(140, 140, 150),
        font=font_sm,
    )
    path = out_dir / "contact_sheet_cf_vs_pruned.png"
    sheet.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Benchmark L0/L1 pruned GI vs CacheFormer")
    parser.add_argument("--h5_file", type=str, default="tmp/c1_scenes/cbox.h5")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/c1_cycles/best.pt")
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--scene_change_every", type=int, default=3)
    parser.add_argument("--refresh_every", type=int, default=3)
    parser.add_argument("--neural_res_scale", type=float, default=0.5)
    parser.add_argument("--direct_res_scale", type=float, default=0.25)
    parser.add_argument(
        "--parallel_refresh",
        action="store_true",
        default=False,
        help="Experimental: Direct∥RF on refresh (often slower with lite Direct; off by default)",
    )
    parser.add_argument("--no_parallel_refresh", action="store_true")
    parser.add_argument(
        "--direct_mode",
        type=str,
        default="always",
        choices=["always", "refresh_only", "stub"],
        help="always=每帧低分 Direct（推荐）；stub=消融",
    )
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--output_dir", type=str, default="out/compare_pruned_l0_l1")
    parser.add_argument("--no_c1_head", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scene + RF...")
    data = load_h5(args.h5_file, device)
    c2w, fov = _ensure_c2w_fov(data)
    data["c2w"], data["fov"] = c2w, fov

    rf = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    rf.to(device)
    head = None if args.no_c1_head else _load_head(args.checkpoint, device)

    print("\n=== CacheFormer ===")
    cf = run_cacheformer(
        rf, data, args.num_frames, args.scene_change_every, device, dtype, args.resolution, out_dir
    )

    print("\n=== L0/L1 pruned ===")
    pipe = PrunedIndirectPipeline(
        rf,
        refresh_every=args.refresh_every,
        neural_res_scale=args.neural_res_scale,
        direct_res_scale=args.direct_res_scale,
        parallel_refresh=(args.parallel_refresh and not args.no_parallel_refresh),
        direct_mode=args.direct_mode,
        head=head,
        auto_align_direct=True,
    ).to(device)
    tag = (
        f"pruned_r{args.refresh_every}_n{args.neural_res_scale}_"
        f"d{args.direct_res_scale}_{args.direct_mode}"
    )
    pr = run_pruned(
        pipe,
        data,
        args.num_frames,
        args.scene_change_every,
        device,
        dtype,
        args.resolution,
        out_dir,
        tag,
    )

    speedup = cf["mean_ms"] / pr["mean_ms"] if pr["mean_ms"] > 0 else 0.0
    report = {
        "config": vars(args),
        "cacheformer": cf,
        "pruned": pr,
        "speedup_vs_cacheformer": speedup,
        "rf_call_reduction": 1.0 - (pr["rf_calls"] / max(cf["rf_calls"], 1)),
    }
    path = out_dir / "report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    sheet = _write_contact_sheet(
        out_dir,
        out_dir / "cacheformer",
        out_dir / tag,
        args.num_frames,
        args.scene_change_every,
        cf["mean_ms"],
        pr["mean_ms"],
        speedup,
        pruned_label=f"{args.direct_mode}",
    )

    print("\n========== SUMMARY ==========")
    print(f"CacheFormer:  mean={cf['mean_ms']:.1f} ms  fps={cf['fps']:.2f}  rf_calls={cf['rf_calls']}")
    print(
        f"Pruned L0/L1: mean={pr['mean_ms']:.1f} ms  fps={pr['fps']:.2f}  "
        f"rf_calls={pr['rf_calls']}  refreshes={pr['refreshes']}  skips={pr['skips']}"
    )
    print(f"Speedup vs CacheFormer: {speedup:.2f}x")
    print(f"RF call reduction: {100 * report['rf_call_reduction']:.0f}%")
    print(f"Report -> {path}")
    print(f"Contact sheet -> {sheet}")


if __name__ == "__main__":
    main()

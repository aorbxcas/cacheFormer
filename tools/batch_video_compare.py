#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续帧（环视）视频案例：RenderFormer / CacheFormer / Runtime Direct 三路对比。

目录:
  tmp/renders_video/{slug}/
    renderformer/   frames/ + video.mp4
    cacheformer/    frames/ + video.mp4
    runtime_direct/ frames/ + video.mp4

用法:
  python tools/batch_video_compare.py
  python tools/batch_video_compare.py --num_frames 16 --fps 12 --resolution 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_vi_cache import _apply_camera_variant, load_h5
from renderformer import RenderFormerRenderingPipeline, ViewIndependentCache
from renderformer.hybrid.runtime_direct import RuntimeDirectRenderer, nvdiffrast_available

# 两个环视视频案例（复用已生成的 H5）
VIDEO_CASES = [
    {
        "slug": "cbox-orbit",
        "h5_slug": "cbox",
        "orbit_start_deg": -35.0,
        "orbit_end_deg": 35.0,
    },
    {
        "slug": "fox-orbit",
        "h5_slug": "fox-in-the-wild",
        "orbit_start_deg": -30.0,
        "orbit_end_deg": 30.0,
    },
]


def _tonemap_hdr(hdr: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(hdr[hdr > 0], 95.0)) if (hdr > 0).any() else 1.0
    ldr = 1.0 - np.exp(-hdr / (scale * 0.6 + 1e-6))
    return (np.clip(ldr, 0, 1) ** (1.0 / 2.2) * 255).astype(np.uint8)


def _orbit_variants(num_frames: int, start_deg: float, end_deg: float) -> list[dict]:
    if num_frames <= 1:
        return [{"name": "f0", "orbit_y_deg": start_deg}]
    out = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        deg = start_deg + (end_deg - start_deg) * t
        out.append({"name": f"frame_{i:03d}", "orbit_y_deg": deg})
    return out


def _single_view(c2w: torch.Tensor, fov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """确保 c2w (1,1,4,4)、fov (1,1,1)。"""
    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0).unsqueeze(0)
    elif c2w.dim() == 3:
        c2w = c2w.unsqueeze(1)
    if fov.dim() == 0:
        fov = fov.view(1, 1, 1)
    elif fov.dim() == 1:
        fov = fov.view(1, -1, 1)
    elif fov.dim() == 2:
        fov = fov.unsqueeze(-1)
    return c2w, fov


def _render_video_sequence(
    slug: str,
    h5_path: Path,
    out_root: Path,
    rf: RenderFormerRenderingPipeline,
    direct: RuntimeDirectRenderer,
    device: torch.device,
    dtype: torch.dtype,
    resolution: int,
    num_frames: int,
    fps: int,
    orbit_start: float,
    orbit_end: float,
) -> dict:
    data = load_h5(str(h5_path), device)
    # 基准相机：取 H5 第一视角
    base_c2w = data["c2w"][:, 0:1, :, :].squeeze(1)  # (1, 4, 4)
    base_fov = data["fov"][:, 0:1, :]  # (1, 1, 1)
    variants = _orbit_variants(num_frames, orbit_start, orbit_end)

    record = {
        "slug": slug,
        "h5": str(h5_path),
        "num_frames": num_frames,
        "fps": fps,
        "orbit_deg": [orbit_start, orbit_end],
        "methods": {},
    }

    vi_cache = ViewIndependentCache(max_entries=4)
    scene_kw_base = {
        "triangles": data["triangles"],
        "texture": data["texture"],
        "mask": data["mask"],
        "vn": data["vn"],
        "resolution": resolution,
    }
    rf_kw_base = {**scene_kw_base, "torch_dtype": dtype}

    for method in ("renderformer", "cacheformer", "runtime_direct"):
        out_dir = out_root / slug / method
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames: list[np.ndarray] = []
        times_ms: list[float] = []
        hits = 0

        if method == "cacheformer":
            vi_cache.clear()

        for i, var in enumerate(variants):
            c2w, fov = _apply_camera_variant(base_c2w, base_fov, var, device, dtype)
            c2w, fov = _single_view(c2w, fov)

            t0 = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize()

            if method == "renderformer":
                hdr = rf.render(**rf_kw_base, c2w=c2w, fov=fov)
                arr = hdr[0, 0].detach().cpu().numpy().astype(np.float32)
            elif method == "cacheformer":
                hdr, info = rf.render(
                    **rf_kw_base,
                    c2w=c2w,
                    fov=fov,
                    vi_cache=vi_cache,
                    return_vi_cache_info=True,
                )
                arr = hdr[0, 0].detach().cpu().numpy().astype(np.float32)
                if info.get("vi_cache_hit"):
                    hits += 1
            else:
                hdr, _depth = direct.render(
                    **scene_kw_base,
                    c2w=c2w,
                    fov=fov,
                )
                arr = hdr[0, 0].detach().cpu().numpy().astype(np.float32)

            if device.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

            ldr = _tonemap_hdr(arr)
            frames.append(ldr)
            imageio.v3.imwrite(frames_dir / f"frame_{i:03d}.png", ldr)
            imageio.v3.imwrite(frames_dir / f"frame_{i:03d}.exr", arr)

        video_path = out_dir / "video.mp4"
        imageio.v3.imwrite(video_path, np.stack(frames), fps=fps, quality=9)

        rec = {
            "video": str(video_path),
            "frames_dir": str(frames_dir),
            "mean_ms": round(float(np.mean(times_ms)), 2),
            "total_ms": round(float(np.sum(times_ms)), 2),
        }
        if method == "cacheformer":
            rec["vi_cache_hits"] = hits
            rec["vi_cache_misses"] = num_frames - hits
        record["methods"][method] = rec

    return record


def main():
    parser = argparse.ArgumentParser(description="Batch orbit video compare")
    parser.add_argument("--tmp_root", type=str, default=str(ROOT / "tmp"))
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--direct_backend", type=str, default="auto", choices=["auto", "nvdiffrast", "lite"])
    parser.add_argument("--cases", nargs="*", default=None, help="video slug list, default: cbox-orbit fox-orbit")
    args = parser.parse_args()

    tmp_root = Path(args.tmp_root)
    out_root = tmp_root / "renders_video"
    cases = args.cases or [c["slug"] for c in VIDEO_CASES]
    case_map = {c["slug"]: c for c in VIDEO_CASES}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.float16
        if args.precision == "fp16"
        else torch.bfloat16
        if args.precision == "bf16"
        else torch.float32
    )

    print(f"=== Video compare: {len(cases)} cases, {args.num_frames} frames @ {args.resolution}px ===")
    print(f"  device={device}, nvd={nvdiffrast_available()}")

    t0 = time.perf_counter()
    rf = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    if device.type == "cuda" and os.name == "posix":
        try:
            from renderformer_liger_kernel import apply_kernels

            apply_kernels(rf.model)
        except ImportError:
            pass
    rf.to(device)
    direct = RuntimeDirectRenderer(backend=args.direct_backend)
    print(f"  model load: {time.perf_counter()-t0:.1f}s, direct={direct.active_backend}")

    manifest = {
        "tmp_root": str(tmp_root),
        "output_root": str(out_root),
        "num_frames": args.num_frames,
        "fps": args.fps,
        "resolution": args.resolution,
        "cases": [],
    }

    for slug in cases:
        cfg = case_map.get(slug)
        if cfg is None:
            print(f"  [{slug}] SKIP unknown case")
            continue
        h5 = tmp_root / "scenes" / cfg["h5_slug"] / f"{cfg['h5_slug']}.h5"
        if not h5.is_file():
            print(f"  [{slug}] SKIP missing h5: {h5}")
            continue
        print(f"  [{slug}] rendering {args.num_frames} frames ...", flush=True)
        try:
            rec = _render_video_sequence(
                slug=slug,
                h5_path=h5,
                out_root=out_root,
                rf=rf,
                direct=direct,
                device=device,
                dtype=dtype,
                resolution=args.resolution,
                num_frames=args.num_frames,
                fps=args.fps,
                orbit_start=cfg["orbit_start_deg"],
                orbit_end=cfg["orbit_end_deg"],
            )
            manifest["cases"].append(rec)
            m = rec["methods"]
            print(
                f"    RF {m['renderformer']['total_ms']:.0f}ms | "
                f"Cache {m['cacheformer']['total_ms']:.0f}ms "
                f"(hits {m['cacheformer'].get('vi_cache_hits', 0)}/{args.num_frames}) | "
                f"Direct {m['runtime_direct']['total_ms']:.0f}ms"
            )
        except Exception as e:
            print(f"    FAIL: {e}")
            manifest["cases"].append({"slug": slug, "error": str(e)})

    manifest_path = out_root / "manifest_video.json"
    out_root.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nDone. {manifest_path}")


if __name__ == "__main__":
    main()

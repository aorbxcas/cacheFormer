#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全渲染管线连续序列帧对比（同场景 orbit 相机）。

管线：
  - rf_baseline      RenderFormer 全量（无 VI 缓存）
  - rf_vicache       CacheFormer：RF + ViewIndependentCache
  - direct_only      Runtime Direct
  - hybrid           Hybrid（Direct + RF decompose + confidence）
  - hybrid_vicache   Hybrid + VI 缓存
  - c1               C1 残差间接头
  - c1_vicache       C1 + VI 缓存

用法:
  python tools/compare_all_pipelines_sequence.py \\
      --h5_file tmp/c1_scenes/cbox.h5 \\
      --checkpoint checkpoints/c1_cycles/best.pt \\
      --num_frames 12 --resolution 256
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_vi_cache import _apply_camera_variant, load_h5
from renderformer import RenderFormerRenderingPipeline, ViewIndependentCache
from renderformer.c1.pipeline import C1ResidualPipeline
from renderformer.c1.pruned_pipeline import PrunedIndirectPipeline
from renderformer.c1.residual_head import ResidualIndirectHead
from renderformer.hybrid.pipeline import HybridFusionPipeline
from renderformer.hybrid.profile import HybridProfile
from renderformer.hybrid.runtime_direct.factory import create_runtime_direct_renderer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _gpu_peak_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def _orbit_sequence(num_frames: int, deg_span: float = 60.0) -> List[dict]:
    out = []
    for i in range(num_frames):
        if num_frames <= 1:
            deg = 0.0
        else:
            deg = -deg_span / 2 + deg_span * i / (num_frames - 1)
        out.append({"name": f"orbit_{deg:+.0f}", "orbit_y_deg": float(deg)})
    return out


def _tonemap(hdr: np.ndarray) -> np.ndarray:
    x = np.clip(hdr.astype(np.float32), 0, None)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    scale = float(np.percentile(x[x > 0], 95)) if (x > 0).any() else 1.0
    ldr = 1.0 - np.exp(-x / (scale * 0.6 + 1e-6))
    return (np.clip(ldr, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def _to_hwc(t: torch.Tensor) -> np.ndarray:
    """Accept [B,nv,H,W,3] or [B,3,H,W] or [H,W,3]."""
    x = t.detach().float().cpu()
    while x.dim() > 3 and x.shape[0] == 1:
        x = x[0]
    if x.dim() == 4 and x.shape[0] <= 4:  # C,H,W or nv,H,W,C
        if x.shape[-1] == 3:
            x = x[0]
        else:
            x = x[0].permute(1, 2, 0)
    elif x.dim() == 3 and x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    return x.numpy().astype(np.float32)


def _abs_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _rel_l1(a: np.ndarray, b: np.ndarray, eps: float = 1e-3) -> float:
    return float(np.mean(np.abs(a - b) / (np.abs(b) + eps)))


def _load_c1_head(path: str, device: torch.device) -> ResidualIndirectHead:
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


def _ensure_c2w_fov(data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    c2w = data["c2w"]
    fov = data["fov"]
    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)
    if c2w.dim() == 3:
        c2w = c2w.unsqueeze(1)
    if fov.dim() == 1:
        fov = fov.view(1, 1, 1)
    elif fov.dim() == 2:
        fov = fov.unsqueeze(-1)
    return c2w, fov


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="All-pipeline continuous sequence compare")
    parser.add_argument("--h5_file", type=str, default="tmp/c1_scenes/cbox.h5")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/c1_cycles/best.pt")
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--output_dir", type=str, default="out/compare_all_pipelines")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--auto_align", action="store_true", default=True)
    parser.add_argument("--no_auto_align", action="store_true")
    args = parser.parse_args()
    auto_align = args.auto_align and not args.no_auto_align

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    out_root = Path(args.output_dir)
    frames_root = out_root / "frames"
    out_root.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading scene {args.h5_file}")
    data = load_h5(args.h5_file, device)
    base_c2w, base_fov = _ensure_c2w_fov(data)
    data["c2w"], data["fov"] = base_c2w, base_fov

    variants = _orbit_sequence(args.num_frames)
    print(f"Loading RF {args.model_id}")
    rf = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    rf.to(device)

    hybrid = HybridFusionPipeline(
        rf,
        profile=HybridProfile(),
        direct_renderer=create_runtime_direct_renderer(backend="lite", ray_chunk=2048),
    ).to(device)
    direct_only = create_runtime_direct_renderer(backend="lite", ray_chunk=2048)

    c1 = None
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_file():
        head = _load_c1_head(str(ckpt_path), device)
        c1 = C1ResidualPipeline(
            rf,
            head,
            direct_renderer=create_runtime_direct_renderer(backend="lite", ray_chunk=2048),
            alpha=1.0,
            use_confidence=True,
            auto_align=auto_align,
        ).to(device)
        print(f"C1 checkpoint: {ckpt_path}")
    else:
        print(f"[WARN] C1 checkpoint missing ({ckpt_path}), skip c1 modes")

    # mode_id -> factory that returns (hdr_hwc, meta) for one frame
    def make_rf(use_cache: bool):
        cache = ViewIndependentCache(max_entries=8) if use_cache else None

        def _run(c2w, fov, clear_cache=False):
            if cache is not None and clear_cache:
                cache.clear()
            out = rf.render(
                triangles=data["triangles"],
                texture=data["texture"],
                mask=data["mask"],
                vn=data["vn"],
                c2w=c2w,
                fov=fov,
                resolution=args.resolution,
                torch_dtype=dtype,
                vi_cache=cache,
                return_vi_cache_info=True,
            )
            imgs, info = out
            return _to_hwc(imgs), dict(info), cache

        return _run

    def make_direct():
        def _run(c2w, fov, clear_cache=False):
            hdr, depth = direct_only.render(
                data["triangles"],
                data["texture"],
                data["vn"],
                data["mask"],
                c2w,
                fov,
                args.resolution,
            )
            return _to_hwc(hdr), {"vi_cache_hit": False}, None

        return _run

    def make_hybrid(use_cache: bool):
        cache = ViewIndependentCache(max_entries=8) if use_cache else None

        def _run(c2w, fov, clear_cache=False):
            if cache is not None and clear_cache:
                cache.clear()
            ctx = hybrid.render(
                triangles=data["triangles"],
                texture=data["texture"],
                mask=data["mask"],
                vn=data["vn"],
                c2w=c2w,
                fov=fov,
                resolution=args.resolution,
                torch_dtype=dtype,
                vi_cache=cache,
                auto_align=auto_align,
            )
            meta = {k: v for k, v in ctx.meta.items() if isinstance(v, (int, float, bool, str, type(None)))}
            return _to_hwc(ctx.hdr_fused), meta, cache

        return _run

    def make_c1(use_cache: bool):
        cache = ViewIndependentCache(max_entries=8) if use_cache else None

        def _run(c2w, fov, clear_cache=False):
            if cache is not None and clear_cache:
                cache.clear()
            ctx = c1.render(
                triangles=data["triangles"],
                texture=data["texture"],
                mask=data["mask"],
                vn=data["vn"],
                c2w=c2w,
                fov=fov,
                resolution=args.resolution,
                torch_dtype=dtype,
                vi_cache=cache,
            )
            meta = {k: v for k, v in ctx.meta.items() if isinstance(v, (int, float, bool, str, type(None)))}
            return _to_hwc(ctx.hdr_fused), meta, cache

        return _run

    def make_pruned(direct_mode: str):
        pipe = PrunedIndirectPipeline(
            rf,
            refresh_every=3,
            neural_res_scale=0.5,
            direct_res_scale=0.25,
            parallel_refresh=False,
            direct_mode=direct_mode,
            head=None,
            auto_align_direct=auto_align,
        ).to(device)

        def _run(c2w, fov, clear_cache=False):
            if clear_cache:
                pipe.reset()
            out = pipe.render(
                data["triangles"],
                data["texture"],
                data["mask"],
                data["vn"],
                c2w,
                fov,
                resolution=args.resolution,
                torch_dtype=dtype,
                scene_key="static",
            )
            meta = {
                k: v
                for k, v in out.meta.items()
                if isinstance(v, (int, float, bool, str, type(None)))
            }
            meta["vi_cache_hit"] = False
            return _to_hwc(out.hdr_fused), meta, None

        return _run

    modes: List[Tuple[str, str, Callable]] = [
        ("rf_baseline", "RF 全量（无缓存）", make_rf(False)),
        ("rf_vicache", "CacheFormer（RF+VI Cache）", make_rf(True)),
        ("direct_only", "Runtime Direct", make_direct()),
        ("hybrid", "Hybrid GI", make_hybrid(False)),
        ("hybrid_vicache", "Hybrid + VI Cache", make_hybrid(True)),
    ]
    if c1 is not None:
        modes.append(("c1", "C1 残差头", make_c1(False)))
        modes.append(("c1_vicache", "C1 + VI Cache", make_c1(True)))
    modes.append(("pruned_always", "L0/L1 Direct@64", make_pruned("always")))
    modes.append(("pruned_stub", "L0/L1 stub", make_pruned("stub")))

    # ---- Pass 1: RF baseline reference frames ----
    print("=== Pass: RF baseline references ===")
    rf_ref_runner = make_rf(False)
    ref_hdrs: List[np.ndarray] = []
    # warmup
    c2w0, fov0 = _apply_camera_variant(base_c2w, base_fov, variants[0], device, dtype)
    for _ in range(args.warmup):
        _sync(device)
        rf_ref_runner(c2w0, fov0)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    results: Dict[str, Any] = {
        "scene": args.h5_file,
        "resolution": args.resolution,
        "num_frames": args.num_frames,
        "device": str(device),
        "precision": args.precision,
        "checkpoint": str(ckpt_path) if ckpt_path.is_file() else None,
        "modes": {},
        "frames": [{"index": i + 1, "name": v["name"], "orbit_y_deg": v["orbit_y_deg"]} for i, v in enumerate(variants)],
    }

    # We'll fill rf_baseline during the mode loop; also keep refs
    mode_frame_records: Dict[str, List[dict]] = {}

    for mode_id, mode_label, runner in modes:
        print(f"=== Mode: {mode_id} ({mode_label}) ===")
        mode_dir = frames_root / mode_id
        mode_dir.mkdir(parents=True, exist_ok=True)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        # warmup：缓存模式在 warmup 末尾保持热缓存，模拟连续序列稳态
        c2w_w, fov_w = _apply_camera_variant(base_c2w, base_fov, variants[0], device, dtype)
        try:
            for wi in range(args.warmup):
                _sync(device)
                runner(c2w_w, fov_w, clear_cache=(wi == 0 and "vicache" in mode_id))
        except RuntimeError as e:
            print(f"  [SKIP] warmup failed: {e}")
            results["modes"][mode_id] = {
                "summary": {
                    "label": mode_label,
                    "error": str(e),
                    "mean_ms": None,
                    "fps": None,
                    "speedup_vs_rf_baseline": None,
                    "mean_abs_l1_vs_rf": None,
                    "mean_rel_l1_vs_rf": None,
                    "cache_hit_rate": None,
                    "gpu_peak_mb": _gpu_peak_mb(device),
                },
                "frames": [],
            }
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        records: List[dict] = []
        times: List[float] = []
        hits = 0
        misses = 0
        aborted = False

        for i, var in enumerate(variants):
            c2w, fov = _apply_camera_variant(base_c2w, base_fov, var, device, dtype)
            try:
                _sync(device)
                t0 = time.perf_counter()
                hdr, meta, cache = runner(c2w, fov, clear_cache=False)
                _sync(device)
                elapsed = time.perf_counter() - t0
            except RuntimeError as e:
                print(f"  [ABORT] frame {i + 1}: {e}")
                aborted = True
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break

            times.append(elapsed)
            if device.type == "cuda" and mode_id in (
                "direct_only",
                "hybrid",
                "hybrid_vicache",
                "c1",
                "c1_vicache",
            ):
                # lite Direct 易碎片化，每帧回收
                torch.cuda.empty_cache()

            hit = bool(meta.get("vi_cache_hit", False))
            if "vicache" in mode_id:
                if hit:
                    hits += 1
                else:
                    misses += 1

            if mode_id == "rf_baseline":
                ref_hdrs.append(hdr.copy())

            q_abs = q_rel = None
            if mode_id != "rf_baseline" and i < len(ref_hdrs):
                ref = ref_hdrs[i]
                if hdr.shape == ref.shape:
                    q_abs = _abs_l1(hdr, ref)
                    q_rel = _rel_l1(hdr, ref)

            png_path = mode_dir / f"frame_{i + 1:02d}.png"
            iio.imwrite(png_path, _tonemap(hdr))
            del hdr

            rec = {
                "frame": i + 1,
                "name": var["name"],
                "orbit_y_deg": var["orbit_y_deg"],
                "elapsed_ms": elapsed * 1000.0,
                "vi_cache_hit": hit,
                "abs_l1_vs_rf": q_abs,
                "rel_l1_vs_rf": q_rel,
                "png": str(png_path.relative_to(out_root)).replace("\\", "/"),
            }
            for k in ("direct_ms", "rf_ms", "total_ms", "head_ms"):
                if k in meta:
                    rec[k] = meta[k]
            records.append(rec)
            print(
                f"  frame {i + 1:02d} {var['name']}: {elapsed * 1000:.1f} ms"
                + (f" hit={hit}" if "vicache" in mode_id else "")
                + (f" absL1={q_abs:.4f}" if q_abs is not None else "")
            )

        if aborted and not times:
            results["modes"][mode_id] = {
                "summary": {
                    "label": mode_label,
                    "error": "aborted",
                    "mean_ms": None,
                    "fps": None,
                    "speedup_vs_rf_baseline": None,
                    "mean_abs_l1_vs_rf": None,
                    "mean_rel_l1_vs_rf": None,
                    "cache_hit_rate": None,
                    "gpu_peak_mb": _gpu_peak_mb(device),
                },
                "frames": records,
            }
            continue

        total = sum(times)
        summary = {
            "label": mode_label,
            "total_sec": total,
            "mean_ms": (total / len(times) * 1000.0) if times else 0.0,
            "fps": (len(times) / total) if total > 0 else 0.0,
            "min_ms": min(times) * 1000.0 if times else 0.0,
            "max_ms": max(times) * 1000.0 if times else 0.0,
            "gpu_peak_mb": _gpu_peak_mb(device),
            "cache_hits": hits if "vicache" in mode_id else None,
            "cache_misses": misses if "vicache" in mode_id else None,
            "cache_hit_rate": (hits / max(hits + misses, 1)) if "vicache" in mode_id else None,
        }
        # quality aggregates
        abs_vals = [r["abs_l1_vs_rf"] for r in records if r["abs_l1_vs_rf"] is not None]
        rel_vals = [r["rel_l1_vs_rf"] for r in records if r["rel_l1_vs_rf"] is not None]
        if abs_vals:
            summary["mean_abs_l1_vs_rf"] = float(np.mean(abs_vals))
            summary["mean_rel_l1_vs_rf"] = float(np.mean(rel_vals))
        else:
            summary["mean_abs_l1_vs_rf"] = 0.0 if mode_id == "rf_baseline" else None
            summary["mean_rel_l1_vs_rf"] = 0.0 if mode_id == "rf_baseline" else None

        # speedup vs rf_baseline
        results["modes"][mode_id] = {"summary": summary, "frames": records}
        mode_frame_records[mode_id] = records

    # fill quality for modes that ran before we had refs? rf_baseline is first, OK.
    # speedup ratios
    base_mean = results["modes"].get("rf_baseline", {}).get("summary", {}).get("mean_ms")
    for mode_id, blob in results["modes"].items():
        m = blob["summary"].get("mean_ms")
        if m and base_mean:
            blob["summary"]["speedup_vs_rf_baseline"] = float(base_mean / m)
        else:
            blob["summary"]["speedup_vs_rf_baseline"] = None
        if blob["summary"].get("error") or not records and mode_id == mode_id:
            pass

    report_path = out_root / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # compact CSV-like table to stdout
    print("\n========== SUMMARY ==========")
    print(f"{'mode':20s} {'mean_ms':>10s} {'fps':>8s} {'speedup':>8s} {'absL1':>10s} {'hit%':>8s}")
    for mode_id, blob in results["modes"].items():
        s = blob["summary"]
        if s.get("mean_ms") is None:
            print(f"{mode_id:20s} {'ERR':>10s} {'-':>8s} {'-':>8s} {'-':>10s} {'-':>8s}")
            continue
        hit = f"{100 * s['cache_hit_rate']:.0f}%" if s.get("cache_hit_rate") is not None else "-"
        abs_s = f"{s['mean_abs_l1_vs_rf']:.4f}" if s.get("mean_abs_l1_vs_rf") is not None else "-"
        sp = f"{s['speedup_vs_rf_baseline']:.2f}x" if s.get("speedup_vs_rf_baseline") else "-"
        print(
            f"{mode_id:20s} {s['mean_ms']:10.1f} {s['fps']:8.2f} {sp:>8s} {abs_s:>10s} {hit:>8s}"
        )
    print(f"\nReport -> {report_path}")
    print(f"Frames -> {frames_root}")


if __name__ == "__main__":
    main()

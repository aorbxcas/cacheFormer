#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三层间接栈 vs CacheFormer：动态场景序列。验收加速>1× 且 absL1<0.08。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_TOOLS = str(ROOT / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from benchmark_vi_cache import _apply_camera_variant, load_h5
from benchmark_multi_dynamic_vs_cf import (
    SCENE_CATALOG,
    FramePlan,
    _apply_texture,
    _cam_variant,
    build_dynamic_plans,
)
from renderformer import RenderFormerRenderingPipeline, ViewIndependentCache
from renderformer.c1.layered_pipeline import LayeredIndirectPipeline


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


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


@torch.no_grad()
def run_cf(rf, data, plans: list[FramePlan], device, dtype, resolution, out_dir: Path):
    cache = ViewIndependentCache(16)
    base_c2w, base_fov = data["c2w"], data["fov"]
    frames_dir = out_dir / "cacheformer"
    frames_dir.mkdir(parents=True, exist_ok=True)

    tex0 = _apply_texture(data["texture"], plans[0])
    c2w, fov = _apply_camera_variant(base_c2w, base_fov, _cam_variant(plans[0]), device, dtype)
    _sync(device)
    rf.render(
        data["triangles"], tex0, data["mask"], data["vn"], c2w, fov,
        resolution=resolution, torch_dtype=dtype, vi_cache=cache, return_vi_cache_info=True,
    )
    cache.clear()

    records, times, hdrs = [], [], []
    for i, plan in enumerate(plans):
        tex = _apply_texture(data["texture"], plan)
        c2w, fov = _apply_camera_variant(base_c2w, base_fov, _cam_variant(plan), device, dtype)
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
        iio.imwrite(frames_dir / f"frame_{i + 1:02d}.png", _tonemap(arr))
        records.append(
            {
                "frame": i + 1,
                "name": plan.name,
                "scene_key": plan.scene_key,
                "elapsed_ms": elapsed,
                "vi_cache_hit": bool(info.get("vi_cache_hit")),
            }
        )
        print(f"  [CF] {plan.name}: {elapsed:.1f} ms hit={info.get('vi_cache_hit')}")

    total = sum(times) / 1000.0
    return {
        "mean_ms": float(np.mean(times)),
        "fps": len(times) / total if total else 0.0,
        "rf_calls": len(plans),
        "frames": records,
        "hdrs": hdrs,
    }


@torch.no_grad()
def run_rf(rf, data, plans: list[FramePlan], device, dtype, resolution, out_dir: Path):
    """纯 RenderFormer：每帧 RF，无 VI Cache。"""
    base_c2w, base_fov = data["c2w"], data["fov"]
    frames_dir = out_dir / "renderformer"
    frames_dir.mkdir(parents=True, exist_ok=True)

    tex0 = _apply_texture(data["texture"], plans[0])
    c2w, fov = _apply_camera_variant(base_c2w, base_fov, _cam_variant(plans[0]), device, dtype)
    _sync(device)
    rf.render(
        data["triangles"], tex0, data["mask"], data["vn"], c2w, fov,
        resolution=resolution, torch_dtype=dtype,
    )

    records, times, hdrs = [], [], []
    for i, plan in enumerate(plans):
        tex = _apply_texture(data["texture"], plan)
        c2w, fov = _apply_camera_variant(base_c2w, base_fov, _cam_variant(plan), device, dtype)
        _sync(device)
        t0 = time.perf_counter()
        hdr = rf.render(
            data["triangles"], tex, data["mask"], data["vn"], c2w, fov,
            resolution=resolution, torch_dtype=dtype,
        )
        _sync(device)
        elapsed = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed)
        arr = _to_hwc(hdr)
        hdrs.append(arr)
        iio.imwrite(frames_dir / f"frame_{i + 1:02d}.png", _tonemap(arr))
        records.append(
            {
                "frame": i + 1,
                "name": plan.name,
                "scene_key": plan.scene_key,
                "elapsed_ms": elapsed,
            }
        )
        print(f"  [RF] {plan.name}: {elapsed:.1f} ms")

    total = sum(times) / 1000.0
    return {
        "mean_ms": float(np.mean(times)),
        "fps": len(times) / total if total else 0.0,
        "rf_calls": len(plans),
        "frames": records,
        "hdrs": hdrs,
    }


@torch.no_grad()
def run_layered(pipe, data, plans: list[FramePlan], device, dtype, resolution, out_dir: Path):
    pipe.reset()
    base_c2w, base_fov = data["c2w"], data["fov"]
    frames_dir = out_dir / "layered"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records, times, hdrs = [], [], []

    for i, plan in enumerate(plans):
        tex = _apply_texture(data["texture"], plan)
        c2w, fov = _apply_camera_variant(base_c2w, base_fov, _cam_variant(plan), device, dtype)
        _sync(device)
        t0 = time.perf_counter()
        out = pipe.render(
            data["triangles"], tex, data["mask"], data["vn"], c2w, fov,
            resolution=resolution, torch_dtype=dtype, scene_key=plan.scene_key,
        )
        _sync(device)
        elapsed = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed)
        arr = _to_hwc(out.hdr_fused)
        hdrs.append(arr)
        iio.imwrite(frames_dir / f"frame_{i + 1:02d}.png", _tonemap(arr))
        records.append(
            {
                "frame": i + 1,
                "name": plan.name,
                "scene_key": plan.scene_key,
                "elapsed_ms": elapsed,
                "refreshed": out.refreshed,
                "reason": out.meta.get("refresh_reason"),
                "hole_ratio": out.meta.get("hole_ratio"),
                "skip_depth_ms": out.meta.get("skip_depth_ms"),
            }
        )
        print(
            f"  [L123] {plan.name}: {elapsed:.1f} ms refresh={out.refreshed} "
            f"reason={out.meta.get('refresh_reason')} hole={out.meta.get('hole_ratio')}"
        )

    total = sum(times) / 1000.0
    return {
        "mean_ms": float(np.mean(times)),
        "fps": len(times) / total if total else 0.0,
        "rf_calls": pipe.stats["rf_calls"],
        "refreshes": pipe.stats["refreshes"],
        "skips": pipe.stats["skips"],
        "adaptive_defers": pipe.stats.get("adaptive_defers", 0),
        "pipe_stats": dict(pipe.stats),
        "frames": records,
        "hdrs": hdrs,
    }


def _compare(cf, layered, rf, out_dir: Path, n: int, *, cached_cf: dict | None = None) -> dict:
    abs_l1_cf, psnr_cf = [], []
    abs_l1_rf, psnr_rf = [], []
    use_cached_cf = cached_cf is not None and cf.get("hdrs") is None
    if use_cached_cf:
        abs_l1_cf = list(
            cached_cf.get("per_frame_abs_l1_vs_cf")
            or cached_cf.get("per_frame_abs_l1")
            or []
        )
        psnr_cf = list(
            cached_cf.get("per_frame_psnr_vs_cf")
            or cached_cf.get("per_frame_psnr")
            or []
        )
        if cached_cf.get("cacheformer_mean_ms") is not None:
            cf = {**cf, "mean_ms": cached_cf["cacheformer_mean_ms"]}
    for i, b in enumerate(layered["hdrs"]):
        if not use_cached_cf:
            abs_l1_cf.append(float(np.mean(np.abs(b - cf["hdrs"][i]))))
            psnr_cf.append(_psnr(b, cf["hdrs"][i]))
        abs_l1_rf.append(float(np.mean(np.abs(b - rf["hdrs"][i]))))
        psnr_rf.append(_psnr(b, rf["hdrs"][i]))
    mean_abs_cf = float(np.mean(abs_l1_cf))
    mean_abs_rf = float(np.mean(abs_l1_rf))
    if cached_cf:
        mean_abs_cf = float(cached_cf.get("mean_abs_l1_vs_cf", mean_abs_cf))
        psnr_cf_mean = float(cached_cf.get("mean_psnr_vs_cf", float(np.mean(psnr_cf))))
        if cached_cf.get("cacheformer_mean_ms") is not None:
            cf = {**cf, "mean_ms": cached_cf["cacheformer_mean_ms"]}
    else:
        psnr_cf_mean = float(np.mean(psnr_cf))
    speedup_cf = cf["mean_ms"] / layered["mean_ms"] if layered["mean_ms"] > 0 else 0.0
    speedup_rf = rf["mean_ms"] / layered["mean_ms"] if layered["mean_ms"] > 0 else 0.0
    report = {
        "renderformer_mean_ms": rf["mean_ms"],
        "cacheformer_mean_ms": cf["mean_ms"],
        "layered_mean_ms": layered["mean_ms"],
        "speedup_vs_renderformer": speedup_rf,
        "speedup_vs_cacheformer": speedup_cf,
        "mean_abs_l1_vs_cf": mean_abs_cf,
        "mean_abs_l1_vs_rf": mean_abs_rf,
        "mean_psnr_vs_cf": psnr_cf_mean,
        "mean_psnr_vs_rf": float(np.mean(psnr_rf)),
        "per_frame_abs_l1_vs_cf": abs_l1_cf,
        "per_frame_abs_l1_vs_rf": abs_l1_rf,
        "per_frame_psnr_vs_cf": psnr_cf,
        "per_frame_psnr_vs_rf": psnr_rf,
        "pass_speed_vs_cf": bool(speedup_cf > 1.0),
        "pass_quality_vs_cf": bool(mean_abs_cf < 0.08),
        "pass_speed_vs_rf": bool(speedup_rf > 1.0),
        "pass_quality_vs_rf": bool(mean_abs_rf < 0.08),
        "pass_speed": bool(speedup_cf > 1.0),
        "pass_quality": bool(mean_abs_cf < 0.08),
        "renderformer": {k: v for k, v in rf.items() if k != "hdrs"},
        "cacheformer": {k: v for k, v in cf.items() if k != "hdrs"},
        "layered": {k: v for k, v in layered.items() if k != "hdrs"},
    }
    try:
        from PIL import Image, ImageDraw, ImageFont

        top = np.concatenate(
            [iio.imread(out_dir / "cacheformer" / f"frame_{i + 1:02d}.png") for i in range(n)],
            axis=1,
        )
        bot = np.concatenate(
            [iio.imread(out_dir / "layered" / f"frame_{i + 1:02d}.png") for i in range(n)],
            axis=1,
        )
        img = Image.fromarray(np.concatenate([top, bot], axis=0))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        draw.text(
            (8, 8),
            f"RF {rf['mean_ms']:.0f} | CF {cf['mean_ms']:.0f} | L123 {layered['mean_ms']:.0f} "
            f"({speedup_rf:.2f}x RF, {speedup_cf:.2f}x CF) | "
            f"absL1 cf={mean_abs_cf:.4f} rf={mean_abs_rf:.4f}",
            fill=(255, 255, 0),
            font=font,
        )
        img.save(out_dir / "contact_sheet.png")
    except Exception as exc:
        report["sheet_error"] = str(exc)
    return report


def _resolve_h5(scene: str, h5_file: str) -> str:
    if h5_file:
        return h5_file if Path(h5_file).is_file() else ""
    candidates = [
        SCENE_CATALOG.get(scene, ""),
        str(ROOT / "tmp" / "scenes" / scene / f"{scene}.h5"),
        {"cbox": str(ROOT / "tmp" / "cbox" / "cbox.h5")}.get(scene, ""),
    ]
    for path in candidates:
        if path and Path(path).is_file():
            return path
    return ""


def _make_pipe(rf) -> LayeredIndirectPipeline:
    return LayeredIndirectPipeline(
        rf,
        refresh_every=3,
        quality_anchor="rf",
        l1_direct_follow=False,
        l2_gate_strength=0.0,
        l2_head_mix=0.0,
        l3_adaptive=True,
        soft_rot_deg=20.0,
        max_skip_run=5,
        max_camera_rot_deg=28.0,
        max_hole_ratio=0.99,
        depth_mode="analytic",
        direct_mode="stub",
        view_follow="reproject",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", default="cbox", help="逗号分隔，见 SCENE_CATALOG；缺 h5 则跳过")
    parser.add_argument(
        "--dynamics",
        default="orbit_only,fov_sweep,roughness_orbit,specular_orbit,irradiance_orbit,roughness_fast,combined",
    )
    parser.add_argument("--h5_file", default="", help="覆盖单场景路径（此时 --scenes 只用第一个 id）")
    parser.add_argument("--model_id", default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--output_root", default="out/compare_layered_vs_cf")
    parser.add_argument(
        "--orbit_span_deg",
        type=float,
        default=60.0,
        help="绕 Y 总转角；缩小可避免把封闭场景转出画幅出现黑边",
    )
    parser.add_argument(
        "--fov_crop",
        type=float,
        default=1.0,
        help="FOV 乘子 <1 略拉近，裁掉三墙工作室外的黑边",
    )
    parser.add_argument(
        "--pipelines",
        default="renderformer,cacheformer,layered",
        help="逗号分隔：renderformer,cacheformer,layered；可只跑子集",
    )
    args = parser.parse_args()
    pipeline_set = {p.strip() for p in args.pipelines.split(",") if p.strip()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    print("Loading...", args.model_id, device)
    rf = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    rf.to(device)

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    dynamics = [d.strip() for d in args.dynamics.split(",") if d.strip()]
    summary = {"config": vars(args), "cases": {}, "all_pass": True}

    for scene in scenes:
        h5 = _resolve_h5(scene, args.h5_file)
        if not h5:
            print(f"[SKIP] {scene}: missing h5")
            continue
        data = load_h5(h5, device)
        data["c2w"], data["fov"] = _ensure_c2w_fov(data)

        for dyn in dynamics:
            plans = build_dynamic_plans(
                dyn,
                args.num_frames,
                orbit_span_deg=args.orbit_span_deg,
                fov_crop=args.fov_crop,
            )
            out_dir = Path(args.output_root) / scene / dyn
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "report.json"
            prev = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}

            if "renderformer" in pipeline_set:
                print(f"\n=== {scene}/{dyn} RenderFormer ===")
                rf_out = run_rf(rf, data, plans, device, dtype, args.resolution, out_dir)
            elif prev.get("renderformer_mean_ms") is not None:
                rf_out = {"mean_ms": prev["renderformer_mean_ms"], "hdrs": None, **prev.get("renderformer", {})}
            else:
                print(f"\n=== {scene}/{dyn} RenderFormer (required) ===")
                rf_out = run_rf(rf, data, plans, device, dtype, args.resolution, out_dir)

            if "cacheformer" in pipeline_set:
                print(f"\n=== {scene}/{dyn} CacheFormer ===")
                cf = run_cf(rf, data, plans, device, dtype, args.resolution, out_dir)
            elif prev.get("cacheformer_mean_ms") is not None:
                cf = {"mean_ms": prev["cacheformer_mean_ms"], "hdrs": None, **prev.get("cacheformer", {})}
            else:
                print(f"\n=== {scene}/{dyn} CacheFormer (required) ===")
                cf = run_cf(rf, data, plans, device, dtype, args.resolution, out_dir)

            if "layered" in pipeline_set:
                print(f"\n=== {scene}/{dyn} Layered L1–L3 ===")
                pipe = _make_pipe(rf).to(device)
                layered = run_layered(pipe, data, plans, device, dtype, args.resolution, out_dir)
            elif prev.get("layered_mean_ms") is not None:
                layered = {"mean_ms": prev["layered_mean_ms"], "hdrs": None, **prev.get("layered", {})}
            else:
                print(f"\n=== {scene}/{dyn} Layered L1–L3 (required) ===")
                pipe = _make_pipe(rf).to(device)
                layered = run_layered(pipe, data, plans, device, dtype, args.resolution, out_dir)

            cached_cf = (
                prev
                if ("cacheformer" not in pipeline_set and prev.get("mean_abs_l1_vs_cf") is not None)
                else None
            )
            if cf.get("hdrs") is None and cached_cf is None:
                print(f"\n=== {scene}/{dyn} CacheFormer (hdr reload) ===")
                cf = run_cf(rf, data, plans, device, dtype, args.resolution, out_dir)
            if layered.get("hdrs") is None:
                print(f"\n=== {scene}/{dyn} Layered L1–L3 (hdr reload) ===")
                pipe = _make_pipe(rf).to(device)
                layered = run_layered(pipe, data, plans, device, dtype, args.resolution, out_dir)
            if rf_out.get("hdrs") is None:
                print(f"\n=== {scene}/{dyn} RenderFormer (hdr reload) ===")
                rf_out = run_rf(rf, data, plans, device, dtype, args.resolution, out_dir)

            report = _compare(cf, layered, rf_out, out_dir, len(plans), cached_cf=cached_cf)
            key = f"{scene}/{dyn}"
            (out_dir / "report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            summary["cases"][key] = report
            ok = report["pass_speed_vs_cf"] and report["pass_quality_vs_cf"]
            summary["all_pass"] = summary["all_pass"] and ok
            print(
                f"  -> {key}: RF {report['renderformer_mean_ms']:.1f} | CF {report['cacheformer_mean_ms']:.1f} "
                f"| L123 {report['layered_mean_ms']:.1f} ms "
                f"({report['speedup_vs_renderformer']:.2f}x vs RF, {report['speedup_vs_cacheformer']:.2f}x vs CF) "
                f"absL1 cf={report['mean_abs_l1_vs_cf']:.4f} rf={report['mean_abs_l1_vs_rf']:.4f} pass={ok}"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    slim = {
        k: {sk: sv for sk, sv in v.items() if sk not in ("cacheformer", "layered", "per_frame_abs_l1", "per_frame_psnr")}
        for k, v in summary["cases"].items()
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# 三层栈 vs CacheFormer / RenderFormer（动态场景）",
        "",
        "| 用例 | RF ms | CF ms | L123 ms | vs RF | vs CF | absL1-CF | absL1-RF | PSNR-CF | 速度 | 质量 |",
        "|------|------:|------:|--------:|------:|------:|---------:|---------:|--------:|:----:|:----:|",
    ]
    for key, r in summary["cases"].items():
        lines.append(
            f"| {key} | {r['renderformer_mean_ms']:.1f} | {r['cacheformer_mean_ms']:.1f} | "
            f"{r['layered_mean_ms']:.1f} | {r['speedup_vs_renderformer']:.2f}x | "
            f"{r['speedup_vs_cacheformer']:.2f}x | {r['mean_abs_l1_vs_cf']:.4f} | "
            f"{r['mean_abs_l1_vs_rf']:.4f} | {r['mean_psnr_vs_cf']:.1f} | "
            f"{'PASS' if r['pass_speed_vs_cf'] else 'FAIL'} | "
            f"{'PASS' if r['pass_quality_vs_cf'] else 'FAIL'} |"
        )
    lines += ["", f"all_pass: {summary['all_pass']}"]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print("->", root / "SUMMARY.md")
    if not summary["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

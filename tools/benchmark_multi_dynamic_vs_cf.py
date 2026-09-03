#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多种动态场景帧序列：质量锁定 pruned vs CacheFormer。

动态类型：
  - orbit_only          仅相机 orbit（材质不变）
  - fov_sweep           FOV 缩放 + 小 orbit
  - roughness_orbit     周期性 roughness度 + orbit（经典）
  - specular_orbit      高光强度阶跃 + orbit
  - irradiance_orbit    自发光/辐照阶跃 + orbit
  - roughness_fast      每 2 帧改 roughness（高频材质变）
  - combined            roughness + FOV + orbit 同时变

用法:
  python tools/benchmark_multi_dynamic_vs_cf.py
  python tools/benchmark_multi_dynamic_vs_cf.py --scenes cbox,veach-mis --dynamics orbit_only,roughness_orbit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
from renderformer.c1.pruned_pipeline import PrunedIndirectPipeline

TEX_DIFFUSE = slice(0, 3)
TEX_SPECULAR = slice(3, 6)
TEX_ROUGHNESS = slice(6, 7)
TEX_IRRADIANCE = slice(10, 13)

SCENE_CATALOG = {
    "cbox": "tmp/c1_scenes/cbox.h5",
    "veach-mis": "tmp/c1_scenes/veach-mis.h5",
    "shader-ball": "tmp/c1_scenes/shader-ball.h5",
    "constant-width": "tmp/c1_scenes/constant-width.h5",
    "cbox-bunny": "tmp/c1_scenes/cbox-bunny.h5",
    "cbox-teapot": "tmp/c1_scenes/cbox-teapot.h5",
    "crystals": "tmp/c1_scenes/crystals.h5",
}


@dataclass
class FramePlan:
    name: str
    orbit_y_deg: float = 0.0
    fov_scale: float = 1.0
    roughness: Optional[float] = None
    specular_scale: Optional[float] = None
    irradiance_scale: Optional[float] = None
    scene_key: str = "s0"


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


def _apply_texture(base: torch.Tensor, plan: FramePlan) -> torch.Tensor:
    tex = base.clone()
    if plan.roughness is not None:
        tex[:, :, TEX_ROUGHNESS, :, :] = plan.roughness
    if plan.specular_scale is not None:
        tex[:, :, TEX_SPECULAR, :, :] = (
            base[:, :, TEX_SPECULAR, :, :] * float(plan.specular_scale)
        ).clamp(0.0, 1.0)
    if plan.irradiance_scale is not None:
        tex[:, :, TEX_IRRADIANCE, :, :] = (
            base[:, :, TEX_IRRADIANCE, :, :] * float(plan.irradiance_scale)
        ).clamp(min=0.0)
    return tex


def build_dynamic_plans(kind: str, num_frames: int) -> List[FramePlan]:
    plans: List[FramePlan] = []
    rough_seq = [0.15, 0.45, 0.75, 0.95]
    spec_seq = [0.4, 1.0, 1.6, 0.7]
    irr_seq = [0.5, 1.0, 1.8, 0.8]

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        orbit = -30.0 + 60.0 * t
        if kind == "orbit_only":
            plans.append(
                FramePlan(
                    name=f"f{i+1:02d}_orbit",
                    orbit_y_deg=orbit,
                    scene_key="static",
                )
            )
        elif kind == "fov_sweep":
            fov = 0.85 + 0.3 * t
            plans.append(
                FramePlan(
                    name=f"f{i+1:02d}_fov{fov:.2f}",
                    orbit_y_deg=orbit * 0.35,
                    fov_scale=fov,
                    scene_key="static",
                )
            )
        elif kind == "roughness_orbit":
            sid = i // 3
            r = rough_seq[sid % len(rough_seq)]
            plans.append(
                FramePlan(
                    name=f"f{i+1:02d}_r{r:.2f}",
                    orbit_y_deg=orbit,
                    roughness=r,
                    scene_key=f"rough_{r:.2f}",
                )
            )
        elif kind == "specular_orbit":
            sid = i // 3
            s = spec_seq[sid % len(spec_seq)]
            plans.append(
                FramePlan(
                    name=f"f{i+1:02d}_spec{s:.1f}",
                    orbit_y_deg=orbit,
                    specular_scale=s,
                    scene_key=f"spec_{s:.1f}",
                )
            )
        elif kind == "irradiance_orbit":
            sid = i // 3
            e = irr_seq[sid % len(irr_seq)]
            plans.append(
                FramePlan(
                    name=f"f{i+1:02d}_irr{e:.1f}",
                    orbit_y_deg=orbit,
                    irradiance_scale=e,
                    scene_key=f"irr_{e:.1f}",
                )
            )
        elif kind == "roughness_fast":
            sid = i // 2
            r = rough_seq[sid % len(rough_seq)]
            plans.append(
                FramePlan(
                    name=f"f{i+1:02d}_fast_r{r:.2f}",
                    orbit_y_deg=orbit,
                    roughness=r,
                    scene_key=f"rough_{r:.2f}",
                )
            )
        elif kind == "combined":
            sid = i // 3
            r = rough_seq[sid % len(rough_seq)]
            fov = 0.9 + 0.2 * ((i % 3) / 2.0)
            plans.append(
                FramePlan(
                    name=f"f{i+1:02d}_comb_r{r:.2f}_fov{fov:.2f}",
                    orbit_y_deg=orbit,
                    fov_scale=fov,
                    roughness=r,
                    scene_key=f"rough_{r:.2f}",
                )
            )
        else:
            raise ValueError(f"unknown dynamic kind: {kind}")
    return plans


def _cam_variant(plan: FramePlan) -> dict:
    return {
        "name": plan.name,
        "orbit_y_deg": plan.orbit_y_deg,
        "fov_scale": plan.fov_scale,
    }


@torch.no_grad()
def run_cf(
    rf,
    data,
    plans: List[FramePlan],
    device,
    dtype,
    resolution,
    out_dir: Path,
) -> Dict[str, Any]:
    cache = ViewIndependentCache(16)
    base_c2w, base_fov = data["c2w"], data["fov"]
    frames_dir = out_dir / "cacheformer"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # warmup
    tex0 = _apply_texture(data["texture"], plans[0])
    c2w, fov = _apply_camera_variant(base_c2w, base_fov, _cam_variant(plans[0]), device, dtype)
    _sync(device)
    rf.render(
        data["triangles"], tex0, data["mask"], data["vn"], c2w, fov,
        resolution=resolution, torch_dtype=dtype, vi_cache=cache, return_vi_cache_info=True,
    )
    cache.clear()

    times, hdrs, records = [], [], []
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
        ms = (time.perf_counter() - t0) * 1000.0
        times.append(ms)
        arr = _to_hwc(hdr)
        hdrs.append(arr)
        iio.imwrite(frames_dir / f"frame_{i+1:02d}.png", _tonemap(arr))
        records.append(
            {
                "frame": i + 1,
                "name": plan.name,
                "scene_key": plan.scene_key,
                "elapsed_ms": ms,
                "vi_cache_hit": bool(info.get("vi_cache_hit")),
            }
        )
        print(f"    [CF] {plan.name}: {ms:.1f} ms hit={info.get('vi_cache_hit')}")

    total = sum(times) / 1000.0
    return {
        "mean_ms": float(np.mean(times)),
        "fps": len(times) / total if total > 0 else 0.0,
        "rf_calls": len(plans),
        "cache_hits": sum(1 for r in records if r["vi_cache_hit"]),
        "frames": records,
        "hdrs": hdrs,
    }


@torch.no_grad()
def run_pruned(
    pipe: PrunedIndirectPipeline,
    data,
    plans: List[FramePlan],
    device,
    dtype,
    resolution,
    out_dir: Path,
) -> Dict[str, Any]:
    pipe.reset()
    base_c2w, base_fov = data["c2w"], data["fov"]
    frames_dir = out_dir / "pruned"
    frames_dir.mkdir(parents=True, exist_ok=True)
    times, hdrs, records = [], [], []

    for i, plan in enumerate(plans):
        tex = _apply_texture(data["texture"], plan)
        c2w, fov = _apply_camera_variant(base_c2w, base_fov, _cam_variant(plan), device, dtype)
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
            scene_key=plan.scene_key,
        )
        _sync(device)
        ms = (time.perf_counter() - t0) * 1000.0
        times.append(ms)
        arr = _to_hwc(out.hdr_fused)
        hdrs.append(arr)
        iio.imwrite(frames_dir / f"frame_{i+1:02d}.png", _tonemap(arr))
        records.append(
            {
                "frame": i + 1,
                "name": plan.name,
                "scene_key": plan.scene_key,
                "elapsed_ms": ms,
                "refreshed": out.refreshed,
                "reason": out.meta.get("refresh_reason"),
                "hole_ratio": out.meta.get("hole_ratio"),
            }
        )
        print(
            f"    [PR] {plan.name}: {ms:.1f} ms refresh={out.refreshed} "
            f"reason={out.meta.get('refresh_reason')}"
        )

    total = sum(times) / 1000.0
    return {
        "mean_ms": float(np.mean(times)),
        "fps": len(times) / total if total > 0 else 0.0,
        "rf_calls": pipe.stats["rf_calls"],
        "refreshes": pipe.stats["refreshes"],
        "skips": pipe.stats["skips"],
        "pipe_stats": dict(pipe.stats),
        "frames": records,
        "hdrs": hdrs,
    }


def _write_case_sheet(out_dir: Path, n: int, cf_mean: float, pr_mean: float, speedup: float, abs_l1: float, psnr: float):
    try:
        from PIL import Image, ImageDraw, ImageFont

        tops, bots = [], []
        for i in range(n):
            tops.append(iio.imread(out_dir / "cacheformer" / f"frame_{i+1:02d}.png"))
            bots.append(iio.imread(out_dir / "pruned" / f"frame_{i+1:02d}.png"))
        sheet = np.concatenate([np.concatenate(tops, axis=1), np.concatenate(bots, axis=1)], axis=0)
        img = Image.fromarray(sheet)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        draw.text(
            (8, 8),
            f"CF {cf_mean:.0f}ms | pruned {pr_mean:.0f}ms ({speedup:.2f}x) | "
            f"absL1={abs_l1:.4f} PSNR={psnr:.1f}dB",
            fill=(255, 255, 0),
            font=font,
        )
        img.save(out_dir / "contact_sheet.png")
    except Exception as e:
        print(f"  [WARN] sheet failed: {e}")


def run_one_case(
    rf,
    scene_name: str,
    h5_path: str,
    dynamic: str,
    args,
    device,
    dtype,
) -> Dict[str, Any]:
    print(f"\n{'='*60}\nSCENE={scene_name}  DYNAMIC={dynamic}\n{'='*60}")
    data = load_h5(h5_path, device)
    data["c2w"], data["fov"] = _ensure_c2w_fov(data)
    plans = build_dynamic_plans(dynamic, args.num_frames)
    case_dir = Path(args.output_dir) / scene_name / dynamic
    case_dir.mkdir(parents=True, exist_ok=True)

    print("--- CacheFormer ---")
    cf = run_cf(rf, data, plans, device, dtype, args.resolution, case_dir)

    print("--- Quality pruned ---")
    pipe = PrunedIndirectPipeline(
        rf,
        refresh_every=args.refresh_every,
        neural_res_scale=1.0,
        direct_res_scale=1.0,
        quality_lock=True,
        direct_mode="stub",
        view_follow="reproject",
        depth_mode=args.depth_mode,
        depth_aux_scale=args.depth_aux_scale,
        max_hole_ratio=args.max_hole_ratio,
        max_camera_rot_deg=args.max_camera_rot_deg,
        reproject_mode="inverse",
        guard_refreshes_after_scene_change=0,
        head=None,
    ).to(device)
    pr = run_pruned(pipe, data, plans, device, dtype, args.resolution, case_dir)

    abs_l1, psnrs = [], []
    for a, b in zip(pr["hdrs"], cf["hdrs"]):
        abs_l1.append(float(np.mean(np.abs(a - b))))
        psnrs.append(_psnr(a, b))
    mean_abs = float(np.mean(abs_l1))
    mean_psnr = float(np.mean(psnrs))
    speedup = cf["mean_ms"] / pr["mean_ms"] if pr["mean_ms"] > 0 else 0.0

    _write_case_sheet(case_dir, len(plans), cf["mean_ms"], pr["mean_ms"], speedup, mean_abs, mean_psnr)

    cf_out = {k: v for k, v in cf.items() if k != "hdrs"}
    pr_out = {k: v for k, v in pr.items() if k != "hdrs"}
    result = {
        "scene": scene_name,
        "dynamic": dynamic,
        "h5": h5_path,
        "cacheformer": cf_out,
        "pruned": pr_out,
        "speedup_vs_cf": speedup,
        "mean_abs_l1_vs_cf": mean_abs,
        "mean_psnr_vs_cf": mean_psnr,
        "per_frame_abs_l1": abs_l1,
        "per_frame_psnr": psnrs,
        "pass_speed": bool(speedup > 1.0),
        "pass_quality_soft": bool(mean_abs < 0.08),
    }
    with open(case_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f">>> {scene_name}/{dynamic}: CF={cf['mean_ms']:.1f}ms  "
        f"PR={pr['mean_ms']:.1f}ms  {speedup:.2f}x  "
        f"absL1={mean_abs:.4f}  PSNR={mean_psnr:.1f}dB  "
        f"rf={pr['rf_calls']}/{cf['rf_calls']}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Multi dynamic scenes vs CacheFormer")
    parser.add_argument(
        "--scenes",
        type=str,
        default="cbox,veach-mis,shader-ball,cbox-bunny,crystals",
        help="逗号分隔场景 id（见 SCENE_CATALOG）",
    )
    parser.add_argument(
        "--dynamics",
        type=str,
        default="orbit_only,fov_sweep,roughness_orbit,specular_orbit,irradiance_orbit,roughness_fast,combined",
        help="逗号分隔动态类型",
    )
    parser.add_argument("--model_id", default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--refresh_every", type=int, default=3)
    parser.add_argument("--depth_mode", default="analytic", choices=["analytic", "raycast"])
    parser.add_argument("--depth_aux_scale", type=float, default=0.5)
    parser.add_argument("--max_hole_ratio", type=float, default=0.35)
    parser.add_argument("--max_camera_rot_deg", type=float, default=28.0)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--output_dir", default="out/compare_multi_dynamic_vs_cf")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    dynamics = [d.strip() for d in args.dynamics.split(",") if d.strip()]

    print(f"Loading RF {args.model_id}")
    rf = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    rf.to(device)

    cases: List[Dict[str, Any]] = []
    for scene in scenes:
        h5 = SCENE_CATALOG.get(scene)
        if h5 is None or not Path(h5).is_file():
            print(f"[SKIP] scene {scene}: missing {h5}")
            continue
        for dyn in dynamics:
            try:
                cases.append(run_one_case(rf, scene, h5, dyn, args, device, dtype))
            except Exception as e:
                print(f"[FAIL] {scene}/{dyn}: {e}")
                cases.append(
                    {
                        "scene": scene,
                        "dynamic": dyn,
                        "error": str(e),
                        "speedup_vs_cf": None,
                        "pass_speed": False,
                        "pass_quality_soft": False,
                    }
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # summary table
    summary = {
        "config": vars(args),
        "cases": cases,
        "aggregates": {
            "num_cases": len(cases),
            "num_pass_speed": sum(1 for c in cases if c.get("pass_speed")),
            "num_pass_quality": sum(1 for c in cases if c.get("pass_quality_soft")),
            "mean_speedup": float(
                np.mean([c["speedup_vs_cf"] for c in cases if c.get("speedup_vs_cf")])
            )
            if any(c.get("speedup_vs_cf") for c in cases)
            else None,
        },
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # markdown + csv
    lines = [
        "# 多动态场景 vs CacheFormer",
        "",
        "| 场景 | 动态类型 | CF ms | pruned ms | vs CF | absL1 | PSNR | RF调用 | 速度过 | 质量过 |",
        "|------|----------|------:|----------:|------:|------:|-----:|-------:|:------:|:------:|",
    ]
    csv_rows = ["scene,dynamic,cf_ms,pruned_ms,speedup,abs_l1,psnr,rf_pruned,rf_cf,pass_speed,pass_quality"]
    for c in cases:
        if c.get("error"):
            lines.append(f"| {c['scene']} | {c['dynamic']} | ERR | - | - | - | - | - | FAIL | FAIL |")
            continue
        cf_ms = c["cacheformer"]["mean_ms"]
        pr_ms = c["pruned"]["mean_ms"]
        lines.append(
            f"| {c['scene']} | {c['dynamic']} | {cf_ms:.1f} | {pr_ms:.1f} | "
            f"{c['speedup_vs_cf']:.2f}x | {c['mean_abs_l1_vs_cf']:.4f} | "
            f"{c['mean_psnr_vs_cf']:.1f} | {c['pruned']['rf_calls']}/{c['cacheformer']['rf_calls']} | "
            f"{'PASS' if c['pass_speed'] else 'FAIL'} | {'PASS' if c['pass_quality_soft'] else 'FAIL'} |"
        )
        csv_rows.append(
            f"{c['scene']},{c['dynamic']},{cf_ms:.3f},{pr_ms:.3f},{c['speedup_vs_cf']:.4f},"
            f"{c['mean_abs_l1_vs_cf']:.6f},{c['mean_psnr_vs_cf']:.3f},"
            f"{c['pruned']['rf_calls']},{c['cacheformer']['rf_calls']},"
            f"{int(c['pass_speed'])},{int(c['pass_quality_soft'])}"
        )

    agg = summary["aggregates"]
    lines += [
        "",
        f"- 用例数: {agg['num_cases']}",
        f"- 速度通过 (>1×CF): {agg['num_pass_speed']}",
        f"- 质量通过 (absL1<0.08): {agg['num_pass_quality']}",
        f"- 平均加速比: {agg['mean_speedup']:.3f}x" if agg["mean_speedup"] else "- 平均加速比: n/a",
        "",
        "每用例分帧与 contact sheet 见对应子目录。",
    ]
    (out_root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    (out_root / "summary.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nSummary -> {out_root / 'SUMMARY.md'}")
    print(f"CSV      -> {out_root / 'summary.csv'}")
    print(f"JSON     -> {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()

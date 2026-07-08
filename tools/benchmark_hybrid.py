#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径 B 消融 benchmark：RF-only vs Hybrid（B0–B4 子集）。

用法:
  python tools/benchmark_hybrid.py --h5_file tmp/cbox/cbox.h5 --resolution 256 --num_views 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmark_vi_cache import _apply_camera_variant, _get_camera_variants, load_h5
from renderformer import RenderFormerRenderingPipeline, ViewIndependentCache
from renderformer.hybrid.pipeline import HybridFusionPipeline
from renderformer.hybrid.profile import HybridProfile


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_rf_only(pipeline, data, variants, n, device, dtype, resolution, vi_cache=None) -> Dict:
    times: List[float] = []
    violations_energy = []
    hits = 0
    for i in range(n):
        c2w, fov = _apply_camera_variant(
            data["c2w"].clone(),
            data["fov"].clone(),
            variants[i % len(variants)],
            device,
            dtype,
        )
        _sync(device)
        t0 = time.perf_counter()
        out, info = pipeline.render(
            data["triangles"],
            data["texture"],
            data["mask"],
            data["vn"],
            c2w,
            fov,
            resolution=resolution,
            torch_dtype=dtype,
            vi_cache=vi_cache,
            return_vi_cache_info=True,
        )
        _sync(device)
        times.append(time.perf_counter() - t0)
        if info.get("vi_cache_hit"):
            hits += 1
        violations_energy.append((out - out.clamp(min=0)).abs().mean().item())
    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "fps": len(times) / sum(times),
        "vi_cache_hits": hits,
        "viol_energy_proxy": sum(violations_energy) / len(violations_energy),
    }


def run_hybrid(hybrid, data, variants, n, device, dtype, resolution, profile: HybridProfile, vi_cache=None) -> Dict:
    times: List[float] = []
    violations = []
    hits = 0
    for i in range(n):
        c2w, fov = _apply_camera_variant(
            data["c2w"].clone(),
            data["fov"].clone(),
            variants[i % len(variants)],
            device,
            dtype,
        )
        _sync(device)
        t0 = time.perf_counter()
        ctx = hybrid.render(
            data["triangles"],
            data["texture"],
            data["mask"],
            data["vn"],
            c2w,
            fov,
            resolution=resolution,
            torch_dtype=dtype,
            vi_cache=vi_cache,
            use_physics_correct=profile.use_physics_correct,
        )
        _sync(device)
        times.append(time.perf_counter() - t0)
        if ctx.meta.get("vi_cache_hit"):
            hits += 1
        violations.append(ctx.violations.get("viol_energy", 0.0))
    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "fps": len(times) / sum(times),
        "vi_cache_hits": hits,
        "viol_energy": sum(violations) / len(violations),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.float16
        if args.precision == "fp16"
        else torch.bfloat16
        if args.precision == "bf16"
        else torch.float32
    )

    data = load_h5(args.h5_file, device)
    variants = _get_camera_variants()

    rf = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    rf.to(device)

    results = {}
    results["B0_rf_only"] = run_rf_only(rf, data, variants, args.num_views, device, dtype, args.resolution)

    profile_b1 = HybridProfile(fixed_alpha=1.0, use_physics_correct=False)
    hybrid_b1 = HybridFusionPipeline(rf, profile_b1)
    results["B1_hybrid_fixed_alpha"] = run_hybrid(
        hybrid_b1, data, variants, args.num_views, device, dtype, args.resolution, profile_b1
    )

    profile_b2 = HybridProfile(use_physics_correct=False)
    hybrid_b2 = HybridFusionPipeline(rf, profile_b2)
    results["B2_hybrid_confidence"] = run_hybrid(
        hybrid_b2, data, variants, args.num_views, device, dtype, args.resolution, profile_b2
    )

    profile_b4 = HybridProfile(use_physics_correct=True)
    vi_cache = ViewIndependentCache(max_entries=4)
    hybrid_b4 = HybridFusionPipeline(rf, profile_b4)
    results["B4_hybrid_full_vi_cache"] = run_hybrid(
        hybrid_b4, data, variants, args.num_views, device, dtype, args.resolution, profile_b4, vi_cache
    )

    print(json.dumps(results, indent=2))
    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

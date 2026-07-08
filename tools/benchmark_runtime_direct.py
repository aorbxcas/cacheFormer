#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime Direct 后端性能对比：nvdiffrast vs lite。

用法:
  python tools/benchmark_runtime_direct.py --h5_file tmp/cbox/cbox.h5
  python tools/benchmark_runtime_direct.py --h5_file tmp/cbox/cbox.h5 --resolution 512 --warmup 2 --iters 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from renderformer.hybrid.data_loader import add_batch_dim, load_h5_scene
from renderformer.hybrid.runtime_direct import create_runtime_direct_renderer, nvdiffrast_available


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_backend(
    backend: str,
    batch: dict,
    resolution: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict:
    renderer = create_runtime_direct_renderer(backend=backend)
    name = type(renderer).__name__

    kwargs = dict(
        triangles=batch["triangles"],
        texture=batch["texture"],
        vn=batch["vn"],
        mask=batch["mask"],
        c2w=batch["c2w"],
        fov=batch["fov"],
        resolution=resolution,
    )

    for _ in range(warmup):
        _sync(device)
        renderer.render(**kwargs)
        _sync(device)

    times = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        hdr, depth = renderer.render(**kwargs)
        _sync(device)
        times.append(time.perf_counter() - t0)

    hdr0 = hdr[0, 0]
    valid = depth[0, 0] > 0
    return {
        "backend": backend,
        "class": name,
        "resolution": resolution,
        "mean_ms": sum(times) / len(times) * 1000.0,
        "min_ms": min(times) * 1000.0,
        "max_ms": max(times) * 1000.0,
        "fps": len(times) / sum(times),
        "hdr_mean": float(hdr0.mean().item()),
        "hdr_max": float(hdr0.max().item()),
        "valid_pixel_frac": float(valid.float().mean().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark runtime direct backends")
    parser.add_argument("--h5_file", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--skip_lite", action="store_true", help="仅测 nvdiffrast")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("需要 CUDA 才能测试 nvdiffrast")
        return

    device = torch.device("cuda")
    data = load_h5_scene(args.h5_file)
    batch = add_batch_dim(data, device)

    results = {
        "nvdiffrast_available": nvdiffrast_available(),
        "device": torch.cuda.get_device_name(0),
        "h5_file": args.h5_file,
    }

    backends = ["nvdiffrast"]
    if not args.skip_lite:
        backends.append("lite")

    for b in backends:
        if b == "nvdiffrast" and not nvdiffrast_available():
            print(f"跳过 {b}: 未安装")
            continue
        print(f"Benchmarking {b} @ {args.resolution} ...")
        try:
            results[b] = bench_backend(
                b, batch, args.resolution, args.warmup, args.iters, device
            )
            r = results[b]
            print(
                f"  {r['class']}: mean={r['mean_ms']:.2f} ms, "
                f"min={r['min_ms']:.2f} ms, fps={r['fps']:.1f}, "
                f"valid={r['valid_pixel_frac']:.2%}"
            )
        except Exception as e:
            results[b] = {"error": str(e)}
            print(f"  {b} FAILED: {e}")

    if "nvdiffrast" in results and "lite" in results:
        if "mean_ms" in results["nvdiffrast"] and "mean_ms" in results["lite"]:
            speedup = results["lite"]["mean_ms"] / results["nvdiffrast"]["mean_ms"]
            results["speedup_nvd_over_lite"] = speedup
            print(f"加速比 (lite/nvd): {speedup:.1f}x")

    print(json.dumps(results, indent=2))
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

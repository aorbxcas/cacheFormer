#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0 验收：可视化 Runtime Direct vs RF direct 分量对比。

用法:
  python tools/vis_direct.py --h5_file tmp/cbox/cbox.h5 --resolution 256
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import imageio
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from renderformer.hybrid.data_loader import add_batch_dim, load_h5_scene
from renderformer.hybrid.runtime_direct import RuntimeDirectRenderer, nvdiffrast_available


def main():
    parser = argparse.ArgumentParser(description="Visualize runtime direct pass")
    parser.add_argument("--h5_file", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "nvdiffrast", "lite"],
        default="auto",
        help="direct 渲染后端（默认 auto：有 nvdiffrast 则用 GPU 光栅）",
    )
    parser.add_argument("--benchmark", action="store_true", help="打印单帧耗时")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backend in ("auto", "nvdiffrast") and device.type != "cuda":
        print("警告: nvdiffrast 需要 CUDA，将回退 lite")
    data = load_h5_scene(args.h5_file)
    batch = add_batch_dim(data, device)

    renderer = RuntimeDirectRenderer(backend=args.backend)
    print(f"Runtime Direct backend: {renderer.active_backend} (nvdiffrast 可用: {nvdiffrast_available()})")

    if args.benchmark and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    hdr_direct, depth = renderer.render(
        batch["triangles"],
        batch["texture"],
        batch["vn"],
        batch["mask"],
        batch["c2w"],
        batch["fov"],
        resolution=args.resolution,
    )
    if args.benchmark and device.type == "cuda":
        torch.cuda.synchronize()
    if args.benchmark:
        ms = (time.perf_counter() - t0) * 1000.0
        valid = (depth[0, 0] > 0).float().mean().item()
        print(f"Render: {ms:.2f} ms @ {args.resolution}x{args.resolution}, valid_pixels={valid:.2%}")

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.h5_file), "direct_vis")
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.h5_file))[0]

    nv = hdr_direct.shape[1]
    for i in range(nv):
        hdr = hdr_direct[0, i].detach().cpu().numpy().astype(np.float32)
        exr = os.path.join(output_dir, f"{base}_view_{i}_direct.exr")
        imageio.v3.imwrite(exr, hdr)

        # 用 exposure tonemap，避免高亮墙面把盒子压成纯黑
        scale = float(np.percentile(hdr[hdr > 0], 95.0)) if (hdr > 0).any() else 1.0
        tonemap = 1.0 - np.exp(-hdr / (scale * 0.6 + 1e-6))
        png = os.path.join(output_dir, f"{base}_view_{i}_direct.png")
        imageio.v3.imwrite(png, (np.clip(tonemap, 0, 1) ** (1.0 / 2.2) * 255).astype(np.uint8))

        d = depth[0, i, ..., 0].detach().cpu().numpy()
        d_vis = d / (np.percentile(d[d > 0], 95) + 1e-6) if (d > 0).any() else d
        imageio.v3.imwrite(
            os.path.join(output_dir, f"{base}_view_{i}_depth.png"),
            (np.clip(d_vis, 0, 1) * 255).astype(np.uint8),
        )
        print(f"Saved {exr} and {png}")


if __name__ == "__main__":
    main()

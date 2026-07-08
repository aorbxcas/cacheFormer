#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径 B 主入口：Runtime Direct + RenderFormer + Confidence Fusion。

用法:
  python tools/infer_hybrid_b.py --h5_file tmp/cbox/cbox.h5
  python tools/infer_hybrid_b.py --h5_file tmp/cbox/cbox.h5 --profile hybrid_profiles/cbox.example.json --vi_cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import imageio
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from renderformer import ViewIndependentCache
from renderformer.hybrid.data_loader import add_batch_dim, load_h5_scene
from renderformer.hybrid.pipeline import HybridFusionPipeline
from renderformer.hybrid.profile import HybridProfile, load_hybrid_profile


def _save_outputs(ctx, output_dir: str, base_name: str, tone_mapper=None):
    os.makedirs(output_dir, exist_ok=True)
    nv = ctx.hdr_fused.shape[1]
    meta_path = os.path.join(output_dir, f"{base_name}_hybrid_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "violations": ctx.violations,
                "meta": {k: v for k, v in ctx.meta.items() if isinstance(v, (int, float, str, bool, type(None)))},
            },
            f,
            indent=2,
        )

    for i in range(nv):
        for tag, tensor in [
            ("fused", ctx.hdr_fused),
            ("neural", ctx.hdr_neural),
            ("direct", ctx.hdr_direct),
            ("indirect", ctx.indirect_neural),
        ]:
            hdr = tensor[0, i].detach().cpu().numpy().astype(np.float32)
            exr_path = os.path.join(output_dir, f"{base_name}_view_{i}_{tag}.exr")
            imageio.v3.imwrite(exr_path, hdr)

        alpha = ctx.alpha[0, i].detach().cpu().numpy()
        if alpha.shape[-1] == 1:
            alpha_vis = np.repeat(alpha, 3, axis=-1)
        else:
            alpha_vis = alpha
        alpha_vis = np.clip(alpha_vis, 0, 1)
        imageio.v3.imwrite(
            os.path.join(output_dir, f"{base_name}_view_{i}_alpha.png"),
            (alpha_vis * 255).astype(np.uint8),
        )

        depth = ctx.depth[0, i].detach().cpu().numpy()
        d = depth[..., 0]
        d_norm = d / (np.percentile(d[d > 0], 95) + 1e-6) if (d > 0).any() else d
        imageio.v3.imwrite(
            os.path.join(output_dir, f"{base_name}_view_{i}_depth.png"),
            (np.clip(d_norm, 0, 1) * 255).astype(np.uint8),
        )

        fused_hdr = ctx.hdr_fused[0, i].detach().cpu().numpy().astype(np.float32)
        if tone_mapper is not None:
            ldr = tone_mapper.hdr_to_ldr(fused_hdr)
        else:
            ldr = np.clip(fused_hdr, 0, 1)
        imageio.v3.imwrite(
            os.path.join(output_dir, f"{base_name}_view_{i}_fused.png"),
            (ldr * 255).astype(np.uint8),
        )
        print(f"Saved view {i} outputs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Path B hybrid inference")
    parser.add_argument("--h5_file", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--profile", type=str, default=None, help="hybrid_profile.json")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--vi_cache", action="store_true")
    parser.add_argument("--auto_align", action="store_true")
    parser.add_argument("--parallel_direct", action="store_true")
    parser.add_argument("--fixed_alpha", type=float, default=None, help="消融：固定 α")
    parser.add_argument("--no_physics", action="store_true")
    parser.add_argument("--tone_mapper", type=str, choices=["none", "agx", "filmic", "pbr_neutral"], default="agx")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    profile = load_hybrid_profile(args.profile) if args.profile else HybridProfile()
    if args.fixed_alpha is not None:
        profile.fixed_alpha = args.fixed_alpha

    pipeline = HybridFusionPipeline.from_pretrained(args.model_id, profile=profile)
    if device.type == "cuda" and os.name == "posix":
        try:
            from renderformer_liger_kernel import apply_kernels

            apply_kernels(pipeline.rf_pipeline.model)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except ImportError:
            pass
    elif device.type == "mps":
        args.precision = "fp32"
    pipeline.to(device)

    tone_mapper = None
    if args.tone_mapper != "none":
        from simple_ocio import ToneMapper

        tm_name = "Khronos PBR Neutral" if args.tone_mapper == "pbr_neutral" else args.tone_mapper
        tone_mapper = ToneMapper(tm_name)

    data = load_h5_scene(args.h5_file)
    batch = add_batch_dim(data, device)
    dtype = (
        torch.float16
        if args.precision == "fp16"
        else torch.bfloat16
        if args.precision == "bf16"
        else torch.float32
    )

    vi_cache = ViewIndependentCache(max_entries=4) if args.vi_cache else None

    ctx = pipeline.render(
        triangles=batch["triangles"],
        texture=batch["texture"],
        mask=batch["mask"],
        vn=batch["vn"],
        c2w=batch["c2w"],
        fov=batch["fov"],
        resolution=args.resolution,
        torch_dtype=dtype,
        vi_cache=vi_cache,
        auto_align=args.auto_align,
        use_physics_correct=not args.no_physics,
        parallel_direct=args.parallel_direct,
    )

    print("Hybrid render done.")
    print("Violations:", ctx.violations)
    print("Meta:", ctx.meta)

    output_dir = args.output_dir or os.path.dirname(args.h5_file) or "output/hybrid"
    base_name = os.path.splitext(os.path.basename(args.h5_file))[0]
    _save_outputs(ctx, output_dir, base_name, tone_mapper)


if __name__ == "__main__":
    main()

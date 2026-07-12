#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C1 推理：Direct + RF + ResidualIndirectHead。

用法:
  python tools/c1_infer.py --h5_file tmp/cbox/cbox.h5 --checkpoint checkpoints/c1/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renderformer import RenderFormerRenderingPipeline, ViewIndependentCache
from renderformer.c1.pipeline import C1ResidualPipeline
from renderformer.c1.residual_head import ResidualIndirectHead
from renderformer.hybrid.data_loader import add_batch_dim, load_h5_scene


def _tonemap(hdr: np.ndarray) -> np.ndarray:
    x = np.clip(hdr, 0, None)
    scale = float(np.percentile(x[x > 0], 95)) if (x > 0).any() else 1.0
    ldr = 1.0 - np.exp(-x / (scale * 0.6 + 1e-6))
    return (np.clip(ldr, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def _load_head(ckpt_path: str, device: torch.device) -> ResidualIndirectHead:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("head_cfg") or {}
    head = ResidualIndirectHead(
        use_neural=bool(cfg.get("use_neural", True)),
        use_depth=bool(cfg.get("use_depth", True)),
        base_channels=int(cfg.get("base_channels", 32)),
        num_blocks=int(cfg.get("num_blocks", 3)),
    )
    head.load_state_dict(ckpt["model"])
    head.to(device).eval()
    return head


def main():
    parser = argparse.ArgumentParser(description="C1 residual inference")
    parser.add_argument("--h5_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--vi_cache", action="store_true")
    parser.add_argument("--confidence", action="store_true", help="使用 Hybrid ConfidenceMap 作为 α")
    parser.add_argument("--auto_align", action="store_true", help="Direct 相对 Neural 自动 scale")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    out_dir = Path(args.output_dir or f"output/c1/{Path(args.h5_file).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rf = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    rf.to(device)
    head = _load_head(args.checkpoint, device)
    pipe = C1ResidualPipeline(
        rf,
        head,
        alpha=args.alpha,
        use_confidence=args.confidence,
        auto_align=args.auto_align,
    ).to(device)

    data = add_batch_dim(load_h5_scene(args.h5_file), device)
    cache = ViewIndependentCache(max_entries=4) if args.vi_cache else None

    ctx = pipe.render(
        triangles=data["triangles"],
        texture=data["texture"],
        mask=data["mask"],
        vn=data["vn"],
        c2w=data["c2w"] if data["c2w"].dim() == 4 else data["c2w"].unsqueeze(1),
        fov=data["fov"] if data["fov"].dim() == 3 else data["fov"].unsqueeze(-1),
        resolution=args.resolution,
        torch_dtype=dtype,
        vi_cache=cache,
    )

    base = Path(args.h5_file).stem
    for tag, tensor in [
        ("fused", ctx.hdr_fused),
        ("direct", ctx.hdr_direct),
        ("neural", ctx.hdr_neural),
        ("indirect", ctx.indirect_pred),
    ]:
        arr = tensor[0, 0].detach().cpu().numpy().astype(np.float32)
        iio.imwrite(out_dir / f"{base}_{tag}.exr", arr)
        iio.imwrite(out_dir / f"{base}_{tag}.png", _tonemap(arr))

    a = ctx.alpha[0, 0].detach().cpu().numpy()
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    iio.imwrite(out_dir / f"{base}_alpha.png", (np.clip(a, 0, 1) * 255).astype(np.uint8))

    with open(out_dir / f"{base}_meta.json", "w", encoding="utf-8") as f:
        json.dump(ctx.meta, f, indent=2)
    print(f"Saved to {out_dir}")
    print("meta:", ctx.meta)


if __name__ == "__main__":
    main()

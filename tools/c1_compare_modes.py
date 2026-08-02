#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C1 vs Hybrid 分解 vs Direct-only 三路对比（同 H5）。

用法:
  python tools/c1_compare_modes.py --h5_file tmp/c1_scenes/cbox.h5 \\
      --checkpoint checkpoints/c1/best.pt --output_dir output/c1/compare
"""

from __future__ import annotations

import argparse
import json
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
from renderformer.hybrid.pipeline import HybridFusionPipeline
from renderformer.hybrid.profile import HybridProfile


def _tonemap(hdr: np.ndarray) -> np.ndarray:
    x = np.clip(hdr, 0, None)
    scale = float(np.percentile(x[x > 0], 95)) if (x > 0).any() else 1.0
    ldr = 1.0 - np.exp(-x / (scale * 0.6 + 1e-6))
    return (np.clip(ldr, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def _load_head(path: str, device: torch.device) -> ResidualIndirectHead:
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


def _save(tag: str, tensor: torch.Tensor, out_dir: Path, base: str):
    arr = tensor[0, 0].detach().float().cpu().numpy()
    iio.imwrite(out_dir / f"{base}_{tag}.exr", arr.astype(np.float32))
    iio.imwrite(out_dir / f"{base}_{tag}.png", _tonemap(arr))


def main():
    parser = argparse.ArgumentParser(description="Compare Direct / Hybrid / C1")
    parser.add_argument("--h5_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--output_dir", type=str, default="output/c1/compare")
    parser.add_argument("--vi_cache", action="store_true")
    parser.add_argument("--confidence", action="store_true")
    parser.add_argument("--auto_align", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.h5_file).stem

    rf = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    rf.to(device)
    data = add_batch_dim(load_h5_scene(args.h5_file), device)
    c2w = data["c2w"] if data["c2w"].dim() == 4 else data["c2w"].unsqueeze(1)
    fov = data["fov"] if data["fov"].dim() == 3 else data["fov"].unsqueeze(-1)
    cache = ViewIndependentCache(4) if args.vi_cache else None

    # --- Hybrid (classic decompose) ---
    hybrid = HybridFusionPipeline(rf, profile=HybridProfile()).to(device)
    ctx_h = hybrid.render(
        triangles=data["triangles"],
        texture=data["texture"],
        mask=data["mask"],
        vn=data["vn"],
        c2w=c2w,
        fov=fov,
        resolution=args.resolution,
        torch_dtype=dtype,
        vi_cache=cache,
        auto_align=args.auto_align,
    )
    _save("hybrid_fused", ctx_h.hdr_fused, out_dir, base)
    _save("hybrid_indirect", ctx_h.indirect_neural, out_dir, base)
    _save("direct", ctx_h.hdr_direct, out_dir, base)
    _save("neural", ctx_h.hdr_neural, out_dir, base)

    # --- C1 ---
    if cache is not None:
        cache.clear()
    head = _load_head(args.checkpoint, device)
    c1 = C1ResidualPipeline(
        rf,
        head,
        alpha=1.0,
        use_confidence=args.confidence,
        auto_align=args.auto_align,
    ).to(device)
    ctx_c = c1.render(
        triangles=data["triangles"],
        texture=data["texture"],
        mask=data["mask"],
        vn=data["vn"],
        c2w=c2w,
        fov=fov,
        resolution=args.resolution,
        torch_dtype=dtype,
        vi_cache=cache,
        hdr_neural=ctx_h.hdr_neural,  # 复用，避免重跑 RF
    )
    _save("c1_fused", ctx_c.hdr_fused, out_dir, base)
    _save("c1_indirect", ctx_c.indirect_pred, out_dir, base)
    if ctx_c.alpha is not None:
        a = ctx_c.alpha[0, 0].detach().cpu().numpy()
        if a.shape[-1] == 1:
            a = np.repeat(a, 3, axis=-1)
        iio.imwrite(out_dir / f"{base}_c1_alpha.png", (np.clip(a, 0, 1) * 255).astype(np.uint8))

    # 简易数值对比（相对 neural）
    def rel(a, b):
        a = a[0, 0].float()
        b = b[0, 0].float()
        return float(torch.mean(torch.abs(a - b) / (torch.abs(b) + 1e-3)).item())

    report = {
        "h5": args.h5_file,
        "hybrid_meta": {k: v for k, v in ctx_h.meta.items() if isinstance(v, (int, float, str, bool))},
        "c1_meta": {k: v for k, v in ctx_c.meta.items() if isinstance(v, (int, float, str, bool, type(None)))},
        "rel_l1_to_neural": {
            "direct": rel(ctx_h.hdr_direct, ctx_h.hdr_neural),
            "hybrid_fused": rel(ctx_h.hdr_fused, ctx_h.hdr_neural),
            "c1_fused": rel(ctx_c.hdr_fused, ctx_h.hdr_neural),
        },
        "rel_l1_indirect_c1_vs_hybrid_decomp": rel(ctx_c.indirect_pred, ctx_h.indirect_neural),
    }
    with open(out_dir / f"{base}_compare_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report["rel_l1_to_neural"], indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

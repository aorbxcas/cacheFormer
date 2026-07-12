#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C1 验证：在 val/train 集上报告间接光与合成图误差；可选导出对比图。

用法:
  python tools/c1_eval.py --data_dir data/c1/bootstrap --checkpoint checkpoints/c1/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renderformer.c1.dataset import C1ResidualDataset, c1_collate
from renderformer.c1.residual_head import ResidualIndirectHead, fuse_direct_indirect


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0:
        return 99.0
    peak = max(torch.quantile(target.reshape(-1), 0.99).item(), 1e-3)
    return 10.0 * float(np.log10((peak ** 2) / mse))


def _rel_l1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> float:
    return float(torch.mean(torch.abs(pred - target) / (torch.abs(target) + eps)).item())


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate C1 residual head")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "all"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--preview_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = "train" if args.split == "all" else args.split
    ds = C1ResidualDataset(args.data_dir, split=split)
    if args.split == "all":
        # 手动用 all 列表
        man = json.loads((Path(args.data_dir) / "manifest.json").read_text(encoding="utf-8"))
        ds.entries = [
            Path(args.data_dir) / e if not Path(e).is_absolute() else Path(e)
            for e in (man.get("all") or [])
        ]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=c1_collate)
    head = _load_head(args.checkpoint, device)

    metrics = {
        "indirect_rel_l1": [],
        "indirect_abs_l1": [],
        "indirect_psnr": [],
        "compose_rel_l1": [],
        "compose_abs_l1": [],
        "compose_psnr": [],
        "baseline_decompose_rel_l1": [],
        "baseline_decompose_abs_l1": [],
        "direct_only_compose_rel_l1": [],
        "direct_only_compose_abs_l1": [],
    }

    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        idx = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            i_pred = head(batch["hdr_direct"], batch["hdr_neural"], batch["depth"])
            i_tgt = batch["indirect_target"]
            fused = fuse_direct_indirect(batch["hdr_direct"], i_pred, 1.0)
            decomp = torch.relu(batch["hdr_neural"] - batch["hdr_direct"])
            direct_as_full = batch["hdr_direct"]

            metrics["indirect_rel_l1"].append(_rel_l1(i_pred, i_tgt))
            metrics["indirect_abs_l1"].append(float(torch.mean(torch.abs(i_pred - i_tgt)).item()))
            metrics["indirect_psnr"].append(_psnr(i_pred, i_tgt))
            metrics["compose_rel_l1"].append(_rel_l1(fused, batch["hdr_gt"]))
            metrics["compose_abs_l1"].append(float(torch.mean(torch.abs(fused - batch["hdr_gt"])).item()))
            metrics["compose_psnr"].append(_psnr(fused, batch["hdr_gt"]))
            metrics["baseline_decompose_rel_l1"].append(_rel_l1(decomp, i_tgt))
            metrics["baseline_decompose_abs_l1"].append(
                float(torch.mean(torch.abs(decomp - i_tgt)).item())
            )
            metrics["direct_only_compose_rel_l1"].append(_rel_l1(direct_as_full, batch["hdr_gt"]))
            metrics["direct_only_compose_abs_l1"].append(
                float(torch.mean(torch.abs(direct_as_full - batch["hdr_gt"])).item())
            )

            if preview_dir is not None:
                import imageio.v3 as iio

                def tm(x):
                    arr = x.permute(1, 2, 0).cpu().numpy()
                    arr = np.clip(arr, 0, None)
                    s = float(np.percentile(arr[arr > 0], 95)) if (arr > 0).any() else 1.0
                    ldr = 1.0 - np.exp(-arr / (s * 0.6 + 1e-6))
                    return (np.clip(ldr, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)

                bsz = i_pred.shape[0]
                for b in range(bsz):
                    iio.imwrite(preview_dir / f"{idx:03d}_pred.png", tm(i_pred[b]))
                    iio.imwrite(preview_dir / f"{idx:03d}_tgt.png", tm(i_tgt[b]))
                    iio.imwrite(preview_dir / f"{idx:03d}_fused.png", tm(fused[b]))
                    idx += 1

    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    summary["num_batches"] = len(metrics["indirect_rel_l1"])
    summary["num_samples"] = len(ds)
    summary["checkpoint"] = args.checkpoint
    summary["split"] = args.split

    print("=== C1 Eval ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.5f}")
        else:
            print(f"  {k}: {v}")

    out_json = args.output_json or str(Path(args.checkpoint).parent / f"eval_{args.split}.json")
    Path(out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()

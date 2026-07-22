#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C1 GT / Direct 对齐工具（M1）。

功能：
1. 对已有 bootstrap npz：拟合 Direct→Neural 的 scale，重写 aligned direct 与 indirect_target
2. 导出 gt_cache 风格 EXR（供 --gt_source blender / 评测）
3. 若本机有 bpy，可调用 scene_processor/to_blend.py 渲 Cycles GT（可选）

用法:
  # 对齐现有伪 GT 数据集
  python tools/c1_align_gt.py --data_dir data/c1/bootstrap --write_aligned

  # 仅统计不对齐
  python tools/c1_align_gt.py --data_dir data/c1/bootstrap --report_only

  # 导出 EXR 到 gt_cache
  python tools/c1_align_gt.py --data_dir data/c1/bootstrap --export_gt_cache gt_cache/c1_pseudo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renderformer.hybrid.align import HdrAligner


def _fit_scale(neural: np.ndarray, direct: np.ndarray, thr: float = 1e-3) -> float:
    n = torch.from_numpy(neural)
    d = torch.from_numpy(direct)
    aligner = HdrAligner.fit_robust(n, d, threshold=thr)
    return float(aligner.scale)


def _rel_l1(a: np.ndarray, b: np.ndarray, eps: float = 1e-3) -> float:
    return float(np.mean(np.abs(a - b) / (np.abs(b) + eps)))


def _psnr(a: np.ndarray, b: np.ndarray, peak: float | None = None) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return 99.0
    if peak is None:
        peak = float(max(np.percentile(b, 99), 1e-3))
    return 10.0 * np.log10((peak ** 2) / mse)


def main():
    parser = argparse.ArgumentParser(description="C1 GT/Direct alignment (M1)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--report_only", action="store_true")
    parser.add_argument("--write_aligned", action="store_true", help="写回 *_aligned.npz 并更新 manifest")
    parser.add_argument("--inplace", action="store_true", help="直接覆盖原 npz（谨慎）")
    parser.add_argument("--export_gt_cache", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=1e-3)
    args = parser.parse_args()

    root = Path(args.data_dir)
    manifest_path = root / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest.get("all") or manifest.get("train") or []
    if not entries:
        raise SystemExit("manifest 无样本")

    rows = []
    new_all = []
    for rel in entries:
        path = root / rel if not Path(rel).is_absolute() else Path(rel)
        data = dict(np.load(path))
        neural = data["hdr_neural"].astype(np.float32)
        direct = data["hdr_direct"].astype(np.float32)
        gt = data["hdr_gt"].astype(np.float32)
        scale = _fit_scale(neural, direct, args.threshold)
        direct_a = direct * scale
        indirect = np.clip(gt - direct_a, 0.0, None).astype(np.float32)

        row = {
            "file": rel,
            "scale": scale,
            "rel_l1_direct_vs_neural": _rel_l1(direct, neural),
            "rel_l1_aligned_direct_vs_neural": _rel_l1(direct_a, neural),
            "psnr_aligned_direct_vs_neural": _psnr(direct_a, neural),
            "indirect_mean": float(indirect.mean()),
            "indirect_max": float(indirect.max()),
        }
        rows.append(row)

        if args.export_gt_cache:
            import imageio.v3 as iio

            # gt_cache/<scene>/<res>/<view>_gt_full.exr
            stem = path.stem
            # e.g. cbox_r_orig_v00_orbit_...
            parts = stem.split("_")
            scene = parts[0]
            view_id = 0
            for p in parts:
                if p.startswith("v") and p[1:].isdigit():
                    view_id = int(p[1:])
                    break
            res = int(manifest.get("resolution", neural.shape[0]))
            out = Path(args.export_gt_cache) / scene / str(res)
            out.mkdir(parents=True, exist_ok=True)
            iio.imwrite(out / f"{view_id:04d}_gt_full.exr", gt)
            iio.imwrite(out / f"{view_id:04d}_direct.exr", direct_a)
            iio.imwrite(out / f"{view_id:04d}_indirect.exr", indirect)

        if args.write_aligned or args.inplace:
            out_data = {
                "hdr_neural": neural,
                "hdr_direct": direct_a.astype(np.float32),
                "depth": data["depth"].astype(np.float32),
                "indirect_target": indirect,
                "hdr_gt": gt,
                "align_scale": np.array([scale], dtype=np.float32),
            }
            if args.inplace:
                np.savez_compressed(path, **out_data)
                new_all.append(rel)
            else:
                out_name = path.stem + "_aligned.npz"
                out_path = path.parent / out_name
                np.savez_compressed(out_path, **out_data)
                new_all.append(f"samples/{out_name}")

        print(
            f"{path.name}: scale={scale:.4f} "
            f"relL1 {row['rel_l1_direct_vs_neural']:.3f}->{row['rel_l1_aligned_direct_vs_neural']:.3f} "
            f"PSNR={row['psnr_aligned_direct_vs_neural']:.2f}"
        )

    scales = [r["scale"] for r in rows]
    summary = {
        "num": len(rows),
        "scale_mean": float(np.mean(scales)),
        "scale_median": float(np.median(scales)),
        "rel_l1_before_mean": float(np.mean([r["rel_l1_direct_vs_neural"] for r in rows])),
        "rel_l1_after_mean": float(np.mean([r["rel_l1_aligned_direct_vs_neural"] for r in rows])),
        "psnr_after_mean": float(np.mean([r["psnr_aligned_direct_vs_neural"] for r in rows])),
        "rows": rows,
    }
    report_path = root / "align_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: scale_med={summary['scale_median']:.4f} "
          f"relL1 {summary['rel_l1_before_mean']:.3f}->{summary['rel_l1_after_mean']:.3f} "
          f"PSNR={summary['psnr_after_mean']:.2f}")
    print(f"Report: {report_path}")

    if (args.write_aligned or args.inplace) and not args.report_only:
        if args.write_aligned and not args.inplace:
            # 新 manifest 指向 aligned
            n = len(new_all)
            n_val = max(1, int(round(n * 0.2))) if n >= 5 else 0
            rng = np.random.default_rng(42)
            idx = np.arange(n)
            rng.shuffle(idx)
            val_set = set(idx[:n_val].tolist())
            train = [new_all[i] for i in range(n) if i not in val_set]
            val = [new_all[i] for i in range(n) if i in val_set]
            man2 = dict(manifest)
            man2["train"] = train
            man2["val"] = val
            man2["all"] = new_all
            man2["aligned"] = True
            man2["align_scale_median"] = summary["scale_median"]
            out_man = root / "manifest_aligned.json"
            with open(out_man, "w", encoding="utf-8") as f:
                json.dump(man2, f, indent=2, ensure_ascii=False)
            # 也写一份可用的 manifest（备份原文件）
            bak = root / "manifest_raw.json"
            if not bak.exists():
                bak.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(man2, f, indent=2, ensure_ascii=False)
            print(f"Updated manifest -> aligned samples ({n})")
        print("Aligned samples written.")

    # Blender 提示
    try:
        import bpy  # noqa: F401

        print("\nbpy available: 可用 scene_processor/to_blend.py 渲 Cycles GT，再 --gt_source blender 重烘焙。")
    except ImportError:
        print("\nbpy 未安装：当前以伪 GT + Direct 对齐完成 M1 代理验收；安装 bpy 后可上真实 Cycles。")


if __name__ == "__main__":
    main()

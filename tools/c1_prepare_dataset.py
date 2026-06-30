#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C1 训练数据准备：从场景 JSON/H5 烘焙
  hdr_neural / hdr_direct / depth / hdr_gt / indirect_target

默认 gt_source=neural：用 RF 全图作伪 GT（无 Blender 时可起步训练）。
若有 Blender GT EXR，可用 --gt_source blender 或 --gt_exr_dir。

用法:
  python tools/c1_prepare_dataset.py --scenes examples/cbox.json examples/room.json \\
      --output_dir data/c1/bootstrap --resolution 256 --views_per_scene 6
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

from benchmark_vi_cache import _apply_camera_variant
from renderformer import RenderFormerRenderingPipeline
from renderformer.hybrid.data_loader import add_batch_dim, load_h5_scene
from renderformer.hybrid.gt_loader import load_gt_full
from renderformer.hybrid.runtime_direct.factory import create_runtime_direct_renderer


def _look_at_to_c2w(pos, look_at, up) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    look_at = np.asarray(look_at, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = look_at - pos
    f = f / (np.linalg.norm(f) + 1e-8)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-8)
    u = np.cross(s, f)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = s
    c2w[:3, 1] = u
    c2w[:3, 2] = -f
    c2w[:3, 3] = pos
    return c2w


def _load_cycles_cams(cycles_root: Path | None, slug: str, resolution: int) -> list[dict] | None:
    if cycles_root is None:
        return None
    idx_path = cycles_root / slug / str(resolution) / "index.json"
    if not idx_path.is_file():
        # 也允许只有 EXR 无 index
        gt_dir = cycles_root / slug / str(resolution)
        if gt_dir.is_dir() and list(gt_dir.glob("*_gt_full.exr")):
            metas = sorted(gt_dir.glob("*_meta.json"))
            cams = []
            for m in metas:
                with open(m, "r", encoding="utf-8") as f:
                    cams.append(json.load(f))
            return cams or None
        return None
    with open(idx_path, "r", encoding="utf-8") as f:
        return json.load(f).get("cameras")


def _rotation_variants(n: int) -> list[dict]:
    if n <= 1:
        return [{"name": "v0", "orbit_y_deg": 0.0}]
    out = []
    for i in range(n):
        deg = -25.0 + 50.0 * i / max(n - 1, 1)
        fov = 0.9 + 0.2 * (i % 3) / 2.0  # 0.9 / 1.0 / 1.1
        out.append({"name": f"orbit_{deg:+.0f}_fov{fov:.2f}", "orbit_y_deg": deg, "fov_scale": fov})
    return out


def _roughness_scales(n: int) -> list[tuple[str, float | None]]:
    """None = keep original texture roughness."""
    if n <= 1:
        return [("r_orig", None)]
    vals = [None, 0.2, 0.55, 0.9][:n]
    names = ["r_orig", "r020", "r055", "r090"]
    return list(zip(names, vals))


def _apply_roughness(texture: torch.Tensor, value: float | None) -> torch.Tensor:
    if value is None:
        return texture
    tex = texture.clone()
    tex[:, :, ROUGHNESS_CHANNEL : ROUGHNESS_CHANNEL + 1, ...] = float(value)
    return tex


def _ensure_h5(scene_arg: str, h5_dir: Path) -> Path:
    p = Path(scene_arg)
    if p.suffix.lower() == ".h5" and p.is_file():
        return p
    if p.suffix.lower() == ".json" and p.is_file():
        h5_dir.mkdir(parents=True, exist_ok=True)
        slug = p.stem
        out_h5 = h5_dir / f"{slug}.h5"
        if not out_h5.is_file():
            import subprocess

            cmd = [
                sys.executable,
                str(ROOT / "scene_processor" / "convert_scene.py"),
                str(p),
                "--output_h5_path",
                str(out_h5),
            ]
            print("Running:", " ".join(cmd))
            subprocess.check_call(cmd, cwd=str(ROOT))
        return out_h5
    raise FileNotFoundError(f"无法解析场景: {scene_arg}")


def _tonemap_preview(hdr: np.ndarray) -> np.ndarray:
    x = np.clip(hdr, 0, None)
    scale = float(np.percentile(x[x > 0], 95)) if (x > 0).any() else 1.0
    ldr = 1.0 - np.exp(-x / (scale * 0.6 + 1e-6))
    return (np.clip(ldr, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Prepare C1 residual training dataset")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=[
            "examples/cbox.json",
            "examples/cbox-teapot.json",
            "examples/cbox-bunny.json",
            "examples/room.json",
            "examples/fox-in-the-wild.json",
        ],
        help="场景 JSON 或 H5 列表",
    )
    parser.add_argument("--output_dir", type=str, default="data/c1/bootstrap")
    parser.add_argument("--h5_dir", type=str, default="tmp/c1_scenes")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--views_per_scene", type=int, default=6)
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument(
        "--gt_source",
        type=str,
        default="neural",
        choices=["neural", "blender", "exr_dir"],
        help="neural=伪GT(RF)；blender=gt_cache；exr_dir=外部目录",
    )
    parser.add_argument("--gt_cache_dir", type=str, default="gt_cache")
    parser.add_argument("--gt_exr_dir", type=str, default=None, help="每样本同名 .exr 的目录")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_preview", action="store_true")
    parser.add_argument(
        "--roughness_variants",
        type=int,
        default=2,
        help="每场景材质 roughness 变体数（含原图）；扩大无多 H5 时的样本量",
    )
    parser.add_argument(
        "--cycles_gt_root",
        type=str,
        default=None,
        help="真实 Cycles GT 根目录（含 <slug>/<res>/index.json）。设置后强制 gt_source=blender 并用 GT 相机",
    )
    args = parser.parse_args()

    if args.cycles_gt_root:
        args.gt_source = "blender"
        args.roughness_variants = 1  # 真实 GT 暂不叠加 roughness 伪变体

    out_root = Path(args.output_dir)
    sample_dir = out_root / "samples"
    preview_dir = out_root / "previews"
    sample_dir.mkdir(parents=True, exist_ok=True)
    if args.save_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    # 先全部转 H5，避免与 RF 争用内存
    h5_list: list[Path] = []
    for scene_arg in args.scenes:
        try:
            h5_list.append(_ensure_h5(scene_arg, Path(args.h5_dir)))
        except Exception as e:
            print(f"[SKIP convert] {scene_arg}: {e}")
    if not h5_list:
        raise SystemExit("没有可用 H5，请检查 --scenes 或内存（convert 可能 OOM）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.precision]

    print(f"Loading RF: {args.model_id}")
    rf = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    rf.to(device)
    direct = create_runtime_direct_renderer(backend="auto")
    variants = _rotation_variants(args.views_per_scene)
    rough_vars = _roughness_scales(args.roughness_variants)
    cycles_root = Path(args.cycles_gt_root) if args.cycles_gt_root else None
    if cycles_root:
        # Cycles GT 目录同时作为 gt_cache_dir
        args.gt_cache_dir = str(cycles_root)

    all_rel: list[str] = []
    meta_rows: list[dict] = []

    for h5_path in h5_list:
        slug = h5_path.stem
        print(f"=== Scene {slug} ({h5_path}) ===")
        raw = load_h5_scene(str(h5_path))
        data = add_batch_dim(raw, device)
        base_c2w = data["c2w"]
        base_fov = data["fov"]
        if base_c2w.dim() == 3:
            base_c2w = base_c2w.unsqueeze(1)
        if base_fov.dim() == 2:
            base_fov = base_fov.unsqueeze(-1)

        cycles_cams = _load_cycles_cams(cycles_root, slug, args.resolution)
        if cycles_cams:
            print(f"  using {len(cycles_cams)} Cycles GT cameras")
            view_plan = [
                {"kind": "cycles", "vi": i, "cam": cam, "name": cam.get("name", f"cycles_{i}")}
                for i, cam in enumerate(cycles_cams)
            ]
        else:
            view_plan = [
                {"kind": "variant", "vi": i, "var": var, "name": var["name"]}
                for i, var in enumerate(variants)
            ]

        for r_name, r_val in rough_vars:
            texture = _apply_roughness(data["texture"], r_val)
            for item in view_plan:
                vi = item["vi"]
                if item["kind"] == "cycles":
                    cam = item["cam"]
                    c2w_np = _look_at_to_c2w(cam["position"], cam["look_at"], cam["up"])
                    c2w = torch.from_numpy(c2w_np).to(device=device, dtype=torch.float32)
                    c2w = c2w.unsqueeze(0).unsqueeze(0)
                    fov = torch.tensor(
                        [[[float(cam["fov"])]]], device=device, dtype=torch.float32
                    )
                    vname = item["name"]
                else:
                    var = item["var"]
                    c2w, fov = _apply_camera_variant(
                        base_c2w[:, 0], base_fov[:, 0:1], var, device, dtype
                    )
                    if c2w.dim() == 3:
                        c2w = c2w.unsqueeze(1)
                    if fov.dim() == 2:
                        fov = fov.unsqueeze(-1)
                    vname = var["name"]

                with torch.no_grad():
                    hdr_n = rf.render(
                        triangles=data["triangles"],
                        texture=texture,
                        mask=data["mask"],
                        vn=data["vn"],
                        c2w=c2w,
                        fov=fov,
                        resolution=args.resolution,
                        torch_dtype=dtype,
                    )
                    hdr_d, depth = direct.render(
                        data["triangles"],
                        texture,
                        data["vn"],
                        data["mask"],
                        c2w,
                        fov,
                        args.resolution,
                    )

                hn = hdr_n[0, 0].detach().float().cpu().numpy()
                hd = hdr_d[0, 0].detach().float().cpu().numpy()
                dep = depth[0, 0].detach().float().cpu().numpy()
                if dep.ndim == 2:
                    dep = dep[..., None]

                if args.gt_source == "neural":
                    gt = hn.copy()
                    gt_tag = "pseudo_neural"
                elif args.gt_source == "blender":
                    gt_arr = load_gt_full(args.gt_cache_dir, slug, args.resolution, vi)
                    if gt_arr is None:
                        print(f"  [WARN] missing Cycles GT, skip: {slug} view {vi}")
                        continue
                    gt = gt_arr.astype(np.float32)
                    if gt.ndim == 3 and gt.shape[-1] > 3:
                        gt = gt[..., :3]
                    if gt.shape[0] != args.resolution or gt.shape[1] != args.resolution:
                        import torch.nn.functional as F

                        t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0)
                        t = F.interpolate(
                            t, size=(args.resolution, args.resolution), mode="bilinear"
                        )
                        gt = t[0].permute(1, 2, 0).numpy()
                    gt_tag = "cycles"
                else:
                    exr_path = Path(args.gt_exr_dir) / f"{slug}_view_{vi:02d}.exr"
                    if not exr_path.is_file():
                        print(f"  [WARN] missing {exr_path}, skip")
                        continue
                    import imageio.v3 as iio

                    gt = iio.imread(exr_path).astype(np.float32)[..., :3]
                    gt_tag = "exr_dir"

                # 将 Cycles GT 对齐到 RF neural 能量空间；Direct 保持与 neural 同空间，不再强行贴 GT
                # （否则 I*=clip(gt-direct) 被压到近零，头会塌成预测全零）
                from renderformer.hybrid.align import HdrAligner

                gt_t = torch.from_numpy(gt)
                hn_t = torch.from_numpy(hn)
                gt_align = HdrAligner.fit_robust(hn_t, gt_t, threshold=1e-4)
                gt = (gt * gt_align.scale + gt_align.bias).astype(np.float32)
                align_scale = float(gt_align.scale)
                direct_scale = 1.0

                indirect = np.clip(gt - hd, 0.0, None).astype(np.float32)
                name = f"{slug}_{r_name}_v{vi:02d}_{vname}.npz".replace(" ", "_")
                np.savez_compressed(
                    sample_dir / name,
                    hdr_neural=hn.astype(np.float32),
                    hdr_direct=hd.astype(np.float32),
                    depth=dep.astype(np.float32),
                    indirect_target=indirect,
                    hdr_gt=gt.astype(np.float32),
                    align_scale=np.array([align_scale, direct_scale], dtype=np.float32),
                )
                rel = f"samples/{name}"
                all_rel.append(rel)
                meta_rows.append(
                    {
                        "file": rel,
                        "scene": slug,
                        "view": vi,
                        "variant": vname,
                        "roughness": r_name,
                        "gt_source": gt_tag,
                        "gt_to_neural_scale": align_scale,
                        "direct_to_gt_scale": direct_scale,
                        "h5": str(h5_path),
                    }
                )
                print(
                    f"  saved {name} gt={gt_tag} "
                    f"gt_scale={align_scale:.4g} d_scale={direct_scale:.4g} "
                    f"I*mean={float(indirect.mean()):.4g}"
                )

                if args.save_preview:
                    import imageio.v3 as iio

                    stem = f"{slug}_{r_name}_v{vi:02d}"
                    iio.imwrite(preview_dir / f"{stem}_neural.png", _tonemap_preview(hn))
                    iio.imwrite(preview_dir / f"{stem}_direct.png", _tonemap_preview(hd))
                    iio.imwrite(preview_dir / f"{stem}_gt.png", _tonemap_preview(gt))
                    iio.imwrite(preview_dir / f"{stem}_indirect.png", _tonemap_preview(indirect))

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(all_rel))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * args.val_ratio))) if len(idx) >= 5 else max(0, len(idx) // 5)
    val_idx = set(idx[:n_val].tolist())
    train = [all_rel[i] for i in range(len(all_rel)) if i not in val_idx]
    val = [all_rel[i] for i in range(len(all_rel)) if i in val_idx]
    if not train:
        train, val = all_rel, []

    manifest = {
        "version": 1,
        "resolution": args.resolution,
        "gt_source_default": args.gt_source,
        "num_samples": len(all_rel),
        "train": train,
        "val": val,
        "all": all_rel,
        "meta": meta_rows,
        "note": (
            "gt_source=neural 时 indirect_target=clamp(RF-Direct,0)，用于无 Blender 时的启动训练；"
            "有 Cycles GT 后请改用 blender/exr_dir 重跑本脚本。"
        ),
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    readme = out_root / "README.md"
    readme.write_text(
        f"""# C1 Bootstrap Dataset

- samples: `{len(all_rel)}`
- resolution: `{args.resolution}`
- gt_source: `{args.gt_source}`
- train/val: `{len(train)}` / `{len(val)}`

## 字段

| key | shape | 含义 |
|-----|-------|------|
| hdr_neural | HxWx3 | 冻结 RF 全图 HDR |
| hdr_direct | HxWx3 | Runtime Direct |
| depth | HxWx1 | Direct 深度 |
| hdr_gt | HxWx3 | GT（neural 伪标签或 Blender） |
| indirect_target | HxWx3 | clamp(gt - direct, 0) |

## 训练

```bash
python tools/c1_train.py --data_dir {out_root.as_posix()} --epochs 20 --batch_size 2
```
""",
        encoding="utf-8",
    )
    print(f"\nDone. manifest -> {out_root / 'manifest.json'} ({len(all_rel)} samples)")


if __name__ == "__main__":
    main()

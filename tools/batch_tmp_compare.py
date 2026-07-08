#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 examples/*.json 批量生成 H5，并用三种方式渲染对比：

  - renderformer/   纯 RenderFormer（无 VI 缓存）
  - cacheformer/    RenderFormer + VI Cache
  - runtime_direct/ nvdiffrast / lite Runtime Direct

目录结构:
  tmp/scenes/{slug}/{slug}.h5
  tmp/renders/{slug}/renderformer|cacheformer|runtime_direct/

用法:
  python tools/batch_tmp_compare.py
  python tools/batch_tmp_compare.py --resolution 256 --scenes cbox room tree
  python tools/batch_tmp_compare.py --convert-only
  python tools/batch_tmp_compare.py --render-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renderformer import RenderFormerRenderingPipeline, ViewIndependentCache
from renderformer.hybrid.data_loader import add_batch_dim, load_h5_scene
from renderformer.hybrid.runtime_direct import RuntimeDirectRenderer, nvdiffrast_available

# 约 10 个代表性案例（跳过 init-template）
DEFAULT_SCENES = [
    "cbox",
    "cbox-bunny",
    "cbox-teapot",
    "cbox-lucy",
    "room",
    "shader-ball",
    "tree",
    "crystals",
    "fox-in-the-wild",
    "compose-scene",
]


def scene_slug(json_stem: str) -> str:
    return json_stem.replace("_", "-").lower()


def h5_path(tmp_root: Path, slug: str) -> Path:
    return tmp_root / "scenes" / slug / f"{slug}.h5"


def render_root(tmp_root: Path, slug: str) -> Path:
    return tmp_root / "renders" / slug


def convert_one(json_path: Path, out_h5: Path, mesh_path: Path) -> dict:
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scene_processor" / "convert_scene.py"),
        str(json_path),
        "--output_h5_path",
        str(out_h5),
        "--mesh_path",
        str(mesh_path),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT / "scene_processor"), capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0 and out_h5.is_file()
    return {
        "json": str(json_path),
        "h5": str(out_h5),
        "ok": ok,
        "sec": round(elapsed, 2),
        "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
    }


def _tonemap_hdr(hdr: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(hdr[hdr > 0], 95.0)) if (hdr > 0).any() else 1.0
    ldr = 1.0 - np.exp(-hdr / (scale * 0.6 + 1e-6))
    return (np.clip(ldr, 0, 1) ** (1.0 / 2.2) * 255).astype(np.uint8)


def _save_rf_views(hdr: torch.Tensor, out_dir: Path, prefix: str = "view") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nv = hdr.shape[1]
    for i in range(nv):
        arr = hdr[0, i].detach().cpu().numpy().astype(np.float32)
        imageio.v3.imwrite(out_dir / f"{prefix}_{i}.exr", arr)
        imageio.v3.imwrite(out_dir / f"{prefix}_{i}.png", _tonemap_hdr(arr))


def _save_direct_views(hdr: torch.Tensor, depth: torch.Tensor, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nv = hdr.shape[1]
    for i in range(nv):
        arr = hdr[0, i].detach().cpu().numpy().astype(np.float32)
        imageio.v3.imwrite(out_dir / f"view_{i}_direct.exr", arr)
        imageio.v3.imwrite(out_dir / f"view_{i}_direct.png", _tonemap_hdr(arr))
        d = depth[0, i, ..., 0].detach().cpu().numpy()
        d_vis = d / (np.percentile(d[d > 0], 95) + 1e-6) if (d > 0).any() else d
        imageio.v3.imwrite(out_dir / f"view_{i}_depth.png", (np.clip(d_vis, 0, 1) * 255).astype(np.uint8))


def render_one_scene(
    slug: str,
    h5_file: Path,
    out_base: Path,
    rf_pipeline: RenderFormerRenderingPipeline,
    direct_renderer: RuntimeDirectRenderer,
    device: torch.device,
    dtype: torch.dtype,
    resolution: int,
    vi_cache: ViewIndependentCache,
    methods: list[str],
) -> dict:
    record = {"slug": slug, "h5": str(h5_file), "views": {}}
    if not h5_file.is_file():
        record["error"] = "h5 missing"
        return record

    data = load_h5_scene(str(h5_file))
    batch = add_batch_dim(data, device)
    nv = batch["c2w"].shape[1]
    scene_kw = dict(
        triangles=batch["triangles"],
        texture=batch["texture"],
        mask=batch["mask"],
        vn=batch["vn"],
        c2w=batch["c2w"],
        fov=batch["fov"],
        resolution=resolution,
    )
    rf_kw = {**scene_kw, "torch_dtype": dtype}

    if "renderformer" in methods:
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()
        hdr_rf = rf_pipeline.render(**rf_kw)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_rf = time.perf_counter() - t0
        rf_dir = out_base / "renderformer"
        _save_rf_views(hdr_rf, rf_dir)
        record["views"]["renderformer_ms"] = round(t_rf * 1000, 2)

    if "cacheformer" in methods:
        vi_cache.clear()
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()
        hdr_cf, info = rf_pipeline.render(**rf_kw, vi_cache=vi_cache, return_vi_cache_info=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_cf = time.perf_counter() - t0
        cf_dir = out_base / "cacheformer"
        _save_rf_views(hdr_cf, cf_dir)
        record["views"]["cacheformer_ms"] = round(t_cf * 1000, 2)
        record["views"]["vi_cache_hit"] = bool(info.get("vi_cache_hit", False))

    if "runtime_direct" in methods:
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()
        hdr_d, depth = direct_renderer.render(**scene_kw)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_rd = time.perf_counter() - t0
        rd_dir = out_base / "runtime_direct"
        _save_direct_views(hdr_d, depth, rd_dir)
        record["views"]["runtime_direct_ms"] = round(t_rd * 1000, 2)

    record["views"]["num_views"] = nv
    record["views"]["resolution"] = resolution
    return record


def main():
    parser = argparse.ArgumentParser(description="Batch H5 + triple render compare")
    parser.add_argument("--tmp_root", type=str, default=str(ROOT / "tmp"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--direct_backend", type=str, default="auto", choices=["auto", "nvdiffrast", "lite"])
    parser.add_argument("--scenes", nargs="*", default=None, help="json stem list, default 10 scenes")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["renderformer", "cacheformer", "runtime_direct"],
        choices=["renderformer", "cacheformer", "runtime_direct"],
        help="subset of render passes",
    )
    args = parser.parse_args()

    tmp_root = Path(args.tmp_root)
    scenes = args.scenes or DEFAULT_SCENES
    examples = ROOT / "examples"

    manifest = {
        "tmp_root": str(tmp_root),
        "resolution": args.resolution,
        "methods": args.methods,
        "scenes": scenes,
        "convert": [],
        "render": [],
    }

    if not args.render_only:
        print(f"=== Phase 1: convert {len(scenes)} JSON -> H5 ===")
        for stem in scenes:
            json_path = examples / f"{stem}.json"
            if not json_path.is_file():
                # allow cbox style paths
                alt = examples / f"{stem.replace('-', '_')}.json"
                json_path = alt if alt.is_file() else json_path
            slug = scene_slug(Path(stem).stem if stem.endswith(".json") else stem)
            out_h5 = h5_path(tmp_root, slug)
            mesh_p = tmp_root / "scenes" / slug / "mesh.obj"
            print(f"  [{slug}] convert ...", end=" ", flush=True)
            if not json_path.is_file():
                print("SKIP (json not found)")
                manifest["convert"].append({"slug": slug, "ok": False, "error": "json not found"})
                continue
            rec = convert_one(json_path, out_h5, mesh_p)
            manifest["convert"].append(rec)
            print("OK" if rec["ok"] else f"FAIL ({rec.get('stderr_tail', '')[:80]})")

    if args.convert_only:
        manifest_path = tmp_root / "renders" / "manifest_convert.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Wrote {manifest_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.float16
        if args.precision == "fp16"
        else torch.bfloat16
        if args.precision == "bf16"
        else torch.float32
    )

    print(f"\n=== Phase 2: render {len(scenes)} scenes @ {args.resolution}x{args.resolution} ===")
    print(f"  device={device}, direct={args.direct_backend}, nvd={nvdiffrast_available()}")

    t_load = time.perf_counter()
    rf = None
    direct = None
    vi_cache = ViewIndependentCache(max_entries=16)
    need_rf = "renderformer" in args.methods or "cacheformer" in args.methods
    need_direct = "runtime_direct" in args.methods
    if need_rf:
        rf = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
        if device.type == "cuda" and os.name == "posix":
            try:
                from renderformer_liger_kernel import apply_kernels

                apply_kernels(rf.model)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except ImportError:
                pass
        rf.to(device)
    if need_direct:
        direct = RuntimeDirectRenderer(backend=args.direct_backend)
    print(f"  model load: {(time.perf_counter()-t_load):.1f}s", end="")
    if direct is not None:
        print(f", direct={direct.active_backend}", end="")
    print()

    for stem in scenes:
        slug = scene_slug(stem)
        h5 = h5_path(tmp_root, slug)
        out_base = render_root(tmp_root, slug)
        print(f"  [{slug}] render ...", end=" ", flush=True)
        try:
            rec = render_one_scene(
                slug,
                h5,
                out_base,
                rf,
                direct,
                device,
                dtype,
                args.resolution,
                vi_cache,
                args.methods,
            )
            manifest["render"].append(rec)
            v = rec.get("views", {})
            print(
                f"RF {v.get('renderformer_ms', '?')}ms | "
                f"Cache {v.get('cacheformer_ms', '?')}ms | "
                f"Direct {v.get('runtime_direct_ms', '?')}ms"
            )
        except Exception as e:
            manifest["render"].append({"slug": slug, "error": str(e)})
            print(f"FAIL: {e}")

    manifest_path = tmp_root / "renders" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone. manifest: {manifest_path}")
    print("Layout:")
    print(f"  H5:     {tmp_root / 'scenes'}")
    print(f"  renders: {tmp_root / 'renders'}/{{scene}}/{{renderformer|cacheformer|runtime_direct}}/")


if __name__ == "__main__":
    main()

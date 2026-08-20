#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态场景连续帧对比：基线（无 VI 缓存） vs VI 缓存管线。

每 scene_change_every 帧切换一次场景（修改材质 roughness），段内仅相机 orbit 变化。
预期 VI 缓存：每段首帧 miss，段内后续帧 hit。

用法:
  python tools/compare_dynamic_scene.py --h5_file tmp/cbox/cbox.h5
  python tools/compare_dynamic_scene.py --h5_file tmp/cbox/cbox.h5 --num_frames 12 --scene_change_every 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmark_vi_cache import _apply_camera_variant, load_h5
from compare_baseline_vs_vi_cache import _get_gpu_memory_mb, _sanitize_filename, _sync
from renderformer import RenderFormerRenderingPipeline, ViewIndependentCache

# texture: [B, N, 13, 32, 32] → diffuse(3) + specular(3) + roughness(1) + normal(3) + irradiance(3)
ROUGHNESS_CHANNEL = 6
SCENE_ROUGHNESS = [0.15, 0.45, 0.75, 0.95]


def _orbit_variant(frame_idx: int, num_frames: int) -> dict:
    if num_frames <= 1:
        deg = 0.0
    else:
        deg = -30.0 + 60.0 * frame_idx / (num_frames - 1)
    return {"name": f"f{frame_idx + 1:02d}_orbit_{deg:+.0f}deg", "orbit_y_deg": deg}


def _scene_texture(base_texture: torch.Tensor, scene_id: int) -> torch.Tensor:
    tex = base_texture.clone()
    r = SCENE_ROUGHNESS[scene_id % len(SCENE_ROUGHNESS)]
    tex[:, :, ROUGHNESS_CHANNEL : ROUGHNESS_CHANNEL + 1, :, :] = r
    return tex


def _tonemap_frame(hdr: np.ndarray) -> np.ndarray:
    x = np.clip(hdr, 0.0, None)
    scale = float(np.percentile(x[x > 0], 95.0)) if (x > 0).any() else 1.0
    ldr = 1.0 - np.exp(-x / (scale * 0.6 + 1e-6))
    return (np.clip(ldr, 0, 1) ** (1.0 / 2.2) * 255).astype(np.uint8)


def run_dynamic_sequence(
    pipeline: RenderFormerRenderingPipeline,
    base_data: dict,
    num_frames: int,
    scene_change_every: int,
    device: torch.device,
    dtype: torch.dtype,
    resolution: int,
    vi_cache: Optional[ViewIndependentCache],
    output_dir: Optional[str] = None,
    run_label: str = "baseline",
    save_frames: int = 0,
) -> Tuple[List[dict], Dict]:
    if vi_cache is not None:
        vi_cache.clear()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    save_dir = None
    if output_dir and save_frames > 0:
        save_dir = os.path.join(output_dir, run_label)
        os.makedirs(save_dir, exist_ok=True)

    records: List[dict] = []
    for i in range(num_frames):
        scene_id = i // scene_change_every
        var = _orbit_variant(i, num_frames)
        texture = _scene_texture(base_data["texture"], scene_id)
        c2w, fov = _apply_camera_variant(
            base_data["c2w"], base_data["fov"], var, device, dtype
        )
        tag = f"scene{scene_id}_frame{i + 1}"
        name = f"{tag}_{var['name']}"

        _sync(device)
        t0 = time.perf_counter()
        rendered, info = pipeline.render(
            triangles=base_data["triangles"],
            texture=texture,
            mask=base_data["mask"],
            vn=base_data["vn"],
            c2w=c2w,
            fov=fov,
            resolution=resolution,
            torch_dtype=dtype,
            vi_cache=vi_cache,
            return_vi_cache_info=True,
        )
        _sync(device)
        elapsed = time.perf_counter() - t0

        rec = {
            "frame": i + 1,
            "scene_id": scene_id,
            "roughness": SCENE_ROUGHNESS[scene_id % len(SCENE_ROUGHNESS)],
            "elapsed_sec": elapsed,
            "vi_cache_hit": info.get("vi_cache_hit", False),
            "tag": tag,
        }
        records.append(rec)

        if save_dir is not None and i < save_frames:
            import imageio

            hdr = rendered[0, 0].detach().cpu().numpy().astype(np.float32)
            png = _tonemap_frame(hdr)
            fname = f"frame_{i + 1:02d}_{_sanitize_filename(name)}.png"
            imageio.v3.imwrite(os.path.join(save_dir, fname), png)

    times = [r["elapsed_sec"] for r in records]
    summary = {
        "total_sec": sum(times),
        "mean_sec": sum(times) / len(times) if times else 0.0,
        "fps": len(times) / sum(times) if sum(times) > 0 else 0.0,
        "gpu_peak_mb": _get_gpu_memory_mb(device),
    }
    if vi_cache is not None:
        s = vi_cache.stats()
        summary.update(
            {
                "cache_hits": s["hits"],
                "cache_misses": s["misses"],
                "cache_hit_rate": s["hit_rate"],
                "cache_entries": s["entries"],
            }
        )
    return records, summary


def main():
    parser = argparse.ArgumentParser(description="动态场景连续帧：基线 vs VI 缓存")
    parser.add_argument("--h5_file", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--scene_change_every", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--vi_cache_max_entries", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="out/compare_dynamic")
    parser.add_argument("--save_frames", type=int, default=0, help="0=全部帧")
    args = parser.parse_args()

    if args.scene_change_every < 1:
        parser.error("--scene_change_every 必须 >= 1")
    save_frames = args.num_frames if args.save_frames == 0 else args.save_frames

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.float16
        if args.precision == "fp16"
        else torch.bfloat16
        if args.precision == "bf16"
        else torch.float32
    )

    num_scenes = (args.num_frames + args.scene_change_every - 1) // args.scene_change_every
    expected_misses = num_scenes
    expected_hits = max(0, args.num_frames - expected_misses)

    print("=" * 60)
    print("动态场景连续帧对比")
    print(f"  帧数={args.num_frames}, 每 {args.scene_change_every} 帧换场景 → {num_scenes} 个场景段")
    print(f"  预期 VI 缓存: miss≈{expected_misses}, hit≈{expected_hits}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    pipeline = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    pipeline.to(device)
    data = load_h5(args.h5_file, device)

    if args.warmup > 0:
        print(f"预热 {args.warmup} 次...")
        for _ in range(args.warmup):
            tex = _scene_texture(data["texture"], 0)
            c2w, fov = _apply_camera_variant(
                data["c2w"], data["fov"], _orbit_variant(0, args.num_frames), device, dtype
            )
            pipeline.render(
                triangles=data["triangles"],
                texture=tex,
                mask=data["mask"],
                vn=data["vn"],
                c2w=c2w,
                fov=fov,
                resolution=args.resolution,
                torch_dtype=dtype,
            )
        _sync(device)

    print("\n【第一轮】基线（无 VI 缓存）")
    rec_b, sum_b = run_dynamic_sequence(
        pipeline,
        data,
        args.num_frames,
        args.scene_change_every,
        device,
        dtype,
        args.resolution,
        vi_cache=None,
        output_dir=args.output_dir,
        run_label="baseline",
        save_frames=save_frames,
    )

    cache = ViewIndependentCache(max_entries=args.vi_cache_max_entries)
    print("\n【第二轮】VI 缓存管线")
    rec_c, sum_c = run_dynamic_sequence(
        pipeline,
        data,
        args.num_frames,
        args.scene_change_every,
        device,
        dtype,
        args.resolution,
        vi_cache=cache,
        output_dir=args.output_dir,
        run_label="with_cache",
        save_frames=save_frames,
    )

    speedup = sum_b["total_sec"] / sum_c["total_sec"] if sum_c["total_sec"] > 0 else 0.0

    print("\n" + "=" * 60)
    print("性能汇总")
    print("=" * 60)
    print(f"  {'指标':<24} {'基线':<16} {'VI缓存':<16}")
    print(f"  {'总耗时(s)':<24} {sum_b['total_sec']:.4f}          {sum_c['total_sec']:.4f}")
    print(f"  {'平均帧(ms)':<24} {sum_b['mean_sec']*1000:.1f}            {sum_c['mean_sec']*1000:.1f}")
    print(f"  {'FPS':<24} {sum_b['fps']:.2f}            {sum_c['fps']:.2f}")
    print(f"  加速比: {speedup:.2f}x")
    if "cache_hits" in sum_c:
        print(
            f"  缓存: hits={sum_c['cache_hits']}, misses={sum_c['cache_misses']}, "
            f"命中率={sum_c['cache_hit_rate']*100:.1f}%, entries={sum_c['cache_entries']}"
        )

    print("\n逐帧明细 (scene_id | roughness | 基线ms | 缓存ms | hit):")
    print(f"  {'帧':>3}  {'段':>4}  {'r':>5}  {'基线':>8}  {'缓存':>8}  hit")
    print("  " + "-" * 42)
    for rb, rc in zip(rec_b, rec_c):
        print(
            f"  {rb['frame']:3d}  {rb['scene_id']:4d}  {rb['roughness']:5.2f}  "
            f"{rb['elapsed_sec']*1000:8.1f}  {rc['elapsed_sec']*1000:8.1f}  {rc['vi_cache_hit']}"
        )

    report = {
        "h5_file": args.h5_file,
        "num_frames": args.num_frames,
        "scene_change_every": args.scene_change_every,
        "num_scenes": num_scenes,
        "expected_misses": expected_misses,
        "baseline": sum_b,
        "with_cache": sum_c,
        "speedup": speedup,
        "frames_baseline": rec_b,
        "frames_cache": rec_c,
    }
    report_path = os.path.join(args.output_dir, "dynamic_compare_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n报告: {report_path}")
    print(f"帧图: {args.output_dir}/baseline/ 与 {args.output_dir}/with_cache/")


if __name__ == "__main__":
    main()

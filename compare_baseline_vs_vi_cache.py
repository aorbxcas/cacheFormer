# -*- coding: utf-8 -*-
"""
对照实验：原模型（无 VI 缓存） vs 启用 VI 缓存的模型。

在相同 h5 场景、相同相机序列下分别跑两轮，采集并对比：
- 总耗时、每帧耗时、FPS、显存占用（若可用）
- 启用缓存时的命中率与加速比

用法:
  python compare_baseline_vs_vi_cache.py --h5_file tmp/cbox/cbox.h5
  python compare_baseline_vs_vi_cache.py --h5_file tmp/cbox/cbox.h5 --num_renders 8 --resolution 512
"""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from renderformer import RenderFormerRenderingPipeline, ViewIndependentCache

# 复用 benchmark 的加载与相机变体逻辑
from benchmark_vi_cache import (
    _apply_camera_variant,
    _get_camera_variants,
    load_h5,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _get_gpu_memory_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def _sanitize_filename(name: str) -> str:
    """用于文件名的安全字符串：空格、斜杠等替换为下划线。"""
    return name.replace(" ", "_").replace("/", "_").replace("°", "deg")


def run_sequence(
    pipeline: RenderFormerRenderingPipeline,
    data: dict,
    variants: list,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    resolution: int,
    vi_cache: Optional[ViewIndependentCache],
    reset_gpu_stats: bool = True,
    output_dir: Optional[str] = None,
    run_label: str = "baseline",
    save_frames: int = 0,
    cache_refresh_interval: int = 0,
) -> Tuple[List[float], Dict]:
    """
    按 variants 前 n 组相机各渲染一帧，返回每帧耗时列表与汇总信息。

    若 output_dir 非空且 save_frames > 0，将前 save_frames 帧导出为 PNG 到 output_dir/run_label/。

    Returns:
        frame_times: 每帧耗时 (秒)
        info: {"total_sec", "mean_sec", "fps", "cache_hits", "cache_misses", "cache_hit_rate", "gpu_peak_mb"}
    """
    if reset_gpu_stats and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if vi_cache is not None:
        vi_cache.clear()

    do_save = output_dir and save_frames > 0
    if do_save:
        save_dir = os.path.join(output_dir, run_label)
        os.makedirs(save_dir, exist_ok=True)

    frame_times: List[float] = []
    refresh_count = 0
    for i in range(n):
        if (
            vi_cache is not None
            and cache_refresh_interval > 0
            and i > 0
            and i % cache_refresh_interval == 0
        ):
            vi_cache.clear()
            refresh_count += 1
        var = variants[i]
        name = var.get("name", str(i + 1))
        c2w, fov = _apply_camera_variant(
            data["c2w"], data["fov"], var, device, dtype
        )
        _sync(device)
        t0 = time.perf_counter()
        out = pipeline.render(
            triangles=data["triangles"],
            texture=data["texture"],
            mask=data["mask"],
            vn=data["vn"],
            c2w=c2w,
            fov=fov,
            resolution=resolution,
            torch_dtype=dtype,
            vi_cache=vi_cache,
            return_vi_cache_info=True,
        )
        rendered_imgs, info = out
        _sync(device)
        elapsed = time.perf_counter() - t0
        frame_times.append(elapsed)

        if do_save and i < save_frames:
            import imageio
            frame = rendered_imgs[0, 0].cpu().numpy().astype(np.float32)
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)
            path = os.path.join(
                save_dir,
                "frame_%02d_%s.png" % (i + 1, _sanitize_filename(name)),
            )
            imageio.v3.imwrite(path, frame)

    total_sec = sum(frame_times)
    mean_sec = total_sec / n if n else 0.0
    fps = 1.0 / mean_sec if mean_sec > 0 else 0.0

    info = {
        "total_sec": total_sec,
        "mean_sec": mean_sec,
        "fps": fps,
        "gpu_peak_mb": _get_gpu_memory_mb(device),
    }
    if vi_cache is not None:
        s = vi_cache.stats()
        info["cache_hits"] = s["hits"]
        info["cache_misses"] = s["misses"]
        info["cache_hit_rate"] = s["hit_rate"]
        info["cache_refresh_interval"] = cache_refresh_interval
        info["cache_refresh_count"] = refresh_count
    return frame_times, info


def main():
    parser = argparse.ArgumentParser(
        description="对照实验：无 VI 缓存（基线） vs 启用 VI 缓存，相同 h5 与相机序列，打印详细性能对比。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5_file", type=str, required=True, help="输入 H5 场景文件路径")
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/renderformer-v1.1-swin-large",
        help="Hugging Face 模型 ID 或本地路径",
    )
    parser.add_argument("--resolution", type=int, default=256, help="渲染分辨率（边长）")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="fp16",
        help="推理精度",
    )
    parser.add_argument(
        "--num_renders",
        type=int,
        default=None,
        metavar="N",
        help="渲染帧数（相机变体数量）；不指定则用全部约 14 组",
    )
    parser.add_argument(
        "--vi_cache_max_entries",
        type=int,
        default=4,
        help="VI 缓存最大条目数（仅影响「启用缓存」一轮）",
    )
    parser.add_argument(
        "--cache_refresh_interval",
        type=int,
        default=0,
        help="缓存轮每 N 帧清空一次缓存（0=不清空；>0 可模拟 hybrid 周期全量刷新）",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="正式计时前每轮预热渲染次数（避免首帧冷启动）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="导出渲染效果图到此目录：baseline/ 与 with_cache/ 子目录下保存 PNG",
    )
    parser.add_argument(
        "--save_frames",
        type=int,
        default=0,
        metavar="K",
        help="导出前 K 帧为 PNG（0=不导出）；需同时指定 --output_dir",
    )
    args = parser.parse_args()

    if args.save_frames > 0 and not args.output_dir:
        parser.error("使用 --save_frames 时需同时指定 --output_dir")
    if args.cache_refresh_interval < 0:
        parser.error("--cache_refresh_interval 必须 >= 0")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    if device.type == "mps":
        dtype = torch.float32
    else:
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[args.precision]

    pipeline = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    pipeline.to(device)
    data = load_h5(args.h5_file, device)

    variants = _get_camera_variants()
    n = min(args.num_renders, len(variants)) if args.num_renders is not None else len(variants)
    n = max(1, n)

    cache = ViewIndependentCache(max_entries=args.vi_cache_max_entries)

    # ----- 预热 -----
    if args.warmup > 0:
        print("预热中 (每轮 %d 次) ..." % args.warmup)
        for _ in range(args.warmup):
            c2w, fov = _apply_camera_variant(
                data["c2w"], data["fov"], variants[0], device, dtype
            )
            pipeline.render(
                triangles=data["triangles"],
                texture=data["texture"],
                mask=data["mask"],
                vn=data["vn"],
                c2w=c2w,
                fov=fov,
                resolution=args.resolution,
                torch_dtype=dtype,
                vi_cache=None,
            )
        _sync(device)
        print("预热完成。\n")

    if args.output_dir and args.save_frames > 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print("渲染效果图将导出到: %s (baseline/ 与 with_cache/ 各前 %d 帧)\n" % (args.output_dir, args.save_frames))

    # ----- 第一轮：基线（无缓存） -----
    print("=" * 60)
    print("【第一轮】基线（原模型，无 VI 缓存）")
    print("=" * 60)
    frame_times_baseline, info_baseline = run_sequence(
        pipeline, data, variants, n, device, dtype, args.resolution,
        vi_cache=None,
        reset_gpu_stats=True,
        output_dir=args.output_dir,
        run_label="baseline",
        save_frames=args.save_frames,
    )
    _sync(device)

    # ----- 第二轮：启用 VI 缓存 -----
    print("\n【第二轮】启用 VI 缓存 (max_entries=%d)" % args.vi_cache_max_entries)
    print("=" * 60)
    frame_times_cached, info_cached = run_sequence(
        pipeline, data, variants, n, device, dtype, args.resolution,
        vi_cache=cache,
        reset_gpu_stats=True,
        output_dir=args.output_dir,
        run_label="with_cache",
        save_frames=args.save_frames,
        cache_refresh_interval=args.cache_refresh_interval,
    )
    _sync(device)

    # ----- 性能对比表 -----
    total_b = info_baseline["total_sec"]
    total_c = info_cached["total_sec"]
    speedup = total_b / total_c if total_c > 0 else 0.0

    print("\n")
    print("=" * 60)
    print("性能对比汇总 (共 %d 帧, resolution=%d, precision=%s)" % (n, args.resolution, args.precision))
    print("=" * 60)

    print("  %-28s  %-20s  %-20s" % ("指标", "原模型（无缓存）", "启用 VI 缓存"))
    print("  " + "-" * 70)
    print("  %-28s  %-20s  %-20s" % ("总耗时 (s)", "%.4f" % total_b, "%.4f" % total_c))
    print("  %-28s  %-20s  %-20s" % (
        "平均每帧耗时 (ms)",
        "%.2f" % (info_baseline["mean_sec"] * 1000),
        "%.2f" % (info_cached["mean_sec"] * 1000),
    ))
    print("  %-28s  %-20s  %-20s" % (
        "FPS (帧/秒)",
        "%.2f" % info_baseline["fps"],
        "%.2f" % info_cached["fps"],
    ))
    if device.type == "cuda":
        print("  %-28s  %-20s  %-20s" % (
            "显存峰值 (MB)",
            "%.1f" % (info_baseline["gpu_peak_mb"] or 0),
            "%.1f" % (info_cached["gpu_peak_mb"] or 0),
        ))
    print("  " + "-" * 70)
    print("  加速比 (基线总耗时 / 缓存总耗时):  %.2fx" % speedup)
    print("  缓存统计 (启用缓存轮):  hits=%d, misses=%d, 命中率=%.1f%%" % (
        info_cached.get("cache_hits", 0),
        info_cached.get("cache_misses", 0),
        (info_cached.get("cache_hit_rate", 0) or 0) * 100,
    ))
    if args.cache_refresh_interval > 0:
        print("  缓存周期清空:            每 %d 帧清空一次, 共清空 %d 次" % (
            args.cache_refresh_interval,
            info_cached.get("cache_refresh_count", 0),
        ))
    print("=" * 60)

    # ----- 每帧耗时明细（第 1 帧为 miss，第 2～n 帧为 hit） -----
    print("\n每帧耗时明细 (s):")
    print("  帧序号    基线(无缓存)    启用缓存    说明")
    print("  " + "-" * 58)
    for i in range(n):
        note = "miss (完整 VI+VD)" if i == 0 else "hit (仅 VD)"
        print("  第 %2d 帧   %12.4f   %12.4f   %s" % (
            i + 1,
            frame_times_baseline[i],
            frame_times_cached[i],
            note,
        ))
    print("  " + "-" * 58)

    if args.output_dir and args.save_frames > 0:
        print("\n渲染效果图已导出:")
        print("  基线（无缓存）:  %s" % os.path.join(args.output_dir, "baseline"))
        print("  启用 VI 缓存:   %s" % os.path.join(args.output_dir, "with_cache"))


if __name__ == "__main__":
    main()

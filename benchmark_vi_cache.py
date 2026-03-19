# -*- coding: utf-8 -*-
"""
VI 缓存效果基准：同一场景、多轮不同相机渲染，验证缓存命中与耗时。

预期：
- 第 1 次：缓存未命中（完整 VI + VD），耗时长
- 第 2～N 次：缓存命中（仅 VD），耗时明显缩短

用法（需已有 h5，且 GPU）:
    python benchmark_vi_cache.py --h5_file tmp/cbox/cbox.h5
    python benchmark_vi_cache.py --h5_file tmp/cbox/cbox.h5 --num_renders 6

说明：
- 仅 batch_size=1 时启用缓存；多轮渲染共用同一场景几何，仅改变 FOV/相机，
  故第 1 次 miss，后续均应 hit，用于验证缓存有效性。
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from renderformer import RenderFormerRenderingPipeline, ViewIndependentCache


def _rotation_y_3x3(angle_deg: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """绕 Y 轴的 3x3 旋转矩阵（角度制）。"""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device, dtype=dtype)
    return R


def _apply_camera_variant(
    c2w: torch.Tensor,
    fov: torch.Tensor,
    variant: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """
    根据 variant 描述修改 c2w 与 fov，返回 (c2w_new, fov_new)。
    c2w 形状 (1, 4, 4)，fov (1, 1, 1)。variant 可含 fov_scale, orbit_y_deg, distance_scale。
    """
    c2w = c2w.clone()
    fov = fov.clone()
    if "fov_scale" in variant:
        fov = fov * variant["fov_scale"]
    # 取第一个样本、第一个视角的 4x4 矩阵（兼容 c2w 形状 (1,4,4) 或 (1,nv,4,4)）
    if c2w[0].dim() == 2:
        M = c2w[0]
        write_back = lambda m: c2w[0].copy_(m)
    else:
        M = c2w[0, 0]
        write_back = lambda m: c2w[0, 0].copy_(m)
    # M 为 (4, 4)，M[:3, 3] 形状 (3,)，可与 3x3 旋转矩阵相乘
    if "orbit_y_deg" in variant:
        R_y = _rotation_y_3x3(variant["orbit_y_deg"], M.device, M.dtype)
        M_new = M.clone()
        M_new[:3, :3] = R_y @ M[:3, :3]
        M_new[:3, 3] = R_y @ M[:3, 3]
        write_back(M_new)
        M = c2w[0, 0] if c2w[0].dim() == 3 else c2w[0]
    if "distance_scale" in variant:
        M = c2w[0, 0] if c2w[0].dim() == 3 else c2w[0]
        M_new = M.clone()
        M_new[:3, 3] = M[:3, 3] * variant["distance_scale"]
        write_back(M_new)
    return c2w, fov


def _get_camera_variants() -> list:
    """返回多组相机参数变体，每项为 dict：name + fov_scale / orbit_y_deg / distance_scale。"""
    return [
        {"name": "原始", "fov_scale": 1.0},
        {"name": "FOV 0.9x", "fov_scale": 0.9},
        {"name": "FOV 1.1x", "fov_scale": 1.1},
        {"name": "FOV 0.85x", "fov_scale": 0.85},
        {"name": "FOV 1.15x", "fov_scale": 1.15},
        {"name": "Orbit Y +15°", "orbit_y_deg": 15.0},
        {"name": "Orbit Y -15°", "orbit_y_deg": -15.0},
        {"name": "Orbit Y +25°", "orbit_y_deg": 25.0},
        {"name": "Orbit Y -20°", "orbit_y_deg": -20.0},
        {"name": "FOV 0.9x + Orbit Y +10°", "fov_scale": 0.9, "orbit_y_deg": 10.0},
        {"name": "FOV 1.1x + Orbit Y -10°", "fov_scale": 1.1, "orbit_y_deg": -10.0},
        {"name": "距离 1.1x (拉远)", "distance_scale": 1.1},
        {"name": "距离 0.9x (推近)", "distance_scale": 0.9},
        {"name": "距离 1.05x + FOV 0.95x", "distance_scale": 1.05, "fov_scale": 0.95},
    ]


def _short_key(key: str, head: int = 12, tail: int = 8) -> str:
    if not key or len(key) <= head + tail:
        return key
    return f"{key[:head]}...{key[-tail:]}"


def _print_cache_detail(cache: ViewIndependentCache) -> None:
    s = cache.stats()
    print(f"    [缓存状态] 当前条目数={s['entries']}/{cache.max_entries}, 累计 hits={s['hits']}, misses={s['misses']}, hit_rate={s['hit_rate']:.2%}")
    if s["entries"] > 0:
        keys = cache.current_keys()
        print(f"    [缓存键列表] {[ _short_key(k) for k in keys ]}")


def load_h5(path: str, device: torch.device):
    import h5py

    with h5py.File(path, "r") as f:
        triangles = torch.from_numpy(np.array(f["triangles"]).astype(np.float32))
        texture = torch.from_numpy(np.array(f["texture"]).astype(np.float32))
        num_tris = triangles.shape[0]
        mask = torch.ones(num_tris, dtype=torch.bool)
        vn = torch.from_numpy(np.array(f["vn"]).astype(np.float32))
        c2w = torch.from_numpy(np.array(f["c2w"]).astype(np.float32))
        fov = torch.from_numpy(np.array(f["fov"]).astype(np.float32))
    return {
        "triangles": triangles.unsqueeze(0).to(device),
        "texture": texture.unsqueeze(0).to(device),
        "mask": mask.unsqueeze(0).to(device),
        "vn": vn.unsqueeze(0).to(device),
        "c2w": c2w.unsqueeze(0).to(device),
        "fov": fov.unsqueeze(0).unsqueeze(-1).to(device),
    }


def main():
    parser = argparse.ArgumentParser(
        description="VI 缓存基准：同一场景多组相机渲染，验证缓存命中与耗时（第 1 次 miss，后续 hit）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
示例:
  python benchmark_vi_cache.py --h5_file tmp/cbox/cbox.h5
  python benchmark_vi_cache.py --h5_file tmp/cbox/cbox.h5 --num_renders 6 --resolution 512
  python benchmark_vi_cache.py --h5_file tmp/cbox/cbox.h5 --no_cache --num_renders 3
  python benchmark_vi_cache.py --h5_file tmp/cbox/cbox.h5 --output_dir out/bench --save_frames 2
        """,
    )
    parser.add_argument(
        "--h5_file",
        type=str,
        required=True,
        help="输入 H5 场景文件路径（需先由 scene_processor/convert_scene.py 生成）",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/renderformer-v1.1-swin-large",
        help="Hugging Face 模型 ID 或本地权重路径",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="渲染分辨率（边长，如 256 表示 256x256）",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="fp16",
        help="推理精度；MPS 下会强制 fp32",
    )
    parser.add_argument(
        "--num_renders",
        type=int,
        default=None,
        metavar="N",
        help="渲染轮数；不指定则使用全部相机变体（约 14 组）",
    )
    parser.add_argument(
        "--vi_cache_max_entries",
        type=int,
        default=4,
        help="VI 缓存最大条目数（LRU 容量）",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="禁用 VI 缓存，每轮都跑完整 VI+VD，用于对比基线耗时",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="可选：保存渲染帧的目录；需配合 --save_frames 使用",
    )
    parser.add_argument(
        "--save_frames",
        type=int,
        default=0,
        metavar="K",
        help="保存前 K 帧为 PNG（0 表示不保存）；需指定 --output_dir",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="仅打印汇总结果，不打印每轮详细缓存信息",
    )
    args = parser.parse_args()

    def log(*a, **kw):
        if not args.quiet:
            print(*a, **kw)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if device.type == "mps":
        dtype = torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    if args.save_frames > 0 and not args.output_dir:
        parser.error("使用 --save_frames 时需同时指定 --output_dir")
    if args.output_dir and args.save_frames > 0:
        os.makedirs(args.output_dir, exist_ok=True)

    pipeline = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    pipeline.to(device)
    data = load_h5(args.h5_file, device)

    variants = _get_camera_variants()
    n = min(args.num_renders, len(variants)) if args.num_renders is not None else len(variants)
    n = max(1, n)
    cache = None if args.no_cache else ViewIndependentCache(max_entries=args.vi_cache_max_entries)
    results = []  # [(pass_idx, name, hit, elapsed_sec, key_short), ...]

    log("")
    log(f"共 {n} 轮渲染，同场景多组相机参数 (FOV / Orbit Y / 距离)，VI 缓存: {'禁用' if args.no_cache else '启用 (max_entries=%d)' % args.vi_cache_max_entries}")
    if not args.no_cache:
        log("  预期：第 1 次 miss，第 2～%d 次 hit" % n)
    log("")

    for i in range(n):
        var = variants[i]
        name = var.get("name", str(i + 1))
        c2w, fov = _apply_camera_variant(
            data["c2w"], data["fov"], var, device, dtype
        )

        log("========== 第 %d 次渲染 [%s] ==========" % (i + 1, name))
        log(f"  resolution={args.resolution}, precision={args.precision}, device={device}")
        if cache is not None:
            log(f"  渲染前缓存: 条目数={cache.stats()['entries']}, hits={cache.hits}, misses={cache.misses}")
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        out = pipeline.render(
            triangles=data["triangles"],
            texture=data["texture"],
            mask=data["mask"],
            vn=data["vn"],
            c2w=c2w,
            fov=fov,
            resolution=args.resolution,
            torch_dtype=dtype,
            vi_cache=cache,
            return_vi_cache_info=True,
        )
        rendered_imgs, info = out
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - t0
        key_short = _short_key(info.get("vi_cache_key") or "")
        results.append((i + 1, name, info["vi_cache_hit"], elapsed, key_short))
        log(f"  结果: vi_cache_hit={info['vi_cache_hit']}, vi_cache_key={key_short}")
        log(f"  耗时: {elapsed:.4f} s")
        if cache is not None:
            _print_cache_detail(cache)
        if args.save_frames > 0 and i < args.save_frames and args.output_dir:
            import imageio
            frame = rendered_imgs[0, 0].cpu().numpy().astype(np.float32)
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)
            path = os.path.join(args.output_dir, "frame_%02d_%s.png" % (i + 1, name.replace(" ", "_").replace("/", "_")))
            imageio.v3.imwrite(path, frame)
            log(f"  已保存: {path}")
        log("")

    # ---------- 汇总 ----------
    print("========== 汇总 ==========")
    for pass_idx, name, hit, elapsed, key_short in results:
        print(f"  第 {pass_idx:2d} 次 [{name:30s}] vi_cache_hit={hit}, 耗时={elapsed:.4f} s")
    if cache is not None:
        print("  缓存统计:", cache.stats())
    else:
        print("  (未使用 VI 缓存)")

    hits = sum(1 for _, _, hit, _, _ in results if hit)
    misses = sum(1 for _, _, hit, _, _ in results if not hit)
    if cache is not None and n >= 2 and misses == 1 and hits == n - 1:
        print("  验证结果: 通过 (第 1 次 miss，后续均 hit)")
    elif cache is not None and n >= 2:
        print("  验证结果: 异常 (预期仅第 1 次 miss，实际 miss=%d, hit=%d)" % (misses, hits))


if __name__ == "__main__":
    main()

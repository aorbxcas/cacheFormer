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
    # 只改第一个样本的 c2w
    c = c2w[0]
    if "orbit_y_deg" in variant:
        R_y = _rotation_y_3x3(variant["orbit_y_deg"], c.device, c.dtype)
        c2w[0, :3, :3] = R_y @ c[:3, :3]
        c2w[0, :3, 3] = R_y @ c[:3, 3]
    if "distance_scale" in variant:
        c2w[0, :3, 3] = c2w[0, :3, 3] * variant["distance_scale"]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", type=str, required=True)
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/renderformer-v1.1-swin-large",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_renders", type=int, default=None, metavar="N",
                        help="渲染次数，默认使用全部相机变体（FOV/轨道/距离等）；指定 N 时仅用前 N 个变体")
    parser.add_argument("--vi_cache_max_entries", type=int, default=4,
                        help="VI cache 最大条目数，默认 4")
    parser.add_argument("--quiet", action="store_true", help="只打印最终结果，不打印每次渲染的详细缓存信息")
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
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipeline = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    pipeline.to(device)
    data = load_h5(args.h5_file, device)

    variants = _get_camera_variants()
    n = min(args.num_renders, len(variants)) if args.num_renders is not None else len(variants)
    n = max(1, n)
    cache = ViewIndependentCache(max_entries=args.vi_cache_max_entries)
    results = []  # [(pass_idx, name, hit, elapsed_sec, key_short), ...]

    log("")
    log(f"共 {n} 轮渲染，同场景多组相机参数 (FOV / Orbit Y / 距离)，预期：第 1 次 miss，第 2～{n} 次 hit")
    log("")

    for i in range(n):
        var = variants[i]
        name = var.get("name", str(i + 1))
        c2w, fov = _apply_camera_variant(
            data["c2w"], data["fov"], var, device, dtype
        )

        log("========== 第 %d 次渲染 [%s] ==========" % (i + 1, name))
        log(f"  resolution={args.resolution}, dtype={dtype}, device={device}")
        log(f"  渲染前缓存: 条目数={cache.stats()['entries']}, hits={cache.hits}, misses={cache.misses}")
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        _, info = pipeline.render(
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
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - t0
        key_short = _short_key(info.get("vi_cache_key") or "")
        results.append((i + 1, name, info["vi_cache_hit"], elapsed, key_short))
        log(f"  结果: vi_cache_hit={info['vi_cache_hit']}, vi_cache_key={key_short}")
        log(f"  耗时: {elapsed:.4f} s")
        _print_cache_detail(cache)
        log("")

    # ---------- 汇总 ----------
    print("========== 汇总 ==========")
    for pass_idx, name, hit, elapsed, key_short in results:
        print(f"  第 {pass_idx:2d} 次 [{name:30s}] vi_cache_hit={hit}, 耗时={elapsed:.4f} s")
    print("  缓存统计:", cache.stats())

    hits = sum(1 for _, _, hit, _, _ in results if hit)
    misses = sum(1 for _, _, hit, _, _ in results if not hit)
    if n >= 2 and misses == 1 and hits == n - 1:
        print("  验证结果: 通过 (第 1 次 miss，后续均 hit)")
    elif n >= 2:
        print("  验证结果: 异常 (预期仅第 1 次 miss，实际 miss=%d, hit=%d)" % (misses, hits))


if __name__ == "__main__":
    main()

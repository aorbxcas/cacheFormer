"""
同一 HDF5 场景：原生（全量 VI + 每视角 VD）与缓存近似路径对比。

- 原生：一次 forward_vi_only 得 vi_gold，再每帧 render_from_vi_seq(vi_gold, …)。静态几何下与 pipeline.render
  多视角一次前向在数学上等价，但按视角拆分可显著降低峰值显存（避免多视角同时占满 VD）。
- 缓存：同一 vi_gold 基准；每帧 approx_vi_local_window + render_from_vi_seq(vi_apx)；vi_cache 链式更新。
- 输出：控制台表、CSV、native/ 与 cache/ 下的 EXR/PNG。

显存不足：降低 --res、减少 --frames，或设置环境变量 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。

attention 在脚本内固定为 sdpa，无需 ATTN_IMPL。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, List

import imageio
import numpy as np
import torch

import renderformer.layers.attention as _rf_attn

_rf_attn.ATTN = "sdpa"

from renderformer import RenderFormerRenderingPipeline
from renderformer.approx.local_window_vi import approx_vi_local_window

try:
    from simple_ocio import ToneMapper
except ImportError:
    ToneMapper = None  # type: ignore


def look_at_to_c2w(
    camera_position: list,
    target_position: list | None = None,
    up_dir: list | None = None,
) -> np.ndarray:
    if target_position is None:
        target_position = [0.0, 0.0, 0.0]
    if up_dir is None:
        up_dir = [0.0, 0.0, 1.0]
    camera_direction = np.array(camera_position) - np.array(target_position)
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    camera_right = np.cross(np.array(up_dir), camera_direction)
    camera_right = camera_right / np.linalg.norm(camera_right)
    camera_up = np.cross(camera_direction, camera_right)
    camera_up = camera_up / np.linalg.norm(camera_up)
    rotation_transform = np.zeros((4, 4))
    rotation_transform[0, :3] = camera_right
    rotation_transform[1, :3] = camera_up
    rotation_transform[2, :3] = camera_direction
    rotation_transform[-1, -1] = 1.0
    translation_transform = np.eye(4)
    translation_transform[:3, -1] = -np.array(camera_position)
    look_at_transform = np.matmul(rotation_transform, translation_transform)
    return np.linalg.inv(look_at_transform)


def load_h5(path: Path) -> dict[str, torch.Tensor]:
    import h5py

    with h5py.File(path, "r") as f:
        triangles = torch.from_numpy(np.array(f["triangles"]).astype(np.float32))
        num_tris = triangles.shape[0]
        texture = torch.from_numpy(np.array(f["texture"]).astype(np.float32))
        mask = torch.ones(num_tris, dtype=torch.bool)
        vn = torch.from_numpy(np.array(f["vn"]).astype(np.float32))
        c2w = torch.from_numpy(np.array(f["c2w"]).astype(np.float32))
        fov = torch.from_numpy(np.array(f["fov"]).astype(np.float32))
    return {
        "triangles": triangles,
        "texture": texture,
        "mask": mask,
        "vn": vn,
        "c2w": c2w,
        "fov": fov,
    }


def preprocess_texture(pipeline: RenderFormerRenderingPipeline, texture: torch.Tensor) -> torch.Tensor:
    t = texture
    if pipeline.config.texture_encode_patch_size == 1 and t.dim() == 5:
        t = t[:, :, :, 0, 0]
    if not pipeline.config.use_ldr:
        t = t.clone()
        t[:, :, -3:] = torch.log10(t[:, :, -3:] + 1.0)
    return t


def build_frame_cameras(
    c2w: torch.Tensor,
    fov: torch.Tensor,
    synthetic_frames: int,
    orbit_arc_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    V = c2w.shape[0]
    fov_np = fov.reshape(-1).cpu().numpy()

    if V == 1 and synthetic_frames > 1:
        c0 = c2w[0].cpu().numpy()
        pos = c0[:3, 3].copy()
        f0 = float(fov_np[0])
        look_at = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        mats: list[np.ndarray] = []
        fovs: list[float] = []
        for i in range(synthetic_frames):
            ang = orbit_arc_deg * (np.pi / 180.0) * (i / max(synthetic_frames - 1, 1))
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            pos_i = R @ pos.astype(np.float64)
            mats.append(look_at_to_c2w(pos_i.tolist(), look_at.tolist(), up.tolist()))
            fovs.append(f0)
        c2w_out = torch.from_numpy(np.stack(mats, axis=0)).float()
        fov_out = torch.tensor(fovs, dtype=torch.float32)
        return c2w_out, fov_out

    n = min(V, synthetic_frames) if synthetic_frames > 0 else V
    return c2w[:n].clone(), fov.reshape(-1)[:n].clone()


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean().item())


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(((a - b) ** 2).mean()).item())


def psnr_hdr(mse_v: float, peak: float) -> float:
    if mse_v <= 1e-20:
        return float("inf")
    return float(10.0 * np.log10((peak * peak) / mse_v))


def union_window_indices(miss_list: List[int], num_tri: int, R: int) -> List[int]:
    win: set[int] = set()
    for i in miss_list:
        lo = max(0, i - R)
        hi = min(num_tri - 1, i + R)
        win.update(range(lo, hi + 1))
    return sorted(win)


def main() -> None:
    epilog = """
示例:
  python test_h5_native_vs_cache.py -i examples/cbox.h5 -o output/h5_cmp --frames 8 --res 512

若无 h5:
  python scene_processor/convert_scene.py examples/cbox.json --output_h5_path examples/cbox.h5
"""
    p = argparse.ArgumentParser(
        description="同一 H5：原生 pipeline.render 与缓存近似路径对比，输出指标与图像。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument("-i", "--input", "--h5_file", dest="h5_file", type=str, default="examples/cbox.h5")
    p.add_argument("-o", "--output", "--output_dir", dest="output_dir", type=str, default="output/h5_native_vs_cache")
    p.add_argument("--model", "--model_id", dest="model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp16")
    p.add_argument("--res", "--resolution", dest="resolution", type=int, default=256)
    p.add_argument("--tone", "--tone_mapper", dest="tone_mapper", type=str, default="agx", choices=["none", "agx", "filmic", "pbr_neutral"])
    p.add_argument("--frames", "--synthetic_frames", dest="synthetic_frames", type=int, default=8)
    p.add_argument("--orbit", "--orbit_arc_deg", dest="orbit_arc_deg", type=float, default=40.0)
    p.add_argument("--miss", "--miss_ratio", dest="miss_ratio", type=float, default=0.08)
    p.add_argument("-R", "--window_radius", type=int, default=4)
    p.add_argument("-K", "--num_refiner_layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.synthetic_frames < 1:
        print("--frames 必须 >= 1", file=sys.stderr)
        sys.exit(1)

    h5_path = Path(args.h5_file)
    if not h5_path.is_file():
        print(f"找不到 H5: {h5_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.output_dir)
    dir_native = out_root / "native"
    dir_cache = out_root / "cache"
    dir_native.mkdir(parents=True, exist_ok=True)
    dir_cache.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    torch_dtype = (
        torch.float16
        if args.precision == "fp16"
        else torch.bfloat16
        if args.precision == "bf16"
        else torch.float32
    )
    if device.type == "mps":
        torch_dtype = torch.float32

    pipeline = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    pipeline.to(device)
    model = pipeline.model

    if device == torch.device("cuda") and os.name == "posix":
        try:
            from renderformer_liger_kernel import apply_kernels

            apply_kernels(pipeline.model)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except ImportError:
            pass

    raw = load_h5(h5_path)
    triangles = raw["triangles"].unsqueeze(0).to(device)
    texture_raw = raw["texture"].unsqueeze(0).to(device)
    mask = raw["mask"].unsqueeze(0).to(device)
    vn = raw["vn"].unsqueeze(0).to(device)
    c2w_stored = raw["c2w"].to(device)
    fov_stored = raw["fov"].to(device)

    texture_proc = preprocess_texture(pipeline, texture_raw)

    c2w_frames, fov_frames = build_frame_cameras(
        c2w_stored,
        fov_stored,
        synthetic_frames=args.synthetic_frames,
        orbit_arc_deg=args.orbit_arc_deg,
    )
    c2w_frames = c2w_frames.to(device)
    fov_frames = fov_frames.to(device)
    num_frames = c2w_frames.shape[0]

    tone_mapper = None
    if args.tone_mapper != "none" and ToneMapper is not None:
        tm = args.tone_mapper
        if tm == "pbr_neutral":
            tm = "Khronos PBR Neutral"
        tone_mapper = ToneMapper(tm)

    stem = h5_path.stem
    skip = model.skip_token_num
    num_tri = mask.shape[1]

    # —— 一次全量 VI（原生与缓存共用基准）——
    t_vi0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch_dtype):
        vi_gold = model.forward_vi_only(
            triangles.reshape(1, -1, 9),
            texture_proc,
            mask,
            vn.reshape(1, -1, 9),
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_vi_full = time.perf_counter() - t_vi0

    _, valid_mask_padded, _ = model.construct_seq(
        triangles.reshape(1, -1, 9),
        texture_proc,
        mask,
        vn.reshape(1, -1, 9),
    )

    vi_cache = vi_gold.clone()
    rows: list[dict[str, Any]] = []
    t_native_vd_sum = 0.0
    t_cache_vi_sum = 0.0
    t_cache_vd_sum = 0.0

    for frame_idx in range(num_frames):
        c2w_1 = c2w_frames[frame_idx : frame_idx + 1].unsqueeze(0)
        fov_1 = fov_frames[frame_idx : frame_idx + 1].unsqueeze(0).unsqueeze(-1)

        # 原生：仅 VD（vi_gold），单视角峰值显存
        t_n0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch_dtype):
            img_native = pipeline.render_from_vi_seq(
                vi_gold,
                valid_mask_padded,
                triangles,
                mask,
                c2w_1,
                fov_1,
                resolution=args.resolution,
                torch_dtype=torch_dtype,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_nat_vd = time.perf_counter() - t_n0
        t_native_vd_sum += t_nat_vd

        n_miss = max(1, int(num_tri * args.miss_ratio))
        perm = torch.randperm(num_tri, device=device, generator=gen)[:n_miss]
        miss = perm.long()
        miss_list = sorted({int(x) for x in miss.tolist()})
        union_win = union_window_indices(miss_list, num_tri, args.window_radius)

        t1 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch_dtype):
            vi_apx = approx_vi_local_window(
                model,
                triangles.reshape(1, -1, 9),
                texture_proc,
                mask,
                vn.reshape(1, -1, 9),
                vi_cache,
                miss,
                window_radius=args.window_radius,
                num_refiner_layers=args.num_refiner_layers,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_apx = time.perf_counter() - t1
        t_cache_vi_sum += t_apx

        t2 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch_dtype):
            img_cache = pipeline.render_from_vi_seq(
                vi_apx,
                valid_mask_padded,
                triangles,
                mask,
                c2w_1,
                fov_1,
                resolution=args.resolution,
                torch_dtype=torch_dtype,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_vd_c = time.perf_counter() - t2
        t_cache_vd_sum += t_vd_c

        vi_cache = vi_apx.detach()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        hdr_n = img_native[0, 0].float().cpu()
        hdr_c = img_cache[0, 0].float().cpu()
        img_mse = mse(hdr_n, hdr_c)
        img_rmse = rmse(hdr_n, hdr_c)
        peak = max(float(hdr_n.max()), float(hdr_c.max()), 1.0)
        img_psnr = psnr_hdr(img_mse, peak)

        miss_idx = skip + miss
        mse_vi_all = mse(vi_apx, vi_gold)
        mse_vi_miss = mse(vi_apx[:, miss_idx, :], vi_gold[:, miss_idx, :])
        rel_l2_miss = float(
            (vi_apx[:, miss_idx, :] - vi_gold[:, miss_idx, :]).norm()
            / (vi_gold[:, miss_idx, :].norm() + 1e-8)
        )

        def save_pair(tag_dir: Path, prefix: str, hdr: torch.Tensor) -> tuple[Path, Path]:
            exr_p = tag_dir / f"{stem}_f{frame_idx:03d}_{prefix}.exr"
            png_p = tag_dir / f"{stem}_f{frame_idx:03d}_{prefix}.png"
            imageio.v3.imwrite(exr_p, hdr.numpy().astype(np.float32))
            if tone_mapper is not None:
                ldr = tone_mapper.hdr_to_ldr(hdr.numpy().astype(np.float32))
            else:
                ldr = np.clip(hdr.numpy(), 0, 1)
            imageio.v3.imwrite(png_p, (ldr * 255).astype(np.uint8))
            return exr_p, png_p

        exr_n, png_n = save_pair(dir_native, "native", hdr_n)
        exr_c, png_c = save_pair(dir_cache, "cache", hdr_c)

        rows.append(
            {
                "frame": frame_idx,
                "n_miss": len(miss_list),
                "union_win_tris": len(union_win),
                "t_vd_native_s": t_nat_vd,
                "t_apx_vi_s": t_apx,
                "t_vd_cache_s": t_vd_c,
                "mse_vi_vs_gold": mse_vi_all,
                "mse_vi_miss": mse_vi_miss,
                "rel_l2_vi_miss": rel_l2_miss,
                "hdr_mse_native_vs_cache": img_mse,
                "hdr_rmse": img_rmse,
                "psnr_hdr": img_psnr,
                "native_exr": str(exr_n),
                "native_png": str(png_n),
                "cache_exr": str(exr_c),
                "cache_png": str(png_c),
            }
        )

    csv_path = out_root / "h5_native_vs_cache_metrics.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # —— 打印 ——
    print("\n### 汇总\n")
    print(f"| H5 | {h5_path} |")
    print(f"| 帧数 | {num_frames} |")
    print(f"| 全量 VI 一次 (s) | {t_vi_full:.4f} |")
    print(f"| 原生 Σ仅VD vi_gold (s) | {t_native_vd_sum:.4f} |")
    print(f"| 缓存 Σ近似VI (s) | {t_cache_vi_sum:.4f} |")
    print(f"| 缓存 Σ仅VD (s) | {t_cache_vd_sum:.4f} |")

    print("\n### 逐帧：原生 vs 缓存图像 / VI\n")
    hdr = (
        "| frame | n_miss | union_win | t_vd_native | t_apx_vi | t_vd_cache | mse_vi | mse_vi_miss | "
        "hdr_mse(n,c) | rmse | PSNR |"
    )
    print(hdr)
    print("|" + "|".join(["---"] * 11) + "|")
    for r in rows:
        print(
            f"| {r['frame']} | {r['n_miss']} | {r['union_win_tris']} | "
            f"{r['t_vd_native_s']:.4f} | {r['t_apx_vi_s']:.4f} | {r['t_vd_cache_s']:.4f} | "
            f"{r['mse_vi_vs_gold']:.4e} | {r['mse_vi_miss']:.4e} | "
            f"{r['hdr_mse_native_vs_cache']:.4e} | {r['hdr_rmse']:.4e} | {r['psnr_hdr']:.2f} |"
        )

    print(f"\n**CSV**: {csv_path.resolve()}")
    print(f"**原生图像目录**: {dir_native.resolve()}")
    print(f"**缓存图像目录**: {dir_cache.resolve()}\n")


if __name__ == "__main__":
    main()

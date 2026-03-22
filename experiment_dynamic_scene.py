"""
动态场景渲染对比实验：① baseline ② block_cache ③ temporal_vi，导出每帧 PNG/EXR 与横向拼接对比图。

两种数据来源：
  A) 文件夹：多帧 H5 视为「真实动态」序列（如视频数据）。
  B) 单 H5 + 合成扰动：同一场景每帧对顶点/纹理等做可控扰动，模拟几何或材质渐变。

示例（仓库根目录下执行；或 --h5_folder 用绝对路径）：
  python experiment_dynamic_scene.py --h5_folder video-data/teaser-scenes/cbox-roughness --max_frames 12 --output_dir output/exp_dynamic_real
  # Windows 本仓库：C:\Users\zhangleipa\.openclaw\workspace\renderformer\video-data\teaser-scenes\cbox-roughness
  python experiment_dynamic_scene.py --h5_file tmp/cbox/cbox.h5 --num_frames 20 --perturb sliding_block --output_dir output/exp_dynamic_syn
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from natsort import natsorted

from renderformer import RenderFormerRenderingPipeline
from renderformer.cache import BlockCache
from renderformer.temporal_vi import TemporalVIConfig, TemporalVIState

try:
    from simple_ocio import ToneMapper
except ImportError:
    ToneMapper = None  # type: ignore


def load_h5_item(file_path: str, padding_length: Optional[int] = None) -> Dict[str, Any]:
    with h5py.File(file_path, "r") as f:
        triangles = torch.from_numpy(np.array(f["triangles"], dtype=np.float32)).float()
        num_tris = triangles.shape[0]
        texture = torch.from_numpy(np.array(f["texture"], dtype=np.float32)).float()
        vn = torch.from_numpy(np.array(f["vn"], dtype=np.float32)).float()
        c2w = torch.from_numpy(np.array(f["c2w"], dtype=np.float32)).float()
        fov = torch.from_numpy(np.array(f["fov"], dtype=np.float32)).float()
        if padding_length is not None:
            pad = padding_length - num_tris
            triangles = torch.cat((triangles, torch.zeros(pad, *triangles.shape[1:])), dim=0)
            texture = torch.cat((texture, torch.zeros(pad, *texture.shape[1:])), dim=0)
            vn = torch.cat((vn, torch.zeros(pad, *vn.shape[1:])), dim=0)
            mask = torch.zeros(padding_length, dtype=torch.bool)
            mask[:num_tris] = True
        else:
            mask = torch.ones(num_tris, dtype=torch.bool)
    return {
        "triangles": triangles,
        "texture": texture,
        "mask": mask,
        "vn": vn,
        "c2w": c2w,
        "fov": fov,
        "file_path": file_path,
        "num_tris": num_tris,
    }


def _perturb_vertex_noise(
    item: Dict[str, Any], rng: np.random.Generator, std: float
) -> Dict[str, Any]:
    t = item["triangles"].clone()
    m = item["mask"]
    noise = torch.from_numpy(rng.standard_normal(t.shape).astype(np.float32)) * std
    w = m[:, None, None].float()
    t = t + noise * w
    out = {**item, "triangles": t}
    return out


def _perturb_global_translate(
    item: Dict[str, Any], step: float, axis: np.ndarray
) -> Dict[str, Any]:
    """每帧沿固定轴平移一步（累积动态）。"""
    t = item["triangles"].clone()
    m = item["mask"]
    delta = torch.from_numpy(axis.astype(np.float32) * float(step))
    t = t + delta * m[:, None, None].float()
    return {**item, "triangles": t}


def _perturb_sliding_block(
    item: Dict[str, Any], block_size: int, shift: float
) -> Dict[str, Any]:
    """每帧将前 block_size 个有效三角沿 +X 推进一步（模拟局部刚体平移）。"""
    t = item["triangles"].clone()
    m = item["mask"]
    n_tot = t.shape[0]
    if n_tot == 0:
        return {**item, "triangles": t}
    end = min(max(1, block_size), n_tot)
    vec = torch.tensor([shift, shift * 0.25, 0.0], dtype=t.dtype)
    for i in range(end):
        if m[i]:
            t[i] = t[i] + vec
    return {**item, "triangles": t}


def _perturb_texture_jitter(
    item: Dict[str, Any], rng: np.random.Generator, std: float
) -> Dict[str, Any]:
    tx = item["texture"].clone()
    m = item["mask"]
    noise = torch.from_numpy(rng.standard_normal(tx.shape).astype(np.float32)) * std
    if tx.dim() == 3:
        w = m[:, None, None].float()
    else:
        w = m[:, None, None, None, None].float()
    tx = tx + noise * w
    tx = torch.clamp(tx, min=0.0)
    return {**item, "texture": tx}


def _perturb_combo(
    item: Dict[str, Any],
    rng: np.random.Generator,
    noise_std: float,
    translate_step: float,
    translate_axis: np.ndarray,
    block_size: int,
    shift: float,
) -> Dict[str, Any]:
    x = _perturb_vertex_noise(item, rng, noise_std * 0.4)
    x = _perturb_global_translate(x, translate_step, translate_axis)
    x = _perturb_sliding_block(x, block_size, shift * 0.5)
    return x


def clone_item(item: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        else:
            out[k] = v
    return out


def apply_perturbation(
    base: Dict[str, Any],
    mode: str,
    rng: np.random.Generator,
    noise_std: float,
    translate_step: float,
    translate_axis: np.ndarray,
    block_size: int,
    sliding_shift: float,
) -> Dict[str, Any]:
    if mode == "none":
        return clone_item(base)
    if mode == "vertex_noise":
        return _perturb_vertex_noise(base, rng, noise_std)
    if mode == "global_translate":
        return _perturb_global_translate(base, translate_step, translate_axis)
    if mode == "sliding_block":
        return _perturb_sliding_block(base, block_size, sliding_shift)
    if mode == "texture_jitter":
        return _perturb_texture_jitter(base, rng, noise_std)
    if mode == "combo":
        return _perturb_combo(
            base, rng, noise_std, translate_step, translate_axis, block_size, sliding_shift
        )
    raise ValueError(f"Unknown perturb: {mode}")


def hdr_to_uint8(
    hdr: np.ndarray, tone_mapper: Optional[Any]
) -> np.ndarray:
    """hdr [H,W,3] float -> uint8 [H,W,3]"""
    if tone_mapper is not None:
        ldr = tone_mapper.hdr_to_ldr(hdr.astype(np.float32))
    else:
        ldr = np.clip(hdr, 0.0, 1.0)
    return (np.clip(ldr, 0, 1) * 255.0).astype(np.uint8)


def hstack_ldr(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, gap: int = 4, gap_value: int = 40
) -> np.ndarray:
    """三图等高横向拼接，高度取最大并居中 pad。"""
    h_max = max(a.shape[0], b.shape[0], c.shape[0])
    wtot = a.shape[1] + b.shape[1] + c.shape[1] + 2 * gap

    def pad_y(x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        if h == h_max:
            return x
        pad_t = (h_max - h) // 2
        pad_b = h_max - h - pad_t
        return np.pad(x, ((pad_t, pad_b), (0, 0), (0, 0)), constant_values=gap_value)

    pa, pb, pc = pad_y(a), pad_y(b), pad_y(c)
    gap_arr = np.full((h_max, gap, 3), gap_value, dtype=np.uint8)
    return np.concatenate([pa, gap_arr, pb, gap_arr, pc], axis=1)


def image_metrics_np(ref: np.ndarray, oth: np.ndarray) -> Dict[str, float]:
    r = ref.reshape(-1).astype(np.float64)
    o = oth.reshape(-1).astype(np.float64)
    d = o - r
    mse = float((d * d).mean())
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "max_abs": float(np.abs(d).max())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic scene: compare ①②③ and export images")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5_folder", type=str, help="Real multi-frame sequence (*.h5)")
    src.add_argument("--h5_file", type=str, help="Base scene; use with --num_frames + --perturb")
    parser.add_argument("--max_frames", type=int, default=None, help="Cap frames (folder mode)")
    parser.add_argument("--num_frames", type=int, default=16, help="Synthetic frame count (file mode)")
    parser.add_argument(
        "--perturb",
        type=str,
        default="vertex_noise",
        choices=[
            "none",
            "vertex_noise",
            "global_translate",
            "sliding_block",
            "texture_jitter",
            "combo",
        ],
        help="Synthetic perturbation per frame (ignored in folder mode)",
    )
    parser.add_argument("--noise_std", type=float, default=0.003, help="vertex_noise / texture jitter scale")
    parser.add_argument("--translate_step", type=float, default=0.002, help="global_translate step per frame index")
    parser.add_argument("--sliding_shift", type=float, default=0.008, help="sliding_block shift scale")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--padding_length", type=int, default=None)
    parser.add_argument("--max_cache_entries", type=int, default=50000)
    parser.add_argument("--full_every_k", type=int, default=8)
    parser.add_argument("--max_consecutive_approx", type=int, default=32)
    parser.add_argument("--changed_block_ratio_threshold", type=float, default=None)
    parser.add_argument("--approx_mode", type=str, choices=["level0", "level1"], default="level0")
    parser.add_argument("--blend_alpha", type=float, default=0.15)
    parser.add_argument("--tone_mapper", type=str, choices=["none", "agx", "filmic", "pbr_neutral"], default="agx")
    parser.add_argument("--views", type=str, default="0", help="Comma-separated view indices to export, e.g. 0 or 0,1")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.h5_folder:
        file_list = natsorted(glob.glob(os.path.join(args.h5_folder, "*.h5")))
        if args.max_frames is not None:
            file_list = file_list[: args.max_frames]
        if not file_list:
            print("No h5 files found.")
            return
        mode_name = "folder_sequence"
        perturb = "none"
    else:
        if not os.path.isfile(args.h5_file):
            print(f"Missing file: {args.h5_file}")
            return
        base_item = load_h5_item(args.h5_file, args.padding_length)
        mode_name = "synthetic"
        perturb = args.perturb
        axis = rng.standard_normal(3).astype(np.float32)
        axis = axis / (np.linalg.norm(axis) + 1e-8)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if device.type == "mps":
        args.precision = "fp32"

    pipeline = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    if device.type == "cuda" and os.name == "posix":
        try:
            from renderformer_liger_kernel import apply_kernels

            apply_kernels(pipeline.model)
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    pipeline.to(device)

    dtype = (
        torch.float16
        if args.precision == "fp16"
        else torch.bfloat16
        if args.precision == "bf16"
        else torch.float32
    )

    tv_cfg = TemporalVIConfig(
        full_every_k=args.full_every_k,
        max_consecutive_approx=args.max_consecutive_approx,
        changed_block_ratio_threshold=args.changed_block_ratio_threshold,
        approx_mode=args.approx_mode,
        blend_alpha=args.blend_alpha,
    )
    block_cache_bc = BlockCache(max_entries=args.max_cache_entries)
    block_cache_tv = BlockCache(max_entries=args.max_cache_entries)
    tv_state = TemporalVIState()

    tone_mapper = None
    if args.tone_mapper != "none" and ToneMapper is not None:
        tname = "Khronos PBR Neutral" if args.tone_mapper == "pbr_neutral" else args.tone_mapper
        tone_mapper = ToneMapper(tname)
    elif args.tone_mapper != "none" and ToneMapper is None:
        print("simple_ocio not installed; using clamp for LDR export.")

    view_indices = [int(x.strip()) for x in args.views.split(",") if x.strip()]

    meta = {
        "mode": mode_name,
        "perturb": perturb,
        "noise_std": args.noise_std,
        "translate_step": args.translate_step,
        "sliding_shift": args.sliding_shift,
        "block_size": args.block_size,
        "seed": args.seed,
        "model_id": args.model_id,
        "resolution": args.resolution,
        "temporal": {
            "full_every_k": args.full_every_k,
            "approx_mode": args.approx_mode,
        },
    }
    if mode_name == "synthetic":
        meta["translate_axis"] = axis.tolist()

    with open(os.path.join(args.output_dir, "experiment_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    records: List[Dict[str, Any]] = []
    print(f"Exporting to {args.output_dir}  (mode={mode_name} perturb={perturb})")

    running: Optional[Dict[str, Any]] = None
    for frame_idx in range(len(file_list) if mode_name == "folder_sequence" else args.num_frames):
        if mode_name == "folder_sequence":
            path = file_list[frame_idx]
            item = load_h5_item(path, args.padding_length)
            tag = f"frame_{frame_idx:04d}"
        else:
            path = args.h5_file
            if running is None:
                running = clone_item(base_item)
            else:
                running = apply_perturbation(
                    running,
                    perturb,
                    rng,
                    args.noise_std,
                    args.translate_step,
                    axis,
                    args.block_size,
                    args.sliding_shift,
                )
            item = running
            tag = f"syn_{frame_idx:04d}"

        triangles = item["triangles"].unsqueeze(0).to(device)
        texture = item["texture"].unsqueeze(0).to(device)
        mask = item["mask"].unsqueeze(0).to(device)
        vn = item["vn"].unsqueeze(0).to(device)
        c2w = item["c2w"].unsqueeze(0).to(device)
        fov = item["fov"].unsqueeze(0).unsqueeze(-1).to(device)

        with torch.no_grad():
            img_b = pipeline.render(
                triangles, texture, mask, vn, c2w, fov, resolution=args.resolution, torch_dtype=dtype
            )
            img_bc, _ = pipeline.render_with_block_cache(
                triangles,
                texture,
                mask,
                vn,
                c2w,
                fov,
                block_cache_bc,
                block_size=args.block_size,
                resolution=args.resolution,
                torch_dtype=dtype,
                verbose=False,
            )
            img_tv, fl_tv = pipeline.render_with_temporal_vi(
                triangles,
                texture,
                mask,
                vn,
                c2w,
                fov,
                block_cache_tv,
                tv_state,
                tv_cfg,
                block_size=args.block_size,
                resolution=args.resolution,
                torch_dtype=dtype,
                verbose=False,
            )

        nb = img_b[0].cpu().numpy().astype(np.float32)
        nbc = img_bc[0].cpu().numpy().astype(np.float32)
        ntv = img_tv[0].cpu().numpy().astype(np.float32)

        rec: Dict[str, Any] = {
            "frame": frame_idx,
            "tag": tag,
            "source": path if mode_name == "folder_sequence" else args.h5_file,
            "vi_path": fl_tv.get("vi_path"),
            "force_reason": fl_tv.get("force_reason"),
        }
        for vi in view_indices:
            if vi >= nb.shape[0]:
                continue
            hb, hbc, htv = nb[vi], nbc[vi], ntv[vi]
            m_bc = image_metrics_np(hb, hbc)
            m_tv = image_metrics_np(hb, htv)
            rec[f"view{vi}_mse_bc"] = m_bc["mse"]
            rec[f"view{vi}_mse_tv"] = m_tv["mse"]
            rec[f"view{vi}_rmse_tv"] = m_tv["rmse"]

            ub = hdr_to_uint8(hb, tone_mapper)
            ubc = hdr_to_uint8(hbc, tone_mapper)
            utv = hdr_to_uint8(htv, tone_mapper)
            prefix = os.path.join(args.output_dir, f"{tag}_view{vi}")
            iio.imwrite(f"{prefix}_01_baseline.png", ub)
            iio.imwrite(f"{prefix}_02_block_cache.png", ubc)
            iio.imwrite(f"{prefix}_03_temporal_vi.png", utv)
            iio.imwrite(f"{prefix}_00_compare_row.png", hstack_ldr(ub, ubc, utv))
            iio.imwrite(f"{prefix}_01_baseline.exr", hb.astype(np.float32))
            iio.imwrite(f"{prefix}_02_block_cache.exr", hbc.astype(np.float32))
            iio.imwrite(f"{prefix}_03_temporal_vi.exr", htv.astype(np.float32))

        records.append(rec)
        print(
            f"  [{frame_idx:04d}] {tag}  VI={fl_tv.get('vi_path')} {fl_tv.get('force_reason')}  "
            f"mse_tv_vs_base={rec.get('view0_mse_tv', 0):.5e}"
        )

    with open(os.path.join(args.output_dir, "per_frame_metrics.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done. {len(records)} frames, metrics in per_frame_metrics.jsonl")
    print(f"命名: *_01_baseline / *_02_block_cache / *_03_temporal_vi / *_00_compare_row (左→右 ①②③)")


if __name__ == "__main__":
    main()

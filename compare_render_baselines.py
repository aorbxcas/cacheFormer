
from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from natsort import natsorted

from renderformer import RenderFormerRenderingPipeline
from renderformer.cache import BlockCache
from renderformer.temporal_vi import TemporalVIConfig, TemporalVIState


def load_h5_item(file_path: str, padding_length: Optional[int] = None) -> Dict[str, Any]:
    with h5py.File(file_path, "r") as f:
        triangles = torch.from_numpy(np.array(f["triangles"])).float()
        num_tris = triangles.shape[0]
        texture = torch.from_numpy(np.array(f["texture"])).float()
        vn = torch.from_numpy(np.array(f["vn"])).float()
        c2w = torch.from_numpy(np.array(f["c2w"])).float()
        fov = torch.from_numpy(np.array(f["fov"])).float()
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_render(
    device: torch.device,
    fn,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, float]:
    _sync(device)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _sync(device)
    return out, time.perf_counter() - t0


def image_metrics(
    ref: torch.Tensor,
    other: torch.Tensor,
) -> Dict[str, float]:
    """相对 ref 的误差指标（linear HDR 全张量）。"""
    r = ref.detach().float().cpu().reshape(-1)
    o = other.detach().float().cpu().reshape(-1)
    diff = o - r
    mse = float((diff * diff).mean().item())
    rmse = float(np.sqrt(mse))
    mx = float(diff.abs().max().item())
    mean_abs_r = float(r.abs().mean().item())
    std_r = float(r.std().item())
    # 相对 RMSE：相对 |ref| 均值，避免除零
    rel_rmse_pct = float(100.0 * rmse / (mean_abs_r + 1e-12))
    # 粗略「动态范围」相对误差：max_abs / (|ref|max + eps)
    rmax = float(r.abs().max().item()) + 1e-12
    max_rel_pct = float(100.0 * mx / rmax)
    return {
        "mse": mse,
        "rmse": rmse,
        "max_abs": mx,
        "mean_abs_ref": mean_abs_r,
        "std_ref": std_r,
        "rel_rmse_pct": rel_rmse_pct,
        "max_abs_over_refmax_pct": max_rel_pct,
    }


def _fmt_pct_saved(t_base: float, t_other: float) -> str:
    """相对 baseline 节省的时间比例（%）；负值表示更慢。"""
    if t_base <= 0:
        return "n/a"
    saved = 100.0 * (1.0 - t_other / t_base)
    return f"{saved:+.1f}%"


def _fmt_speedup(t_base: float, t_other: float) -> str:
    if t_other <= 0:
        return "n/a"
    return f"{t_base / t_other:.2f}x"


W = 118  # 分隔线宽度


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline render vs block_cache vs temporal_vi (timing + image error vs baseline)"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5_folder", type=str, help="Directory of *.h5 frames")
    src.add_argument("--h5_file", type=str, help="Single H5; use with --runs to repeat virtual frames")
    parser.add_argument("--max_frames", type=int, default=None, help="Only first N frames (folder mode)")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Single-file mode: repeat the same H5 this many times as virtual frames",
    )
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--padding_length", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--max_cache_entries", type=int, default=50000)
    parser.add_argument("--full_every_k", type=int, default=8)
    parser.add_argument("--max_consecutive_approx", type=int, default=32)
    parser.add_argument("--changed_block_ratio_threshold", type=float, default=None)
    parser.add_argument("--approx_mode", type=str, choices=["level0", "level1"], default="level0")
    parser.add_argument("--blend_alpha", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=1, help="Warmup forward passes before timing")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印逐帧长表（仍会打印开头图例与末尾三组汇总）",
    )
    args = parser.parse_args()

    if args.h5_file:
        file_list = [args.h5_file]
        if args.runs < 1:
            parser.error("--runs must be >= 1")
        virtual_repeats = args.runs
    else:
        file_list = natsorted(glob.glob(os.path.join(args.h5_folder, "*.h5")))
        if not file_list:
            print(f"No *.h5 in {args.h5_folder}")
            return
        if args.max_frames is not None:
            file_list = file_list[: args.max_frames]
        virtual_repeats = 1

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

    print("=" * W)
    print("对比实验  ① baseline = 原始 pipeline.render（无块缓存、无 VI 近似）")
    print("          ② block_cache = 仅块级 construct_seq 缓存 + 全量 VI")
    print("          ③ temporal_vi = 块缓存 + 跨帧 VI 近似/周期全算")
    print("-" * W)
    print(f"  device={device}  dtype={dtype}  resolution={args.resolution}  block_size={args.block_size}")
    print(
        f"  temporal: full_every_k={args.full_every_k}  max_consecutive_approx={args.max_consecutive_approx}  "
        f"approx_mode={args.approx_mode}  blend_alpha={args.blend_alpha}"
    )
    if args.changed_block_ratio_threshold is not None:
        print(f"  changed_block_ratio_threshold={args.changed_block_ratio_threshold}")
    print("=" * W)

    # Warmup (baseline)
    wpath = file_list[0]
    witem = load_h5_item(wpath, args.padding_length)
    wt = witem["triangles"].unsqueeze(0).to(device)
    wtx = witem["texture"].unsqueeze(0).to(device)
    wm = witem["mask"].unsqueeze(0).to(device)
    wvn = witem["vn"].unsqueeze(0).to(device)
    wc2w = witem["c2w"].unsqueeze(0).to(device)
    wfov = witem["fov"].unsqueeze(0).unsqueeze(-1).to(device)
    for _ in range(max(0, args.warmup)):
        with torch.no_grad():
            _ = pipeline.render(
                wt, wtx, wm, wvn, wc2w, wfov, resolution=args.resolution, torch_dtype=dtype
            )
    _sync(device)

    rows: List[Dict[str, Any]] = []
    t_base_total = t_bc_total = t_tv_total = 0.0
    # 两套 LRU，避免同帧内 temporal 复用 block_cache 刚写入的命中，计时更独立
    block_cache_bc = BlockCache(max_entries=args.max_cache_entries)
    block_cache_tv = BlockCache(max_entries=args.max_cache_entries)
    tv_state = TemporalVIState()

    frame_idx = 0
    for path in file_list:
        for rep in range(virtual_repeats):
            item = load_h5_item(path, args.padding_length)
            triangles = item["triangles"].unsqueeze(0).to(device)
            texture = item["texture"].unsqueeze(0).to(device)
            mask = item["mask"].unsqueeze(0).to(device)
            vn = item["vn"].unsqueeze(0).to(device)
            c2w = item["c2w"].unsqueeze(0).to(device)
            fov = item["fov"].unsqueeze(0).unsqueeze(-1).to(device)

            img_b, t_b = time_render(
                device,
                pipeline.render,
                triangles,
                texture,
                mask,
                vn,
                c2w,
                fov,
                resolution=args.resolution,
                torch_dtype=dtype,
            )

            out_bc, t_bc = time_render(
                device,
                pipeline.render_with_block_cache,
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
            img_bc, _fl_bc_stats = out_bc

            out_tv, t_tv = time_render(
                device,
                pipeline.render_with_temporal_vi,
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
            img_tv, fl_tv = out_tv

            met_bc = image_metrics(img_b, img_bc)
            met_tv = image_metrics(img_b, img_tv)

            t_base_total += t_b
            t_bc_total += t_bc
            t_tv_total += t_tv

            label = os.path.basename(path)
            if virtual_repeats > 1:
                label = f"{label}#r{rep}"

            row = {
                "idx": frame_idx,
                "file": label,
                "t_base_ms": t_b * 1000,
                "t_bc_ms": t_bc * 1000,
                "t_tv_ms": t_tv * 1000,
                "mse_bc": met_bc["mse"],
                "rmse_bc": met_bc["rmse"],
                "maxd_bc": met_bc["max_abs"],
                "rel_rmse_bc_pct": met_bc["rel_rmse_pct"],
                "mse_tv": met_tv["mse"],
                "rmse_tv": met_tv["rmse"],
                "maxd_tv": met_tv["max_abs"],
                "rel_rmse_tv_pct": met_tv["rel_rmse_pct"],
                "mean_abs_ref": met_bc["mean_abs_ref"],
                "vi_path": fl_tv.get("vi_path"),
                "tv_reason": fl_tv.get("force_reason"),
            }
            rows.append(row)

            if not args.quiet:
                if frame_idx == 0:
                    print()
                    print(
                        "逐帧  列说明: t = 耗时ms(①|②|③); 省 = 相对①省时%%(②,③); × = 相对①加速比(②,③); "
                        "rmse/rel/max = 相对①的 RMSE、rel_RMSE%%、max|diff|"
                    )
                    print("-" * W)
                sav_bc = _fmt_pct_saved(t_b, t_bc)
                sav_tv = _fmt_pct_saved(t_b, t_tv)
                sp_bc = _fmt_speedup(t_b, t_bc)
                sp_tv = _fmt_speedup(t_b, t_tv)
                print(
                    f"  #{frame_idx:03d}  {label[:26]:26s}  "
                    f"t {row['t_base_ms']:7.2f}|{row['t_bc_ms']:7.2f}|{row['t_tv_ms']:7.2f}  "
                    f"省{sav_bc:>7}{sav_tv:>7}  "
                    f"×{sp_bc:>5}{sp_tv:>5}  "
                    f"②rmse={met_bc['rmse']:.3e} rel%={met_bc['rel_rmse_pct']:.3f} mx={met_bc['max_abs']:.3e}  "
                    f"③rmse={met_tv['rmse']:.3e} rel%={met_tv['rel_rmse_pct']:.3f} mx={met_tv['max_abs']:.3e}  "
                    f"VI={fl_tv.get('vi_path'):5s} {fl_tv.get('force_reason')}"
                )
            frame_idx += 1

    n = len(rows)
    if n == 0:
        print("No frames processed.")
        return

    if args.quiet:
        print(f"\n（--quiet：已跳过 {n} 帧的逐行长表，以下为汇总）\n")

    print()
    print("=" * W)
    print("【汇总 1】耗时对比（相对原始 baseline）")
    print("=" * W)
    tb, tbc, ttv = t_base_total, t_bc_total, t_tv_total
    denom = tb if tb > 1e-9 else 1e-9
    mean_b, mean_bc, mean_tv = tb / n * 1000, tbc / n * 1000, ttv / n * 1000

    def row_timing(name: str, total: float, mean_ms: float) -> None:
        pct_vs_base = 100.0 * total / denom
        saved = 100.0 * (1.0 - total / denom)
        sp = tb / total if total > 1e-9 else 0.0
        print(
            f"  {name:22s}  总计 {total:10.4f}s  |  均 {mean_ms:8.3f} ms/帧  |  "
            f"占 baseline {pct_vs_base:6.2f}%  |  较 baseline 省时 {saved:+6.2f}%  |  加速 {sp:.3f}x"
        )

    print(f"  帧数: {n}")
    print("-" * W)
    row_timing("① baseline (原始)", tb, mean_b)
    row_timing("② block_cache", tbc, mean_bc)
    row_timing("③ temporal_vi", ttv, mean_tv)
    print("-" * W)
    print(
        f"  对比结论:  BC 相对 baseline  总省时 {_fmt_pct_saved(tb, tbc)}  平均加速 {_fmt_speedup(tb, tbc)}"
    )
    print(
        f"            TV 相对 baseline  总省时 {_fmt_pct_saved(tb, ttv)}  平均加速 {_fmt_speedup(tb, ttv)}"
    )
    if tbc > 1e-9:
        print(
            f"            TV 相对 BC 路径    总省时 {_fmt_pct_saved(tbc, ttv)}  加速 {_fmt_speedup(tbc, ttv)} "
            f"(TV 多跳过 VI 近似帧时更明显)"
        )

    ratios_bc = [
        r["t_bc_ms"] / r["t_base_ms"] for r in rows if r["t_base_ms"] > 1e-6
    ]
    ratios_tv = [
        r["t_tv_ms"] / r["t_base_ms"] for r in rows if r["t_base_ms"] > 1e-6
    ]
    if ratios_bc:
        print("-" * W)
        print(
            f"  单帧耗时比 ②/①:  min={min(ratios_bc):.4f}  max={max(ratios_bc):.4f}  mean={float(np.mean(ratios_bc)):.4f}"
        )
        print(
            f"  单帧耗时比 ③/①:  min={min(ratios_tv):.4f}  max={max(ratios_tv):.4f}  mean={float(np.mean(ratios_tv)):.4f}"
        )
        ibc = int(np.argmin(ratios_bc))
        it = int(np.argmin(ratios_tv))
        print(
            f"  ②相对①最快帧: idx={rows[ibc]['idx']}  file={rows[ibc]['file'][:40]}  ②/①={ratios_bc[ibc]:.4f}"
        )
        print(
            f"  ③相对①最快帧: idx={rows[it]['idx']}  file={rows[it]['file'][:40]}  "
            f"③/①={ratios_tv[it]:.4f}  VI={rows[it]['vi_path']}"
        )

    mse_bc_mean = float(np.mean([r["mse_bc"] for r in rows]))
    mse_tv_mean = float(np.mean([r["mse_tv"] for r in rows]))
    rmse_bc_mean = float(np.mean([r["rmse_bc"] for r in rows]))
    rmse_tv_mean = float(np.mean([r["rmse_tv"] for r in rows]))
    rel_bc_mean = float(np.mean([r["rel_rmse_bc_pct"] for r in rows]))
    rel_tv_mean = float(np.mean([r["rel_rmse_tv_pct"] for r in rows]))
    max_bc_max = float(np.max([r["maxd_bc"] for r in rows]))
    max_tv_max = float(np.max([r["maxd_tv"] for r in rows]))
    mean_ref = float(np.mean([r["mean_abs_ref"] for r in rows]))

    approx_rows = [r for r in rows if r["vi_path"] == "approx"]
    full_rows = [r for r in rows if r["vi_path"] == "full"]

    print()
    print("=" * W)
    print("【汇总 2】图像数值 vs baseline（linear HDR，越小越接近原始）")
    print("=" * W)
    print(
        f"  {'路径':<16}  {'MSE 均值':>14}  {'RMSE 均值':>14}  "
        f"{'rel_RMSE% 均值':>16}  {'max|diff| 最大':>16}"
    )
    print("-" * W)
    print(
        f"  {'② block_cache':<16}  {mse_bc_mean:14.6e}  {rmse_bc_mean:14.6e}  "
        f"{rel_bc_mean:16.6f}  {max_bc_max:16.6e}"
    )
    print(
        f"  {'③ temporal_vi':<16}  {mse_tv_mean:14.6e}  {rmse_tv_mean:14.6e}  "
        f"{rel_tv_mean:16.6f}  {max_tv_max:16.6e}"
    )
    print("-" * W)
    print(f"  baseline |像素|均值(仅参考量级): {mean_ref:.6e}")
    print(
        f"   fidelity: BC 应 ≈0；TV 在 approx 帧会偏大；TV 全序列 MSE 较 BC 放大 "
        f"{(mse_tv_mean / (mse_bc_mean + 1e-30)):.2f}x（几何/参数相关）"
    )

    if approx_rows:
        mse_a = float(np.mean([r["mse_tv"] for r in approx_rows]))
        rmse_a = float(np.mean([r["rmse_tv"] for r in approx_rows]))
        rel_a = float(np.mean([r["rel_rmse_tv_pct"] for r in approx_rows]))
        print()
        print("  —— 仅 VI=approx 的帧（相对 baseline 误差更敏感）——")
        print(f"     帧数={len(approx_rows)}  MSE均值={mse_a:.6e}  RMSE均值={rmse_a:.6e}  rel_RMSE%均值={rel_a:.4f}")
    if full_rows and approx_rows:
        mse_f = float(np.mean([r["mse_tv"] for r in full_rows]))
        print(f"     对照: VI=full 的帧数={len(full_rows)}  TV MSE均值={mse_f:.6e}（应接近 BC）")

    approx_frames = len(approx_rows)
    full_frames = len(full_rows)
    print()
    print("=" * W)
    print("【汇总 3】Temporal VI 调度与块缓存")
    print("=" * W)
    print(f"  VI 全算帧: {full_frames}  |  VI 近似帧: {approx_frames}  |  近似占比: {100.0 * approx_frames / n:.1f}%")
    print(f"  强制全算原因统计 force_reason_hist: {dict(tv_state.force_reason_hist)}")
    st_bc = block_cache_bc.stats()
    st_tv = block_cache_tv.stats()
    print(
        f"  BlockCache(BC 专用)  查询 hit={st_bc['hits']} miss={st_bc['misses']}  "
        f"命中率={st_bc['hit_rate']:.2%}  条目={st_bc['size']}  内存≈{st_bc['memory_mb']:.2f} MB"
    )
    print(
        f"  BlockCache(TV 专用)  查询 hit={st_tv['hits']} miss={st_tv['misses']}  "
        f"命中率={st_tv['hit_rate']:.2%}  条目={st_tv['size']}  内存≈{st_tv['memory_mb']:.2f} MB"
    )
    print("=" * W)
    print("说明: ① 原始 render；② 仅块缓存（与①应几乎一致）；③ 块缓存+跨帧 VI 近似（approx 帧会快但偏离①）。")
    print("Done.")


if __name__ == "__main__":
    main()

"""
批量渲染 H5 视频序列：块缓存 + 跨帧近似 VI（周期/阈值强制全算）。
用法示例（--h5_folder 可为相对路径或绝对路径）：
  python batch_infer_temporal_vi.py --h5_folder video-data/teaser-scenes/cbox-roughness \\
    --output_dir output/videos/cbox-temporal --full_every_k 8 --approx_mode level0
  # Windows 本仓库官方序列: C:\Users\zhangleipa\.openclaw\workspace\renderformer\video-data\teaser-scenes\cbox-roughness
"""
import os
import glob
import torch
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import argparse
import imageio
import h5py

from renderformer import RenderFormerRenderingPipeline
from renderformer.cache import BlockCache
from renderformer.temporal_vi import TemporalVIConfig, TemporalVIState
from simple_ocio import ToneMapper


def load_h5_item(file_path: str, padding_length=None):
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


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference: block cache + temporal VI (approx / periodic full)"
    )
    parser.add_argument("--h5_folder", type=str, required=True, help="Folder with *.h5 frames")
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--max_cache_entries", type=int, default=50000)
    parser.add_argument("--padding_length", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--tone_mapper", type=str, choices=["none", "agx", "filmic", "pbr_neutral"], default="none")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument(
        "--full_every_k",
        type=int,
        default=8,
        help="每完成 K 帧后强制全算 VI；0=不按周期（仍可用块变化/连续近似上限触发）",
    )
    parser.add_argument(
        "--max_consecutive_approx",
        type=int,
        default=32,
        help="连续近似 VI 帧数上限，超过则强制全算",
    )
    parser.add_argument(
        "--changed_block_ratio_threshold",
        type=float,
        default=None,
        help="块哈希相对上一帧变化比例超过该值则强制全算；不设则关闭（例: 0.01 表示变化>1%% 的块即全算）",
    )
    parser.add_argument(
        "--approx_mode",
        type=str,
        choices=["level0", "level1"],
        default="level0",
        help="level0: 复用上次 VI；level1: alpha*LayerNorm(seq)+(1-alpha)*VI_ref",
    )
    parser.add_argument(
        "--blend_alpha",
        type=float,
        default=0.15,
        help="level1 混合系数",
    )

    args = parser.parse_args()

    file_list = natsorted(glob.glob(os.path.join(args.h5_folder, "*.h5")))
    if not file_list:
        print(f"No *.h5 in {args.h5_folder}")
        return

    print(f"[temporal_vi] H5 count={len(file_list)} folder={args.h5_folder}")
    print(
        f"[temporal_vi] full_every_k={args.full_every_k} max_consecutive_approx={args.max_consecutive_approx} "
        f"changed_block_ratio_threshold={args.changed_block_ratio_threshold} "
        f"approx_mode={args.approx_mode} blend_alpha={args.blend_alpha}"
    )
    print("-" * 72)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    pipeline = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    if device.type == "cuda" and os.name == "posix":
        try:
            from renderformer_liger_kernel import apply_kernels

            apply_kernels(pipeline.model)
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        args.precision = "fp32"
    pipeline.to(device)

    tone_mapper = None
    if args.tone_mapper != "none":
        tmap = "Khronos PBR Neutral" if args.tone_mapper == "pbr_neutral" else args.tone_mapper
        tone_mapper = ToneMapper(tmap)
        print(f"Tone mapper: {tmap}")

    dtype = (
        torch.float16
        if args.precision == "fp16"
        else torch.bfloat16
        if args.precision == "bf16"
        else torch.float32
    )
    output_dir = args.output_dir or args.h5_folder
    os.makedirs(output_dir, exist_ok=True)

    block_cache = BlockCache(max_entries=args.max_cache_entries)
    tv_cfg = TemporalVIConfig(
        full_every_k=args.full_every_k,
        max_consecutive_approx=args.max_consecutive_approx,
        changed_block_ratio_threshold=args.changed_block_ratio_threshold,
        approx_mode=args.approx_mode,
        blend_alpha=args.blend_alpha,
    )
    tv_state = TemporalVIState()

    video_frames = [] if args.save_video else None
    frame_logs = []

    if not args.quiet:
        print(
            "[temporal_vi] columns: frame | file | vi_path | force | reason | blocks | block_hit% | consec_approx"
        )
        print("-" * 72)

    for frame_idx, file_path in enumerate(tqdm(file_list, desc="Render")):
        item = load_h5_item(file_path, args.padding_length)
        num_tris = item["num_tris"]
        triangles = item["triangles"].unsqueeze(0).to(device)
        texture = item["texture"].unsqueeze(0).to(device)
        mask = item["mask"].unsqueeze(0).to(device)
        vn = item["vn"].unsqueeze(0).to(device)
        c2w = item["c2w"].unsqueeze(0).to(device)
        fov = item["fov"].unsqueeze(0).unsqueeze(-1).to(device)

        rendered, flog = pipeline.render_with_temporal_vi(
            triangles=triangles,
            texture=texture,
            mask=mask,
            vn=vn,
            c2w=c2w,
            fov=fov,
            block_cache=block_cache,
            temporal_state=tv_state,
            temporal_cfg=tv_cfg,
            block_size=args.block_size,
            resolution=args.resolution,
            torch_dtype=dtype,
            verbose=not args.quiet,
        )
        flog["file"] = os.path.basename(file_path)
        flog["frame_idx"] = frame_idx
        frame_logs.append(flog)

        if not args.quiet:
            bhr = flog.get("block_hit_rate", 0.0) * 100
            print(
                f"  [frame {frame_idx:4d}] {flog['file']:28s} vi={flog['vi_path']:5s} "
                f"force={str(flog['force_full']):5s} reason={flog['force_reason']:22s} "
                f"blocks={flog['num_blocks']:3d} block_hit%={bhr:5.1f} "
                f"consec_apx={tv_state.consecutive_approx}"
            )

        if args.save_video and video_frames is not None:
            nv = c2w.shape[1]
            for vi in range(nv):
                hdr = rendered[0, vi].cpu().numpy().astype(np.float32)
                ldr = tone_mapper.hdr_to_ldr(hdr) if tone_mapper else np.clip(hdr, 0, 1)
                ldr = (ldr * 255).astype(np.uint8)
                video_frames.append(ldr)
                base = os.path.splitext(os.path.basename(file_path))[0]
                imageio.v3.imwrite(os.path.join(output_dir, f"{base}_view_{vi}.png"), ldr)
                imageio.v3.imwrite(os.path.join(output_dir, f"{base}_view_{vi}.exr"), hdr)

    bc = block_cache.stats()
    full_ct = tv_state.cumulative.get("full_vi", 0)
    apx_ct = tv_state.cumulative.get("approx_vi", 0)

    print()
    print("=" * 72)
    print("  Temporal VI + Block cache 汇总")
    print("=" * 72)
    print(f"  总帧数                  {len(frame_logs)}")
    print(f"  VI 全算次数             {full_ct}")
    print(f"  VI 近似次数             {apx_ct}")
    print(f"  块缓存总查询命中        {bc['hits']} / {bc['hits'] + bc['misses']}  ({bc['hit_rate']:.2%})")
    print(f"  块缓存条目 / 内存       {bc['size']}  /  {bc['memory_mb']:.2f} MB")
    print(f"  强制全算原因 histogram  {dict(tv_state.force_reason_hist)}")
    print(f"  末帧状态                {tv_state.summary()}")
    print("=" * 72)

    if frame_logs:
        print(f"  首帧: {frame_logs[0]['vi_path']} reason={frame_logs[0]['force_reason']}")
        print(f"  末帧: {frame_logs[-1]['vi_path']} reason={frame_logs[-1]['force_reason']}")

    print()
    if args.save_video and video_frames:
        video_path = os.path.join(output_dir, "video.mp4")
        imageio.v3.imwrite(video_path, np.array(video_frames), fps=24, quality=9)
        print(f"Saved: {output_dir} -> {video_path}")
    else:
        print(f"Output dir: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()

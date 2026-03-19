"""
批量渲染 H5 视频序列，使用块级缓存；逐帧打印缓存数据，结束时输出命中汇总。
用法示例：
  python batch_infer_with_cache.py --h5_folder video-data/teaser-scenes/cbox-roughness --output_dir output/videos/cbox-roughness-cache
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
from simple_ocio import ToneMapper


def load_h5_item(file_path: str, padding_length=None):
    """Load one H5 file to tensors (no batch dim)."""
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
    parser = argparse.ArgumentParser(description="Batch inference with block cache; print cache stats and summary")
    parser.add_argument("--h5_folder", type=str, required=True, help="Folder containing *.h5 files (e.g. video-data/teaser-scenes/cbox-roughness)")
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--block_size", type=int, default=256, help="Triangles per block")
    parser.add_argument("--max_cache_entries", type=int, default=50000)
    parser.add_argument("--padding_length", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--tone_mapper", type=str, choices=["none", "agx", "filmic", "pbr_neutral"], default="none")
    parser.add_argument("--quiet", action="store_true", help="Less per-frame print, only summary")
    args = parser.parse_args()

    file_list = natsorted(glob.glob(os.path.join(args.h5_folder, "*.h5")))
    if not file_list:
        print(f"No *.h5 in {args.h5_folder}")
        return
    print(f"[cache] Found {len(file_list)} H5 files in {args.h5_folder}")
    print(f"[cache] block_size={args.block_size} max_cache_entries={args.max_cache_entries}")
    print("-" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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

    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16 if args.precision == "bf16" else torch.float32
    output_dir = args.output_dir or args.h5_folder
    os.makedirs(output_dir, exist_ok=True)

    block_cache = BlockCache(max_entries=args.max_cache_entries)
    frame_stats = []
    video_frames = [] if args.save_video else None

    if not args.quiet:
        print("[cache] 逐帧缓存数据 (per-frame cache data): frame | file | tri | blocks | hit | miss | frame_hit% | cache_size | memory_mb")
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

        hits_before = block_cache.stats()["hits"]
        misses_before = block_cache.stats()["misses"]

        rendered, stats = pipeline.render_with_block_cache(
            triangles=triangles,
            texture=texture,
            mask=mask,
            vn=vn,
            c2w=c2w,
            fov=fov,
            block_cache=block_cache,
            block_size=args.block_size,
            resolution=args.resolution,
            torch_dtype=dtype,
            verbose=not args.quiet,
        )

        hits_after = stats["hits"]
        misses_after = stats["misses"]
        frame_hits = hits_after - hits_before
        frame_misses = misses_after - misses_before
        blocks_this_frame = frame_hits + frame_misses
        frame_hit_rate = (frame_hits / blocks_this_frame * 100) if blocks_this_frame else 0.0

        frame_stats.append({
            "frame": os.path.basename(file_path),
            "idx": frame_idx,
            "num_tris": num_tris,
            "blocks": blocks_this_frame,
            "hits": frame_hits,
            "misses": frame_misses,
            "hit_rate": frame_hit_rate,
        })

        if not args.quiet:
            print(
                f"  [frame {frame_idx:4d}] {os.path.basename(file_path):24s} "
                f"tri={num_tris:5d} blocks={blocks_this_frame:3d} "
                f"hit={frame_hits:3d} miss={frame_misses:3d} "
                f"frame_hit%={frame_hit_rate:5.1f}  cache_size={stats['size']:5d} memory_mb={stats['memory_mb']:.2f}"
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

    final = block_cache.stats()
    total_hits = final["hits"]
    total_misses = final["misses"]
    total_queries = total_hits + total_misses
    overall_hit_rate = (total_hits / total_queries * 100) if total_queries else 0.0

    print()
    print("=" * 72)
    print("  缓存命中汇总 (Cache Summary)")
    print("=" * 72)
    print(f"  总帧数           Total frames:     {len(frame_stats)}")
    print(f"  总查询次数       Total block lookups: {total_queries}  (hits + misses)")
    print(f"  总命中           Total hits:       {total_hits}")
    print(f"  总未命中         Total misses:     {total_misses}")
    print(f"  整体命中率       Overall hit rate:  {overall_hit_rate:.2f}%")
    print(f"  缓存条目数       Cache entries:    {final['size']}")
    print(f"  缓存占用         Cache memory:     {final['memory_mb']:.2f} MB")
    print("=" * 72)

    if frame_stats:
        first_frame = frame_stats[0]
        last_frame = frame_stats[-1]
        print(f"  首帧 {first_frame['frame']}: blocks={first_frame['blocks']} hit={first_frame['hits']} miss={first_frame['misses']} (预期全未命中)")
        print(f"  末帧 {last_frame['frame']}: blocks={last_frame['blocks']} hit={last_frame['hits']} miss={last_frame['misses']} (同场景预期高命中)")

    print()
    if args.save_video and video_frames:
        video_path = os.path.join(output_dir, "video.mp4")
        imageio.v3.imwrite(video_path, np.array(video_frames), fps=24, quality=9)
        print(f"Output and video saved to: {output_dir} -> {video_path}")
    else:
        print(f"Output dir: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()

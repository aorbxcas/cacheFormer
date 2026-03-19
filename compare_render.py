"""
视频渲染对比：原模型（无缓存） vs 带块缓存的模型。
在同一 H5 序列上先后跑两遍，打印耗时、帧率、缓存统计等对比数据。
用法：
  python compare_render.py --h5_folder video-data/teaser-scenes/cbox-roughness
"""
import os
import glob
import time
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
        "num_tris": num_tris,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare original vs cached video rendering")
    parser.add_argument("--h5_folder", type=str, required=True, help="Folder of *.h5 (e.g. video-data/teaser-scenes/cbox-roughness)")
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--max_cache_entries", type=int, default=50000)
    parser.add_argument("--padding_length", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="Save images/video from cached run here (optional)")
    parser.add_argument("--save_outputs", action="store_true", help="Save images and video from cached run")
    parser.add_argument("--tone_mapper", type=str, choices=["none", "agx", "filmic", "pbr_neutral"], default="none")
    args = parser.parse_args()

    file_list = natsorted(glob.glob(os.path.join(args.h5_folder, "*.h5")))
    if not file_list:
        print(f"No *.h5 in {args.h5_folder}")
        return

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

    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16 if args.precision == "bf16" else torch.float32
    tone_mapper = None
    if args.tone_mapper != "none":
        tmap = "Khronos PBR Neutral" if args.tone_mapper == "pbr_neutral" else args.tone_mapper
        tone_mapper = ToneMapper(tmap)

    num_frames = len(file_list)
    print()
    print("=" * 72)
    print("  视频渲染对比测试  Original vs Block-Cache")
    print("=" * 72)
    print(f"  序列: {args.h5_folder}")
    print(f"  帧数: {num_frames}  分辨率: {args.resolution}  block_size: {args.block_size}")
    print("=" * 72)

    # ---------- 1) 原模型（无缓存） ----------
    print("\n[1/2] 原模型（无缓存）渲染中...")
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    for file_path in tqdm(file_list, desc="Original"):
        item = load_h5_item(file_path, args.padding_length)
        tri = item["triangles"].unsqueeze(0).to(device)
        tex = item["texture"].unsqueeze(0).to(device)
        msk = item["mask"].unsqueeze(0).to(device)
        vn = item["vn"].unsqueeze(0).to(device)
        c2w = item["c2w"].unsqueeze(0).to(device)
        fov = item["fov"].unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            _ = pipeline.render(
                triangles=tri,
                texture=tex,
                mask=msk,
                vn=vn,
                c2w=c2w,
                fov=fov,
                resolution=args.resolution,
                torch_dtype=dtype,
            )
    torch.cuda.synchronize() if device.type == "cuda" else None
    time_original = time.perf_counter() - t0
    fps_original = num_frames / time_original if time_original > 0 else 0
    ms_per_frame_original = time_original / num_frames * 1000

    # ---------- 2) 带块缓存 ----------
    print("\n[2/2] 带块缓存渲染中...")
    block_cache = BlockCache(max_entries=args.max_cache_entries)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    video_frames = [] if args.save_outputs else None
    for file_path in tqdm(file_list, desc="Cached"):
        item = load_h5_item(file_path, args.padding_length)
        tri = item["triangles"].unsqueeze(0).to(device)
        tex = item["texture"].unsqueeze(0).to(device)
        msk = item["mask"].unsqueeze(0).to(device)
        vn = item["vn"].unsqueeze(0).to(device)
        c2w = item["c2w"].unsqueeze(0).to(device)
        fov = item["fov"].unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            rendered, _ = pipeline.render_with_block_cache(
                triangles=tri,
                texture=tex,
                mask=msk,
                vn=vn,
                c2w=c2w,
                fov=fov,
                block_cache=block_cache,
                block_size=args.block_size,
                resolution=args.resolution,
                torch_dtype=dtype,
                verbose=False,
            )
        if args.save_outputs and video_frames is not None:
            hdr = rendered[0, 0].cpu().numpy().astype(np.float32)
            ldr = tone_mapper.hdr_to_ldr(hdr) if tone_mapper else np.clip(hdr, 0, 1)
            video_frames.append((ldr * 255).astype(np.uint8))
    torch.cuda.synchronize() if device.type == "cuda" else None
    time_cached = time.perf_counter() - t0
    fps_cached = num_frames / time_cached if time_cached > 0 else 0
    ms_per_frame_cached = time_cached / num_frames * 1000

    cache_stats = block_cache.stats()
    speedup = time_original / time_cached if time_cached > 0 else 0

    # ---------- 打印对比表 ----------
    print()
    print("=" * 72)
    print("  对比结果 (Comparison Results)")
    print("=" * 72)
    print()
    print("  ┌─────────────────────────────┬──────────────────┬──────────────────┐")
    print("  │ 指标                        │ 原模型 (无缓存)   │ 带块缓存         │")
    print("  ├─────────────────────────────┼──────────────────┼──────────────────┤")
    print(f"  │ 总耗时 (s)                  │ {time_original:16.2f} │ {time_cached:16.2f} │")
    print(f"  │ 平均每帧 (ms)               │ {ms_per_frame_original:16.2f} │ {ms_per_frame_cached:16.2f} │")
    print(f"  │ 帧率 (FPS)                  │ {fps_original:16.2f} │ {fps_cached:16.2f} │")
    print("  └─────────────────────────────┴──────────────────┴──────────────────┘")
    print()
    print("  ┌─────────────────────────────┬──────────────────┐")
    print("  │ 加速比 (原模型耗时/缓存耗时)  │ {:>16.2f}x │".format(speedup))
    print("  └─────────────────────────────┴──────────────────┘")
    print()
    print("  ┌─────────────────────────────┬──────────────────┐")
    print("  │ 缓存统计 (仅带缓存运行)      │                  │")
    print("  ├─────────────────────────────┼──────────────────┤")
    print(f"  │ 总查询次数 (block lookups)   │ {cache_stats['hits'] + cache_stats['misses']:16d} │")
    print(f"  │ 命中 (hits)                 │ {cache_stats['hits']:16d} │")
    print(f"  │ 未命中 (misses)             │ {cache_stats['misses']:16d} │")
    print(f"  │ 命中率 (%)                  │ {cache_stats['hit_rate'] * 100:15.2f}% │")
    print(f"  │ 缓存条目数                  │ {cache_stats['size']:16d} │")
    print(f"  │ 缓存占用 (MB)               │ {cache_stats['memory_mb']:15.2f} MB │")
    print("  └─────────────────────────────┴──────────────────┘")
    print()
    print("=" * 72)

    if args.save_outputs and video_frames:
        out_dir = args.output_dir or os.path.join(args.h5_folder, "compare_output")
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, "video_cached.mp4")
        imageio.v3.imwrite(video_path, np.array(video_frames), fps=24, quality=9)
        print(f"  带缓存渲染视频已保存: {video_path}")
        print("=" * 72)
    print()


if __name__ == "__main__":
    main()

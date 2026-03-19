"""
Inference with block-level cache. Run twice on the same scene to see cache hits on the second run.
Prints cache stats and rendering data (shape, dtype, value range).
"""
import os
import sys
import torch
import h5py
import argparse
import numpy as np
import imageio

from renderformer import RenderFormerRenderingPipeline
from renderformer.cache import BlockCache
from simple_ocio import ToneMapper


def load_single_h5_data(file_path):
    with h5py.File(file_path, "r") as f:
        triangles = torch.from_numpy(np.array(f["triangles"]).astype(np.float32))
        num_tris = triangles.shape[0]
        texture = torch.from_numpy(np.array(f["texture"]).astype(np.float32))
        mask = torch.ones(num_tris, dtype=torch.bool)
        vn = torch.from_numpy(np.array(f["vn"]).astype(np.float32))
        c2w = torch.from_numpy(np.array(f["c2w"]).astype(np.float32))
        fov = torch.from_numpy(np.array(f["fov"]).astype(np.float32))
        data = {
            "triangles": triangles,
            "texture": texture,
            "mask": mask,
            "c2w": c2w,
            "fov": fov,
            "vn": vn,
        }
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Infer with block-level cache; print cache and render stats"
    )
    parser.add_argument("--h5_file", type=str, required=True, help="Path to input H5 file")
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/renderformer-v1.1-swin-large",
        help="Model ID or local path",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="fp16",
        help="Precision for inference",
    )
    parser.add_argument("--resolution", type=int, default=512, help="Resolution")
    parser.add_argument("--block_size", type=int, default=256, help="Triangles per block for cache")
    parser.add_argument("--max_cache_entries", type=int, default=10000, help="LRU cache max entries")
    parser.add_argument("--output_dir", type=str, default=None, help="Save images here")
    parser.add_argument("--tone_mapper", type=str, choices=["none", "agx", "filmic", "pbr_neutral"], default="none")
    parser.add_argument("--runs", type=int, default=2, help="Number of render runs (2 = first miss, second hit)")
    args = parser.parse_args()

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
        print("MPS: forcing fp32")
    pipeline.to(device)

    if args.tone_mapper != "none":
        tmap_name = "Khronos PBR Neutral" if args.tone_mapper == "pbr_neutral" else args.tone_mapper
        tone_mapper = ToneMapper(tmap_name)
        print(f"Tone mapper: {tmap_name}")
    else:
        tone_mapper = None

    data = load_single_h5_data(args.h5_file)
    triangles = data["triangles"].unsqueeze(0).to(device)
    texture = data["texture"].unsqueeze(0).to(device)
    mask = data["mask"].unsqueeze(0).to(device)
    vn = data["vn"].unsqueeze(0).to(device)
    c2w = data["c2w"].unsqueeze(0).to(device)
    fov = data["fov"].unsqueeze(0).unsqueeze(-1).to(device)

    dtype = (
        torch.float16
        if args.precision == "fp16"
        else torch.bfloat16
        if args.precision == "bf16"
        else torch.float32
    )

    block_cache = BlockCache(max_entries=args.max_cache_entries)
    num_tris = triangles.shape[1]
    print(f"Scene: {args.h5_file}  num_triangles={num_tris}  block_size={args.block_size}")
    print("-" * 60)

    for run in range(args.runs):
        print(f"\n--- Run {run + 1}/{args.runs} ---")
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
            verbose=True,
        )
        print(f"Cache stats: {stats}")
        r = rendered[0]
        print(
            f"Rendering data: shape={r.shape} dtype={r.dtype} "
            f"min={r.min().item():.4f} max={r.max().item():.4f} mean={r.mean().item():.4f}"
        )
        if run == args.runs - 1 and args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(args.h5_file))[0]
            nv = c2w.shape[1]
            for i in range(nv):
                hdr = rendered[0, i].cpu().numpy().astype(np.float32)
                if tone_mapper:
                    ldr = tone_mapper.hdr_to_ldr(hdr)
                else:
                    ldr = np.clip(hdr, 0, 1)
                ldr = (ldr * 255).astype(np.uint8)
                imageio.v3.imwrite(
                    os.path.join(args.output_dir, f"{base}_view_{i}.png"), ldr
                )
            print(f"Saved images to {args.output_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()

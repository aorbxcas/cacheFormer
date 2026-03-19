# -*- coding: utf-8 -*-
"""
VI 缓存效果简易基准：同一场景、两个不同相机各渲染一次。

预期：
- 第 1 次：缓存未命中（完整 VI + VD）
- 第 2 次：缓存命中（仅 VD + 射线等）

用法（需已有 h5，且 GPU）:
    python benchmark_vi_cache.py --h5_file tmp/cbox/cbox.h5

说明：
- 仅 batch_size=1 时启用缓存；本脚本固定为单场景双视角分两次 render 调用，
  第二次仅改变 c2w/fov，几何与第一次相同，故应命中 VI 缓存。
"""

import argparse
import os
import time

import numpy as np
import torch

from renderformer import RenderFormerRenderingPipeline, ViewIndependentCache


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
    args = parser.parse_args()

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

    # 两次渲染：第二次微调 FOV，场景三角不变 → 应命中 VI 缓存
    c2w_a = data["c2w"].clone()
    fov_a = data["fov"].clone()
    c2w_b = data["c2w"].clone()
    fov_b = data["fov"].clone() * 0.95

    cache = ViewIndependentCache(max_entries=4)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    _, info0 = pipeline.render(
        triangles=data["triangles"],
        texture=data["texture"],
        mask=data["mask"],
        vn=data["vn"],
        c2w=c2w_a,
        fov=fov_a,
        resolution=args.resolution,
        torch_dtype=dtype,
        vi_cache=cache,
        return_vi_cache_info=True,
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.perf_counter()

    _, info1 = pipeline.render(
        triangles=data["triangles"],
        texture=data["texture"],
        mask=data["mask"],
        vn=data["vn"],
        c2w=c2w_b,
        fov=fov_b,
        resolution=args.resolution,
        torch_dtype=dtype,
        vi_cache=cache,
        return_vi_cache_info=True,
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    t2 = time.perf_counter()

    print("第 1 次 vi_cache_hit:", info0["vi_cache_hit"], "耗时(s):", t1 - t0)
    print("第 2 次 vi_cache_hit:", info1["vi_cache_hit"], "耗时(s):", t2 - t1)
    print("缓存统计:", cache.stats())


if __name__ == "__main__":
    main()

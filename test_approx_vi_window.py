"""
对比全量 VI (forward_vi_only) 与近似思路1：miss 邻域 + 局部窗口 attention + 少量层。

运行前建议强制 SDPA（避免 flash-attn 与自定义 2D mask 不兼容）::

    set ATTN_IMPL=sdpa   # Windows CMD
    $env:ATTN_IMPL='sdpa'  # PowerShell

或在下方 import renderformer 之前设置 os.environ（本脚本已设置默认值）。
"""

from __future__ import annotations

import os
import time
from typing import Optional

# attention 模块在 import 时读取 ATTN_IMPL
os.environ.setdefault("ATTN_IMPL", "sdpa")

import torch

from renderformer.approx.local_window_vi import approx_vi_local_window
from renderformer.models.config import RenderFormerConfig
from renderformer.models.renderformer import RenderFormer


def _random_scene_batch(
    config: RenderFormerConfig,
    device: torch.device,
    num_tri: int,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    patch = config.texture_encode_patch_size
    ch = config.texture_channels
    tri = torch.randn(1, num_tri, 9, device=device, generator=generator)
    tex = torch.randn(1, num_tri, ch, patch, patch, device=device, generator=generator)
    mask = torch.ones(1, num_tri, dtype=torch.bool, device=device)
    vn = torch.randn(1, num_tri, 9, device=device, generator=generator)
    return tri, tex, mask, vn


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torch.manual_seed(0)
    gen = torch.Generator(device=device).manual_seed(0)

    config = RenderFormerConfig(
        num_layers=12,
        num_heads=6,
        latent_dim=768,
        num_register_tokens=8,
        texture_encode_patch_size=8,
        dropout=0.0,
        pe_type="rope",
    )
    model = RenderFormer(config).to(device=device, dtype=dtype)
    model.eval()

    num_tri = 40
    tri, tex, mask, vn = _random_scene_batch(config, device, num_tri, generator=gen)

    t0 = time.perf_counter()
    vi_full = model.forward_vi_only(tri, tex, mask, vn)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_full = time.perf_counter() - t0

    skip = model.skip_token_num
    miss_ratio = 0.12
    n_miss = max(1, int(num_tri * miss_ratio))
    perm = torch.randperm(num_tri, device=device, generator=gen)[:n_miss]
    miss = perm.long()

    vi_cached = vi_full.clone()
    noise = torch.randn_like(vi_cached) * 0.25
    vi_cached[:, skip:, :] = vi_full[:, skip:, :] + noise[:, skip:, :]

    R = 3
    K = 2

    t1 = time.perf_counter()
    vi_approx = approx_vi_local_window(
        model,
        tri,
        tex,
        mask,
        vn,
        vi_cached,
        miss,
        window_radius=R,
        num_refiner_layers=K,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_apx = time.perf_counter() - t1

    def mse(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(((a - b) ** 2).mean().item())

    win: set[int] = set()
    for i in miss.tolist():
        for j in range(max(0, i - R), min(num_tri - 1, i + R) + 1):
            win.add(j)
    win_list = sorted(win)

    tri_idx_full = torch.tensor([skip + j for j in range(num_tri)], device=device)
    win_idx = torch.tensor([skip + j for j in win_list], device=device)
    miss_idx = skip + miss

    mse_all = mse(vi_approx, vi_full)
    mse_win = mse(vi_approx[:, win_idx, :], vi_full[:, win_idx, :])
    mse_miss = mse(vi_approx[:, miss_idx, :], vi_full[:, miss_idx, :])

    rel_miss = float(
        (vi_approx[:, miss_idx, :] - vi_full[:, miss_idx, :]).norm()
        / (vi_full[:, miss_idx, :].norm() + 1e-8)
    )

    print("=== 近似思路1：局部窗口 VI（对比全量） ===")
    print(f"device={device}, ATTN_IMPL={os.environ.get('ATTN_IMPL', '')}")
    print(f"num_tri={num_tri}, skip(reg)={skip}, num_layers(full)={config.num_layers}")
    print(f"miss_count={n_miss}, miss_ratio={miss_ratio:.3f}, window_radius_R={R}, refiner_layers_K={K}")
    print(f"union_window_tri_count={len(win_list)} (子序列总长 L_sub={skip + len(win_list)})")
    print(f"time_forward_vi_only_sec={t_full:.6f}")
    print(f"time_approx_vi_local_window_sec={t_apx:.6f}")
    print(f"mse(vi_approx, vi_full) all_tokens={mse_all:.6e}")
    print(f"mse on union_window_tri_tokens={mse_win:.6e}")
    print(f"mse on miss_tri_tokens_only={mse_miss:.6e}")
    print(f"relative_l2_on_miss_rows={rel_miss:.6f}")
    print("(说明：K 层 + 局部 mask 不等于 12 层全局 VI，误差预期显著；脚本用于打印基线与耗时。)")


if __name__ == "__main__":
    main()

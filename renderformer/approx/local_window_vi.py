"""
近似思路 1：对 miss 邻域子序列做带局部窗口约束的少量 VI 层，窗外 token 的 VI 行沿用缓存。

子序列的 K 层前向在「已有 VI 特征」vi_cached 的切片上进行（而非 construct_seq 的初始嵌入），
避免在窗口内用「仅 K 层 + 局部」从嵌入重算并覆盖全量 VI，导致首帧即与 vi_gold 严重不一致。

与全量 12 层全局 attention 仍不等价；用于可接受的近似与耗时对比实验。
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union

import torch
import torch.nn as nn

from renderformer.encodings.rope import freqs_to_cos_sin
from renderformer.layers.attention import TransformerEncoder
from renderformer.models.renderformer import RenderFormer


def build_triangle_local_attn_bias(
    orig_tri_idx: torch.Tensor,
    window_radius: int,
    device: torch.device,
) -> torch.Tensor:
    """
    构造加性 attention 偏置：禁止的 (query, key) 为 large negative。

    orig_tri_idx: [Ls]，寄存器位置为 -1，三角 token 为全局三角下标 0..N-1。
    规则：凡涉及寄存器的位置对允许互看；三角-三角仅当 |i-j|<=R。
    """
    Ls = int(orig_tri_idx.shape[0])
    oa = orig_tri_idx.unsqueeze(1).expand(Ls, Ls)
    ob = orig_tri_idx.unsqueeze(0).expand(Ls, Ls)
    tri_both = (oa >= 0) & (ob >= 0)
    local_tri = tri_both & ((oa - ob).abs() <= window_radius)
    any_reg = (oa < 0) | (ob < 0)
    allowed = local_tri | any_reg
    neg_large = torch.finfo(torch.float32).min / 4
    return torch.zeros(Ls, Ls, device=device, dtype=torch.float32).masked_fill(~allowed, neg_large)


def _run_layers_subset(
    transformer: TransformerEncoder,
    layers: Iterable[nn.Module],
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    triangle_pos: torch.Tensor,
    attn_bias_2d: torch.Tensor,
) -> torch.Tensor:
    if transformer.rope_dim is not None:
        rope_freqs = transformer.rope_emb.get_triangle_freqs(triangle_pos)
        rope_cos, rope_sin = freqs_to_cos_sin(rope_freqs, head_dim=transformer.head_dim)
    else:
        rope_cos = rope_sin = None
    for layer in layers:
        x = layer(
            x,
            src_key_padding_mask=valid_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            attn_bias_2d=attn_bias_2d,
            force_sdpa=True,
        )
    return x


@torch.no_grad()
def approx_vi_local_window(
    model: RenderFormer,
    tri_vpos_list: torch.Tensor,
    texture_patch_list: torch.Tensor,
    valid_mask: torch.Tensor,
    vns: torch.Tensor,
    vi_cached: torch.Tensor,
    miss_tri_indices: Union[torch.Tensor, Sequence[int]],
    window_radius: int = 4,
    num_refiner_layers: int = 2,
) -> torch.Tensor:
    """
    以 vi_cached 为基底；对 miss 三角在半径 window_radius 内的并集子序列上，
    取 vi_cached 在该子序列上的切片作为 K 层输入（在已有 VI 上修正），
    仅跑前 num_refiner_layers 层 VI，且自注意力受局部窗口限制；其余行保持 vi_cached。

    Args:
        model: RenderFormer
        tri_vpos_list, texture_patch_list, valid_mask, vns: 与 construct_seq 一致
        vi_cached: [B, skip+N, D]，通常为上帧或缓存的 VI 输出
        miss_tri_indices: 当前帧判定为 cache miss 的三角下标（0..N-1，未含寄存器偏移）
        window_radius: 三角网格索引上的邻域半径 R
        num_refiner_layers: 使用的 VI 层数 K（取 model.transformer 的前 K 层）

    Returns:
        vi_out: [B, skip+N, D]
    """
    device = tri_vpos_list.device
    skip = model.skip_token_num
    num_tri = valid_mask.shape[1]

    if isinstance(miss_tri_indices, torch.Tensor):
        miss_list = miss_tri_indices.detach().long().flatten().tolist()
    else:
        miss_list = list(miss_tri_indices)
    miss_list = sorted({int(m) for m in miss_list if 0 <= int(m) < num_tri})

    _, valid_padded, tri_pos = model.construct_seq(
        tri_vpos_list, texture_patch_list, valid_mask, vns
    )
    vi_out = vi_cached.clone()

    if not miss_list:
        return vi_out

    win: set[int] = set()
    for i in miss_list:
        lo = max(0, i - window_radius)
        hi = min(num_tri - 1, i + window_radius)
        win.update(range(lo, hi + 1))
    wsorted = sorted(win)

    idx_list = list(range(skip)) + [skip + w for w in wsorted]
    idx_tensor = torch.tensor(idx_list, device=device, dtype=torch.long)

    # 在已有 VI（如 vi_gold）子序列上跑 K 层，而不是从初始嵌入 seq 重算，
    # 否则窗口内会被「仅 K 层」表示覆盖，与窗外全量 VI 混用会破坏第一帧。
    x_win = vi_cached[:, idx_tensor, :].clone()
    tri_win = tri_pos[:, idx_tensor, :]
    valid_win = valid_padded[:, idx_tensor]

    orig = torch.tensor([-1] * skip + wsorted, device=device, dtype=torch.long)
    attn_bias = build_triangle_local_attn_bias(orig, window_radius, device)

    n_layers = len(model.transformer.layers)
    k = max(0, min(int(num_refiner_layers), n_layers))
    if k == 0:
        return vi_out

    layers = model.transformer.layers[:k]
    x_ref = _run_layers_subset(model.transformer, layers, x_win, valid_win, tri_win, attn_bias)

    vi_out[:, :skip] = x_ref[:, :skip]
    for wi, w in enumerate(wsorted):
        vi_out[:, skip + w] = x_ref[:, skip + wi]

    return vi_out

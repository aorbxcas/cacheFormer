from typing import List, Optional, Tuple

import numpy as np
import torch

from renderformer.temporal_vi.state import TemporalVIConfig, TemporalVIState


def decide_force_full(
    state: TemporalVIState,
    cfg: TemporalVIConfig,
    num_tris: int,
    block_keys_curr: List[bytes],
    seq_len_curr: int,
) -> Tuple[bool, str]:
    """
    在渲染当前帧之前调用（使用 state.frame_index 作为「当前是第几帧」的序号）。
    block_keys_curr: 本帧按块顺序的哈希列表。
    seq_len_curr: 本帧 seq_full 长度 L（应与 vi_ref 的 L 一致才可近似）。
    """
    fi = state.frame_index

    if state.vi_ref_np is None:
        return True, "cold_start"

    if state.num_tris >= 0 and num_tris != state.num_tris:
        return True, "tri_count_change"

    if state.vi_ref_np.shape[1] != seq_len_curr:
        return True, "seq_len_mismatch"

    if cfg.changed_block_ratio_threshold is not None and state.last_block_keys is not None:
        if len(block_keys_curr) != len(state.last_block_keys):
            return True, "block_count_change"
        changed = sum(1 for a, b in zip(block_keys_curr, state.last_block_keys) if a != b)
        n = len(block_keys_curr)
        ratio = changed / max(n, 1)
        if ratio > cfg.changed_block_ratio_threshold:
            return True, "block_change_ratio"

    if cfg.full_every_k > 0 and (fi - state.last_full_frame) >= cfg.full_every_k:
        return True, "periodic_k"

    if state.consecutive_approx >= cfg.max_consecutive_approx:
        return True, "max_consecutive_approx"

    return False, "approx_ok"


def apply_vi_approximation(
    seq_curr: torch.Tensor,
    vi_ref_np: np.ndarray,
    cfg: TemporalVIConfig,
    latent_dim: int,
) -> torch.Tensor:
    """
    seq_curr: [B, L, D] 当前帧拼接后的序列（与进 VI 前一致）。
    vi_ref_np: [B, L, D] CPU float32。
    """
    if cfg.approx_mode == "level0":
        t = torch.from_numpy(vi_ref_np).to(seq_curr.device, dtype=seq_curr.dtype)
        return t

    if cfg.approx_mode == "level1":
        ref = torch.from_numpy(vi_ref_np).to(seq_curr.device, dtype=torch.float32)
        s = seq_curr.float()
        s_norm = torch.nn.functional.layer_norm(s, (latent_dim,))
        alpha = float(cfg.blend_alpha)
        out = alpha * s_norm + (1.0 - alpha) * ref
        return out.to(seq_curr.dtype)

    raise ValueError(f"Unknown approx_mode: {cfg.approx_mode}")

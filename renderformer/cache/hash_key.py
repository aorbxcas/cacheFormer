"""
Block-level hash key for cache. Key = 128-bit fingerprint of (tri_vpos, texture summary, vn).
"""
import hashlib
import numpy as np
from typing import Union

import torch


def _to_bytes_f32(x: Union[np.ndarray, torch.Tensor]) -> bytes:
    """Convert array to contiguous float32 bytes for hashing."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return np.ascontiguousarray(x.astype(np.float32)).tobytes()


def compute_block_hash(
    tri_vpos: Union[torch.Tensor, np.ndarray],
    texture: Union[torch.Tensor, np.ndarray],
    vn: Union[torch.Tensor, np.ndarray],
    quantize_bits: int = 0,
) -> bytes:
    """
    Compute 128-bit (16 bytes) hash for a block of triangles.
    View- and camera-independent; only geometry + material.

    Args:
        tri_vpos: [1, N, 9] or [N, 9] - triangle vertex positions (flattened)
        texture: [1, N, C, H, W] or [N, C, H, W] - texture; summarized by mean over H,W if 5D
        vn: [1, N, 3, 3] or [N, 9] - vertex normals
        quantize_bits: 0 = no quantization; >0 quantize to reduce float noise (e.g. 12)

    Returns:
        16-byte key for use in cache.
    """
    if isinstance(tri_vpos, torch.Tensor):
        tri_vpos = tri_vpos.detach().cpu().float().numpy()
    if isinstance(texture, torch.Tensor):
        texture = texture.detach().cpu().float().numpy()
    if isinstance(vn, torch.Tensor):
        vn = vn.detach().cpu().float().numpy()

    tri_vpos = np.asarray(tri_vpos, dtype=np.float32)
    texture = np.asarray(texture, dtype=np.float32)
    vn = np.asarray(vn, dtype=np.float32)

    if tri_vpos.ndim == 3:
        tri_vpos = tri_vpos.reshape(-1, 9)
    if vn.ndim == 4:
        vn = vn.reshape(-1, 9)
    elif vn.ndim == 3:
        vn = vn.reshape(-1, 9)

    # Texture: reduce spatial dims to keep hash small and robust
    if texture.ndim == 5:
        # [..., C, H, W] -> mean over H, W -> [..., N, C]
        texture = texture.reshape(texture.shape[0], texture.shape[1], -1).mean(axis=-1)
    if texture.ndim == 4:
        texture = texture.reshape(texture.shape[0], -1)
    texture = texture.astype(np.float32)

    if quantize_bits > 0:
        scale = float(1 << quantize_bits)
        tri_vpos = (tri_vpos * scale).round() / scale
        vn = (vn * scale).round() / scale
        texture = (texture * scale).round() / scale

    h = hashlib.md5()
    h.update(_to_bytes_f32(tri_vpos))
    h.update(_to_bytes_f32(vn))
    h.update(_to_bytes_f32(texture))
    return h.digest()[:16]

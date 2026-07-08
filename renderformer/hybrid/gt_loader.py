from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def gt_full_path(
    gt_cache_dir: str | Path,
    scene_fp: str,
    resolution: int,
    view_id: int,
) -> Path:
    return Path(gt_cache_dir) / scene_fp / str(resolution) / f"{view_id:04d}_gt_full.exr"


def load_gt_full(
    gt_cache_dir: str | Path,
    scene_fp: str,
    resolution: int,
    view_id: int,
) -> Optional[np.ndarray]:
    """
    G0：离线 Blender GT（仅评测）。文件不存在时返回 None。
    """
    path = gt_full_path(gt_cache_dir, scene_fp, resolution, view_id)
    if not path.is_file():
        return None
    try:
        import imageio.v3 as iio
    except ImportError:
        return None
    return iio.imread(path).astype(np.float32)

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TemporalVIConfig:
    """调度近似 VI 与强制全算 VI 的策略配置。"""

    full_every_k: int = 8
    """距离上次全算已满 K 个「已渲染帧」则强制全算；0 表示不按周期触发。"""

    max_consecutive_approx: int = 32
    """连续近似帧数上限，超过则强制全算（安全网）。"""

    changed_block_ratio_threshold: Optional[float] = None
    """本帧相对上一帧，块哈希变化比例超过该值则全算；None 表示不启用。"""

    approx_mode: str = "level0"
    """level0: 直接使用上次 VI 输出；level1: 与当前 construct_seq 做逐 token 混合。"""

    blend_alpha: float = 0.15
    """level1 时 vi_approx = alpha * LayerNorm(seq_curr) + (1-alpha) * vi_ref（float 域）。"""


@dataclass
class TemporalVIState:
    """跨帧状态；由调用方在视频序列上持久化同一实例。"""

    vi_ref_np: Optional[np.ndarray] = None
    """上次全算 VI 输出，CPU float32，形状 [1, L, D]。"""

    last_block_keys: Optional[List[bytes]] = None
    """上一帧各块哈希，用于变化率。"""

    num_tris: int = -1
    """上一帧有效三角数（用于检测拓扑变化）。"""

    frame_index: int = 0
    """已成功完成的渲染次数（下一帧到来前的计数）。"""

    last_full_frame: int = -10**9
    """最近一次全算时的 frame_index（与 frame_index 同语义）。"""

    consecutive_approx: int = 0
    """当前连续近似帧数；全算后清零。"""

    cumulative: Dict[str, int] = field(
        default_factory=lambda: {
            "full_vi": 0,
            "approx_vi": 0,
        }
    )
    force_reason_hist: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "last_full_frame": self.last_full_frame,
            "consecutive_approx": self.consecutive_approx,
            "has_vi_ref": self.vi_ref_np is not None,
            "vi_ref_shape": None if self.vi_ref_np is None else list(self.vi_ref_np.shape),
            "num_tris": self.num_tris,
            "cumulative_full_vi": self.cumulative.get("full_vi", 0),
            "cumulative_approx_vi": self.cumulative.get("approx_vi", 0),
            "force_reason_hist": dict(self.force_reason_hist),
        }

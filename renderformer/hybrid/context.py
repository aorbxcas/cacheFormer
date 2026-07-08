from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import torch


@dataclass
class HybridRenderContext:
    """路径 B 单帧/多视角混合渲染上下文。"""

    hdr_neural: torch.Tensor
    hdr_direct: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    indirect_neural: torch.Tensor
    hdr_fused: torch.Tensor
    violations: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

# -*- coding: utf-8 -*-
"""
视图无关（View-Independent）阶段输出的 LRU 缓存。

设计原因：
- VI 阶段仅依赖三角网格 + 材质纹理 + 法线 + mask，与相机无关；
- 视频或多视角下场景不变时，重复计算 12 层 Transformer 是冗余的；
- 缓存键必须覆盖与 VI 输入一致的数据，否则会出现错误复用。

注意：
- 当前实现面向 batch_size=1；多 batch 且场景不同时需按样本分 key（可扩展）。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch


def scene_fingerprint(
    triangles: torch.Tensor,
    texture: torch.Tensor,
    vn: torch.Tensor,
    mask: torch.Tensor,
) -> str:
    """
    为「当前帧喂给 VI 的几何+材质」生成稳定哈希键。

    原因：
    - 任意顶点/材质/可见性变化都必须产生新键，否则会错误命中旧缓存；
    - 使用 CPU float32 连续内存字节，避免 GPU 非确定性排序；
    - 不包含相机参数（c2w/fov），因为 VI 与相机无关。

    Args:
        triangles: [B, N, 3, 3] 或已 flatten 前的网格顶点
        texture: 与 pipeline 在调用模型前一致（含 log10 编码后的光照通道时须一致）
        vn: 顶点法线，形状与模型输入一致
        mask: [B, N] bool，有效三角掩码

    Returns:
        64 字符十六进制 SHA256 摘要
    """
    h = hashlib.sha256()
    # 按固定顺序拼接，保证同一场景键稳定
    for tensor in (triangles, texture, vn):
        t = tensor.detach().cpu().to(torch.float32).contiguous()
        h.update(t.numpy().tobytes())
    m = mask.detach().cpu().to(torch.bool).contiguous()
    h.update(m.numpy().tobytes())
    return h.hexdigest()


class ViewIndependentCache:
    """
    LRU：显存/内存有限时淘汰最久未使用的场景 VI 特征。

    存储内容：
    - seq_vi: VI 阶段输出 [1, L, D]，与推理 autocast  dtype 一致（如 fp16）；
    - valid_mask_padded: [1, L] bool，与 seq_vi 同行，供 VD 使用。

    原因：
    - 仅存 CPU 张量可降低 GPU 占用；命中时再 .to(device)；
    - detach 后无计算图，避免显存泄漏。
    """

    def __init__(self, max_entries: int = 8):
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        # key -> (seq_vi_cpu, mask_padded_cpu)
        self._store: "OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """LRU：访问即移到队尾（最近使用）。"""
        if key not in self._store:
            self.misses += 1
            return None
        self.hits += 1
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, seq_vi: torch.Tensor, valid_mask_padded: torch.Tensor) -> None:
        """写入缓存；超容量时淘汰队首（最久未使用）。"""
        self._store[key] = (seq_vi, valid_mask_padded)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "entries": len(self._store),
        }

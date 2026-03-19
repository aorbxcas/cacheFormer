"""
LRU block cache: key = block_hash (16 bytes), value = (tri_emb_slice) for that block.
"""
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


class BlockCache:
    """LRU cache for block embeddings. Keys are 16-byte hashes; values are numpy arrays (tri_emb)."""

    def __init__(self, max_entries: int = 10000, max_memory_mb: Optional[float] = None):
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self._cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: bytes) -> Optional[torch.Tensor]:
        """Get block embedding for key; return None if miss. Moves to end (recent)."""
        if key not in self._cache:
            self._misses += 1
            return None
        self._hits += 1
        val = self._cache.pop(key)
        self._cache[key] = val  # move to end
        return torch.from_numpy(val.copy())

    def put(self, key: bytes, tri_emb: torch.Tensor) -> None:
        """Store block embedding. Evict oldest if over capacity."""
        tri_emb_np = tri_emb.detach().cpu().float().numpy()
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = tri_emb_np
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        if self.max_memory_mb is not None:
            total_mb = sum(v.nbytes for v in self._cache.values()) / (1024 * 1024)
            while total_mb > self.max_memory_mb and self._cache:
                self._cache.popitem(last=False)
                total_mb = sum(v.nbytes for v in self._cache.values()) / (1024 * 1024)

    def invalidate_block(self, key: bytes) -> bool:
        """Remove one entry; return True if key was present."""
        if key in self._cache:
            self._cache.pop(key)
            return True
        return False

    def stats(self) -> Dict[str, Any]:
        """Return hit/miss counts and cache size."""
        total_bytes = sum(v.nbytes for v in self._cache.values())
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "memory_mb": round(total_bytes / (1024 * 1024), 2),
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0,
        }

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0

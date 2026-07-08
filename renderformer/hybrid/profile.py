from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class HybridProfile:
    version: int = 2
    mode: str = "path_b_runtime_direct"
    scene_fp: str = ""
    source_json: str = ""
    h5_path: str = ""
    direct_scale_s: float = 1.0
    direct_bias_b: float = 0.0
    fitted_against: str = ""
    runtime_backend: str = "auto"
    shadow_map_size: int = 1024
    brdf: str = "ggx_simplified"
    confidence: Dict[str, float] = field(default_factory=lambda: {
        "w0": 2.0,
        "w1": 4.0,
        "w2": 1.0,
        "w3": 2.0,
        "w4": 0.0,
    })
    use_physics_correct: bool = True
    fixed_alpha: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridProfile":
        alignment = data.get("alignment", {})
        runtime = data.get("runtime_direct", {})
        return cls(
            version=int(data.get("version", 2)),
            mode=str(data.get("mode", "path_b_runtime_direct")),
            scene_fp=str(data.get("scene_fp", "")),
            source_json=str(data.get("source_json", "")),
            h5_path=str(data.get("h5_path", "")),
            direct_scale_s=float(alignment.get("direct_scale_s", 1.0)),
            direct_bias_b=float(alignment.get("direct_bias_b", 0.0)),
            fitted_against=str(alignment.get("fitted_against", "")),
            runtime_backend=str(runtime.get("backend", "auto")),
            shadow_map_size=int(runtime.get("shadow_map_size", 1024)),
            brdf=str(runtime.get("brdf", "ggx_simplified")),
            confidence=dict(data.get("confidence", {})) or cls().confidence,
            use_physics_correct=bool(data.get("use_physics_correct", True)),
            fixed_alpha=data.get("fixed_alpha"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "scene_fp": self.scene_fp,
            "source_json": self.source_json,
            "h5_path": self.h5_path,
            "alignment": {
                "direct_scale_s": self.direct_scale_s,
                "direct_bias_b": self.direct_bias_b,
                "fitted_against": self.fitted_against,
            },
            "runtime_direct": {
                "backend": self.runtime_backend,
                "shadow_map_size": self.shadow_map_size,
                "brdf": self.brdf,
            },
            "confidence": self.confidence,
            "use_physics_correct": self.use_physics_correct,
            "fixed_alpha": self.fixed_alpha,
        }


def load_hybrid_profile(path: str | Path) -> HybridProfile:
    with open(path, "r", encoding="utf-8") as f:
        return HybridProfile.from_dict(json.load(f))


def save_hybrid_profile(profile: HybridProfile, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)

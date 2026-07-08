from __future__ import annotations

import warnings
from typing import Any, Literal, Union

BackendName = Literal["auto", "nvdiffrast", "lite"]


def nvdiffrast_available() -> bool:
    try:
        import nvdiffrast.torch as dr  # noqa: F401

        return True
    except ImportError:
        return False


def create_runtime_direct_renderer(
    backend: BackendName = "auto",
    shadow_map_size: int = 1024,
    **kwargs: Any,
):
    """
    创建 Runtime Direct 渲染器。

    Args:
        backend: auto | nvdiffrast | lite
        shadow_map_size: nvdiffrast shadow map 分辨率
        **kwargs: 传给具体 backend（ambient, scene_center, ...）
    """
    if backend == "auto":
        backend = "nvdiffrast" if nvdiffrast_available() else "lite"

    if backend == "nvdiffrast":
        if not nvdiffrast_available():
            warnings.warn(
                "nvdiffrast 未安装，回退到 R0-lite。请执行: "
                "pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git",
                UserWarning,
                stacklevel=2,
            )
            backend = "lite"
        else:
            from renderformer.hybrid.runtime_direct.renderer_nvd import NvdiffrastDirectRenderer

            return NvdiffrastDirectRenderer(
                shadow_map_size=shadow_map_size,
                **kwargs,
            )

    from renderformer.hybrid.runtime_direct.renderer_lite import RuntimeDirectRendererLite

    return RuntimeDirectRendererLite(**kwargs)


class RuntimeDirectRenderer:
    """统一接口：按 backend 委托到 nvdiffrast 或 lite。"""

    def __init__(
        self,
        backend: BackendName = "auto",
        shadow_map_size: int = 1024,
        **kwargs: Any,
    ):
        self.backend = backend
        self._impl = create_runtime_direct_renderer(
            backend=backend,
            shadow_map_size=shadow_map_size,
            **kwargs,
        )

    @property
    def active_backend(self) -> str:
        return type(self._impl).__name__

    def render(self, *args, **kwargs):
        return self._impl.render(*args, **kwargs)

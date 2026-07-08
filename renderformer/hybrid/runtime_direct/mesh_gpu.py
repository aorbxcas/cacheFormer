from __future__ import annotations

from dataclasses import dataclass

import torch

from renderformer.hybrid.runtime_direct.mesh_buffer import MeshBuffer


@dataclass
class FlatMeshGPU:
    """展平三角网格，供 nvdiffrast 光栅化。"""

    vertices: torch.Tensor
    faces: torch.Tensor
    global_tri_ids: torch.Tensor
    shadow_vertices: torch.Tensor
    shadow_faces: torch.Tensor

    @classmethod
    def from_mesh_buffer(cls, mesh: MeshBuffer) -> "FlatMeshGPU":
        visible_idx = mesh.mask.nonzero(as_tuple=False).squeeze(-1)
        tris = mesh.triangles[visible_idx]
        t = tris.shape[0]
        vertices = tris.reshape(-1, 3).contiguous()
        faces = torch.arange(t * 3, device=vertices.device, dtype=torch.int32).reshape(t, 3)
        global_tri_ids = visible_idx

        sh_idx = mesh.shadow_caster_mask.nonzero(as_tuple=False).squeeze(-1)
        sh_tris = mesh.triangles[sh_idx]
        st = sh_tris.shape[0]
        shadow_vertices = sh_tris.reshape(-1, 3).contiguous()
        shadow_faces = torch.arange(st * 3, device=vertices.device, dtype=torch.int32).reshape(st, 3)

        return cls(
            vertices=vertices,
            faces=faces,
            global_tri_ids=global_tri_ids,
            shadow_vertices=shadow_vertices,
            shadow_faces=shadow_faces,
        )

    def tri_id_map_from_rast(self, rast: torch.Tensor) -> torch.Tensor:
        """rast [1,H,W,4] → global tri id [H,W]，背景为 -1。

        nvdiffrast 的 triangle_id 为 1-based，0 表示背景。
        """
        # nvdiffrast: w 通道 = triangle_id（1-based，0=背景）
        local = rast[0, ..., 3].long() - 1
        h, w = local.shape
        out = torch.full((h, w), -1, device=local.device, dtype=torch.long)
        valid = local >= 0
        out[valid] = self.global_tri_ids[local[valid]]
        return out

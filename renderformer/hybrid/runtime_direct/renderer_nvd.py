from __future__ import annotations

from typing import Optional, Tuple

import torch

from renderformer.hybrid.runtime_direct.camera import light_mvp, project_world_to_light_ndc, world_to_clip
from renderformer.hybrid.runtime_direct.lights import EmissiveLight, extract_emissive_lights
from renderformer.hybrid.runtime_direct.mesh_buffer import MeshBuffer
from renderformer.hybrid.runtime_direct.mesh_gpu import FlatMeshGPU
from renderformer.hybrid.runtime_direct.shading import sample_shadow_map, shade_surface_direct


class NvdiffrastDirectRenderer:
    """
    R0-nvd：nvdiffrast CUDA 光栅 + shadow map。

    需安装 nvdiffrast 与 NVIDIA GPU。
    """

    def __init__(
        self,
        shadow_map_size: int = 1024,
        shadow_bias: float = 5e-4,
        min_emissive: float = 1.0,
        ambient: float = 0.0,
        scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        light_extent: float = 2.5,
        prefer_gl: bool = False,
    ):
        import nvdiffrast.torch as dr

        self._dr = dr
        if prefer_gl:
            self.glctx = dr.RasterizeGLContext(output_db=False)
        else:
            self.glctx = dr.RasterizeCudaContext()
        self.shadow_map_size = shadow_map_size
        self.shadow_bias = shadow_bias
        self.min_emissive = min_emissive
        self.ambient = ambient
        self.scene_center = torch.tensor(scene_center, dtype=torch.float32)
        self.light_extent = light_extent

    def render(
        self,
        triangles: torch.Tensor,
        texture: torch.Tensor,
        vn: torch.Tensor,
        mask: torch.Tensor,
        c2w: torch.Tensor,
        fov: torch.Tensor,
        resolution: int = 512,
        depth_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mesh = MeshBuffer.from_scene_tensors(triangles, vn, texture, mask)
        flat = FlatMeshGPU.from_mesh_buffer(mesh)
        lights = extract_emissive_lights(
            mesh,
            min_emissive=self.min_emissive,
            scene_center=self.scene_center.to(triangles.device),
        )

        bs, nv = c2w.shape[0], c2w.shape[1]
        if fov.dim() == 2:
            fov = fov.unsqueeze(-1)

        hdr_views = []
        depth_views = []
        for b in range(bs):
            for v in range(nv):
                hdr, depth = self._render_single_view(
                    mesh=mesh,
                    flat=flat,
                    lights=lights,
                    c2w=c2w[b, v],
                    fov_deg=fov[b, v, 0],
                    resolution=resolution,
                )
                if depth_only:
                    hdr = torch.zeros_like(hdr)
                hdr_views.append(hdr)
                depth_views.append(depth)

        hdr = torch.stack(hdr_views, dim=0).reshape(bs, nv, resolution, resolution, 3)
        depth = torch.stack(depth_views, dim=0).reshape(bs, nv, resolution, resolution, 1)
        return hdr, depth

    def _render_single_view(
        self,
        mesh: MeshBuffer,
        flat: FlatMeshGPU,
        lights: list[EmissiveLight],
        c2w: torch.Tensor,
        fov_deg: torch.Tensor,
        resolution: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dr = self._dr
        device = c2w.device
        dtype = c2w.dtype
        H = W = resolution

        pos_clip = world_to_clip(flat.vertices.to(dtype).float(), c2w, fov_deg)
        rast, _ = dr.rasterize(self.glctx, pos_clip.unsqueeze(0), flat.faces, (H, W))

        world_pos, _ = dr.interpolate(
            flat.vertices.unsqueeze(0).to(dtype),
            rast,
            flat.faces,
        )
        world_pos = world_pos[0]

        tri_map = flat.tri_id_map_from_rast(rast)
        cam_pos = c2w[:3, 3]
        view_dir = cam_pos - world_pos
        view_dir = torch.nn.functional.normalize(view_dir, dim=-1)

        shadow_vis = torch.ones((H, W, 1), device=device, dtype=dtype)
        if lights and flat.shadow_vertices.shape[0] > 0:
            shadow_vis = self._shadow_visibility(
                flat=flat,
                world_pos=world_pos,
                lights=lights,
                device=device,
                dtype=dtype,
            )

        hdr = shade_surface_direct(
            mesh=mesh,
            global_tri=tri_map,
            world_pos=world_pos,
            view_dir=view_dir,
            lights=lights,
            shadow_visible=shadow_vis,
            ambient=self.ambient,
        )

        valid = tri_map >= 0
        dist = torch.linalg.norm(world_pos - cam_pos, dim=-1, keepdim=True)
        depth = torch.where(valid.unsqueeze(-1), dist, torch.zeros_like(dist))

        # nvdiffrast 光栅行 0 在 OpenGL 底边；RF RayGenerator / lite 行 0 在图像顶边
        hdr = torch.flip(hdr, dims=[0])
        depth = torch.flip(depth, dims=[0])
        return hdr, depth

    def _shadow_visibility(
        self,
        flat: FlatMeshGPU,
        world_pos: torch.Tensor,
        lights: list[EmissiveLight],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        dr = self._dr
        H, W, _ = world_pos.shape
        light = lights[0]
        target = self.scene_center.to(device=device, dtype=dtype)
        mvp = light_mvp(
            light.position.to(device=device, dtype=dtype),
            target,
            half_extent=self.light_extent,
        )

        pos_light_clip = project_world_to_light_ndc(
            flat.shadow_vertices.to(dtype).float(),
            mvp,
        )[0]
        sh_rast, _ = dr.rasterize(
            self.glctx,
            pos_light_clip.unsqueeze(0),
            flat.shadow_faces,
            (self.shadow_map_size, self.shadow_map_size),
        )
        shadow_depth = sh_rast[..., 2:3].detach()

        _, light_uv, light_z = project_world_to_light_ndc(
            world_pos.reshape(-1, 3),
            mvp,
        )
        light_uv = light_uv.reshape(H, W, 2)
        light_z = light_z.reshape(H, W, 1)

        return sample_shadow_map(shadow_depth, light_uv, light_z, bias=self.shadow_bias)

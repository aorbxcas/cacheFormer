from typing import Optional

import torch

from renderformer.cache import BlockCache, compute_block_hash
from renderformer.models.renderformer import RenderFormer
from renderformer.utils.ray_generator import RayGenerator
from renderformer.utils.transform import trans_to_cam_coord


def _partition_blocks(num_tris: int, block_size: int):
    """Yield (start, end) slices for blocks. Last block may be smaller."""
    for start in range(0, num_tris, block_size):
        end = min(start + block_size, num_tris)
        yield start, end


class RenderFormerRenderingPipeline:
    def __init__(self, model: RenderFormer):
        self.model = model
        self.config = model.config
        self.ray_generator = RayGenerator().to(model.device)

    @classmethod
    def from_pretrained(cls, model_id: str):
        model = RenderFormer.from_pretrained(model_id)
        model.eval()
        return cls(model)

    @property
    def device(self):
        return self.model.device

    def to(self, device: torch.device):
        self.model.to(device)
        self.ray_generator.to(device)

    def render(
        self,
        triangles,
        texture,
        mask,
        vn,
        c2w,
        fov,
        resolution: int = 512,
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Render images using the RenderFormer model
        
        Args:
            model: RenderFormer model
            config: RenderFormerConfig object
            triangles: Triangle data tensor [bs, num_tris, 3, 3] - vertices of triangles
            texture: Texture data tensor [bs, num_tris, C, texture_size, texture_size] or [bs, num_tris, C]
            mask: Mask data tensor [bs, num_tris] - boolean mask indicating valid triangles
            vn: Vertex normal vectors tensor [bs, num_tris, 3, 3] - normal vectors of triangles
            c2w: Camera-to-world matrix tensor [bs, num_views, 4, 4]
            fov: Field of view tensor [bs, num_views, 1] - in degrees
            resolution: Render resolution (default: 512)
            torch_dtype: PyTorch dtype for inference, default is torch.float16

        Returns:
            torch.Tensor: Rendered HDR image tensor [bs, num_views, H, W, 3]
        """

        bs, nv = c2w.shape[0], c2w.shape[1]

        # Process data according to config
        # If texture patch size is 1, simplify the texture tensor
        # texture: [bs, num_tris, C, texture_size, texture_size] -> [bs, num_tris, C]
        if self.config.texture_encode_patch_size == 1 and texture.dim() == 5:
            texture = texture[:, :, :, 0, 0]

        # Log encode lighting if not learning LDR directly
        if not self.config.use_ldr:
            texture[:, :, -3:] = torch.log10(texture[:, :, -3:] + 1.)

        # Handle view transformation
        if self.config.turn_to_cam_coord:
            # Reshape for transformation
            # c2w: [bs, nv, 4, 4] -> [bs*nv, 4, 4]
            # triangles: [bs, num_tris, 3, 3] -> [bs*nv, num_tris, 3, 3] by repeating
            c2w_reshaped = c2w.reshape(-1, 4, 4)
            triangles_repeated = torch.repeat_interleave(triangles, nv, dim=0)
            
            tris_for_view_tf, c2w_for_view_tf, _ = trans_to_cam_coord(
                c2w_reshaped,
                triangles_repeated
            )
            # Reshape back
            # c2w_for_view_tf: [bs*nv, 4, 4] -> [bs, nv, 4, 4]
            # tris_for_view_tf: [bs*nv, num_tris, 3, 3] -> [bs, nv, num_tris, 3, 3]
            c2w_for_view_tf = c2w_for_view_tf.reshape(bs, nv, 4, 4)
            tris_for_view_tf = tris_for_view_tf.reshape(bs, nv, -1, 3, 3)
        else:
            # Expand triangles for each view
            # triangles: [bs, num_tris, 3, 3] -> [bs, nv, num_tris, 3, 3]
            tris_for_view_tf = triangles.unsqueeze(1).expand(-1, nv, -1, -1, -1)
            c2w_for_view_tf = c2w

        # Generate rays
        # rays_o, rays_d: [bs, nv, H, W, 3]
        rays_o, rays_d = self.ray_generator(c2w_for_view_tf, fov / 180. * torch.pi, resolution)

        # Set precision
        assert torch_dtype in [torch.bfloat16, torch.float16, torch.float32], f"Invalid precision: {torch_dtype}\nChoose from: torch.bfloat16, torch.float16, torch.float32"
        tf32_view_tf = torch_dtype == torch.bfloat16 or torch_dtype == torch.float16

        # Perform rendering
        # Flatten triangles: [bs, num_tris, 3, 3] -> [bs, num_tris*9]
        # Flatten vn: [bs, num_tris, 3, 3] -> [bs, num_tris*9]
        # Flatten tri_vpos_view_tf: [bs, nv, num_tris, 3, 3] -> [bs, nv, num_tris*9]
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch_dtype):
            rendered_imgs = self.model(
                triangles.reshape(bs, -1, 9),
                texture,
                mask,
                vn.reshape(bs, -1, 9),
                rays_o=rays_o,
                rays_d=rays_d,
                tri_vpos_view_tf=tris_for_view_tf.reshape(bs, nv, -1, 9),
                tf32_view_tf=tf32_view_tf,
            )

        # Process output
        # rendered_imgs: [bs, nv, C, H, W] -> [bs, nv, H, W, C]
        rendered_imgs = rendered_imgs.permute(0, 1, 3, 4, 2)

        # Log decode lighting if needed
        if not self.config.use_ldr:
            rendered_imgs = torch.pow(10., rendered_imgs) - 1.

        return rendered_imgs

    def render_with_block_cache(
        self,
        triangles: torch.Tensor,
        texture: torch.Tensor,
        mask: torch.Tensor,
        vn: torch.Tensor,
        c2w: torch.Tensor,
        fov: torch.Tensor,
        block_cache: BlockCache,
        block_size: int = 256,
        resolution: int = 512,
        torch_dtype: torch.dtype = torch.float16,
        verbose: bool = True,
    ) -> tuple:
        """
        Render using block-level cache (encode triangles per block, cache by block hash;
        reassemble and run global transformer + view_transformer).
        Returns (rendered_imgs, cache_stats_dict).
        """
        bs, nv = c2w.shape[0], c2w.shape[1]
        num_tris = triangles.shape[1]
        skip = self.model.skip_token_num

        # Same preprocessing as render()
        if self.config.texture_encode_patch_size == 1 and texture.dim() == 5:
            texture = texture[:, :, :, 0, 0]
        if not self.config.use_ldr:
            texture = texture.clone()
            texture[:, :, -3:] = torch.log10(texture[:, :, -3:] + 1.0)

        if self.config.turn_to_cam_coord:
            c2w_reshaped = c2w.reshape(-1, 4, 4)
            triangles_repeated = torch.repeat_interleave(triangles, nv, dim=0)
            tris_for_view_tf, c2w_for_view_tf, _ = trans_to_cam_coord(
                c2w_reshaped, triangles_repeated
            )
            c2w_for_view_tf = c2w_for_view_tf.reshape(bs, nv, 4, 4)
            tris_for_view_tf = tris_for_view_tf.reshape(bs, nv, -1, 3, 3)
        else:
            tris_for_view_tf = triangles.unsqueeze(1).expand(-1, nv, -1, -1, -1)
            c2w_for_view_tf = c2w

        rays_o, rays_d = self.ray_generator(
            c2w_for_view_tf, fov / 180.0 * torch.pi, resolution
        )
        tf32_view_tf = torch_dtype in (torch.bfloat16, torch.float16)
        tri_vpos = triangles.reshape(bs, -1, 9)
        vn_flat = vn.reshape(bs, -1, 9)

        # Block cache path: only support bs=1 for clarity
        if bs != 1:
            if verbose:
                print("[block_cache] batch size > 1, falling back to normal render")
            out = self.render(
                triangles=triangles,
                texture=texture,
                mask=mask,
                vn=vn,
                c2w=c2w,
                fov=fov,
                resolution=resolution,
                torch_dtype=torch_dtype,
            )
            return out, block_cache.stats()

        tri_emb_list = []
        for start, end in _partition_blocks(num_tris, block_size):
            tri_vpos_b = tri_vpos[:, start:end, :]
            texture_b = texture[:, start:end]
            if texture_b.dim() == 3:
                p = self.config.texture_encode_patch_size
                texture_b = texture_b.unsqueeze(-1).unsqueeze(-1).expand(
                    -1, -1, -1, p, p
                )
            vn_b = vn_flat[:, start:end, :]
            mask_b = mask[:, start:end]

            block_key = compute_block_hash(
                tri_vpos_b.cpu().numpy(),
                texture_b.cpu().numpy(),
                vn_b.cpu().numpy(),
            )
            cached = block_cache.get(block_key)
            if cached is not None:
                cached = cached.to(self.device, dtype=torch.float32)
                tri_emb_list.append(cached)
            else:
                with torch.no_grad(), torch.autocast(
                    device_type=self.device.type, dtype=torch.float32
                ):
                    seq_b, valid_b, tri_vpos_b_proc = self.model.construct_seq(
                        tri_vpos_b, texture_b, mask_b, vn_b
                    )
                tri_emb_b = seq_b[:, skip:, :].detach().float()
                block_cache.put(block_key, tri_emb_b)
                tri_emb_list.append(tri_emb_b)

        tri_emb_full = torch.cat(tri_emb_list, dim=1)
        reg_tokens = self.model.reg_tokens.expand(bs, -1, -1)
        seq_full = torch.cat([reg_tokens, tri_emb_full], dim=1)
        tri_vpos_full = tri_vpos
        valid_full = mask
        tri_vpos_list, valid_mask_padded = self.model.process_tri_vpos_list(
            tri_vpos_full, valid_full
        )

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=torch_dtype
        ):
            rendered_imgs = self.model.forward_from_sequence(
                seq_full,
                valid_mask_padded,
                tri_vpos_list,
                rays_o=rays_o,
                rays_d=rays_d,
                tri_vpos_view_tf=tris_for_view_tf.reshape(bs, nv, -1, 9),
                tf32_view_tf=tf32_view_tf,
            )

        rendered_imgs = rendered_imgs.permute(0, 1, 3, 4, 2)
        if not self.config.use_ldr:
            rendered_imgs = torch.pow(10.0, rendered_imgs) - 1.0

        stats = block_cache.stats()
        if verbose:
            print(
                f"[block_cache] blocks={len(tri_emb_list)} block_size={block_size} "
                f"hits={stats['hits']} misses={stats['misses']} "
                f"hit_rate={stats['hit_rate']:.2%} size={stats['size']} memory_mb={stats['memory_mb']}"
            )
            print(
                f"[render] output shape={rendered_imgs.shape} "
                f"dtype={rendered_imgs.dtype} min={rendered_imgs.min().item():.4f} max={rendered_imgs.max().item():.4f}"
            )
        return rendered_imgs, stats

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)

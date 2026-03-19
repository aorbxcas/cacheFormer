import torch
from typing import TYPE_CHECKING, Any, Dict, Optional

from renderformer.models.renderformer import RenderFormer
from renderformer.utils.ray_generator import RayGenerator
from renderformer.utils.transform import trans_to_cam_coord

if TYPE_CHECKING:
    from renderformer.cache.vi_cache import ViewIndependentCache


class RenderFormerRenderingPipeline:
    """
    推理管线：数据预处理 →（可选）VI 缓存 → RenderFormer。

    VI 缓存策略说明：
    - 仅当 batch_size=1 且传入 vi_cache 时启用；多 batch 场景需按样本分 key，此处避免误用；
    - 指纹在「与模型 VI 输入一致」的张量上计算（含非 LDR 时对 texture 光照通道的 log10）；
    - 命中则跳过 12 层视图无关 Transformer，仅执行视图相关分支。
    """

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
        torch_dtype: torch.dtype = torch.float16,
        vi_cache: Optional["ViewIndependentCache"] = None,
        return_vi_cache_info: bool = False,
    ):
        """
        渲染 HDR 线性图像。

        Args:
            triangles, texture, mask, vn, c2w, fov: 与原版一致
            resolution, torch_dtype: 与原版一致
            vi_cache: 可选；ViewIndependentCache 实例，用于跨帧/跨视角复用 VI 特征
            return_vi_cache_info: 为 True 时额外返回本次是否命中 VI 缓存（便于实验统计）

        Returns:
            默认: [bs, nv, H, W, 3]
            若 return_vi_cache_info: (tensor, {"vi_cache_hit": bool, "vi_cache_key": str|None})
        """
        from renderformer.cache.vi_cache import scene_fingerprint

        bs, nv = c2w.shape[0], c2w.shape[1]

        # 与模型 config 一致：patch 为 1 时压成 [B,N,C]
        if self.config.texture_encode_patch_size == 1 and texture.dim() == 5:
            texture = texture[:, :, :, 0, 0]

        # 非 LDR 训练时，光照通道与 VI 一致需先 log 编码（与后续 forward 输入一致，指纹也必须在此之后算）
        if not self.config.use_ldr:
            texture = texture.clone()
            texture[:, :, -3:] = torch.log10(texture[:, :, -3:] + 1.0)

        # 相机系：将场景变换到各视角相机坐标（供射线与 VD 三角位置）
        if self.config.turn_to_cam_coord:
            c2w_reshaped = c2w.reshape(-1, 4, 4)
            triangles_repeated = torch.repeat_interleave(triangles, nv, dim=0)
            tris_for_view_tf, c2w_for_view_tf, _ = trans_to_cam_coord(
                c2w_reshaped,
                triangles_repeated,
            )
            c2w_for_view_tf = c2w_for_view_tf.reshape(bs, nv, 4, 4)
            tris_for_view_tf = tris_for_view_tf.reshape(bs, nv, -1, 3, 3)
        else:
            tris_for_view_tf = triangles.unsqueeze(1).expand(-1, nv, -1, -1, -1)
            c2w_for_view_tf = c2w

        rays_o, rays_d = self.ray_generator(
            c2w_for_view_tf, fov / 180.0 * torch.pi, resolution
        )

        assert torch_dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ), f"Invalid precision: {torch_dtype}"
        tf32_view_tf = torch_dtype in (torch.bfloat16, torch.float16)

        tri_flat = triangles.reshape(bs, -1, 9)
        vn_flat = vn.reshape(bs, -1, 9)
        tris_view_flat = tris_for_view_tf.reshape(bs, nv, -1, 9)

        vi_cache_hit = False
        cache_key: Optional[str] = None
        # 仅单 batch 启用：多场景 batch 共用同一缓存键会错误复用
        use_vi_cache = vi_cache is not None and bs == 1
        if vi_cache is not None and bs != 1:
            import warnings
            warnings.warn(
                "ViewIndependentCache 仅在 batch_size=1 时生效；当前 bs=%d，本帧将走完整 VI+VD。"
                % bs,
                UserWarning,
                stacklevel=2,
            )

        if use_vi_cache:
            cache_key = scene_fingerprint(triangles, texture, vn, mask)
            entry = vi_cache.get(cache_key)
        else:
            entry = None

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=torch_dtype
        ):
            if entry is not None:
                # ---------- 缓存命中：仅 VD ----------
                vi_cache_hit = True
                seq_vi, mask_p = entry
                seq_vi = seq_vi.to(self.device, non_blocking=True)
                mask_p = mask_p.to(self.device, non_blocking=True)
                rendered_imgs = self.model(
                    tri_flat,
                    texture,
                    mask,
                    vn_flat,
                    rays_o=rays_o,
                    rays_d=rays_d,
                    tri_vpos_view_tf=tris_view_flat,
                    tf32_view_tf=tf32_view_tf,
                    cached_seq_vi=seq_vi,
                    cached_valid_mask_padded=mask_p,
                )
            elif use_vi_cache:
                # ---------- 未命中：VI + VD，再写入 LRU ----------
                seq_vi, mask_p = self.model.encode_view_independent(
                    tri_flat, texture, mask, vn_flat
                )
                rendered_imgs = self.model.decode_view_dependent(
                    seq_vi,
                    mask_p,
                    rays_o,
                    rays_d,
                    tris_view_flat,
                    mask,
                    tf32_view_tf=tf32_view_tf,
                )
                vi_cache.put(
                    cache_key,
                    seq_vi.detach().cpu(),
                    mask_p.detach().cpu(),
                )
            else:
                # ---------- 无缓存或 batch>1：与原版相同 ----------
                rendered_imgs = self.model(
                    tri_flat,
                    texture,
                    mask,
                    vn_flat,
                    rays_o=rays_o,
                    rays_d=rays_d,
                    tri_vpos_view_tf=tris_view_flat,
                    tf32_view_tf=tf32_view_tf,
                )

        rendered_imgs = rendered_imgs.permute(0, 1, 3, 4, 2)

        if not self.config.use_ldr:
            rendered_imgs = torch.pow(10.0, rendered_imgs) - 1.0

        if return_vi_cache_info:
            info: Dict[str, Any] = {
                "vi_cache_hit": vi_cache_hit,
                "vi_cache_key": cache_key,
            }
            return rendered_imgs, info
        return rendered_imgs

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)

import torch
from torch import nn
from torch.amp import autocast

from huggingface_hub import PyTorchModelHubMixin

from renderformer.encodings.nerf_encoding import NeRFEncoding
from renderformer.layers.attention import TransformerEncoder
from renderformer.models.view_transformer import ViewTransformer
from renderformer.models.config import RenderFormerConfig


class RenderFormer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: RenderFormerConfig):
        super(RenderFormer, self).__init__()
        self.config = config

        if self.config.pe_type == 'nerf':
            # vertex PE and projections
            self.tri_vpos_pe = NeRFEncoding(
                in_dim=9,
                num_frequencies=self.config.vertex_pe_num_freqs,
                include_input=True
            )
            # triangle pe projection
            self.tri_encoding_proj = nn.Linear(
                self.tri_vpos_pe.get_out_dim(),
                self.config.latent_dim
            )
            # reuse this config for ablation...
            if self.config.vn_encoder_norm_type == 'layer_norm':
                self.tri_encoding_norm = nn.LayerNorm(self.config.latent_dim)
            elif self.config.vn_encoder_norm_type == 'rms_norm':
                self.tri_encoding_norm = nn.RMSNorm(self.config.latent_dim)
            elif self.config.vn_encoder_norm_type == 'none':
                self.tri_encoding_norm = nn.Identity()
            self.rope_dim = None
        elif self.config.pe_type == 'rope':
            self.rope_dim = self.config.vertex_pe_num_freqs
        else:
            raise ValueError(f"Invalid positional encoding type: {self.config.pe_type}")

        if self.config.use_vn_encoder:
            self.vn_pe = NeRFEncoding(
                in_dim=9,
                num_frequencies=self.config.vn_pe_num_freqs,
                include_input=True
            )
            self.vn_encoding_proj = nn.Linear(
                self.vn_pe.get_out_dim(),
                self.config.latent_dim
            )
            if self.config.vn_encoder_norm_type == 'layer_norm':
                self.vn_encoder_norm = nn.LayerNorm(self.config.latent_dim)
            elif self.config.vn_encoder_norm_type == 'rms_norm':
                self.vn_encoder_norm = nn.RMSNorm(self.config.latent_dim)
            elif self.config.vn_encoder_norm_type == 'none':
                self.vn_encoder_norm = nn.Identity()
            else:
                raise ValueError(f"Invalid vertex normal normalization type: {self.config.vn_encoder_norm_type}")

        # texture encoder
        self.texture_encoder = nn.Linear(
            self.config.texture_channels * self.config.texture_encode_patch_size * self.config.texture_encode_patch_size,
            self.config.latent_dim
        )
        if self.config.texture_encoder_norm_type == 'layer_norm':
            self.texture_encoder_norm = nn.LayerNorm(self.config.latent_dim)
        elif self.config.texture_encoder_norm_type == 'rms_norm':
            self.texture_encoder_norm = nn.RMSNorm(self.config.latent_dim)
        else:
            raise ValueError(f"Invalid texture encoder normalization type: {self.config.texture_encoder_norm_type}")

        # learnable tokens
        self.tri_token = nn.Parameter(torch.randn(1, 1, self.config.latent_dim))
        self.reg_tokens = nn.Parameter(torch.randn(1, self.config.num_register_tokens, self.config.latent_dim))
        self.skip_token_num = self.config.num_register_tokens

        # core radiosity transformer（视图无关：仅依赖场景三角+材质）
        self.transformer = TransformerEncoder(
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.latent_dim,
            ffn_hidden_dim=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation=self.config.activation,
            norm_type=self.config.norm_type,
            norm_first=self.config.norm_first,
            rope_dim=self.rope_dim,
            rope_type=self.config.rope_type,
            bias=self.config.bias,
            qk_norm=self.config.view_indep_qk_norm,
            rope_double_max_freq=self.config.rope_double_max_freq
        )

        # view transformer（视图相关：射线 + 相机系三角位置）
        self.view_transformer = ViewTransformer(config)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    @autocast(device_type="cuda", dtype=torch.float32)  # avoid bf16 for its low precision
    def process_tri_vpos_list(self, tri_vpos_list, valid_mask):
        """
        Process tri_vpos_list for RoPE positional encoding.

        :param tri_vpos_list: [batch_size, max_num_tri, 9], padded
        :param valid_mask: [batch_size, max_num_tri]
        :return: processed tri_vpos_list, updated valid_mask
        """
        mask_weight = (valid_mask.float() / (valid_mask.sum(dim=1, keepdim=True) + 1e-5))[..., None]
        weighted_tri_pos = mask_weight * tri_vpos_list
        center_pos = weighted_tri_pos.sum(dim=1).reshape(-1, 3, 3).mean(dim=1, keepdim=True).repeat(1, self.skip_token_num, 3)
        tri_vpos_list = torch.cat([center_pos, tri_vpos_list], dim=1)

        # construct valid mask, things you want is True
        valid_mask = torch.cat([
            torch.ones((tri_vpos_list.size(0), self.skip_token_num), dtype=torch.bool, device=valid_mask.device),
            valid_mask
        ], dim=1)

        return tri_vpos_list, valid_mask

    def construct_seq(self, tri_vpos_list, texture_patch_list, valid_mask, vns):
        """
        From input triangle list + texture patches, construct the sequence for transformer.

        :param tri_vpos_list: [batch_size, max_num_tri, 9], padded
        :param texture_patch_list: [batch_size, max_num_tri, texture_channel, patch_size, patch_size], padded
        :param valid_mask: [batch_size, max_num_tri]
        :param vns: [batch_size, max_num_tri, 3, 3], padded
        :return: [batch_size, max_num_tri + 2, latent_dim]
        """
        batch_size = tri_vpos_list.size(0)

        # vertex normal encoding
        if self.config.use_vn_encoder:
            vn_emb = self.vn_encoder_norm(self.vn_encoding_proj(self.vn_pe(vns)))
        else:
            vn_emb = 0.

        # texture encoding
        tri_tex_emb = self.texture_encoder_norm(self.texture_encoder(
            texture_patch_list.reshape(texture_patch_list.size(0), texture_patch_list.size(1), -1)
        ))

        # construct sequence
        tokens = []
        tokens.append(self.reg_tokens.expand(batch_size, -1, -1))

        if self.config.pe_type == 'nerf':
            tri_vpos_pe = self.tri_vpos_pe(tri_vpos_list)
            tri_emb = self.tri_encoding_norm(self.tri_encoding_proj(tri_vpos_pe)) + self.tri_token + tri_tex_emb + vn_emb
            tokens.append(tri_emb)
        elif self.config.pe_type == 'rope':
            tri_emb = self.tri_token + tri_tex_emb + vn_emb
            tokens.append(tri_emb)
        else:
            raise ValueError(f"Invalid positional encoding type: {self.config.pe_type}")

        seq = torch.cat(tokens, dim=1)

        # pad triangle pos (for RoPE) and valid mask (for all)
        # use center pos for RoPE on auxiliary tokens
        tri_vpos_list, valid_mask = self.process_tri_vpos_list(tri_vpos_list, valid_mask)

        return seq, valid_mask, tri_vpos_list

    def encode_view_independent(
        self,
        tri_vpos_list: torch.Tensor,
        texture_patch_list: torch.Tensor,
        valid_mask: torch.Tensor,
        vns: torch.Tensor,
    ):
        """
        视图无关阶段：三角序列编码 + 12 层 Transformer。

        为何单独抽出：
        - 该段输出仅由场景几何/材质决定，与相机无关，可跨帧/跨视角缓存；
        - 与 decode_view_dependent 拼接后与原始 forward 数学等价。

        Returns:
            seq_vi: [B, L, latent_dim]，L = num_register_tokens + num_tris
            valid_mask_padded: [B, L]，与 seq_vi 对齐的 padding mask（True=有效）
        """
        seq, valid_mask_padded, tri_vpos_for_rope = self.construct_seq(
            tri_vpos_list, texture_patch_list, valid_mask, vns
        )
        seq_vi = self.transformer(
            seq,
            src_key_padding_mask=valid_mask_padded,
            triangle_pos=tri_vpos_for_rope,
        )
        return seq_vi, valid_mask_padded

    def decode_view_dependent(
        self,
        seq_vi: torch.Tensor,
        valid_mask_padded: torch.Tensor,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        tri_vpos_view_tf: torch.Tensor,
        valid_mask: torch.Tensor,
        tf32_view_tf: bool = False,
    ) -> torch.Tensor:
        """
        视图相关阶段：按视角复制 VI token，用相机系三角位置做 RoPE，再 ViewTransformer。

        为何不能缓存：
        - rays_o/rays_d、tri_vpos_view_tf 随相机变化，每帧必须重算。

        Args:
            seq_vi: VI 输出 [B, L, D]（未按视角展开）
            valid_mask_padded: [B, L]
            rays_o: [B, num_views, 3]
            rays_d: [B, num_views, H, W, 3]
            tri_vpos_view_tf: [B, num_views, num_tris, 9]
            valid_mask: 原始三角有效掩码 [B, num_tris]（未含 register）
            tf32_view_tf: 低精度推理时 ViewTransformer 使用 tf32 路径

        Returns:
            [B, num_views, C, H, W]
        """
        batch_size, num_views = rays_o.size(0), rays_o.size(1)
        # 多视角共享同一份 VI 特征：在 batch 维上按视角重复
        seq = seq_vi.repeat_interleave(num_views, dim=0)
        rays_o = rays_o.view(-1, *rays_o.shape[2:])
        rays_d = rays_d.view(-1, *rays_d.shape[2:])
        tri_vpos_view_tf = tri_vpos_view_tf.reshape(-1, *tri_vpos_view_tf.shape[2:])
        valid_mask_expanded = valid_mask.repeat_interleave(num_views, dim=0)
        valid_mask_padded_expanded = valid_mask_padded.repeat_interleave(num_views, dim=0)
        pos_seq, _ = self.process_tri_vpos_list(tri_vpos_view_tf, valid_mask_expanded)

        res = self.view_transformer(
            rays_o,
            rays_d,
            seq,
            pos_seq,
            valid_mask_padded_expanded,
            tf32_mode=tf32_view_tf,
        )
        res = res.view(batch_size, num_views, *res.size()[1:])
        return res

    def forward(
        self,
        tri_vpos_list,
        texture_patch_list,
        valid_mask,
        vns,
        rays_o,
        rays_d,
        tri_vpos_view_tf,
        tf32_view_tf=False,
        cached_seq_vi=None,
        cached_valid_mask_padded=None,
    ):
        """
        完整前向；若提供 cached_seq_vi 则跳过 VI，仅执行 VD（用于缓存命中）。

        Args:
            cached_seq_vi / cached_valid_mask_padded:
                必须同时提供或同时为 None。命中时无需再传几何纹理给 VI（仍建议
                由上层传齐以保持 API 兼容；VI 路径下这些参数仍用于未缓存分支）。

        原因：
        - 默认路径与旧版完全一致，保证预训练权重行为不变；
        - 缓存路径避免重复执行 construct_seq + transformer。
        """
        if cached_seq_vi is not None:
            if cached_valid_mask_padded is None:
                raise ValueError("cached_valid_mask_padded is required when cached_seq_vi is set")
            return self.decode_view_dependent(
                cached_seq_vi,
                cached_valid_mask_padded,
                rays_o,
                rays_d,
                tri_vpos_view_tf,
                valid_mask,
                tf32_view_tf=tf32_view_tf,
            )

        seq_vi, valid_mask_padded = self.encode_view_independent(
            tri_vpos_list, texture_patch_list, valid_mask, vns
        )
        return self.decode_view_dependent(
            seq_vi,
            valid_mask_padded,
            rays_o,
            rays_d,
            tri_vpos_view_tf,
            valid_mask,
            tf32_view_tf=tf32_view_tf,
        )

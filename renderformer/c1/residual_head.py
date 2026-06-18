# -*- coding: utf-8 -*-
"""C1-a：图像域残差间接光头（冻结 RF，只训本模块）。"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualIndirectHead(nn.Module):
    """
    输入通道（NCHW）：
      - hdr_neural [3]  （可选，可在配置中关闭）
      - hdr_direct  [3]
      - depth       [1] （可选）
    输出：
      - I_pred [3]  （线性 HDR 间接光，ReLU 保证非负）
    """

    def __init__(
        self,
        use_neural: bool = True,
        use_depth: bool = True,
        base_channels: int = 32,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.use_neural = use_neural
        self.use_depth = use_depth
        in_ch = 3  # always direct
        if use_neural:
            in_ch += 3
        if use_depth:
            in_ch += 1

        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks):
            layers.extend(
                [
                    nn.Conv2d(base_channels, base_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(base_channels, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        hdr_direct: torch.Tensor,
        hdr_neural: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hdr_direct / hdr_neural: [B,3,H,W] 或 [B,nv,H,W,3]（自动转 NCHW）
            depth: [B,1,H,W] 或 [B,nv,H,W,1]
        Returns:
            I_pred: 与输入同布局（若输入为 BHWC 视图布局则还原）
        """
        layout_bhwc = hdr_direct.dim() == 5
        if layout_bhwc:
            b, nv, h, w, _ = hdr_direct.shape
            d = hdr_direct.reshape(b * nv, h, w, 3).permute(0, 3, 1, 2).contiguous()
            n = None
            parts = [d]
            if self.use_neural:
                if hdr_neural is None:
                    raise ValueError("use_neural=True 时需要 hdr_neural")
                n = hdr_neural.reshape(b * nv, h, w, 3).permute(0, 3, 1, 2).contiguous()
                parts.append(n)
            if self.use_depth:
                if depth is None:
                    z = torch.zeros(b * nv, 1, h, w, device=d.device, dtype=d.dtype)
                else:
                    z = depth.reshape(b * nv, h, w, -1)[..., :1]
                    z = z.permute(0, 3, 1, 2).contiguous()
                parts.append(z)
            x = torch.cat(parts, dim=1)
            delta = self.net(x)
            if n is not None:
                base = torch.relu(n - d)
                y = torch.relu(base + delta)
            else:
                y = torch.nn.functional.softplus(delta)
            return y.permute(0, 2, 3, 1).reshape(b, nv, h, w, 3)

        # NCHW
        parts = [hdr_direct]
        n = None
        if self.use_neural:
            if hdr_neural is None:
                raise ValueError("use_neural=True 时需要 hdr_neural")
            n = hdr_neural
            parts.append(n)
        if self.use_depth:
            if depth is None:
                z = torch.zeros(
                    hdr_direct.shape[0],
                    1,
                    hdr_direct.shape[2],
                    hdr_direct.shape[3],
                    device=hdr_direct.device,
                    dtype=hdr_direct.dtype,
                )
            else:
                z = depth if depth.shape[1] == 1 else depth[:, :1]
            parts.append(z)
        x = torch.cat(parts, dim=1)
        delta = self.net(x)
        # 以 RF 分解为基座学修正量，避免输出塌成全零；neural dropout 时退化为 softplus
        if n is not None and torch.any(n != 0):
            base = torch.relu(n - hdr_direct)
            return torch.relu(base + delta)
        return torch.nn.functional.softplus(delta)


def fuse_direct_indirect(
    hdr_direct: torch.Tensor,
    indirect: torch.Tensor,
    alpha: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    if isinstance(alpha, (float, int)):
        return hdr_direct + float(alpha) * indirect
    if alpha.dim() == hdr_direct.dim() - 1:
        alpha = alpha.unsqueeze(-1)
    return hdr_direct + alpha * indirect

# 块缓存与 Temporal VI：设计说明、实验原理与代码结构

本文档与**系统架构图**（自顶向下：场景输入 → 块级缓存层 → Temporal VI 调度 → 视图相关层）一致，说明**设计思路**、**实验原理**、**关键代码分层引用**，并给出**实验目的**与**实验结果记录模板**。更完整的公式与图集见 [TECHNICAL_DOCUMENT.md](./TECHNICAL_DOCUMENT.md)。

---

## 1. 架构图与模块对应关系

| 图中模块 | 实现职责 |
|----------|----------|
| 输入 / 预处理 | `RenderFormerRenderingPipeline`：`texture` log、相机坐标、`RayGenerator` |
| 块划分 + MD5 指纹 | `_partition_blocks`、`compute_block_hash` |
| LRU 与 `construct_seq` | `BlockCache`、`model.construct_seq`（仅 miss） |
| 拼接 `seq_full` + RoPE/mask | `reg_tokens`、`torch.cat`、`process_tri_vpos_list` |
| `decide_force_full` | `renderformer/temporal_vi/policy.py` |
| 全算 VI / 近似 VI | `forward_vi_only`、`apply_vi_approximation`、`vi_ref` |
| 视图层（不可跳过） | `forward_view_only` → `ViewTransformer`（DPT / 线性头） |

**模块分工（先读这段）**：先把三角形编成向量 → 再做**整场景**上的全局光计算 → 再按**当前相机**把图画出来。**块缓存**只管第一步里「哪些小块不用重算」；**近似与完整重算**只管第二步里「这一帧要不要做完整的全局光网络」。两步前后相接，解决的问题不同，可以叠在一起用。

---

## 2. 设计思路

### 2.1 推理在逻辑上分三段

1. **三角形编码**：把每个三角形（形状、法线、材质贴图等）变成一串高维向量，供后面网络使用。  
2. **全局光传输（视图无关）**：让所有三角形的向量互相「看见」彼此，得到已经混在一起的全局光照信息。这一步计算量大，且**和当前相机朝向无关**（相机不进入这一步的输入）。  
3. **按视角成图（视图相关）**：用**当前帧的相机和每条光线**，把上一步的结果解码成图像。相机一变，这一步必须重新跑。

官方实现相当于：每一帧对**整段**三角形**一次性**做完第 1 步，再做完第 2 步，再做完第 3 步。

---

### 2.2 机制一：按块的编码缓存（解决「第 1 步里谁可以抄近路」）

**想法**：把三角形按固定大小切成很多**小块**。每个小块只看自己的几何和材质，算出一个「小块专属的编码结果」。若**另一帧、另一场景里**同一个小块的几何和材质没变，这一步的结果就不必再算，直接从表里取。

**为什么不能把缓存放在第 2 步之后？**  
第 2 步之后，每个三角形的向量已经跟**全场景所有三角形**混在一起了；这时再按「单个小块」去复用就会对错——邻居变了，自己的全局结果也变。所以只能缓存**第 2 步之前**、**只由本小块决定**的那部分结果。

**和相机的关系**：编码用的指纹里**不包含相机**。因此**相机怎么动**，只要小块几何和材质不变，第 1 步仍可以命中缓存。

---

### 2.3 机制二：全局光计算的「完整重算」与「近似复用」（解决「第 2 步能不能偶尔偷懒」）

**想法**：第 2 步是最重的全局网络。若连续多帧里**场景本身几乎不变**，理论上第 2 步的输出也应几乎不变，则可以**偶尔跳过**整段网络，改用**上一次完整算出来的结果**当作本帧输入；隔一段时间或一旦发现场景变了，再**强制做一次完整的第 2 步**，更新那份「上次完整结果」，避免误差累积或跟不上变化。

**两种工作模式（语义上）**：

- **完整重算**：照常跑完全局光网络，结果可信，与不做近似时一致（在同样数值设置下）。  
- **近似帧**：不跑这段网络，用**最近一次完整重算**保存下来的张量顶替；省时间，但**只在场景真的没怎么变时才合理**。

**何时必须回到完整重算（与机制一如何配合）**：例如首帧没有「上次结果」、三角形数量或序列长度变了、**小块指纹相对上一帧变太多**（说明材质或几何在变）、隔了固定帧数要校正、连续近似太多次要兜底等。其中「小块变了多少」来自**机制一**为每个小块算好的指纹列表——所以**块级缓存不仅为省第 1 步服务，还为「场景变没变」提供信号，用来触发第 2 步的完整重算**。

---

### 2.4 两种机制的关系（核心）

| 维度 | 块级编码缓存 | 全局光的近似 / 完整重算 |
|------|----------------|-------------------------|
| 作用阶段 | 第 1 步：三角形 → 向量 | 第 2 步：全局光网络 |
| 省下的主要计算 | 重复的小块编码 | 整段全局光网络（仅近似帧） |
| 正确性 | 与官方路径可对齐（推理等价设计） | 近似帧**不保证**与每帧全算一致 |
| 和相机 | 无关（指纹不含相机） | 无关（该步本身不读相机） |
| 数据流位置 | 先做 | 在拼接好的整段序列上做 |

**串联关系**：每一帧总是先经过（带或不带缓存的）**第 1 步**，拼成整场景序列，再进入**第 2 步**的「完整重算或近似」分支，最后**第 3 步**一定执行。  
**配合关系**：机制一降低「重复编码」成本，并在动态场景里用**块变化比例**帮助决定机制二该不该**结束近似、回到完整重算**；机制二在静态或慢变场景里进一步降低「重复全局光」成本，但**不能替代**机制一对「几何材质变了」的感知——二者一环套一环，而不是互相替代。

---

### 2.5 视图层为何始终参与

无论第 1 步是否命中缓存、第 2 步是完整还是近似，**当前帧的相机和图像上每条光线**都要参与解码，因此**按视角成图这一段不能跳过**。

---

## 3. 实验原理

### 3.1 对照路径

| 代号 | 路径 | 用途 |
|------|------|------|
| ① | `pipeline.render()` | 官方基线，无块缓存、无 VI 近似 |
| ② | `render_with_block_cache` | 验证块缓存与 ① 的数值接近度与耗时 |
| ③ | `render_with_temporal_vi` | 验证近似 VI 的耗时与相对 ① 的误差 |

### 3.2 可观测指标

- **正确性 / 保真**：相对 ① 的 HDR 全张量 **MSE、RMSE、max|diff|**（`compare_render_baselines.py`）；或导出 PNG/EXR 目视（`experiment_dynamic_scene.py`）。
- **性能**：`cuda.synchronize` 后的墙钟时间；块缓存 **hit/miss、hit_rate**；Temporal **`vi_path`（full/approx）、`force_reason`**。
- **调度行为**：`TemporalVIState.force_reason_hist`、`full_every_k`、可选 `changed_block_ratio_threshold`。

### 3.3 实验假设（可证伪）

- **H1**：静态或慢变几何下，② 与 ① 误差应接近 0（允许混合精度路径差）。
- **H2**：③ 在 **approx** 帧可能快于 ②，但相对 ① 误差在动态场景会增大。
- **H3**：块命中率高**不必然**带来总帧时间下降（VI+View 主导且块路径有固定开销时可能为负收益）。

---

## 4. 代码结构（关键片段分块）

### 4.1 流水线入口与依赖

`RenderFormerRenderingPipeline` 聚合 `BlockCache`、`TemporalVIConfig/State` 与策略函数。

```1:14:renderformer/pipelines/rendering_pipeline.py
from typing import Any, Dict

import torch

from renderformer.cache import BlockCache, compute_block_hash
from renderformer.models.renderformer import RenderFormer
from renderformer.temporal_vi import (
    TemporalVIConfig,
    TemporalVIState,
    apply_vi_approximation,
    decide_force_full,
)
from renderformer.utils.ray_generator import RayGenerator
from renderformer.utils.transform import trans_to_cam_coord
```

### 4.2 块划分

按三角形索引均匀切片，末块可不足 `block_size`。

```17:21:renderformer/pipelines/rendering_pipeline.py
def _partition_blocks(num_tris: int, block_size: int):
    """Yield (start, end) slices for blocks. Last block may be smaller."""
    for start in range(0, num_tris, block_size):
        end = min(start + block_size, num_tris)
        yield start, end
```

### 4.3 块键：几何+材质指纹（与相机无关）

```18:36:renderformer/cache/hash_key.py
def compute_block_hash(
    tri_vpos: Union[torch.Tensor, np.ndarray],
    texture: Union[torch.Tensor, np.ndarray],
    vn: Union[torch.Tensor, np.ndarray],
    quantize_bits: int = 0,
) -> bytes:
    """
    Compute 128-bit (16 bytes) hash for a block of triangles.
    View- and camera-independent; only geometry + material.
    ...
    Returns:
        16-byte key for use in cache.
    """
```

### 4.4 块缓存循环：哈希 → get → miss 则 `construct_seq` → put

```347:379:renderformer/pipelines/rendering_pipeline.py
        tri_emb_list = []
        block_keys: list = []
        for start, end in _partition_blocks(num_tris, block_size):
            tri_vpos_b = tri_vpos[:, start:end, :]
            texture_b = texture[:, start:end]
            ...
            block_key = compute_block_hash(
                tri_vpos_b.cpu().numpy(),
                texture_b.cpu().numpy(),
                vn_b.cpu().numpy(),
            )
            block_keys.append(block_key)
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
```

### 4.5 配置与跨帧状态（Temporal）

```7:47:renderformer/temporal_vi/state.py
@dataclass
class TemporalVIConfig:
    """调度近似 VI 与强制全算 VI 的策略配置。"""

    full_every_k: int = 8
    ...
    changed_block_ratio_threshold: Optional[float] = None
    ...
    approx_mode: str = "level0"
    ...


@dataclass
class TemporalVIState:
    """跨帧状态；由调用方在视频序列上持久化同一实例。"""

    vi_ref_np: Optional[np.ndarray] = None
    ...
    last_block_keys: Optional[List[bytes]] = None
    ...
    consecutive_approx: int = 0
```

### 4.6 强制全算判定（与图中决策节点一致）

```9:47:renderformer/temporal_vi/policy.py
def decide_force_full(
    state: TemporalVIState,
    cfg: TemporalVIConfig,
    num_tris: int,
    block_keys_curr: List[bytes],
    seq_len_curr: int,
) -> Tuple[bool, str]:
    ...
    if state.vi_ref_np is None:
        return True, "cold_start"
    ...
    if cfg.full_every_k > 0 and (fi - state.last_full_frame) >= cfg.full_every_k:
        return True, "periodic_k"

    if state.consecutive_approx >= cfg.max_consecutive_approx:
        return True, "max_consecutive_approx"

    return False, "approx_ok"
```

### 4.7 近似 VI（level0 / level1）

```50:70:renderformer/temporal_vi/policy.py
def apply_vi_approximation(
    seq_curr: torch.Tensor,
    vi_ref_np: np.ndarray,
    cfg: TemporalVIConfig,
    latent_dim: int,
) -> torch.Tensor:
    ...
    if cfg.approx_mode == "level0":
        t = torch.from_numpy(vi_ref_np).to(seq_curr.device, dtype=seq_curr.dtype)
        return t

    if cfg.approx_mode == "level1":
        ref = torch.from_numpy(vi_ref_np).to(seq_curr.device, dtype=torch.float32)
        s = seq_curr.float()
        s_norm = torch.nn.functional.layer_norm(s, (latent_dim,))
        alpha = float(cfg.blend_alpha)
        out = alpha * s_norm + (1.0 - alpha) * ref
        return out.to(seq_curr.dtype)
```

### 4.8 模型：VI / View 拆分（支撑全算与仅 View）

```246:259:renderformer/models/renderformer.py
    def forward_vi_only(
        self,
        seq: torch.Tensor,
        valid_mask_padded: torch.Tensor,
        tri_vpos_list: torch.Tensor,
    ) -> torch.Tensor:
        """
        View-independent stage only: 12-layer TransformerEncoder.
        ...
        """
        return self.transformer(
            seq, src_key_padding_mask=valid_mask_padded, triangle_pos=tri_vpos_list
        )
```

```261:289:renderformer/models/renderformer.py
    def forward_view_only(
        self,
        seq_after_vi: torch.Tensor,
        valid_mask_padded: torch.Tensor,
        tri_vpos_list: torch.Tensor,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        tri_vpos_view_tf: torch.Tensor,
        tf32_view_tf: bool = False,
    ) -> torch.Tensor:
        """
        View-dependent stage only, given VI output seq_after_vi (same layout as after forward_vi_only).
        """
        batch_size, num_views = rays_o.size(0), rays_o.size(1)
        seq = seq_after_vi.repeat_interleave(num_views, dim=0)
        ...
        res = self.view_transformer(
            rays_o,
            rays_d,
            seq,
            pos_seq,
            valid_mask_padded,
            tf32_mode=tf32_view_tf,
        )
```

### 4.9 单帧内：决策 → VI → View（Temporal 主路径）

```391:449:renderformer/pipelines/rendering_pipeline.py
        force_full, reason = decide_force_full(
            temporal_state,
            temporal_cfg,
            num_tris,
            block_keys,
            seq_len_curr,
        )
        ...
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=torch_dtype
        ):
            if force_full:
                seq_vi = self.model.forward_vi_only(
                    seq_full, valid_mask_padded, tri_vpos_list
                )
                temporal_state.vi_ref_np = (
                    seq_vi.detach().float().cpu().numpy().copy()
                )
                ...
            else:
                seq_vi = apply_vi_approximation(
                    seq_full,
                    temporal_state.vi_ref_np,
                    temporal_cfg,
                    latent_dim,
                )
                ...
            rendered_imgs = self.model.forward_view_only(
                seq_vi,
                valid_mask_padded,
                tri_vpos_list,
                rays_o=rays_o,
                rays_d=rays_d,
                tri_vpos_view_tf=tris_for_view_tf.reshape(bs, nv, -1, 9),
                tf32_view_tf=tf32_view_tf,
            )
```

---

## 5. 实验目的（撰写示例）

可按课题直接选用或改写：

1. **验证块级缓存在视频序列上的命中率与相对官方路径的数值一致性**，并量化固定开销对总耗时的影响。  
2. **验证 Temporal VI 在「周期全算 + 近似帧」策略下相对 ①/② 的加速与画质折衷**，并扫描 `full_every_k`、`approx_mode`、`changed_block_ratio_threshold`。  
3. **在动态场景**（真实多 H5 或 `experiment_dynamic_scene.py` 合成扰动）下观察 **approx 帧**相对 baseline 的误差演化。  
4. **为工程落地提供默认参数建议**（如：仅追求与官方一致则使用 ① 或 `full_every_k=1` 仅保留块缓存）。

---

## 6. 实验结果模板

以下为 Markdown 表，实验完成后直接填入；可与 `compare_render_baselines.py` 汇总段、`per_frame_metrics.jsonl` 对照。

### 6.1 环境与数据

| 项 | 内容 |
|----|------|
| 日期 | YYYY-MM-DD |
| GPU / 驱动 / CUDA | |
| PyTorch 版本 | |
| 模型 `model_id` | |
| 精度 `fp16/bf16/fp32` | |
| 分辨率 | |
| 数据路径 | （例：`video-data/teaser-scenes/cbox-roughness`） |
| 帧数 / 场景说明 | |

### 6.2 参数

| 参数 | 取值 |
|------|------|
| `block_size` | |
| `max_cache_entries` | |
| `full_every_k` | |
| `max_consecutive_approx` | |
| `changed_block_ratio_threshold` | |
| `approx_mode` / `blend_alpha` | |

### 6.3 耗时汇总（相对 ①）

| 路径 | 总时间 (s) | 均时间 (ms/帧) | 相对 ① 省时 (%) | 相对 ① 加速比 |
|------|------------|----------------|-----------------|---------------|
| ① baseline | | | 0 | 1.00× |
| ② block_cache | | | | |
| ③ temporal_vi | | | | |

### 6.4 数值误差（相对 ①，linear HDR）

| 路径 | MSE 均值 | RMSE 均值 | rel_RMSE% 均值 | max\|diff\| 最大 |
|------|----------|-----------|----------------|-----------------|
| ② block_cache | | | | |
| ③ temporal_vi | | | | |

（可选）**仅 VI=approx 子集**：MSE / RMSE / rel_RMSE% 均值：____  

### 6.5 调度与缓存统计

| 项 | 数值 |
|----|------|
| VI full 帧数 / approx 帧数 | |
| `force_reason_hist` | （粘贴 JSON 或表） |
| BlockCache（BC 路径）末：hits / misses / hit_rate | |
| BlockCache（TV 路径，若单独统计） | |

### 6.6 结论与下一步

- **结论**：（是否达到加速目标；② 与 ① 是否一致；③ 可接受误差范围）  
- **下一步**：（调参方向 / Profiler 关注点 / 是否需改分块策略等）

---

## 7. 相关脚本与文档

| 资源 | 说明 |
|------|------|
| `compare_render_baselines.py` | ①②③ 自动对比 + 控制台汇总表 |
| `experiment_dynamic_scene.py` | 动态序列或合成扰动 + 导出对比图 |
| `batch_infer_temporal_vi.py` | 整目录 Temporal 推理 + 统计 |
| [TECHNICAL_DOCUMENT.md](./TECHNICAL_DOCUMENT.md) | 汇总技术文档 |
| [TEMPORAL_VI.md](./TEMPORAL_VI.md) | Temporal 参数与运行说明 |

---

*文档版本：与当前仓库实现一致；架构图若单独维护于 drawio/PNG，可在本节首行增加「图文件路径」便于组内同步。*

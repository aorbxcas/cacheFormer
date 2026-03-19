# 实验记录：同场景视图无关（VI）缓存

## 一、同场景实验

### 1. 实验思想

- **初衷**：在视图无关层（VI）与视图相关层（VD）之间插入一层缓存，存储并识别「在 VI 中已经出现过且计算过的三角形所对应的 Token」。在后续帧或其它场景中，当再次出现相同或相似三角形时，直接从缓存中读取该 Token，跳过 VI 的 Token 计算，从而减少重复计算、加速多帧/多视角渲染。
- **理想情况**：若每个三角形的 VI 输出只依赖于该三角形自身的几何与材质，则可以用「三角形（或小块）内容」做哈希 key，实现跨帧、跨场景的细粒度复用。

### 2. 实际发现与原因分析

- **现象**：在按「三角形或其它小单位」做缓存的实验中，**缓存命中率始终为 0**。
- **原因**：在现有 RenderFormer 架构下，**第一层视图无关层对每个三角形计算 Token 时，该三角形的输出会受到全局上下文影响**。VI 阶段的 12 层 Transformer 是在「整段序列（reg_tokens + 全部三角形 token）」上做自注意力，因此：
  - 单个三角形的 VI 输出不仅与自身几何/材质有关，还与场景中其它三角形、光照与遮挡关系等有关；
  - 无法用「按三角形（或其它小单位）的哈希 key」做到「不同场景、相同三角形复用同一份缓存」；
  - 即：**视图无关阶段的输出并不是按单位独立的**，强行按小单位缓存会导致 key 无法在不同上下文下复用，从而命中率为 0。

### 3. 最小解决方案（同场景假设）

- **思路**：不再追求跨场景复用，**假设场景完全不变（仅相机参数如 c2w、fov 变化）**。此时：
  - VI 的输入（triangles、texture、vn、mask）不变；
  - VI 的完整输出与「当前场景」一一对应，可整段缓存。
- **做法**：
  - 以**整场景**为单位做缓存：缓存 key = 与 VI 输入一致的数据指纹（triangles + texture + vn + mask），value = 该场景的 VI 输出序列及对应 valid_mask；
  - 不改变 Transformer 模型结构，仅在 pipeline 中：若当前帧指纹命中缓存则跳过 12 层 VI，直接用缓存的 seq 走 VD；未命中则完整 VI+VD 并写入缓存。
- **适用边界**：
  - **适用**：同场景、多视角/多帧（仅改相机），如编辑器预览、固定场景的相机漫游；
  - **不适用**：场景几何或材质随时间/帧变化（每帧 key 不同，无法命中），实际应用主要限于「场景不变、仅相机变化」的场合（如游戏引擎编辑器预览）。

---

### 4. 实验步骤

#### 4.1 场景指纹与缓存接口设计

**目的**：保证「同一场景」在任意时刻得到同一个 key，且 key 不依赖相机，与 VI 输入严格一致。

**实现要点**：

- 指纹在「与模型 VI 输入一致」的张量上计算：包含 triangles、texture、vn、mask；**不包含** c2w、fov（VI 与相机无关）。若模型使用非 LDR 路径，需在算指纹前对 texture 的光照通道做与 pipeline 一致的 `log10` 编码，否则会误命中/误未命中。
- 使用 CPU、float32、连续内存的字节做 SHA256，避免设备或顺序差异导致 key 不稳定。

**相关代码**（`renderformer/cache/vi_cache.py`）：

```python
def scene_fingerprint(
    triangles: torch.Tensor,
    texture: torch.Tensor,
    vn: torch.Tensor,
    mask: torch.Tensor,
) -> str:
    """为「当前帧喂给 VI 的几何+材质」生成稳定哈希键。不包含相机参数。"""
    h = hashlib.sha256()
    for tensor in (triangles, texture, vn):
        t = tensor.detach().cpu().to(torch.float32).contiguous()
        h.update(t.numpy().tobytes())
    m = mask.detach().cpu().to(torch.bool).contiguous()
    h.update(m.numpy().tobytes())
    return h.hexdigest()
```

- 缓存类 `ViewIndependentCache`：LRU，key → `(seq_vi_cpu, valid_mask_padded_cpu)`；命中时将张量迁回当前设备再送入 VD。存 CPU 以控制显存占用。

#### 4.2 Pipeline 中接入 VI 缓存

**目的**：在单样本（batch_size=1）时根据指纹查缓存；命中则跳过 VI，只跑 VD；未命中则跑完整 VI+VD 并写回缓存。

**实现要点**：

- 仅在 `batch_size=1` 时启用缓存；batch>1 时多场景共用同一 key 会错误复用，因此强制走完整 VI+VD 并打 warning。
- 在调用模型前完成与 VI 输入一致的数据处理（含 texture 的 log 编码），再计算 `scene_fingerprint`；命中时调用 `model(..., cached_seq_vi=..., cached_valid_mask_padded=...)`，未命中时先 `model.encode_view_independent(...)` 再 `model.decode_view_dependent(...)` 并 `vi_cache.put(...)`。

**相关代码**（`renderformer/pipelines/rendering_pipeline.py` 片段）：

```python
if use_vi_cache:
    cache_key = scene_fingerprint(triangles, texture, vn, mask)
    entry = vi_cache.get(cache_key)
else:
    entry = None

if entry is not None:
    vi_cache_hit = True
    seq_vi, mask_p = entry
    seq_vi = seq_vi.to(self.device, non_blocking=True)
    mask_p = mask_p.to(self.device, non_blocking=True)
    rendered_imgs = self.model(..., cached_seq_vi=seq_vi, cached_valid_mask_padded=mask_p)
elif use_vi_cache:
    seq_vi, mask_p = self.model.encode_view_independent(tri_flat, texture, mask, vn_flat)
    rendered_imgs = self.model.decode_view_dependent(seq_vi, mask_p, ...)
    vi_cache.put(cache_key, seq_vi.detach().cpu(), mask_p.detach().cpu())
else:
    rendered_imgs = self.model(...)  # 无缓存或 batch>1
```

**效果**：同场景下首帧 miss、后续帧 hit，VD 阶段耗时远小于完整 VI+VD，整体渲染时间明显下降。

#### 4.3 同场景多相机验证（benchmark_vi_cache）

**目的**：用「同一 H5 场景、多组不同相机参数（FOV/轨道角/距离）」连续渲染，验证「第 1 次 miss、第 2～N 次 hit」且缓存统计正确。

**步骤**：

1. 加载同一 H5（triangles、texture、vn、mask 不变），仅按预设变体修改 c2w、fov（如 FOV 0.9x/1.1x、Orbit Y ±15°、距离 0.9x/1.1x 等）。
2. 创建 `ViewIndependentCache(max_entries=4)`，对每种相机变体调用一次 `pipeline.render(..., vi_cache=cache, return_vi_cache_info=True)`。
3. 记录每次的 `vi_cache_hit`、耗时、缓存条目数与 hit/miss 统计。

**命令示例**：

```bash
python benchmark_vi_cache.py --h5_file tmp/cbox/cbox.h5
python benchmark_vi_cache.py --h5_file tmp/cbox/cbox.h5 --num_renders 6
```

**预期结果**：第 1 次渲染为 miss（完整 VI+VD），第 2～N 次均为 hit（仅 VD）；汇总输出中 `vi_cache_hit=True` 仅首帧为 False，其余为 True，且验证通过提示「第 1 次 miss，后续均 hit」。

#### 4.4 批量视频推理中启用 VI 缓存（batch_infer）

**目的**：在按文件夹逐帧渲染视频时，对「同场景多帧」自动复用 VI 缓存，观察命中率与加速比。

**步骤**：

1. 使用 `batch_infer.py`，传入 `--h5_folder`（同一场景多帧时，每帧 H5 的几何/材质一致、仅相机可能不同）或单场景多帧数据。
2. 加上 `--vi_cache`，并可选 `--vi_cache_max_entries`（默认 64）；启用时脚本会强制 `batch_size=1`，避免多 batch 误用缓存。
3. 运行后查看日志与（若有）性能 JSON 中的 `vi_cache_hits`、`vi_cache_misses`、`vi_cache_hit_rate`。

**命令示例**：

```bash
python batch_infer.py --h5_folder path/to/same_scene_frames/ --vi_cache --vi_cache_max_entries 64 --output_dir output/vi_cached
```

**效果**：若文件夹内为「同一场景、仅相机变化」的连续帧，首帧 miss，后续帧大量 hit，整体 VI 计算次数显著减少；若每帧场景都变，则命中率接近 0，与「最小解决方案」的适用边界一致。

#### 4.5 小结

| 步骤 | 内容 | 原因 | 效果 |
|------|------|------|------|
| 4.1 | 场景指纹 + LRU 缓存接口 | 保证同场景同 key、不依赖相机 | 可复现、稳定的缓存键与存储 |
| 4.2 | Pipeline 内按 key 查/存、命中则只跑 VD | 避免同场景重复跑 12 层 VI | 同场景多帧/多视角明显加速 |
| 4.3 | benchmark_vi_cache 同场景多相机 | 验证命中逻辑与统计正确 | 确认第 1 次 miss、后续 hit |
| 4.4 | batch_infer --vi_cache | 真实多帧流水线中观察命中率 | 同场景视频渲染加速；异场景则命中率≈0 |

整体结论：**按三角形或小块缓存的方案因 VI 的全局依赖而无法得到有效命中；改为「同场景整段 VI 缓存」后，在「场景不变、仅相机变化」的设定下可稳定命中并加速，适合编辑器预览等固定场景的交互渲染。**

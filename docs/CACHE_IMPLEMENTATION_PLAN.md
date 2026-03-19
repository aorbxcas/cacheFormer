# RenderFormer 视频渲染缓存机制 — 实现方案细节

基于开题思路中「以三角形或 mesh 等单元为单位、哈希键存储、视图无关层复用」的目标，本文档给出与现有代码结构对齐的缓存实现方案细节。

---

## 一、目标与约束

- **目标**：在视频逐帧渲染时，减少重复的视图无关层计算，并支持跨帧、跨场景的块级复用。
- **约束**：
  - 缓存键与**相机无关**，仅依赖几何 + 材质（三角形顶点、法线、纹理/BRDF）。
  - 支持**按块（物体/mesh/空间块）**的精细失效与复用。
  - 显存有限，需 LRU（+ 可选 LFU）淘汰与容量上限。

---

## 二、为何不宜在「视图无关层输出后」按三角形缓存

当前视图无关层是**整序列上的全局 self-attention**：每个三角形 token 的输出都依赖**所有**其他三角形，因此：

- 视图无关层**输出**的每个 768D 向量已是**全局属性**（光传输、遮挡、间接光等耦合在一起），**无法再按三角形拆开**作为独立可复用单元。
- 若强行按三角形切片缓存：同一几何+材质的三角形在不同场景（不同邻居、不同光照）下，其输出 768D 不同，键若只含该三角形则语义错误；键若含整场景则退化为「整场景一个 key」，无法跨场景复用。

因此：**缓存应插入在视图无关层之前**，对「一组三角形」做编码得到**块级中间表示**，该表示仅依赖该块自身（几何+材质），再对块表示做全局 attention。这样块表示可安全地以「块指纹」为 key 缓存与跨场景复用。

---

## 三、推荐方案：块级缓存（插入在视图无关层前）

### 3.1 思路概述

- **插入位置**：在现有「整序列 construct_seq + 12 层 Transformer」**之前**，增加一层**块级编码**。
- **流程**：按「一组三角形」（一个物体、一个 mesh、或空间块）分组 → 块内小范围 attention 得到**块表示** → 对**块表示**做全局 attention（替代原先对三角形 token 的全局 attention）→ 后续与视图相关层衔接不变。
- **缓存**：key = 该块的指纹（mesh + 材质 / 几何+材质），value = 该块的**中间表示**（块编码器输出）。不同场景里相同物体/相同块可复用同一 value。

**特点**（相对其他粒度）：

| 对比 | 块级缓存（本方案） | 整场景一个 key | 每三角形一个 key |
|------|-------------------|----------------|-------------------|
| 粒度 | 块（物体/mesh/空间块） | 整帧 | 单三角形 |
| 跨场景复用 | ✅ 相同块可复用 | ❌ 仅同场景同帧 | ⚠️ 需 (场景, tri) 否则语义错误 |
| 失效/键设计 | 按块失效，键可控 | 简单但粗 | 细但键/语义复杂 |
| 模型与训练 | **需改结构 + 训练** | 可不改模型 | 需改结构 + 训练 |

块级方案比「整场景」细、有跨场景复用；比「每三角形」粗，实现与 key 设计相对可控，但**需要动模型并重新训练**（见下）。

### 3.2 新管线结构（需改模型）

```
tri_vpos_list, texture_patch_list, valid_mask, vns
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 块划分：按 object_id / mesh_id / 空间块 将三角形分组       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 块编码（可缓存）：                                        │
│    - 每块内 construct_seq 风格编码 → 块内 token 序列          │
│    - 块内小范围 self-attention（或 cross-attention + pool）   │
│    - 输出：每块一个 768D 块表示 block_repr [num_blocks, 768]  │
│    - Cache: key = hash(块几何+材质), value = block_repr       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 全局 attention（视图无关）：                                │
│    - 输入：块表示序列 [reg_tokens, block_1, ..., block_K]      │
│    - 12 层 Transformer  over 块表示（不再 over 三角形）       │
│    - 输出：块级 seq_out [num_blocks+skip, 768]                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 与视图相关层衔接：                                         │
│    - 若 view_transformer 仍需要「每三角形」的 token：          │
│      需从块级 seq_out 反哺/广播回三角形级（见 3.4）            │
│    - 否则可直接用块级 seq_out 与 rays 做 cross-attention      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
view_transformer(rays, seq_out, ...) → 像素
```

### 3.3 块的定义与键值设计

- **块（block）**：一组三角形的集合。可选定义方式：
  - **按物体**：一个 object_id / 一个 mesh 对应一块；
  - **按 mesh + 材质**：同一 mesh 同一材质为一块（材质变化则不同块）；
  - **按空间**：将场景空间划分成格子，每格内三角形为一块（适合大场景）。

- **缓存键**：该块的指纹，与相机、场景中其他块无关。
  - 输入：该块内所有三角形的 `tri_vpos`、`texture`、`vn`（与当前 construct_seq 输入一致）。
  - 规范化：顶点顺序固定、可选抖动容忍量化；块内三角形按稳定序（如 mesh 内索引）拼接。
  - 哈希：128 位（如 MD5/xxHash-128），得到 `block_key`。

- **缓存值**：该块编码器的输出 —— **块表示** `block_repr`，形状 `[1, 768]` 或 `[1, 1, 768]`（每块一个向量）。
  - 由「块内 token 构建 + 块内 attention + 聚合（如 [CLS]、mean、或最后一层 [CLS]）」得到。
  - 仅依赖该块自身几何与材质，故可跨场景复用：同一 mesh+材质在不同场景中命中同一 key，直接复用 `block_repr`。

- **查询与写入**：
  - 对当前帧每个块，先算 `block_key`，查缓存；命中则用缓存的 `block_repr`，未命中则跑块编码器，得到 `block_repr` 后写入缓存（并登记到反向索引，见第五节）。

### 3.4 与现有 view_transformer 的衔接

当前 view_transformer 的输入是**三角形级**的 `seq`（长度 num_tri + skip）。改为块级后有两种衔接方式：

- **方式 A（需改 view_transformer）**：view_transformer 改为以**块表示**为 context：即 ray tokens 与「块表示序列」做 cross-attention，不再与三角形序列做。这样无需反哺到三角形级，但 view_transformer 的输入维度和训练方式需一起改。
- **方式 B（保留三角形级接口）**：全局 attention 输出仍是块级 `[num_blocks, 768]`；在送入 view_transformer 前，将每个块的 768D **广播/复制**到该块内所有三角形，得到 `[num_tris, 768]`，再拼上 reg_tokens 等，形状与现有一致。这样 view_transformer 无需改输入接口，但三角形级信息是块级的重复，细节会损失，适合作为过渡或消融。

训练时需决定：是采用 A（端到端块级→视图相关）还是 B（块级→广播到三角形→现有视图相关），并相应设计损失与数据。

### 3.5 训练与实现代价

- **模型改动**：新增块划分逻辑、块编码器（块内 construct_seq + 块内 attention + 聚合）、将原「三角形级 12 层」改为「块表示级 12 层」；视衔接方式改或保留 view_transformer 的输入。
- **训练**：需要重新训练或微调。数据仍可用现有 RenderFormer 数据；前向需支持「按块编码 → 块级全局 attention → 视图相关」的 pipeline，损失可与原版一致（渲染图像与 GT 的 loss）。
- **推理**：块编码阶段对每个块先查缓存；未命中再算并回填缓存；全局 attention 与视图相关层与训练时一致。

---

## 四、缓存粒度与键值设计（汇总）

### 4.1 块级方案（推荐，见第三节）

- **键**：`block_key` = 128 位，hash(块内 tri_vpos || texture || vn)，规范化后拼接再哈希。
- **值**：块表示 `block_repr`，形状 `[768]` 或 `[1, 768]`，fp16/bf16 存储。
- **失效**：某块几何或材质变化 → 仅使该块的 `block_key` 失效（删除该 key 的缓存条目）；其他块不受影响。

### 4.2 备选：整场景级（不改模型时）

若暂不改模型与训练，可仅做**推理侧**的整场景缓存（原文档方案 A）：

- **键**：整帧场景哈希（所有三角形+材质）。
- **值**：整段视图无关层输出 `seq_out`。
- **特点**：实现简单，无跨场景复用；任意三角形/材质变化则整场景失效。

### 4.3 不推荐：视图无关层输出后按三角形缓存

视图无关层输出已带全局属性，单三角形 768D 无法脱离场景独立复用，见第二节；若键含整场景则退化为整场景缓存，故不推荐。

---

## 五、缓存存储与淘汰策略

### 5.1 数据结构（块级方案）

- **主缓存**：以「块指纹」为 key 的键值存储。
  - **Key**：128 位块哈希 `block_key`（16 字节）。
  - **Value**：该块的中间表示：
    - `block_repr`: [768] 或 [1, 768]，fp16/bf16（每块一条，显存远小于整段 seq_out）

- **显存管理**：
  - 使用**固定大小显存池**：预分配若干 768 维向量的 buffer，每条缓存引用池中一块；或「键 → 显存 offset」由统一分配器管理。
  - 块级缓存条目多、单条小，总显存可控（例如 1 万块 × 768 × 2 字节 ≈ 15MB）。

### 5.2 LRU + 可选 LFU（与开题一致）

- **LRU**：用 `OrderedDict` 或双向链表 + 哈希表实现，查询/插入 O(1)；每次 get 移到最近使用端；容量超限时从最久未使用端淘汰。
- **LFU 混合**：为每条维护 `(last_access_time, access_count)`；显存达阈值时优先淘汰低频且久未使用的条目（如 `score = -access_count - λ * recency`）。

### 5.3 容量与显存估算

- 单条块表示：768 × 2 字节 ≈ 1.5KB。若缓存 1 万块 ≈ 15MB；10 万块 ≈ 150MB。
- 设定 `max_cache_entries` 或 `max_cache_memory_mb`，插入后检查，超限则按 5.2 淘汰。

---

## 六、失效机制（块级精准删除）

开题要求：「材质/顶点动态变化时，精确删除与变化点相关的 LRU 数据」。

### 6.1 块级方案下的失效

- 块级缓存下，**一个 key 对应一个块**（一个物体/mesh/空间块）。某块几何或材质变化后，该块的指纹 `block_key` 会变，下次查询自然未命中，旧 key 若未被其他场景引用可被 LRU 自然淘汰。
- 若需**主动失效**（例如已知某 mesh 被编辑，希望立刻释放其旧缓存）：维护可选的反向索引 `block_id → block_key`（或 `mesh_id → set(block_key)`），在变化检测到某块时，主动从缓存中删除该 `block_key` 的条目。

### 6.2 变化检测接口（与开题「异步检测层」对应）

- **输入**：当前帧的块/三角形/材质数据，或引擎事件「某 object/mesh 已变更」。
- **输出**：发生变化的 block_id 或 mesh_id。
- 实现方式：管线内按块做轻量 diff（块指纹变化即视为该块变化）；或由外部事件注入。收到变化后调用 `cache.invalidate_block(block_key)` 或 `cache.invalidate_blocks_by_mesh(mesh_id)`，删除对应条目。

这样即可实现「仅淘汰与某材质/某 mesh 相关的一块（或数块）缓存条目」，其余块保留并可跨场景复用。

---

## 七、渲染管线中的集成流程（块级缓存伪代码）

```text
def render_with_block_cache(triangles, texture, mask, vn, c2w, fov, ...):
    # 1) 块划分：得到每块包含的三角形索引
    block_slices = partition_into_blocks(triangles, mask, ...)  # e.g. by mesh_id / object_id

    # 2) 逐块：算 block_key，查缓存；未命中则跑块编码器，并写回缓存
    block_reprs = []
    for block_id, (tri_idx_start, tri_idx_end) in enumerate(block_slices):
        block_key = compute_block_hash(tri_vpos[tri_idx_start:tri_idx_end], texture[...], vn[...])
        cached = cache.get(block_key)
        if cached is not None:
            block_reprs.append(cached)
        else:
            block_repr = block_encoder(tri_vpos[...], texture[...], vn[...])  # 块内 attention
            cache.put(block_key, block_repr.detach().half(), block_id=block_id)
            block_reprs.append(block_repr)

    # 3) 将块表示组装成序列，做全局 attention（视图无关，over 块）
    block_seq = stack(block_reprs)  # [num_blocks, 768]，加上 reg_tokens 等
    seq_out = global_transformer(block_seq, ...)  # 12 层 over 块表示

    # 4) 与视图相关层衔接：按 3.4 方式 A 或 B（块级→view_transformer 或 广播回三角形级）
    # 5) view_transformer(rays, seq_out, ...) → 像素
    ...
```

- `compute_block_hash`：见 3.3，仅用该块内三角形几何+材质，规范化后 128 位哈希。
- `cache.put(block_key, block_repr, block_id=...)`：可选，用于反向索引以便按 block_id/mesh_id 主动失效。
- 若收到「某 mesh 变化」事件，调用 `cache.invalidate_blocks_by_mesh(mesh_id)` 或对受影响块重算 `block_key` 并删除旧 key。

---

## 八、与现有代码的对接点

- **块级方案（需改模型）**：  
  - 在 `renderformer/models/renderformer.py`（或新模块如 `renderformer_block.py`）中：新增块划分、块编码器（块内 construct_seq + 块内 attention + 聚合）、以及「块表示序列 → 12 层 Transformer」的路径；视 3.4 选择改 view_transformer 输入为块级，或保留三角形级接口并在中间做块→三角形广播。  
  - Pipeline 中：先按块查缓存、未命中跑块编码器并回写，再组块序列、跑全局 Transformer、最后 view_transformer。
- **batch_infer.py**：  
  - 调用带块缓存的 `pipeline.render_with_block_cache(...)`；可选在循环外挂变化检测或事件回调，调用 `cache.invalidate_blocks_by_mesh(...)` 等。
- **HDF5 数据**：  
  - 视频多帧若来自同一场景，许多块的几何+材质不变，对应 `block_key` 可跨帧、跨 batch 命中；不同场景中相同物体（相同 mesh+材质）也可复用同一 block_key 的缓存。

---

## 九、精度门控（可选）

- 块编码命中时：直接使用缓存的块表示（fp16/bf16），加速后续全局 attention 与视图相关层。
- 块编码未命中时：用 fp32 跑块编码器并写入缓存（可存为 fp16 省显存）。在 pipeline 中根据「该块是否命中」可对单块选择精度，或简单按「本帧是否全命中」切换整帧精度。

---

## 十、实现顺序建议

1. **Phase 1（块级主方案）**：模型与块编码  
   - 定义块划分（如按 mesh_id / object_id）；  
   - 实现块编码器（块内 construct_seq 风格编码 + 块内小范围 attention + 聚合为 768D）；  
   - 将原「三角形级 12 层」改为「块表示级 12 层」；  
   - 按 3.4 选定与 view_transformer 的衔接方式（块级输入或广播回三角形级），并打通训练/推理前向。

2. **Phase 2**：块级缓存与哈希  
   - 实现 `compute_block_hash`（块内几何+材质规范化 + 128 位哈希）；  
   - 实现 LRU 缓存（Key=block_key，Value=block_repr）；  
   - 在 pipeline 中插入「按块查缓存 → 未命中跑块编码并回写 → 组块序列 → 全局 Transformer → view_transformer」。

3. **Phase 3**：显存与淘汰、失效  
   - 固定显存池 + 容量上限，LRU + 可选 LFU；  
   - 可选反向索引，实现 `invalidate_blocks_by_mesh` 与变化检测/事件对接。

4. **Phase 4（可选）**：若不改模型，可先做**整场景级**推理缓存（见 4.2）验证管线，再切到块级；精度门控、CUDA Graph 等可按需加。

---

## 十一、文件与模块建议

| 模块 | 路径建议 | 职责 |
|------|----------|------|
| 块划分与块编码器 | `renderformer/models/block_encoder.py` 或扩展现有 model | 块内 token 构建、块内 attention、聚合为 block_repr |
| 块级 RenderFormer / 管线 | `renderformer/models/renderformer_block.py` 或 pipeline 扩展 | 块序列 → 全局 Transformer → 与 view_transformer 衔接 |
| 块哈希键 | `renderformer/cache/hash_key.py` | 块内几何+材质规范化 + 128 位 block_key |
| 块缓存存储与 LRU/LFU | `renderformer/cache/block_cache.py` | Key=block_key, Value=block_repr，显存池、淘汰、可选反向索引 |
| 管线集成 | `renderformer/pipelines/rendering_pipeline.py` | `render_with_block_cache`、按块查/写缓存 |
| 视频批处理 | `batch_infer.py` | 使用带块缓存的 pipeline，可选变化检测与失效 |

以上方案将缓存插入在**视图无关层之前**，以「一组三角形」的块为单位做编码与缓存，块表示可跨场景复用；比整场景一个 key 细、比每三角形一个 key 粗且实现可控，但需改模型并重新训练，与开题中的「几何-材质联合哈希」「LRU+LFU」「精准失效」「精度门控」一致，并解决了「视图无关层输出带全局属性、难以按三角形分离复用」的问题。

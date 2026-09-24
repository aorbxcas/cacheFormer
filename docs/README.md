# CacheFormer 文档总览

> **入口文档**（从这里读起）  
> **仓库定位**：在冻结官方 RenderFormer 权重的前提下，先完成 **Direct / Indirect 语义分离**（Hybrid 合成合同 + C1 可学习间接）；**效率提升**是分离之后才允许引入缓存、降频、warp、小头等技术的**结果**。  
> **讲解叙事**：[`talk_script_pipeline.md`](./talk_script_pipeline.md)

---

## 0. 一页读懂项目

**因果（请按此读）**：

```text
RF 一锅端（D+I 缠死）
    → 语义分离：L = D + α·I     ← 主线（Hybrid / C1）
    → I 可过期 / 可学 / 可缓冲；D 可每帧跟相机
    → 允许：VI Cache · 隔帧填充 · Warp · C1 头 · α 降权
    → 表现为：更快、更跟手（结果）
```

```text
官方 RenderFormer
  三角 token → VI → VD → 全图 N（D/I 未分离）
       │
       ▼
语义分离
  ├─ Hybrid：Runtime Direct + clamp(N−D,0) + α
  └─ C1：监督 I*，小头学间接（间接成为一等公民）
       │  分离之后才长出
       ▼
调度与复用（结果层）
  CacheFormer VI Cache · Layered L1–L3（少跑 RF / warp）
  验收数字：快于 CF、absL1<0.08 —— 验证「分离后的复用」可行
```

| 名称 | 在叙事中的位置 | 主要代码 |
|------|----------------|----------|
| **RenderFormer** | 冻结核；一锅端 GI | `renderformer/pipelines/` |
| **Hybrid / 方案四** | **语义**：合成合同 \(D+\alpha I\) | `renderformer/hybrid/` |
| **C1** | **语义**：\(I\) 可学习 | `renderformer/c1/residual_head.py`、`tools/c1_*` |
| **CacheFormer VI Cache** | 分离后允许的技术 | `ViewIndependentCache` |
| **Layered L1–L3** | 分离后允许的调度/复用（结果） | `layered_pipeline.py` |
| **项目 C** | 质量旋钮（α / 时域等） | 设计文档 |

**主线文档**：[`C1_residual_indirect_head.md`](./C1_residual_indirect_head.md)、[`scheme4_hybrid_gi_plan.md`](./scheme4_hybrid_gi_plan.md)。  
**结果层实测**：[`layered_indirect_three_layer.md`](./layered_indirect_three_layer.md)。  
**口述讲稿**：[`talk_script_pipeline.md`](./talk_script_pipeline.md)。

---

## 1. 文档地图（按阅读顺序）

### 1.1 必读（工程落地）

| 文档 | 内容 | 何时读 |
|------|------|--------|
| **本文** | 全局层次、模块关系、命令入口 | 新人或整合回顾 |
| [`talk_script_pipeline.md`](./talk_script_pipeline.md) | **讲解稿**：语义分离主线（口述/答辩） | 分享、答辩、带人 |
| [`C1_residual_indirect_head.md`](./C1_residual_indirect_head.md) | 残差头原理、训练、语义合同 | 训头 / DI–II 分离 |
| [`layered_indirect_three_layer.md`](./layered_indirect_three_layer.md) | 分离后的调度/warp 与 vs CF/RF 实测 | 改跳过帧 / 跑基准 |
| [`engine_domain_perf_directions.md`](./engine_domain_perf_directions.md) | 进引擎时的调度 / 缓存失效 / 预算 | 产品化与帧率目标 |

### 1.2 架构与背景（方案来源）

| 文档 | 内容 |
|------|------|
| [`scheme4_hybrid_gi_plan.md`](./scheme4_hybrid_gi_plan.md) | Confidence-Aware Hybrid：路径 B、R0 Direct、α、里程碑 |
| [`scheme4_paper_reading_guide.md`](./scheme4_paper_reading_guide.md) | 相关论文索引（NRC/NIRC 等） |
| [`project_c_quality_controlled_neural_gi.md`](./project_c_quality_controlled_neural_gi.md) | 质量可控模块设计（violation / 时域等） |
| [`实验记录_同场景VI缓存.md`](./实验记录_同场景VI缓存.md) | VI Cache 实验与结论 |

### 1.3 代码与 Skill

| 路径 | 说明 |
|------|------|
| `renderformer/c1/residual_head.py` | C1 残差头（语义） |
| `renderformer/hybrid/` | Runtime Direct、Hybrid 融合（语义） |
| `renderformer/c1/layered_pipeline.py` | 分离后的调度栈（结果） |
| `renderformer/c1/pruned_pipeline.py` | 剪枝基类 / 重投影 |
| `tools/c1_*.py` | C1 数据 / 训练 / 推理 |
| `tools/benchmark_layered_vs_cf.py` | 结果层动态基准 |
| `.cursor/skills/layered-indirect-gi/SKILL.md` | Agent 质量锁定口径 |

---

## 2. 管线层次：数据流合同

### 2.1 官方 RF（分离前）

```text
H5 → Tokenize → VI (± Cache) → VD → HDR N   （D/I 缠在 N 里）
```

### 2.2 Hybrid（语义：合成合同）

```text
              ┌─ Runtime Direct ──► D
H5 ───────────┤
              └─ RF(N) ──► I = clamp(N−D, 0)
                              L = D + α·I
```

要点：α / Direct **不决定**是否跳过 RF；只决定合成信任度。此步打开「I 可降权 / 可与 D 不同步」的门。

### 2.3 C1（语义：学习合同）

```text
监督：I* = clamp(L_GT − D, 0)
推理：I_pred = ReLU(ReLU(N−D) + Δ)   或 头逐步独立
合成：L = D + α · I_pred
```

要点：间接成为一等公民；**头本身不加速**；跳过帧默认不用未训练头。

### 2.4 Layered L1–L3（结果：分离后的填充与复用）

```text
每帧 → L3 是否更新 I 相关量？
  是：RF(±VI Cache) → 写 fused/I/depth/c2w
  否：L1 warp(缓冲) → L2(可选) → 输出
```

**闸门**（验证复用是否伤外观）：`speedup > 1` vs CF，`mean absL1 < 0.08`，全分辨率。

---

## 3. 模块职责边界（避免搞反）

| 模块 | 负责 | **不**负责 |
|------|------|------------|
| Runtime Direct | 直射身份、硬阴影 | 决定是否 skip RF |
| Confidence α | 信多少间接 | 决定是否 skip RF |
| C1 残差头 | 间接学得好不好 | 单独加速 |
| L3 调度 | 分离后何时更新 I | 重新定义 GI 物理 |
| L1 Warp | 复用旧 I/融合图 | 新算神经间接 |
| VI Cache | 同场景省 VI | 三角级 LRU（命中率≈0） |

---

## 4. 常用命令

```bash
# C1 语义线
python tools/c1_prepare_dataset.py --output_dir data/c1/bootstrap --resolution 256
python tools/c1_train.py --data_dir data/c1/bootstrap --config c1_profiles/default.json
python tools/c1_infer.py --h5_file tmp/cbox/cbox.h5 --checkpoint checkpoints/c1/best.pt --vi_cache

# 分离后的调度结果（vs CF / RF）
python tools/benchmark_layered_vs_cf.py --h5_file tmp/cbox/cbox.h5 \
  --pipelines renderformer,cacheformer,layered
```

离线模型：`$env:HF_HUB_OFFLINE='1'`（PowerShell）。

---

## 5. 验收与指标

| 指标 | 在叙事中的角色 | 闸门 / 用法 |
|------|----------------|-------------|
| **语义可演示** | Direct / Indirect / α 可分开展示 | compare_modes / infer |
| **speedup** | **结果**：分离后复用是否真省算力 | vs CF **> 1** |
| **absL1 / PSNR** | **约束**：复用 I 时别伤太多 | vs CF absL1 **\< 0.08** |
| **RF 次数** | 结果侧观察 | 越少通常越快 |

---

## 6. 路线图（文档视角）

| 阶段 | 叙事位置 | 状态 |
|------|----------|------|
| Hybrid Direct + α | 语义：合成合同 | 已落地 |
| C1-a 头 + 数据管线 | 语义：I 可学 | 已落地 |
| VI Cache | 分离后允许的技术 | 已落地 |
| Layered L1–L3 vs CF/RF | 结果层验收 | 已落地 |
| C1-b 特征头 | 语义加深 | 未做 |
| 引擎异步 I_buffer | 产品形态 | 设计 |

---

## 7. 文档维护约定

1. **改叙事主线（分离 vs 效率因果）** → 先改 **本文** + `talk_script_pipeline.md` + canvas。  
2. **改残差头 / 训练** → 同步 `C1_residual_indirect_head.md`。  
3. **改调度或基准闸门** → 同步 `layered_indirect_three_layer.md` + skill。  
4. 分册开头链接回本文；实测表以各分册为准。

# 三层间接光性能栈：论证总结、设计与验收

> **版本**：1.1  
> **状态**：已实现；主验收 `tools/benchmark_layered_vs_cf.py`（vs CacheFormer，可选 vs 纯 RenderFormer）  
> **项目总览**：[`docs/README.md`](./README.md)  
> **代码**：`renderformer/c1/layered_pipeline.py`、`renderformer/c1/skip_fast_term.py`  
> **关联**：[`C1_residual_indirect_head.md`](./C1_residual_indirect_head.md)、[`scheme4_hybrid_gi_plan.md`](./scheme4_hybrid_gi_plan.md)、[`engine_domain_perf_directions.md`](./engine_domain_perf_directions.md)  
> **Agent skill**：`.cursor/skills/layered-indirect-gi/SKILL.md`

---

## 0. 在全项目中的位置

```text
RenderFormer / CacheFormer（每帧或每帧+VI Cache）
        │
        ▼
Layered L1–L3  ← 本文（少跑 RF：调度 + warp）
        │
        ├─ 可选挂 C1 残差头（L2 mix；未训练默认关）
        └─ 可选 hybrid：D_now + α·warp(I)（不对标 CF 色差）
```

| 问题 | 谁回答 |
|------|--------|
| 为何少跑 Transformer 仍像 GI？ | 本文 §1–3（借调度，不借 PT） |
| Indirect 如何变成可训练量？ | [C1](./C1_residual_indirect_head.md) |
| Direct+α 拼盘从哪来？ | [方案四](./scheme4_hybrid_gi_plan.md) |
| 进引擎怎么排帧预算？ | [engine_domain](./engine_domain_perf_directions.md) |

---

## 1. 辩论结论（为何这样分层）

下列是实现前的关键澄清，避免把模块职责搞反。

### 1.1 Direct / 残差头不是「跳过 RF」的检测器

跳过判定只看：**缓冲是否空、场景指纹、相机转角、隔帧预算、重投影空洞比**。Direct 与残差头都不参与这个开关。

| 模块 | 真正贡献 |
|------|----------|
| **Runtime Direct** | 0-bounce 直射（硬阴影、纹理）；产品形态下应每帧跟相机。质量对标 CacheFormer 时默认 `stub`，避免经典 Direct 与 RF Direct 配方差造成 absL1 虚高。 |
| **残差头（C1）** | 把间接变成可学习量：\(I=\mathrm{ReLU}(\mathrm{ReLU}(N-D)+\Delta)\)。当前刷新帧仍依赖整图 \(N\)，**本身不加速**。跳过帧默认不用未训练头。 |
| **语义分离** | 允许 \(L=D_{\mathrm{now}}+\alpha\cdot I_{\mathrm{stale}}\)。Indirect 可过期、可 warp、可降权；Direct 可跟帧。 |

分离后的 \(I\) **仍然来自 RF**：\(I=\mathrm{clamp}(N-D,0)\) 或头修正。跳过帧 warp 的是「以前的 RF 间接 / 融合图」，不是另一套网络现算的间接。

### 1.2 相对「生来只出间接」的神经 GI（NRC / NIRC / 屏幕空间头）

优势不在间接网络更强，而在运行条件：

- 零在线训练、零路径追踪，借用预训练全 GI Transformer 当间接先验。
- RF 的 VI 与相机无关，同场景可缓存；GBuffer 小网络没有等价物。
- α / violation 只关间接，硬阴影可降级到 Direct-only。

短板：刷新帧仍要先出整图 \(N\) 再拆 \(I\)；4096 面与 VD 耗时仍在。

因此性能上应 **借别人的调度，不借 PT/在线训**：NRC 的持久 I cache、NIRC 的快项每帧/慢项稀疏、NeRFBuff 的历史 warp、ReFrame 的中间特征复用（此处对应 VI Cache）。

### 1.3 三层与论文思想的映射

| 层 | 合同 | 借自 | 本仓库默认（质量对标 CF） |
|----|------|------|---------------------------|
| **L1** | Direct 跟帧；I 用缓冲 warp | NRC：D 不进 cache | `quality_anchor=rf`：warp **RF 融合图**（stub Direct），保证 vs CF 几乎无色差；`hybrid` 才走 \(D_{\mathrm{now}}+\mathrm{warp}(I)\) |
| **L2** | 跳过帧廉价维护 I | NIRC 快项；屏幕空间头 / **C1 mix** | 默认恒等 warp（`l2_gate_strength=0`）；可选亮度门 / 头 mix |
| **L3** | 少跑 Transformer | ReFrame / VI cache / 时间 Two-Level | VI Cache + **小运动推迟 interval 刷新**（`max_skip_run` 兜底） |

验收口径：**效率 > 纯 CacheFormer（每帧 RF+VI），absL1&lt;0.08，全分辨率，禁止半分辨率凑速度。**  
相对纯 RenderFormer（无 VI Cache）通常约 **3×**（材质动态、静止相机批次）。

---

## 2. 数据流

```text
刷新帧（慢项）：
  RF (± VI Cache 命中则跳过 VI)
  quality_anchor=rf  →  fused = N，写入 I/fused/depth/c2w 缓冲
  quality_anchor=hybrid → fused = D + α I_pred

跳过帧（快项）：
  L1  inverse bilinear 重投影 fused / I（解析平面深度）
  L1' 可选 nvd Direct 跟帧（仅 hybrid 锚点）
  L2  可选 lighting gate / 残差头 mix
  不调用 RenderFormer
```

核心公式（hybrid 产品合同）：

\[
L_t = L^{\mathrm{direct}}_t + \alpha\cdot \mathcal{W}(I_{t-k})
\]

质量对标 CacheFormer 时 \(L^{\mathrm{direct}}\) 不替换 RF 外观，等价于 \(\mathcal{W}(L^{\mathrm{RF}}_{t-k})\)。

与纯 RF / CF 对照见总览 [`docs/README.md`](./README.md) §2–3。

---

## 3. 实现要点

### L1 — `LayeredIndirectPipeline` + inverse reproject

- 跳过帧：`reproject_inverse_bilinear` + 解析平面深度（轮廓可能有锯齿；大 orbit/FOV 易出画幅黑边）。
- `l1_direct_follow=True` 且 `quality_anchor=hybrid`：nvdiffrast Direct 每帧，再与 warp(I) 合成。
- 对标 CF 时关闭 Direct 跟帧，避免 GGX Direct 与 RF Direct 系统性偏差。

### L2 — `skip_fast_term.py`

- `closed_form_lighting_gate`：\(I \leftarrow I\cdot\mathrm{clamp}(L_D^{\mathrm{now}}/L_D^{\mathrm{warp}})\)，`strength=0` 为恒等。
- `mix_residual_head`：仅当存在 C1 checkpoint 且 `l2_head_mix>0`。未训练头不得默认开启。

### L3 — 自适应刷新

在指纹未变时：

1. 转角 ≥ `max_camera_rot_deg` → 强制 RF  
2. 距上次刷新 ≥ `max_skip_run` → 强制 RF  
3. 距上次 ≥ `refresh_every` **且** 转角 ≥ `soft_rot_deg` → RF  
4. 否则 skip（推迟 interval）→ 统计 `adaptive_defers`

材质/几何指纹变仍立即刷新（与 CacheFormer 一样必须重跑 VI）。

---

## 4. 怎么跑

```bash
# 默认：CF + Layered；也可加上纯 RF
python tools/benchmark_layered_vs_cf.py --h5_file tmp/cbox/cbox.h5 \
  --model_id microsoft/renderformer-v1.1-swin-large \
  --pipelines renderformer,cacheformer,layered

# 无画幅黑边：相机静止 + 材质动态（多场景）
python tools/benchmark_layered_vs_cf.py \
  --scenes shader-ball,room,tree \
  --dynamics roughness_orbit,specular_orbit,irradiance_orbit \
  --orbit_span_deg 0 --fov_crop 0.88 \
  --output_root out/compare_layered_vs_cf_noborder
```

默认动态类型（`build_dynamic_plans`）：`orbit_only`、`fov_sweep`、`roughness_orbit`、`specular_orbit`、`irradiance_orbit`、`roughness_fast`、`combined`。

报告：`out/compare_layered_vs_cf/SUMMARY.md`（及 `*_noborder`）。

### 实测摘要（GTX 1660 SUPER · RF Large · 256²）

**批次 A · cbox · 7 动态 · 解析平面 warp · vs CF**

| 用例 | CF ms | L123 ms | vs CF | absL1 | 闸门 |
|------|------:|--------:|------:|------:|:----:|
| orbit_only | 1209 | 687 | 1.76× | 0.012 | 过 |
| fov_sweep | 1219 | 692 | 1.76× | 0.018 | 过 |
| roughness_orbit | 2757 | 2295 | 1.20× | 0.008 | 过 |
| specular_orbit | 2829 | 2312 | 1.22× | 0.009 | 过 |
| irradiance_orbit | 2777 | 2321 | 1.20× | 0.008 | 过 |
| roughness_fast | 2802 | 2444 | 1.15× | 0.005 | 过 |
| combined | 2813 | 2342 | 1.20× | 0.021 | 过 |

**批次 C · 多场景 · 相机静止 · 材质动态 · vs RF / CF**

| 场景×动态 | RF ms | CF ms | L123 ms | vs RF | vs CF | absL1 |
|-----------|------:|------:|--------:|------:|------:|------:|
| shader-ball（三种） | ~15700 | ~5900 | ~5250 | **~3.0×** | ~1.12× | ~0.006 |
| room（三种） | ~9060 | ~3585 | ~3030 | **~3.0×** | ~1.18× | ~0.013 |
| tree（三种） | ~5195 | ~2200 | ~1740 | **~3.0×** | ~1.27× | ~0.007 |

跳过帧稳态约 3–11 ms。刷新帧 absL1=0；跳过帧为 warp 误差。完整表见 `out/compare_layered_vs_cf*/SUMMARY.md`。

产品向 hybrid（不作为 vs CF 质量闸）：

```python
LayeredIndirectPipeline(rf, quality_anchor="hybrid", l1_direct_follow=True, direct_mode="always")
```

---

## 5. 明确不做

- 运行时 PT / NRC 在线训  
- 未训练残差头当跳过帧默认快项  
- 半分辨率 RF 再上采样（质量锁定）  
- 按三角 LRU 缓存 VI（全局 attention 命中率≈0）

---

## 6. 下一步（仍未做）

- C1-b：VI 特征 → 小头出 I，刷新帧也可跳过 VD  
- 运动向量 warp 替代解析平面深度  
- 跳过帧 GBuffer 条件化小头（需训练，且 mix 要小）

与 C1 里程碑 M4/M5、引擎文档对齐，见 [`docs/README.md`](./README.md) §6。

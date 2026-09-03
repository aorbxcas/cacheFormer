# C1 路线：冻结 RenderFormer + 残差间接光头

> **版本**：1.3  
> **状态**：P0–P2 **质量锁定路径**：全分辨率 RF 刷新 + 重投影跟视图；动态序列约 **1.7× CF**、absL1≈0.006  
> **关联**：[`scheme4_hybrid_gi_plan.md`](./scheme4_hybrid_gi_plan.md)、[`project_c_quality_controlled_neural_gi.md`](./project_c_quality_controlled_neural_gi.md)、[`实验记录_同场景VI缓存.md`](./实验记录_同场景VI缓存.md)、[`engine_domain_perf_directions.md`](./engine_domain_perf_directions.md)  
> **定位**：在「不从零训练大模型」的前提下，让 **Direct / Indirect 在语义上真正分开**，并继续沿用 CacheFormer 的缓存与调度思想。

---

## 实现入口（代码）

| 模块 | 路径 |
|------|------|
| 残差头 | `renderformer/c1/residual_head.py` |
| 损失 / Dataset / 推理管线 | `renderformer/c1/losses.py` / `dataset.py` / `pipeline.py` |
| 数据准备 | `tools/c1_prepare_dataset.py` |
| GT/Direct 对齐（M1） | `tools/c1_align_gt.py` |
| 训练 | `tools/c1_train.py` |
| 验证 | `tools/c1_eval.py` |
| 三路对比 | `tools/c1_compare_modes.py` |
| 推理（含 confidence α） | `tools/c1_infer.py` |
| 默认配置 | `c1_profiles/default.json` |

```bash
# 1) 烘焙训练集（默认伪 GT = RF，无 Blender 可跑）
python tools/c1_prepare_dataset.py --output_dir data/c1/bootstrap --resolution 256 --views_per_scene 6 --save_preview

# 1b) Direct 对齐 + 导出 gt_cache EXR
python tools/c1_align_gt.py --data_dir data/c1/bootstrap --write_aligned --export_gt_cache gt_cache/c1_pseudo

# 2) 训练残差头
python tools/c1_train.py --data_dir data/c1/bootstrap --config c1_profiles/default.json --epochs 40

# 2b) 验证
python tools/c1_eval.py --data_dir data/c1/bootstrap --checkpoint checkpoints/c1/best.pt --split val

# 3) 推理（Hybrid confidence α）
python tools/c1_infer.py --h5_file tmp/cbox/cbox.h5 --checkpoint checkpoints/c1/best.pt --vi_cache --confidence --auto_align

# 4) Direct / Hybrid / C1 对比
python tools/c1_compare_modes.py --h5_file tmp/c1_scenes/cbox.h5 --checkpoint checkpoints/c1/best.pt --auto_align
```

有 Cycles GT 时：`--gt_source blender --gt_cache_dir gt_cache` 或 `--gt_source exr_dir --gt_exr_dir ...` 重跑准备脚本。

---

## 0. 通俗讲解：C1 在解决什么问题？

### 0.1 用做菜类比

可以把一张有全局光照的图想成一道菜：

| 部分 | 像什么 | 谁来做更合适 |
|------|--------|----------------|
| **直接光（Direct）** | 锅里最基础的火候、硬阴影、物体被灯直接照到的部分 | 传统光栅 / 阴影图（快、稳、跟得上物体移动） |
| **间接光（Indirect）** | 墙上的色溢、软反射、光线弹几下才看见的氛围 | 神经网络（贵，但擅长「看起来像 GI」） |

**官方 RenderFormer** 像是一个「整锅端」的大厨：一锅里把直射和弹射全煮好了端上来。好吃，但：

- 物体一动，整锅往往要重煮（VI 缓存容易 miss）；
- 你没法只让他「只负责勾芡（间接光）」，直射部分他也全程参与计算。

**当前 CacheFormer Hybrid** 像是：大厨仍煮整锅，旁边再摆一碟「物理直射」，吃的时候把两碟拼盘。拼盘能修阴影、控越界，但 **大厨的工时并没有省下来**——该跑的 VI/VD 还是都跑了。

**C1** 想改成：

> 物理厨师每帧炒「直射」；  
> 神经学徒只学「差多少间接光」；  
> 最后：`成品 = 直射 + 间接残差`。

这样：

1. **语义上分开**：网络目标就是间接，不再是「全图再减去直射」的事后分解；  
2. **工程上好调度**：物体狂动时，直射跟着跑；间接可以降频率、降分辨率、靠缓存；  
3. **训练上扛得住**：不重训 2～4 亿参数的大模型，只训一个小头（或 LoRA）。

### 0.2 一句话目标

> **用可承受的单卡微调，把「全 GI 网络」变成「间接残差网络」，让 Direct 跟玩法走、Indirect 可缓存可降频，并保留 CacheFormer 的加速与融合思想。**

### 0.3 C1 不是什么

| 不是 | 说明 |
|------|------|
| 不是从零训练 RenderFormer | 官方训练集与训练代码未完整开源，个人算力也达不到 |
| 不是保证每帧满配电影级间接光 | 游戏场景优先「可玩、可预览、可收敛」 |
| 不是取代经典 GI（Lumen/DDGI 等） | 研究/插件级路径，先证明 DI/II 分离 + 缓存调度成立 |

---

## 1. 背景与动机

### 1.1 现状瓶颈

1. **整场景 VI 缓存**：仅当几何+材质完全不变时命中；物体位移 → 持续 miss。  
2. **Hybrid 分解融合**：`I ≈ clamp(H_neural − H_direct, 0)` 改善画质与体感，但 **RF 仍算全 GI**。  
3. **开销大头**：同场景命中后仍约 ~240ms/帧量级在 VD；场景变化时 VI+VD 全开。

### 1.2 为何需要 C1

| 诉求 | Hybrid 现状 | C1 |
|------|-------------|-----|
| Direct / Indirect 计算分离 | 否（只分离合成） | **是（监督与输出均为间接）** |
| 物体运动时少跑神经腿 | 靠调度勉强 | 调度 + 小头，语义更干净 |
| 单卡可训 | 无需训 | **小头 / LoRA 可训** |
| 保留 CacheFormer | 已有 | **VI/块缓存、α 融合、降频仍适用** |

---

## 2. 原理（技术版）

### 2.1 分解

完整线性 HDR：

\[
L = L_{\mathrm{direct}} + L_{\mathrm{indirect}}
\]

监督信号（训练时）：

\[
I^{\*} = \mathrm{clamp}\big(L_{\mathrm{GT}} - L_{\mathrm{direct}}^{\mathrm{classic}},\, 0\big)
\]

或使用渲染器 Indirect AOV（若可得）。

推理：

\[
L_{\mathrm{out}} = L_{\mathrm{direct}}^{\mathrm{classic}} + \alpha \cdot I_{\mathrm{pred}}
\]

其中 \(\alpha \in [0,1]\) 可为固定值或现有 `ConfidenceMap`。

### 2.2 数据流

```text
                    ┌─ Runtime Direct ──────────────► L_direct
场景 (mesh/H5) ─────┤
                    │  RF (冻结权重)
                    │    ├─ VI  (可 + VI/块缓存)
                    │    └─ VD / 中层特征
                    └─ Residual Head (可训练) ──► I_pred
                              │
                              ▼
                    L_out = L_direct + α · I_pred
```

### 2.3 两种实现档位

#### C1-a：图像域残差头（先做）

| 项 | 说明 |
|----|------|
| 输入 | 低分 `H_neural`、可选 `L_direct`、depth |
| 结构 | 1×1 校正 / 浅 U-Net / 小型 CNN |
| 监督 | `I*` |
| 优点 | 实现快、易 debug |
| 风险 | 可能「学会抄 RF−Direct」，RF 降频后变差 → 需 ablating |

#### C1-b：特征域残差头（更贴 CacheFormer）

| 项 | 说明 |
|----|------|
| 输入 | 冻结 RF 的 VI token / VD 中层特征 + Direct |
| 结构 | 轻量 decoder |
| 优点 | 静物可只复用 VI 缓存再过小头；更像「间接专用支路」 |
| 成本 | 要改 pipeline 暴露中间特征 |

**推荐顺序**：C1-a 验证数据与损失 → C1-b 接缓存。

### 2.4 与「全 GI 减 Direct」的本质区别

| | 现 Hybrid | C1 |
|--|-----------|-----|
| 网络训练目标 | 无（用官方全 GI） | **最小化 ‖I_pred − I*‖** |
| 推理依赖 | 必须有较准的 H_neural | 理想情况下头可逐步独立 |
| 长期 | 无法去掉全量 RF | 可走向「低成本特征 + 头」或隔帧只跑头 |

---

## 3. 训练与数据要求

### 3.1 单条样本

| 字段 | 来源 |
|------|------|
| 场景表示 | JSON → H5（现有管线） |
| `L_direct` | `runtime_direct` 或 Blender Direct AOV |
| `L_GT` | Blender Cycles beauty（同相机、同曝光约定） |
| `I*` | `clamp(L_GT − L_direct, 0)` |

### 3.2 规模建议（单卡原型）

| 级别 | 规模 | 用途 |
|------|------|------|
| 冒烟 | 几十场景 × 数视角 | 管道跑通 |
| 起步 | 50～200 场景 × 4～8 视角 | 目标 0 |
| 像样 | 500～2000 场景 × 多视角 | 目标 1 |

域约束：三角数、灯光、相机尽量落在 RenderFormer 训练分布内（见仓库 README）。

### 3.3 损失（建议）

- 主损失：对 `I_pred` 与 `I*` 的 L1 / 相对 L1  
- 可选：对 `L_direct + I_pred` 与 `L_GT` 的重建损失  
- 可选：高亮 / violation 区域加权  

### 3.4 显存策略（8GB 级消费卡）

1. **两阶段**：先离线烘焙低分 RF 特征或 `H_neural`，再只训头；  
2. 训练分辨率先 128/256；batch 1 + 梯度累积；  
3. RF 全程 `torch.no_grad()` + fp16。

---

## 4. 预期目标（可验收）

### 目标 0 — 可行性（约 2～4 周）

- [ ] 数据管线：H5 + Direct + GT → `I*`  
- [ ] 冻结 RF，残差头 loss 下降  
- [ ] `Direct + I_pred`  visibly 优于「仅 Direct」  

### 目标 1 — 质量

相对基线：

| 基线 | 预期关系 |
|------|----------|
| 仅 Direct | C1 明显更好（有间接） |
| 现 Hybrid | 接近；硬阴影不更差；violation 不更高 |
| 纯 RF | Direct/阴影更好；极端间接允许略弱 |

指标：PSNR / FLIP（全图 + 间接区域）、violation rate、动态序列观感。

### 目标 2 — 效率（体现「分开」）

| 场景 | 预期 |
|------|------|
| Direct 每帧 | 成本保持可忽略相对 RF |
| 神经腿 | 小头 ≪ 全 RF；隔帧 / 低分后平均帧耗时下降 |
| 静景 | VI 缓存 + 头，接近「跳过重复 VI」 |
| 动景 | 不承诺满间接每帧；承诺交互跟手，静止后补全 |

### 非目标

- 复现官方 RF 训练规模或全面超越 Large 权重  
- 首期支持透明 / SSS / 复杂环境光  
- 替代引擎内置 Lumen/DDGI 的生产方案  

---

## 5. 与 CacheFormer 模块的衔接

| CacheFormer 能力 | C1 中用法 |
|------------------|-----------|
| `ViewIndependentCache` | 静物 / 编辑器漫游时缓存 VI，供头使用 |
| 块缓存（v2 思想） | 局部材质/几何微变时减 construct_seq |
| `HybridFusionPipeline` / α | `L_direct + α·I_pred`，violation 降权 |
| Direct 先行 / 并行 | 游戏交互帧默认策略 |
| 动态场景对比脚本 | 评测「物体动 + 间接隔帧」 |

---

## 6. 游戏引擎使用场景下的改进空间

> 假设产品形态是：**编辑器预览 / 运行时近似 GI 插件**，帧率与跟手优先于离线电影级 Indirect。  
> **汇总与优先级**：见 [`engine_domain_perf_directions.md`](./engine_domain_perf_directions.md)。

### 6.1 对 C1 的针对性改进

| 改进 | 动机 | 做法草案 |
|------|------|----------|
| **交互档 / 质量档** | 拖拽要跟手，静下来要好看 | 交互：仅 Direct（或 Direct+上帧 warp 间接）；松手/静止：跑 RF 特征+头 |
| **间接更新预算** | 引擎有固定 ms 预算 | 每帧最多更新 N% 像素或固定 tile；其余复用上一帧 II |
| **屏幕空间间接** | 头不必吃全分辨率 | II @ 1/4～1/2，双边/棋盘上采样；Direct 全分辨率 |
| **GBuffer 条件化** | 引擎本就有深度/法线/运动向量 | 头输入加 depth、N、motion vector、Roughness，少依赖全量 RF |
| **残差对「玩法灯」敏感** | 动态灯常见 | 训练加入动态 emissive / 局部灯；或灯变时强制刷新 II |
| **与引擎 Tonemap 对齐** | 避免 HDR 合完再崩 | 在线性空间融合，tone map 交给引擎 |

### 6.2 对 CacheFormer 本身的引擎向改进

| 改进 | 现状问题 | 引擎向改法 |
|------|----------|------------|
| **缓存键粒度** | 整场景指纹过粗 | 按 Actor/组件 / 静态几何 vs 动态几何 分 key；静物永命中 |
| **世界分区缓存** | 大关卡无法整场景 VI | 按 cell / sublevel 缓存 VI；玩家移动只激活邻域 |
| **运动向量驱动失效** | 物体一动全 miss | 仅 invalidate 运动物体相关 token/块；静物保留（需 C1-b 或分层 VI） |
| **时间切片 VI/VD** | 一帧算不完 | 多帧摊销：帧 0 编码，帧 1 半层 VI，帧 2 VD… |
| **异步计算队列** | 现同步 Python 管线 | Direct 在渲染线程；神经在 async compute；上一帧 II 合成 |
| **固定点 / 探针混合** | 纯 feed-forward 难覆盖超大场景 | 远景 DDGI/探针，近景 C1；或室内用 RF 域 |
| **内容管线** | H5/JSON 偏研究 | 导入 FBX/glTF → 自动 remesh 到预算三角 → 烘焙 H5 子集 |
| **降级策略** | miss 就全量 RF | miss → Direct only → 低分 II → 满分 II（分级） |

### 6.3 引擎场景下的推荐产品形态

```text
运行时每帧：
  1. 引擎光栅：Albedo / Depth / Shadow → L_direct
  2. 查「间接缓冲」（上一帧或异步结果）
  3. L = L_direct + α * I_buffer

异步 / 降频：
  - 静物 VI 缓存命中 → 只更新动态相关 + 小头
  - 或：每 N 帧全量刷新 I_buffer
  - 相机大裁/传送 → 强制 miss，短时 Direct-only
```

这样 **CacheFormer 的缓存思想**变成引擎里的「间接光缓冲 + 分区失效」；**C1** 变成「填间接缓冲的学习器」，而不是每帧替代整个渲染器。

### 6.4 优先级建议（游戏引擎假设）

| 优先级 | 项 | 归属 |
|--------|----|------|
| P0 | Direct 每帧 + 间接缓冲复用 / 隔帧 | CacheFormer 调度 |
| P0 | 交互档 Direct-only | CacheFormer / 引擎集成 |
| P1 | C1-a 残差头（证明 II 可学） | C1 |
| P1 | 静/动几何分缓存键 | CacheFormer |
| P2 | C1-b 特征头 + VI 缓存喂头 | C1 + Cache |
| P2 | 屏幕空间低分 II + motion warp | 引擎向 |
| P3 | 大世界分区 / 与 DDGI 混合 | 产品化 |

---

## 7. 里程碑草案

| 阶段 | 交付 | 退出标准 | 状态 |
|------|------|----------|------|
| M0 | 本文档 + 数据格式约定 | 评审通过 | **完成** |
| M1 | GT/Direct 对齐脚本，导出 Direct/GT/I* / gt_cache | align_report 可复现 | **完成（伪 GT + 真实 Cycles GT）** |
| M2 | C1-a 训练 + 推理接 confidence α | 目标 0；eval/compare 可跑 | **完成（真实 GT 上 decompose 基线可打）** |
| M3 | 隔帧 II + 质量锁定对比 | 超 CF + 全分辨率无明显劣化 | **P0–P2 质量路径已达成（analytic 重投影）** |

### 本阶段终局（质量锁定 P0–P2）

约束：**禁止半分辨率输出**；`quality_lock=True` 强制 `neural/direct_res_scale=1.0`。

| 配置 | mean ms | vs CF | absL1 vs CF | PSNR |
|------|---------|-------|-------------|------|
| CacheFormer | ~347 | 1× | — | — |
| **quality stub + reproject + analytic depth** | **~204** | **1.70×** | **0.006** | **~49 dB** |
| stub 全分辨率（冻结跳过帧，无重投影） | ~167 | 1.94× | 跳过帧不跟相机 | — |

```bash
python tools/benchmark_quality_p0_p2.py --h5_file tmp/c1_scenes/cbox.h5 \
  --direct_mode stub --view_follow reproject --depth_mode analytic --no_c1_head
```

要点：
- **L0**：每 N 帧 / 换场景全分辨率刷新 RF；跳过帧不跑 Transformer  
- **跟视图**：全分辨率深度重投影融合缓冲（非降采样）  
- **深度**：默认 `analytic`（快）；`raycast` 更准但刷新更贵  
- **P1**：刷新帧叠 VI；相机转角过大强制 refresh  
- **P2**：可选 `--use_c1_head` 把残差头写入 I；基准含 absL1/PSNR  

### 本阶段终局（stub @ full-res，速度上限消融）

动态序列（cbox，12 帧，每 3 帧换 roughness）：

| 配置 | mean ms | vs CacheFormer | 说明 |
|------|---------|----------------|------|
| CacheFormer | ~325 | 1× | 每帧 RF+VI |
| **pruned_stub neural_res=1.0 freeze** | **~167** | **~1.94×** | 刷新全分辨率 RF；跳过帧冻结 |

```bash
python tools/benchmark_pruned_l0_l1.py --h5_file tmp/c1_scenes/cbox.h5 \
  --refresh_every 3 --neural_res_scale 1.0 --direct_mode stub --no_c1_head
```

后续：nvdiffrast 真 Direct 接 `always`；本阶段验收以「全分辨率 + 超 CF + absL1 小」为准。

### M3 剪枝实现（L0 / L1）

| 层级 | 代码 | 行为 |
|------|------|------|
| **L0 调度剪枝** | `renderformer/c1/pruned_pipeline.py` | 换场景 / 每 N 帧才跑 RF；其余帧 **不调用** Transformer，复用 `I_buffer` |
| **L1 刷新降本** | 同上 `neural_res_scale` / `direct_res_scale` | 刷新半分辨率 RF；每帧低分 lite Direct |
| **Benchmark** | `tools/benchmark_pruned_l0_l1.py` | 动态 roughness 序列 vs CacheFormer |

```bash
# 动态场景：L0/L1 + 低分 Direct(每帧) + 半分辨率 RF(刷新)，串行（默认）
python tools/benchmark_pruned_l0_l1.py --h5_file tmp/c1_scenes/cbox.h5 \
  --refresh_every 3 --neural_res_scale 0.5 --direct_res_scale 0.25 --direct_mode always
```

- `direct_mode=always` + `direct_res_scale=0.25`：每帧跟相机的 Direct（lite@64）  
- `parallel_refresh`：**默认关**；lite Direct + RF 多流实测会严重退化  
- `stub`：神经侧消融上限（跳过帧 ~0 成本）

**动态序列实测**（cbox，12 帧，每 3 帧改 roughness，256²，`out/compare_pruned_l0_l1_fast_direct_seq2`）：

| 配置 | mean ms | vs CacheFormer | RF 调用 |
|------|---------|----------------|---------|
| CacheFormer (RF+VI 每帧) | ~331 | 1× | 12 |
| L0/L1 always Direct@64 + RF@128 刷新 | **~188** | **1.76×** | 4 |
| 同配置 Direct@32 | ~190 | 1.74× | 4 |
| stub（无 Direct，神经上限） | ~135 | 2.45× | 4 |
| 并行 refresh（已弃用默认） | ~980 | 0.34× | 4 |

跳过帧约 **50 ms**（仅 Direct）；刷新帧约 **450 ms**（Direct + RF@128）。瓶颈已从 RF 转移到 lite Direct；真光栅 Direct 后跳过帧有望进一步逼近 stub。

| M4 | C1-b 或静动分缓存 PoC | 静物 hit、动物更新的统计 | 待做 |
| M5（可选） | 引擎插件原型（Unity/Unreal 其一） | 编辑器内跟手预览 | 待做 |

---

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 头只抄 `RF−Direct` | 训练时随机丢弃/扰动 H_neural；强制看 Direct+几何；对比「无 RF 输入」ablation |
| Direct 与 Cycles 不对齐 | 统一色彩空间、曝光；auto_align；或一律用同一 Direct 源做监督与推理 |
| 8GB OOM | 离线烘焙特征；降分；梯度检查点 |
| 游戏动态光超出 RF 域 | 缩小演示域；或动态光只进 Direct，II 用保守 α |
| 期望过高 | 验收以目标 0/1/2 为准，不与官方 Large 全量对打 |

---

## 9. 总结

| 问题 | 回答 |
|------|------|
| C1 原理（通俗） | 物理管直射，小网络学「还差多少间接」，再拼起来 |
| C1 目标 | 语义分离 DI/II；单卡可训；接上缓存与降频 |
| 和现 Hybrid | Hybrid 是拼盘；C1 是让神经学徒改行只做勾芡 |
| 游戏引擎 | 把 C1 做成「间接缓冲生成器」，CacheFormer 做成「分区缓存 + 交互降级」；Direct 永远跟帧 |

---

## 10. 参考

- RenderFormer（SIGGRAPH 2025）与官方推理权重  
- 方案四 Hybrid：[`scheme4_hybrid_gi_plan.md`](./scheme4_hybrid_gi_plan.md)  
- NRC / RTXGI：经典 Direct + 神经 radiance cache 的业界范式  
- NIRC / Two-Level MC：神经近似 + 残差分解叙事  
- 本仓库 VI 缓存实验：[`实验记录_同场景VI缓存.md`](./实验记录_同场景VI缓存.md)

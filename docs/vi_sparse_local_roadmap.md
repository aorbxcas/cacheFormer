# VI Sparse-Local 改造路线图

本文档给出基于“局部稀疏 VI + 增量更新”的工程落地规格与实验步骤，目标是在场景小幅变化时减少全量 VI 重算成本。

## 1. 目标

- 在保持现有 `VD` 路径不变的前提下，引入可控近似的 VI 增量更新。
- 对“同场景、局部变化”的连续帧取得稳定加速。
- 保留全量回退路径，避免累计误差失控。

## 2. 分阶段交付

### Phase A（已可开始）

- 强化缓存命名空间隔离：避免不同模型/精度/注意力后端错复用。
- 在批量推理中加入周期性缓存清空参数，作为 hybrid 回退基线。

### Phase B（下一步实现）

- 新增 `SparseVIEncoder`（局部邻域 + 全局 token 桥接）。
- 提供 `vi_mode=full|sparse_local|hybrid` 运行开关。
- 支持 dirty 三角索引输入与 k-hop 邻域扩展。

### Phase C（训练）

- 使用 full VI 作为 teacher 做蒸馏。
- 目标：token 误差可控 + 图像误差可控 + 稳定时序表现。

### Phase D（上线策略）

- 默认 `hybrid`：小变化走 sparse，大变化自动回退 full。
- 每 N 帧强制全量刷新一次，抑制漂移。

## 3. 模型改造规格（Design 1）

### 3.1 邻域定义

- 每个三角 token 只 attend 到：
  - `k-hop` 邻域 token
  - 全局 register tokens
- 全局 tokens 仍可访问全体三角 token，用于跨区域信息汇聚。

建议初值：

- `k_hop=2`
- `max_neighbors=32`
- `num_global_tokens` 复用现有 register token

### 3.2 增量更新接口（建议）

- `encode_view_independent_sparse(...) -> (seq_vi, valid_mask_padded, state)`
- `update_view_independent_sparse(prev_state, dirty_idx, ...) -> (seq_vi_new, state_new)`

### 3.3 Hybrid 回退条件（建议）

- `dirty_ratio > 0.15`：回退全量 VI
- 每 `30` 帧：强制全量刷新
- 质量哨兵超阈值（例如低分辨率 PSNR 下降超限）：回退全量

## 4. 训练数据要求

### MVP 规模

- 100~300 个场景
- 每场景 20~80 视角
- 每场景 10~30 段局部扰动序列（材质/几何）

### 数据关键属性

- 可追踪三角索引（能定位 dirty 三角）
- 同场景连续小改动（增量学习核心）
- 覆盖多网格规模与材质复杂度

## 5. 损失函数建议

- `L_vi = ||seq_sparse - seq_full||_2`
- `L_img = L1(img_sparse, img_full) + LPIPS`
- `L_temp`（可选）：时间一致性损失

总损失：

- `L = λ1 * L_vi + λ2 * L_img + λ3 * L_temp`

## 6. 实验步骤

### 实验 1：离线可行性（无训练）

1. 实现 sparse 推理路径（不改变权重）。
2. 固定数据集，比较 full vs sparse：
   - `VI MSE`
   - `HDR PSNR/RMSE`
   - `VI 时间` 与 `总时间`
3. 网格化扫描参数：`k_hop`、`max_neighbors`、`dirty_ratio`。

### 实验 2：蒸馏训练

1. teacher = 当前 full VI。
2. student = sparse VI。
3. 训练后评估：
   - 相对 full 的画质下降
   - 推理加速比
   - 长序列稳定性（是否漂移）

### 实验 3：Hybrid 策略验证

1. 打开回退阈值与周期全量刷新。
2. 对长序列（300+ 帧）压测。
3. 统计：
   - 平均/95P 时延
   - 回退触发率
   - 画质随帧曲线

## 7. 当前仓库优先修改位置

- `renderformer/models/renderformer.py`：VI 路径新增 sparse/hybrid 分支
- `renderformer/pipelines/rendering_pipeline.py`：策略开关与运行态调度
- `renderformer/cache/vi_cache.py`：状态缓存与命名空间策略
- `batch_infer.py`：实验参数、统计与回退控制
- `test_h5_native_vs_cache.py`：扩展对 sparse/hybrid 的对比指标

## 8. 成功标准（建议）

- 在 dirty ratio <= 10% 的序列上：
  - 平均 VI 耗时降低 >= 35%
  - 画质相对 full 降幅 <= 0.5 dB（PSNR）
  - 无明显时序闪烁增加

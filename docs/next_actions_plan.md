# 下一步行动方案（按优先级）

本文档给出接下来 2~4 周可执行的改进路径，目标是在不破坏现有推理链路的前提下，逐步落地 VI 增量优化。

## 第 0 步（今天可完成）

- 跑一次基线：
  - `python compare_baseline_vs_vi_cache.py --h5_file <your.h5> --num_renders 8`
- 跑一次带周期刷新的 hybrid 近似基线：
  - `python compare_baseline_vs_vi_cache.py --h5_file <your.h5> --num_renders 8 --cache_refresh_interval 4`
- 记录：
  - 总耗时、平均帧耗时、缓存命中率、加速比

## 第 1 步（1~3 天）

- 在 `batch_infer.py` 统一你们的线上参数：
  - 开启 `--vi_cache`
  - 按业务场景设置 `--vi_cache_full_refresh_interval`（例如 0 / 16 / 32）
  - 如果做 AB 实验，设置不同 `--vi_cache_runtime_tag`
- 目标：
  - 形成稳定的同场景视频渲染基线报表（JSON + 日志）

## 第 2 步（1 周）

- 实现 `SparseVIEncoder` 最小版（仅 inference，先不训练）：
  - 邻域定义：`k_hop=2`
  - 最大邻居：`max_neighbors=32`
  - 运行开关：`vi_mode=full|sparse_local`
- 用现有对比脚本扩展指标：
  - `VI MSE`
  - `HDR PSNR / RMSE`
  - `VI 时间占比`

## 第 3 步（1~2 周）

- 蒸馏训练（teacher=full VI, student=sparse VI）：
  - 先 token 对齐损失，再叠加图像损失
  - 样本覆盖 dirty ratio: 1% / 5% / 10% / 20%
- 通过标准（建议）：
  - PSNR 降幅 <= 0.5 dB
  - VI 时间降低 >= 35%

## 第 4 步（上线前）

- 上线 `hybrid` 策略：
  - 小变化走 sparse
  - 大变化回退 full
  - 每 N 帧强制 full 刷新
- 做 300+ 帧长序列压测，输出：
  - 平均/95 分位时延
  - 回退触发率
  - 画质随帧曲线

## 常用命令模板

- 同场景缓存基准：
  - `python benchmark_vi_cache.py --h5_file <your.h5> --num_renders 10`
- 基线 vs 缓存：
  - `python compare_baseline_vs_vi_cache.py --h5_file <your.h5> --num_renders 10`
- 基线 vs hybrid（周期刷新）：
  - `python compare_baseline_vs_vi_cache.py --h5_file <your.h5> --num_renders 10 --cache_refresh_interval 5`
- 批量推理（含缓存）：
  - `python batch_infer.py --h5_folder <folder> --vi_cache --vi_cache_full_refresh_interval 16`


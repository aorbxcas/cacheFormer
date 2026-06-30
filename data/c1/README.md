# C1 训练数据与代码说明

本目录由 `tools/c1_prepare_dataset.py` 生成（默认 gitignore `data/c1/` 大数据）。

## 快速开始

```bash
# 1. 单独转 H5（避免与 RF 同时占内存）
python scene_processor/convert_scene.py examples/cbox.json --output_h5_path tmp/c1_scenes/cbox.h5

# 2. 烘焙样本（伪 GT = RF；有 Blender 时加 --gt_source blender）
python tools/c1_prepare_dataset.py \
  --scenes tmp/c1_scenes/cbox.h5 tmp/c1_scenes/veach-mis.h5 \
  --output_dir data/c1/bootstrap \
  --views_per_scene 4 --roughness_variants 2 --save_preview

# 2b. Direct 对齐（推荐）
python tools/c1_align_gt.py --data_dir data/c1/bootstrap --write_aligned --export_gt_cache gt_cache/c1_pseudo

# 3. 训练
python tools/c1_train.py --data_dir data/c1/bootstrap --epochs 40

# 4. 验证 / 对比 / 推理
python tools/c1_eval.py --data_dir data/c1/bootstrap --checkpoint checkpoints/c1/best.pt --split val
python tools/c1_compare_modes.py --h5_file tmp/c1_scenes/cbox.h5 --checkpoint checkpoints/c1/best.pt --auto_align --confidence
python tools/c1_infer.py --h5_file tmp/c1_scenes/cbox.h5 --checkpoint checkpoints/c1/best.pt --confidence --auto_align
```

## 样本字段

| key | shape | 含义 |
|-----|-------|------|
| `hdr_neural` | H×W×3 | 冻结 RF 全图 |
| `hdr_direct` | H×W×3 | Runtime Direct（对齐后为 scaled） |
| `depth` | H×W×1 | Direct 深度 |
| `hdr_gt` | H×W×3 | GT（伪标签或 Cycles） |
| `indirect_target` | H×W×3 | `clamp(gt - direct, 0)` |

## 注意

伪 GT（`gt_source=neural`）下，`indirect_target` 与 `relu(neural-direct)` 几乎相同，残差头难以超过「简单分解」。要验证 C1 独立建模能力，请接入 Cycles beauty 后重烘焙。

详见 `docs/C1_residual_indirect_head.md`。

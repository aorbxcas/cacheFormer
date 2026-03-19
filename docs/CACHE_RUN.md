# 块级缓存测试运行说明

## 1. 准备场景 H5

若尚无 H5 场景文件，先做场景转换（与官方用法一致）：

```bash
# 从 examples 选一个场景并转换为 H5
python scene_processor/convert_scene.py examples/cbox.json --output_h5_path tmp/cbox/cbox.h5
```

视频序列可直接使用已有 H5 目录，例如：`video-data/teaser-scenes/cbox-roughness`。

## 2. 单场景多次渲染（验证命中）

使用脚本 `infer_with_cache.py`：对同一场景渲染多次，第二次起会命中块缓存并打印统计信息。

```bash
# 默认渲染 2 次：第 1 次全未命中，第 2 次全命中
python infer_with_cache.py --h5_file tmp/cbox/cbox.h5 --output_dir output/cbox_cache

# 可选参数
# --block_size 256       每个块的三角形数量（默认 256）
# --max_cache_entries 10000  LRU 最大条目数
# --runs 3               渲染次数（多次可观察命中率）
# --resolution 512       分辨率
# --precision fp16       fp16 / bf16 / fp32
```

## 3. 完整 H5 视频序列 + 缓存统计与汇总

使用 `batch_infer_with_cache.py` 对整段 H5 目录逐帧渲染，**逐帧打印缓存数据**，结束时输出**命中汇总**。

```bash
# 示例：cbox-roughness 整段序列
python batch_infer_with_cache.py --h5_folder video-data/teaser-scenes/cbox-roughness --output_dir output/videos/cbox-roughness-cache

# 可选
# --block_size 256         每块三角形数
# --max_cache_entries 50000 缓存最大条目
# --quiet                   少打逐行日志，只保留最后汇总
# --save_video              是否合成 video.mp4（默认 True）
```

控制台会输出：
- **逐帧**：帧号、文件名、三角形数、本帧块数、本帧 hit/miss、本帧命中率、当前缓存条数与占用。
- **结尾汇总**：总帧数、总查询次数、总命中/未命中、整体命中率、缓存条目数、缓存占用；并提示首帧（预期全 miss）与末帧（同场景预期高命中）。

## 4. 控制台输出说明（单场景 infer_with_cache）

- **Run 1**：各块均为 cache miss，会执行块编码并写入缓存；打印 `hits=0 misses=<块数>`。
- **Run 2**：各块命中缓存，只做查表与重组；打印 `hits=<块数> misses=0`，`hit_rate=100%`。
- **Rendering data**：输出张量形状、dtype、min/max/mean，便于核对数值范围。

示例：

```
Scene: tmp/cbox/cbox.h5  num_triangles=XXXX  block_size=256
------------------------------------------------------------

--- Run 1/2 ---
[block_cache] blocks=... block_size=256 hits=0 misses=... hit_rate=0.00% size=... memory_mb=...
[render] output shape=torch.Size([1, 1, 512, 512, 3]) dtype=torch.float32 min=... max=...
Cache stats: {'hits': 0, 'misses': ..., 'size': ..., 'memory_mb': ..., 'hit_rate': 0.0}
Rendering data: shape=torch.Size([1, 512, 512, 3]) ...

--- Run 2/2 ---
[block_cache] blocks=... hits=... misses=0 hit_rate=100.00% ...
...
```

## 5. 与无缓存推理对比

无缓存单张图推理（官方脚本）：

```bash
python infer.py --h5_file tmp/cbox/cbox.h5 --output_dir output/cbox
```

`infer_with_cache.py` 在第二次及以后运行同一 H5 时，会跳过块编码计算，仅做缓存查找与全局 Transformer + 视图相关层，用于验证缓存路径与统计输出。

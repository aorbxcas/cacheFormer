# 块级缓存测试运行说明

## 1. 准备场景 H5

若尚无 H5 场景文件，先做场景转换（与官方用法一致）：

```bash
# 从 examples 选一个场景并转换为 H5
python scene_processor/convert_scene.py examples/cbox.json --output_h5_path tmp/cbox/cbox.h5
```

## 2. 运行带缓存的推理（推荐）

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

## 3. 控制台输出说明

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

## 4. 与无缓存推理对比

无缓存单张图推理（官方脚本）：

```bash
python infer.py --h5_file tmp/cbox/cbox.h5 --output_dir output/cbox
```

`infer_with_cache.py` 在第二次及以后运行同一 H5 时，会跳过块编码计算，仅做缓存查找与全局 Transformer + 视图相关层，用于验证缓存路径与统计输出。

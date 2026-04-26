import glob
import json
import os
import time
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from typing import Any, Dict, List, Optional
import argparse
import imageio
from tqdm import tqdm

from renderformer import RenderFormerRenderingPipeline, ViewIndependentCache
from renderformer.layers.attention import ATTN as ATTN_IMPL
from simple_ocio import ToneMapper


class TriangleRenderH5Dataset(Dataset):
    def __init__(self, h5_folder_path: str, padding_length: Optional[int] = None):
        self.file_list = glob.glob(os.path.join(h5_folder_path, '*.h5'))
        self.file_list = natsorted(self.file_list)
        print(f'Found {len(self.file_list)} h5 files in {h5_folder_path}')
        self.padding_length = padding_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        with h5py.File(file_path, 'r') as f:
            triangles = torch.from_numpy(np.array(f['triangles'])).float()
            num_tris = triangles.shape[0]
            texture = torch.from_numpy(np.array(f['texture'])).float()
            vn = torch.from_numpy(np.array(f['vn'])).float()
            c2w = np.array(f['c2w']).astype(np.float32)
            fov = np.array(f['fov']).astype(np.float32)

            if self.padding_length is not None:
                triangles = torch.concatenate((triangles, torch.zeros(
                    (self.padding_length - num_tris, *triangles.shape[1:]))), dim=0)
                texture = torch.concatenate((texture, torch.zeros(
                    (self.padding_length - num_tris, *texture.shape[1:]))), dim=0)
                vn = torch.concatenate((vn, torch.zeros(
                    (self.padding_length - num_tris, *vn.shape[1:]))), dim=0)
                mask = torch.zeros(self.padding_length, dtype=torch.bool)
                mask[:num_tris] = True
            else:
                mask = torch.ones(num_tris, dtype=torch.bool)

            data = {
                'triangles': triangles,
                'texture': texture,
                'mask': mask,
                'c2w': torch.from_numpy(c2w).float(),
                'fov': torch.from_numpy(fov).float(),
                'vn': vn,
                'file_path': file_path
            }
        return data


def main():
    parser = argparse.ArgumentParser(description="Batch inference using triangle radiosity transformer model")
    parser.add_argument("--h5_folder", type=str, required=True, help="Path to the folder containing input H5 files")
    parser.add_argument("--model_id", type=str, help="Model ID on Hugging Face or local path", default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--precision", type=str, choices=['bf16', 'fp16', 'fp32'], default='fp16', 
                        help="Precision for inference")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for inference")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--padding_length", type=int, default=None, help="Padding length for inference")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Output directory for rendered images (default: same as input folder)")
    parser.add_argument("--save_video", dest="save_video", action="store_true",
                        help="Merge rendered images into a video at video.mp4.")
    parser.add_argument("--no_save_video", dest="save_video", action="store_false",
                        help="Disable merging rendered images into video.")
    parser.set_defaults(save_video=True)
    parser.add_argument("--tone_mapper", type=str, choices=['none', 'agx', 'filmic', 'pbr_neutral'], default='none', help="Tone mapper for inference")
    parser.add_argument("--perf_json", type=str, default=None,
                        help="Write performance report JSON (default: <output_dir>/render_perf.json)")
    parser.add_argument("--no_perf", action="store_true", help="Disable performance stats and JSON report")
    parser.add_argument("--vi_cache", action="store_true",
                        help="Enable VI cache (view-independent stage); 启用时强制 batch_size=1，同场景多帧/多视角会命中缓存加速")
    parser.add_argument("--vi_cache_max_entries", type=int, default=64,
                        help="VI cache 最大条目数 (default: 64)")
    parser.add_argument("--vi_cache_full_refresh_interval", type=int, default=0,
                        help="每 N 个 batch 清空一次 VI 缓存（0=不清空，用于 hybrid fallback 基线）")
    parser.add_argument("--vi_cache_runtime_tag", type=str, default="",
                        help="可选：附加在 VI 缓存命名空间中的自定义标签，用于实验隔离")
    args = parser.parse_args()

    if args.vi_cache and args.batch_size != 1:
        args.batch_size = 1
        print("VI cache 仅支持 batch_size=1，已自动设为 1")
    if args.vi_cache_full_refresh_interval < 0:
        parser.error("--vi_cache_full_refresh_interval 必须 >= 0")

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    use_cuda = device.type == "cuda"

    # Load model configuration and weights
    t_load0 = time.perf_counter()
    pipeline = RenderFormerRenderingPipeline.from_pretrained(args.model_id)

    if device == torch.device('cuda') and os.name == 'posix':  # avoid windows
        from renderformer_liger_kernel import apply_kernels
        apply_kernels(pipeline.model)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device == torch.device('mps'):
        args.precision = 'fp32'
        print("bf16 and fp16 will cause too large error in MPS, force using fp32 instead.")

    pipeline.to(device)
    model_ready_sec = time.perf_counter() - t_load0

    # Tone mapper
    if args.tone_mapper != 'none':
        if args.tone_mapper == 'pbr_neutral':
            args.tone_mapper = 'Khronos PBR Neutral'
        tone_mapper = ToneMapper(args.tone_mapper)
        print(f"Using {args.tone_mapper} tone mapper")

    # Create dataset and dataloader
    dataset = TriangleRenderH5Dataset(args.h5_folder, args.padding_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    # Set output directory
    output_dir = args.output_dir if args.output_dir is not None else args.h5_folder
    os.makedirs(output_dir, exist_ok=True)

    if args.save_video:
        video_frames = []

    vi_cache: Optional[ViewIndependentCache] = None
    vi_cache_hits, vi_cache_misses = 0, 0
    if args.vi_cache:
        vi_cache = ViewIndependentCache(max_entries=args.vi_cache_max_entries)
        print(f"VI cache 已启用，max_entries={args.vi_cache_max_entries}")
        if args.vi_cache_full_refresh_interval > 0:
            print(f"VI cache 周期清空已启用，每 {args.vi_cache_full_refresh_interval} 个 batch 清空一次")

    vi_cache_runtime_info = {
        "model_id": args.model_id,
        "precision": args.precision,
        "attention_impl": ATTN_IMPL,
        "runtime_tag": args.vi_cache_runtime_tag,
    }

    perf: Dict[str, Any] = {}
    batch_infer_ms: List[float] = []
    batch_e2e_ms: List[float] = []
    batch_h2d_ms: List[float] = []
    batch_save_ms: List[float] = []
    if not args.no_perf and use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Batch inference
    print(f"Starting batch inference on {len(dataset)} files with batch size {args.batch_size}")
    t_loop0 = time.perf_counter()
    pbar = tqdm(dataloader)
    for batch_idx, batch in enumerate(pbar):
        batch_size = batch['triangles'].shape[0]
        t_b0 = time.perf_counter()

        # Move data to device
        t0 = time.perf_counter()
        triangles = batch['triangles'].to(device)
        texture = batch['texture'].to(device)
        mask = batch['mask'].to(device)
        vn = batch['vn'].to(device)
        c2w = batch['c2w'].to(device)
        fov = batch['fov'].unsqueeze(-1).to(device)
        file_paths = batch['file_path']
        if use_cuda:
            torch.cuda.synchronize()
        h2d_ms = (time.perf_counter() - t0) * 1000.0

        # Perform inference (with optional VI cache when batch_size=1)
        t1 = time.perf_counter()
        use_vi_cache_this_batch = vi_cache is not None and batch_size == 1
        if (
            use_vi_cache_this_batch
            and args.vi_cache_full_refresh_interval > 0
            and batch_idx > 0
            and batch_idx % args.vi_cache_full_refresh_interval == 0
        ):
            vi_cache.clear()
        if use_vi_cache_this_batch:
            out = pipeline(
                triangles=triangles,
                texture=texture,
                mask=mask,
                vn=vn,
                c2w=c2w,
                fov=fov,
                resolution=args.resolution,
                torch_dtype=torch.float16 if args.precision == 'fp16' else torch.bfloat16 if args.precision == 'bf16' else torch.float32,
                vi_cache=vi_cache,
                vi_cache_runtime_info=vi_cache_runtime_info,
                return_vi_cache_info=True,
            )
            rendered_imgs, vi_info = out
            if vi_info["vi_cache_hit"]:
                vi_cache_hits += 1
            else:
                vi_cache_misses += 1
        else:
            rendered_imgs = pipeline(
                triangles=triangles,
                texture=texture,
                mask=mask,
                vn=vn,
                c2w=c2w,
                fov=fov,
                resolution=args.resolution,
                torch_dtype=torch.float16 if args.precision == 'fp16' else torch.bfloat16 if args.precision == 'bf16' else torch.float32
            )
        if use_cuda:
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t1) * 1000.0

        # Save outputs
        t2 = time.perf_counter()
        for i in range(batch_size):
            file_path = file_paths[i]
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            nv = c2w.shape[1]
            for view_idx in range(nv):
                hdr_img = rendered_imgs[i, view_idx].cpu().numpy().astype(np.float32)
                if args.tone_mapper != 'none':
                    ldr_img = tone_mapper.hdr_to_ldr(hdr_img)
                else:
                    ldr_img = np.clip(hdr_img, 0, 1)
                ldr_img = (ldr_img * 255).astype(np.uint8)

                hdr_path = os.path.join(output_dir, f"{base_name}_view_{view_idx}.exr")
                ldr_path = os.path.join(output_dir, f"{base_name}_view_{view_idx}.png")

                imageio.v3.imwrite(hdr_path, hdr_img)
                imageio.v3.imwrite(ldr_path, ldr_img)

                if args.save_video:
                    video_frames.append(ldr_img)

        if use_cuda:
            torch.cuda.synchronize()
        save_ms = (time.perf_counter() - t2) * 1000.0
        e2e_ms = (time.perf_counter() - t_b0) * 1000.0

        if not args.no_perf:
            batch_h2d_ms.append(h2d_ms)
            batch_infer_ms.append(infer_ms)
            batch_save_ms.append(save_ms)
            batch_e2e_ms.append(e2e_ms)
            n_frames_batch = batch_size * c2w.shape[1]
            fps_inst = n_frames_batch / (e2e_ms / 1000.0) if e2e_ms > 0 else 0.0
            postfix = f"infer {infer_ms:.0f}ms | {fps_inst:.1f} fps (batch)"
            if vi_cache is not None and (vi_cache_hits + vi_cache_misses) > 0:
                postfix += f" | VI hit {vi_cache_hits}/{vi_cache_hits + vi_cache_misses}"
            pbar.set_postfix_str(postfix, refresh=False)

    loop_sec = time.perf_counter() - t_loop0
    num_views = 0
    if len(dataset) > 0:
        with h5py.File(dataset.file_list[0], "r") as f:
            num_views = int(np.array(f["c2w"]).shape[0])
    num_frames = len(dataset) * num_views

    video_encode_sec = 0.0
    if args.save_video:
        t_vid0 = time.perf_counter()
        video_frames = np.array(video_frames)
        video_path = os.path.join(output_dir, 'video.mp4')
        imageio.v3.imwrite(video_path, video_frames, fps=24, quality=9)
        video_encode_sec = time.perf_counter() - t_vid0
        print(f"Video saved to: {video_path}")

    print(f"Output saved to: {output_dir}")

    if not args.no_perf:
        n_batches = len(batch_e2e_ms)
        total_infer_sec = sum(batch_infer_ms) / 1000.0
        total_h2d_sec = sum(batch_h2d_ms) / 1000.0
        total_save_sec = sum(batch_save_ms) / 1000.0
        perf = {
            "device": str(device),
            "model_id": args.model_id,
            "model_load_sec": round(model_ready_sec, 4),
            "inference_loop_wall_sec": round(loop_sec, 4),
            "video_encode_sec": round(video_encode_sec, 4),
            "num_h5_files": len(dataset),
            "num_views_per_frame": num_views,
            "num_output_images": num_frames,
            "batch_size": args.batch_size,
            "resolution": args.resolution,
            "precision": args.precision,
            "num_batches": n_batches,
            "time_infer_sec": round(total_infer_sec, 4),
            "time_h2d_sec": round(total_h2d_sec, 4),
            "time_save_io_sec": round(total_save_sec, 4),
            "throughput_fps_e2e_loop": round(num_frames / loop_sec, 4) if loop_sec > 0 else 0.0,
            "throughput_fps_infer_only": round(num_frames / total_infer_sec, 4) if total_infer_sec > 0 else 0.0,
            "avg_ms_per_batch_infer": round(sum(batch_infer_ms) / n_batches, 3) if n_batches else 0.0,
            "avg_ms_per_batch_e2e": round(sum(batch_e2e_ms) / n_batches, 3) if n_batches else 0.0,
            "avg_ms_per_image_infer": round(total_infer_sec * 1000.0 / num_frames, 4) if num_frames else 0.0,
        }
        if vi_cache is not None:
            total_vi = vi_cache_hits + vi_cache_misses
            perf["vi_cache_hits"] = vi_cache_hits
            perf["vi_cache_misses"] = vi_cache_misses
            perf["vi_cache_hit_rate"] = round(vi_cache_hits / total_vi, 4) if total_vi else 0.0
            perf["vi_cache_full_refresh_interval"] = args.vi_cache_full_refresh_interval
            perf["vi_cache_runtime_tag"] = args.vi_cache_runtime_tag
        if use_cuda:
            perf["peak_gpu_memory_allocated_mib"] = round(
                torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
            perf["peak_gpu_memory_reserved_mib"] = round(
                torch.cuda.max_memory_reserved() / (1024 ** 2), 2)
        else:
            perf["peak_gpu_memory_allocated_mib"] = None
            perf["peak_gpu_memory_reserved_mib"] = None

        perf_path = args.perf_json or os.path.join(output_dir, "render_perf.json")
        with open(perf_path, "w", encoding="utf-8") as pf:
            json.dump(perf, pf, indent=2, ensure_ascii=False)

        print("\n========== 性能统计 (Performance) ==========")
        print(f"  模型加载 Model load:     {perf['model_load_sec']:.2f} s")
        print(f"  推理循环 wall (含IO):   {perf['inference_loop_wall_sec']:.2f} s")
        print(f"  其中 纯推理累计:        {perf['time_infer_sec']:.2f} s")
        print(f"  其中 H2D 累计:          {perf['time_h2d_sec']:.2f} s")
        print(f"  其中 存盘/后处理累计:   {perf['time_save_io_sec']:.2f} s")
        if args.save_video:
            print(f"  视频编码 video encode:  {perf['video_encode_sec']:.2f} s")
        print(f"  输出图张数:             {num_frames}")
        print(f"  端到端吞吐:             {perf['throughput_fps_e2e_loop']:.2f} 张/s (整段循环)")
        print(f"  推理吞吐:               {perf['throughput_fps_infer_only']:.2f} 张/s (仅 forward)")
        print(f"  平均每张推理:           {perf['avg_ms_per_image_infer']:.2f} ms")
        if use_cuda:
            print(f"  GPU 峰值显存 allocated: {perf['peak_gpu_memory_allocated_mib']} MiB")
            print(f"  GPU 峰值显存 reserved:  {perf['peak_gpu_memory_reserved_mib']} MiB")
        if vi_cache is not None:
            print(f"  VI 缓存命中:            {perf.get('vi_cache_hits', 0)} / {perf.get('vi_cache_hits', 0) + perf.get('vi_cache_misses', 0)} (hit_rate={perf.get('vi_cache_hit_rate', 0):.2%})")
        print(f"  详细 JSON:              {perf_path}")
        print("============================================\n")


if __name__ == '__main__':
    main()

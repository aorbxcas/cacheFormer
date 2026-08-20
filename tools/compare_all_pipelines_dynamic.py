#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景变化连续序列：全管线对比。

每 scene_change_every 帧切换材质 roughness（场景指纹变 → VI 应 miss），
段内仅相机 orbit 变化（应 hit）。

管线同 compare_all_pipelines_sequence.py。

用法:
  python tools/compare_all_pipelines_dynamic.py \\
      --h5_file tmp/c1_scenes/cbox.h5 \\
      --num_frames 12 --scene_change_every 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_vi_cache import _apply_camera_variant, load_h5
from compare_baseline_vs_vi_cache import _get_gpu_memory_mb, _sync
from renderformer import RenderFormerRenderingPipeline, ViewIndependentCache
from renderformer.c1.pipeline import C1ResidualPipeline
from renderformer.c1.pruned_pipeline import PrunedIndirectPipeline
from renderformer.c1.residual_head import ResidualIndirectHead
from renderformer.hybrid.pipeline import HybridFusionPipeline
from renderformer.hybrid.profile import HybridProfile
from renderformer.hybrid.runtime_direct.factory import create_runtime_direct_renderer

# texture: [B, N, 13, 32, 32]
ROUGHNESS_CHANNEL = 6
SCENE_ROUGHNESS = [0.15, 0.45, 0.75, 0.95]


def _orbit_variant(frame_idx: int, num_frames: int) -> dict:
    if num_frames <= 1:
        deg = 0.0
    else:
        deg = -30.0 + 60.0 * frame_idx / (num_frames - 1)
    return {"name": f"f{frame_idx + 1:02d}_orbit_{deg:+.0f}deg", "orbit_y_deg": deg}


def _scene_texture(base_texture: torch.Tensor, scene_id: int) -> torch.Tensor:
    tex = base_texture.clone()
    r = SCENE_ROUGHNESS[scene_id % len(SCENE_ROUGHNESS)]
    tex[:, :, ROUGHNESS_CHANNEL : ROUGHNESS_CHANNEL + 1, :, :] = r
    return tex


def _gpu_peak_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def _tonemap(hdr: np.ndarray) -> np.ndarray:
    x = np.clip(hdr.astype(np.float32), 0, None)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    scale = float(np.percentile(x[x > 0], 95)) if (x > 0).any() else 1.0
    ldr = 1.0 - np.exp(-x / (scale * 0.6 + 1e-6))
    return (np.clip(ldr, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def _to_hwc(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().cpu()
    while x.dim() > 3 and x.shape[0] == 1:
        x = x[0]
    if x.dim() == 4 and x.shape[0] <= 4:
        if x.shape[-1] == 3:
            x = x[0]
        else:
            x = x[0].permute(1, 2, 0)
    elif x.dim() == 3 and x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    return x.numpy().astype(np.float32)


def _abs_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _rel_l1(a: np.ndarray, b: np.ndarray, eps: float = 1e-3) -> float:
    return float(np.mean(np.abs(a - b) / (np.abs(b) + eps)))


def _ensure_c2w_fov(data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    c2w = data["c2w"]
    fov = data["fov"]
    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)
    if c2w.dim() == 3:
        c2w = c2w.unsqueeze(1)
    if fov.dim() == 1:
        fov = fov.view(1, 1, 1)
    elif fov.dim() == 2:
        fov = fov.unsqueeze(-1)
    return c2w, fov


def _load_c1_head(path: str, device: torch.device) -> ResidualIndirectHead:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("head_cfg") or {}
    head = ResidualIndirectHead(
        use_neural=bool(cfg.get("use_neural", True)),
        use_depth=bool(cfg.get("use_depth", True)),
        base_channels=int(cfg.get("base_channels", 32)),
        num_blocks=int(cfg.get("num_blocks", 3)),
    )
    head.load_state_dict(ckpt["model"])
    return head.to(device).eval()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="All-pipeline dynamic scene-change compare")
    parser.add_argument("--h5_file", type=str, default="tmp/c1_scenes/cbox.h5")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/c1_cycles/best.pt")
    parser.add_argument("--model_id", type=str, default="microsoft/renderformer-v1.1-swin-large")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--scene_change_every", type=int, default=3)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--output_dir", type=str, default="out/compare_all_dynamic")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--auto_align", action="store_true", default=True)
    parser.add_argument("--no_auto_align", action="store_true")
    parser.add_argument(
        "--modes",
        type=str,
        default=(
            "rf_baseline,rf_vicache,direct_only,hybrid,hybrid_vicache,"
            "c1,c1_vicache,pruned_always,pruned_stub"
        ),
        help="逗号分隔管线 id",
    )
    args = parser.parse_args()
    auto_align = args.auto_align and not args.no_auto_align
    want = {m.strip() for m in args.modes.split(",") if m.strip()}

    if args.scene_change_every < 1:
        parser.error("--scene_change_every >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    out_root = Path(args.output_dir)
    frames_root = out_root / "frames"
    out_root.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    num_scenes = (args.num_frames + args.scene_change_every - 1) // args.scene_change_every
    expected_misses = num_scenes
    expected_hits = max(0, args.num_frames - expected_misses)

    print("=" * 60)
    print("场景变化全管线对比")
    print(f"  帧={args.num_frames}, 每 {args.scene_change_every} 帧换 roughness → {num_scenes} 段")
    print(f"  roughness 序列={SCENE_ROUGHNESS[:num_scenes]}")
    print(f"  预期 VI: miss≈{expected_misses}, hit≈{expected_hits}")
    print("=" * 60)

    data = load_h5(args.h5_file, device)
    base_c2w, base_fov = _ensure_c2w_fov(data)
    data["c2w"], data["fov"] = base_c2w, base_fov
    base_tex = data["texture"]

    print(f"Loading RF {args.model_id}")
    rf = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    rf.to(device)

    direct_impl = create_runtime_direct_renderer(backend="lite", ray_chunk=2048)
    hybrid = HybridFusionPipeline(rf, profile=HybridProfile(), direct_renderer=direct_impl).to(device)

    c1 = None
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_file() and ({"c1", "c1_vicache"} & want):
        head = _load_c1_head(str(ckpt_path), device)
        c1 = C1ResidualPipeline(
            rf,
            head,
            direct_renderer=create_runtime_direct_renderer(backend="lite", ray_chunk=2048),
            alpha=1.0,
            use_confidence=True,
            auto_align=auto_align,
        ).to(device)

    def plan_frame(i: int) -> Tuple[dict, torch.Tensor, int, float]:
        scene_id = i // args.scene_change_every
        var = _orbit_variant(i, args.num_frames)
        tex = _scene_texture(base_tex, scene_id)
        r = SCENE_ROUGHNESS[scene_id % len(SCENE_ROUGHNESS)]
        return var, tex, scene_id, r

    def make_rf(use_cache: bool):
        cache = ViewIndependentCache(max_entries=16) if use_cache else None

        def _run(c2w, fov, texture, clear_cache=False):
            if cache is not None and clear_cache:
                cache.clear()
            imgs, info = rf.render(
                triangles=data["triangles"],
                texture=texture,
                mask=data["mask"],
                vn=data["vn"],
                c2w=c2w,
                fov=fov,
                resolution=args.resolution,
                torch_dtype=dtype,
                vi_cache=cache,
                return_vi_cache_info=True,
            )
            return _to_hwc(imgs), dict(info)

        return _run

    def make_direct():
        def _run(c2w, fov, texture, clear_cache=False):
            hdr, _depth = direct_impl.render(
                data["triangles"], texture, data["vn"], data["mask"], c2w, fov, args.resolution
            )
            return _to_hwc(hdr), {"vi_cache_hit": False}

        return _run

    def make_hybrid(use_cache: bool):
        cache = ViewIndependentCache(max_entries=16) if use_cache else None

        def _run(c2w, fov, texture, clear_cache=False):
            if cache is not None and clear_cache:
                cache.clear()
            ctx = hybrid.render(
                triangles=data["triangles"],
                texture=texture,
                mask=data["mask"],
                vn=data["vn"],
                c2w=c2w,
                fov=fov,
                resolution=args.resolution,
                torch_dtype=dtype,
                vi_cache=cache,
                auto_align=auto_align,
            )
            meta = {k: v for k, v in ctx.meta.items() if isinstance(v, (int, float, bool, str, type(None)))}
            return _to_hwc(ctx.hdr_fused), meta

        return _run

    def make_c1(use_cache: bool):
        cache = ViewIndependentCache(max_entries=16) if use_cache else None

        def _run(c2w, fov, texture, clear_cache=False):
            if cache is not None and clear_cache:
                cache.clear()
            ctx = c1.render(
                triangles=data["triangles"],
                texture=texture,
                mask=data["mask"],
                vn=data["vn"],
                c2w=c2w,
                fov=fov,
                resolution=args.resolution,
                torch_dtype=dtype,
                vi_cache=cache,
            )
            meta = {k: v for k, v in ctx.meta.items() if isinstance(v, (int, float, bool, str, type(None)))}
            return _to_hwc(ctx.hdr_fused), meta

        return _run

    def make_pruned(direct_mode: str):
        pipe = PrunedIndirectPipeline(
            rf,
            refresh_every=args.scene_change_every,
            neural_res_scale=0.5,
            direct_res_scale=0.25,
            parallel_refresh=False,
            direct_mode=direct_mode,
            head=None,
            auto_align_direct=auto_align,
        ).to(device)

        def _run(c2w, fov, texture, clear_cache=False):
            if clear_cache:
                pipe.reset()
            r = float(texture[0, 0, ROUGHNESS_CHANNEL, 0, 0].item())
            out = pipe.render(
                data["triangles"],
                texture,
                data["mask"],
                data["vn"],
                c2w,
                fov,
                resolution=args.resolution,
                torch_dtype=dtype,
                scene_key=f"rough_{r:.2f}",
            )
            meta = {
                k: v
                for k, v in out.meta.items()
                if isinstance(v, (int, float, bool, str, type(None)))
            }
            meta["vi_cache_hit"] = False
            meta["refreshed"] = out.refreshed
            return _to_hwc(out.hdr_fused), meta

        return _run

    catalog: List[Tuple[str, str, Callable]] = []
    if "rf_baseline" in want:
        catalog.append(("rf_baseline", "RF 全量（无缓存）", make_rf(False)))
    if "rf_vicache" in want:
        catalog.append(("rf_vicache", "CacheFormer（RF+VI）", make_rf(True)))
    if "direct_only" in want:
        catalog.append(("direct_only", "Runtime Direct", make_direct()))
    if "hybrid" in want:
        catalog.append(("hybrid", "Hybrid GI", make_hybrid(False)))
    if "hybrid_vicache" in want:
        catalog.append(("hybrid_vicache", "Hybrid + VI", make_hybrid(True)))
    if "c1" in want and c1 is not None:
        catalog.append(("c1", "C1 残差", make_c1(False)))
    if "c1_vicache" in want and c1 is not None:
        catalog.append(("c1_vicache", "C1 + VI", make_c1(True)))
    if "pruned_always" in want:
        catalog.append(("pruned_always", "L0/L1 Direct@64", make_pruned("always")))
    if "pruned_stub" in want:
        catalog.append(("pruned_stub", "L0/L1 stub（无 Direct）", make_pruned("stub")))

    frame_meta = []
    for i in range(args.num_frames):
        var, _tex, sid, r = plan_frame(i)
        frame_meta.append(
            {
                "index": i + 1,
                "name": var["name"],
                "orbit_y_deg": var["orbit_y_deg"],
                "scene_id": sid,
                "roughness": r,
            }
        )

    results: Dict[str, Any] = {
        "scene": args.h5_file,
        "resolution": args.resolution,
        "num_frames": args.num_frames,
        "scene_change_every": args.scene_change_every,
        "num_scenes": num_scenes,
        "expected_misses": expected_misses,
        "expected_hits": expected_hits,
        "device": str(device),
        "precision": args.precision,
        "checkpoint": str(ckpt_path) if ckpt_path.is_file() else None,
        "frames": frame_meta,
        "modes": {},
    }

    ref_hdrs: List[np.ndarray] = []

    for mode_id, mode_label, runner in catalog:
        print(f"\n=== Mode: {mode_id} ({mode_label}) ===")
        mode_dir = frames_root / mode_id
        mode_dir.mkdir(parents=True, exist_ok=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        # warmup：不清缓存尾帧（对 vicache：先 clear 再填 scene0）
        var0, tex0, _, _ = plan_frame(0)
        c2w0, fov0 = _apply_camera_variant(base_c2w, base_fov, var0, device, dtype)
        try:
            for wi in range(args.warmup):
                _sync(device)
                runner(c2w0, fov0, tex0, clear_cache=(wi == 0 and "vicache" in mode_id))
        except RuntimeError as e:
            print(f"  [SKIP] {e}")
            results["modes"][mode_id] = {
                "summary": {"label": mode_label, "error": str(e), "mean_ms": None},
                "frames": [],
            }
            continue

        # 正式跑：vicache 从冷启动 clear，使第 1 帧 miss、换场景 miss
        records: List[dict] = []
        times: List[float] = []
        hits = misses = 0
        aborted = False

        for i in range(args.num_frames):
            var, tex, sid, r = plan_frame(i)
            c2w, fov = _apply_camera_variant(base_c2w, base_fov, var, device, dtype)
            clear = i == 0 and ("vicache" in mode_id or mode_id.startswith("pruned"))
            try:
                _sync(device)
                t0 = time.perf_counter()
                hdr, meta = runner(c2w, fov, tex, clear_cache=clear)
                _sync(device)
                elapsed = time.perf_counter() - t0
            except RuntimeError as e:
                print(f"  [ABORT] frame {i + 1}: {e}")
                aborted = True
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break

            times.append(elapsed)
            hit = bool(meta.get("vi_cache_hit", False))
            if "vicache" in mode_id:
                if hit:
                    hits += 1
                else:
                    misses += 1

            if mode_id == "rf_baseline":
                ref_hdrs.append(hdr.copy())

            q_abs = q_rel = None
            if mode_id != "rf_baseline" and i < len(ref_hdrs) and hdr.shape == ref_hdrs[i].shape:
                q_abs = _abs_l1(hdr, ref_hdrs[i])
                q_rel = _rel_l1(hdr, ref_hdrs[i])

            png = mode_dir / f"frame_{i + 1:02d}_s{sid}.png"
            iio.imwrite(png, _tonemap(hdr))
            del hdr

            if device.type == "cuda" and mode_id in (
                "direct_only",
                "hybrid",
                "hybrid_vicache",
                "c1",
                "c1_vicache",
                "pruned_always",
            ):
                torch.cuda.empty_cache()

            rec = {
                "frame": i + 1,
                "scene_id": sid,
                "roughness": r,
                "name": var["name"],
                "orbit_y_deg": var["orbit_y_deg"],
                "elapsed_ms": elapsed * 1000.0,
                "vi_cache_hit": hit,
                "abs_l1_vs_rf": q_abs,
                "rel_l1_vs_rf": q_rel,
                "png": str(png.relative_to(out_root)).replace("\\", "/"),
            }
            records.append(rec)
            print(
                f"  f{i + 1:02d} s{sid} r={r:.2f} {var['name']}: {elapsed * 1000:.1f} ms"
                + (f" hit={hit}" if "vicache" in mode_id else "")
                + (f" absL1={q_abs:.4f}" if q_abs is not None else "")
            )

        if not times:
            results["modes"][mode_id] = {
                "summary": {"label": mode_label, "error": "aborted", "mean_ms": None},
                "frames": records,
            }
            continue

        total = sum(times)
        # 分段：换场景帧 vs 段内帧
        miss_ms = [r["elapsed_ms"] for r in records if "vicache" in mode_id and not r["vi_cache_hit"]]
        hit_ms = [r["elapsed_ms"] for r in records if "vicache" in mode_id and r["vi_cache_hit"]]
        abs_vals = [r["abs_l1_vs_rf"] for r in records if r["abs_l1_vs_rf"] is not None]

        summary = {
            "label": mode_label,
            "total_sec": total,
            "mean_ms": total / len(times) * 1000.0,
            "fps": len(times) / total,
            "min_ms": min(times) * 1000.0,
            "max_ms": max(times) * 1000.0,
            "gpu_peak_mb": _gpu_peak_mb(device),
            "cache_hits": hits if "vicache" in mode_id else None,
            "cache_misses": misses if "vicache" in mode_id else None,
            "cache_hit_rate": (hits / max(hits + misses, 1)) if "vicache" in mode_id else None,
            "mean_miss_ms": float(np.mean(miss_ms)) if miss_ms else None,
            "mean_hit_ms": float(np.mean(hit_ms)) if hit_ms else None,
            "mean_abs_l1_vs_rf": float(np.mean(abs_vals)) if abs_vals else (0.0 if mode_id == "rf_baseline" else None),
            "aborted": aborted,
        }
        results["modes"][mode_id] = {"summary": summary, "frames": records}

    base_mean = results["modes"].get("rf_baseline", {}).get("summary", {}).get("mean_ms")
    cf_mean = results["modes"].get("rf_vicache", {}).get("summary", {}).get("mean_ms")
    for mid, blob in results["modes"].items():
        m = blob["summary"].get("mean_ms")
        blob["summary"]["speedup_vs_rf_baseline"] = (
            float(base_mean / m) if m and base_mean else None
        )
        blob["summary"]["speedup_vs_cacheformer"] = (
            float(cf_mean / m) if m and cf_mean else None
        )

    report_path = out_root / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # contact sheet: key frames × all modes
    try:
        from PIL import Image, ImageDraw, ImageFont

        key_frames = [
            i
            for i in range(args.num_frames)
            if i % args.scene_change_every == 0 or i % args.scene_change_every == args.scene_change_every - 1
        ]
        if not key_frames:
            key_frames = list(range(min(4, args.num_frames)))
        mode_ids = [m for m, _, _ in catalog if m in results["modes"] and results["modes"][m]["summary"].get("mean_ms")]
        sample = None
        for mid in mode_ids:
            p = frames_root / mid / f"frame_01_s0.png"
            if p.is_file():
                sample = iio.imread(p)
                break
        if sample is not None and mode_ids:
            h, w = sample.shape[:2]
            label_w, gap = 150, 4
            sheet_w = label_w + len(key_frames) * (w + gap) + gap
            sheet_h = len(mode_ids) * (h + gap) + gap + 28
            sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 28))
            draw = ImageDraw.Draw(sheet)
            try:
                font = ImageFont.truetype("arial.ttf", 11)
            except OSError:
                font = ImageFont.load_default()
            for row, mid in enumerate(mode_ids):
                s = results["modes"][mid]["summary"]
                y = gap + row * (h + gap)
                sp = s.get("speedup_vs_cacheformer")
                label = f"{mid}\n{s['mean_ms']:.0f}ms"
                if sp:
                    label += f"\n{sp:.2f}xCF"
                draw.multiline_text((6, y + 8), label, fill=(210, 210, 220), font=font, spacing=2)
                for col, fi in enumerate(key_frames):
                    sid = fi // args.scene_change_every
                    fp = frames_root / mid / f"frame_{fi + 1:02d}_s{sid}.png"
                    x = label_w + gap + col * (w + gap)
                    if fp.is_file():
                        sheet.paste(Image.fromarray(iio.imread(fp)), (x, y))
                    if row == 0:
                        draw.text((x + 4, sheet_h - 22), f"f{fi + 1} s{sid}", fill=(160, 160, 170), font=font)
            sheet_path = out_root / "contact_sheet_scene_changes.png"
            sheet.save(sheet_path)
            print(f"Contact sheet -> {sheet_path}")
    except Exception as e:
        print(f"[WARN] contact sheet failed: {e}")

    print("\n========== SUMMARY ==========")
    print(
        f"{'mode':18s} {'mean_ms':>9s} {'fps':>7s} {'vsRF':>8s} {'vsCF':>8s} "
        f"{'absL1':>8s} {'hit%':>6s}"
    )
    for mid, blob in results["modes"].items():
        s = blob["summary"]
        if s.get("mean_ms") is None:
            print(f"{mid:18s} {'ERR':>9s}")
            continue
        hit = f"{100 * s['cache_hit_rate']:.0f}%" if s.get("cache_hit_rate") is not None else "-"
        abs_s = f"{s['mean_abs_l1_vs_rf']:.4f}" if s.get("mean_abs_l1_vs_rf") is not None else "-"
        sp = f"{s['speedup_vs_rf_baseline']:.2f}x" if s.get("speedup_vs_rf_baseline") else "-"
        sc = f"{s['speedup_vs_cacheformer']:.2f}x" if s.get("speedup_vs_cacheformer") else "-"
        print(
            f"{mid:18s} {s['mean_ms']:9.1f} {s['fps']:7.2f} {sp:>8s} {sc:>8s} "
            f"{abs_s:>8s} {hit:>6s}"
        )
    print(f"\nReport -> {report_path}")
    print(f"Frames -> {frames_root}")


if __name__ == "__main__":
    main()

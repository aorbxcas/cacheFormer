#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用本机 Blender Cycles 渲染真实 GT，写入 gt_cache/<scene>/<res>/<view>_gt_full.exr

用法:
  python tools/c1_render_cycles_gt.py --scenes examples/cbox.json --blender "D:/download/Blender/blender.exe"
  python tools/c1_render_cycles_gt.py --scenes examples/cbox.json examples/veach-mis.json --spp 64 --views 4
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scene_processor"))

from dacite import Config, from_dict
from scene_config import SceneConfig
from scene_mesh import generate_scene_mesh


def _find_blender(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path(r"D:\download\Blender\blender.exe"),
            Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"),
            Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
            Path(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"),
        ]
    )
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "找不到 blender.exe。请安装 Blender 或用 --blender 指定路径。"
    )


def _orbit_cameras(base_cam: dict, n: int, include_base: bool = True) -> list[dict]:
    """绕场景 up 轴均匀取 n 个视角（含 0°），避免 base 与 orbit_+0 重复。"""
    pos0 = np.array(base_cam["position"], dtype=np.float64)
    look = np.array(base_cam["look_at"], dtype=np.float64)
    up = np.array(base_cam.get("up", [0.0, 0.0, 1.0]), dtype=np.float64)
    up = up / (np.linalg.norm(up) + 1e-8)
    if n <= 1:
        return [{**dict(base_cam), "name": "base"}]
    # 均匀覆盖 [-20, +20]，奇数视图时自然含 0°
    angles = np.linspace(-20.0, 20.0, n)
    out = []
    for deg in angles:
        a = math.radians(float(deg))
        rel = pos0 - look
        c, s = math.cos(a), math.sin(a)
        rel_rot = rel * c + np.cross(up, rel) * s + up * np.dot(up, rel) * (1.0 - c)
        pos = look + rel_rot
        name = "base" if abs(deg) < 1e-6 else f"orbit_{deg:+.0f}"
        out.append(
            {
                "position": pos.tolist(),
                "look_at": look.tolist(),
                "up": up.tolist(),
                "fov": float(base_cam["fov"]),
                "name": name,
            }
        )
    return out


def prepare_scene(scene_json: Path, work_root: Path) -> tuple[Path, Path, str]:
    with open(scene_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = from_dict(data_class=SceneConfig, data=raw, config=Config(check_types=True, strict=True))
    slug = scene_json.stem
    mesh_dir = work_root / slug
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = mesh_dir / "scene.obj"
    if not (mesh_dir / "split").is_dir() or not any((mesh_dir / "split").glob("*.obj")):
        print(f"[MESH] generating {slug} ...")
        generate_scene_mesh(cfg, str(mesh_path), str(scene_json.parent))
    else:
        print(f"[MESH] reuse {mesh_dir / 'split'}")
    return mesh_dir, scene_json, slug


def main():
    parser = argparse.ArgumentParser(description="Render Cycles GT for C1")
    parser.add_argument("--scenes", nargs="+", default=["examples/cbox.json"])
    parser.add_argument("--blender", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="gt_cache/cycles")
    parser.add_argument("--work_dir", type=str, default="tmp/c1_gt_meshes")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--spp", type=int, default=128)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--device", type=str, default="CUDA")
    parser.add_argument("--emission_scale", type=float, default=0.001, help="Cycles 灯光强度缩放，对齐 RF 量级")
    args = parser.parse_args()

    blender = _find_blender(args.blender)
    print(f"Blender: {blender}")
    worker = ROOT / "tools" / "c1_blender_worker.py"
    work_root = Path(args.work_dir)
    out_root = Path(args.output_root)

    for scene_arg in args.scenes:
        scene_json = Path(scene_arg)
        if not scene_json.is_file():
            print(f"[SKIP] missing {scene_json}")
            continue
        mesh_dir, scene_json, slug = prepare_scene(scene_json, work_root)

        with open(scene_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        base_cam = raw["cameras"][0]
        cams = _orbit_cameras(base_cam, args.views)
        cam_path = mesh_dir / "cameras.json"
        with open(cam_path, "w", encoding="utf-8") as f:
            json.dump(cams, f, indent=2)

        out_dir = out_root / slug / str(args.resolution)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(blender),
            "--background",
            "--python",
            str(worker),
            "--",
            "--scene_json",
            str(scene_json.resolve()),
            "--mesh_dir",
            str(mesh_dir.resolve()),
            "--cameras_json",
            str(cam_path.resolve()),
            "--output_dir",
            str(out_dir.resolve()),
            "--resolution",
            str(args.resolution),
            "--spp",
            str(args.spp),
            "--device",
            args.device,
            "--emission_scale",
            str(args.emission_scale),
        ]
        print("[CMD]", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"[FAIL] blender exit {r.returncode} for {slug}")
            continue

        # index
        index = {
            "scene": slug,
            "resolution": args.resolution,
            "spp": args.spp,
            "views": len(cams),
            "cameras": cams,
            "gt_dir": str(out_dir),
        }
        with open(out_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        print(f"[OK] {slug}: {len(list(out_dir.glob('*_gt_full.exr')))} EXR -> {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
在 Blender 内运行的 Cycles GT worker（无 bpy_helper 依赖）。

调用方式（由 c1_render_cycles_gt.py 启动）:
  blender --background --python tools/c1_blender_worker.py -- \\
      --scene_json examples/cbox.json \\
      --mesh_dir tmp/c1_gt_meshes/cbox \\
      --cameras_json tmp/c1_gt_meshes/cbox/cameras.json \\
      --output_dir gt_cache/cycles/cbox/256 \\
      --resolution 256 --spp 128
"""

from __future__ import annotations

import json
import math
import os
import sys


def _parse_args(argv):
    # blender 会把脚本参数放在 "--" 之后
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--scene_json", required=True)
    p.add_argument("--mesh_dir", required=True, help="含 split/<obj>.obj 的目录上级（mesh.obj 同级）")
    p.add_argument("--cameras_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--spp", type=int, default=128)
    p.add_argument("--device", type=str, default="CUDA", choices=["CUDA", "OPTIX", "HIP", "METAL", "CPU"])
    p.add_argument(
        "--emission_scale",
        type=float,
        default=1.0,
        help="乘到 emissive 强度上；若与 RF 差数量级可试 1e-3~1e-2",
    )
    return p.parse_args(argv)


def _look_at_c2w(pos, target, up):
    """Blender camera: local -Z looks forward, +Y is up. Matches RF/H5 c2w."""
    import numpy as np

    pos = np.asarray(pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - pos
    f = f / (np.linalg.norm(f) + 1e-8)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-8:
        # forward || up：换一个兜底 up
        alt = np.array([0.0, 1.0, 0.0]) if abs(up[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
        s = np.cross(f, alt)
    s = s / (np.linalg.norm(s) + 1e-8)
    u = np.cross(s, f)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = s
    c2w[:3, 1] = u
    c2w[:3, 2] = -f
    c2w[:3, 3] = pos
    return c2w


def _reset_scene(bpy):
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _principled_material(bpy, name, diffuse, specular, roughness, emissive):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    em = [float(x) for x in (emissive or [0.0, 0.0, 0.0])[:3]]
    em_max = max(em) if em else 0.0
    if em_max > 0:
        # 与 to_blend / bpy_helper 一致：纯 Emission，避免 Principled 发射被衰减
        emission = nodes.new("ShaderNodeEmission")
        col = tuple(c / em_max for c in em)
        emission.inputs["Color"].default_value = (*col, 1.0)
        emission.inputs["Strength"].default_value = em_max
        links.new(emission.outputs["Emission"], out.inputs["Surface"])
        return mat
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Base Color"].default_value = (*[float(x) for x in diffuse[:3]], 1.0)
    bsdf.inputs["Roughness"].default_value = float(roughness)
    spec = float(specular[0]) if specular else 0.0
    for key in ("Specular IOR Level", "Specular"):
        if key in bsdf.inputs:
            try:
                bsdf.inputs[key].default_value = max(spec, 0.0)
            except Exception:
                pass
            break
    return mat


def _import_obj(bpy, path):
    """导入已烘焙到世界坐标的 split OBJ；保持 RF 的 Y-forward / Z-up，禁止 Blender 默认轴向重映射。"""
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(
            filepath=path,
            forward_axis="Y",
            up_axis="Z",
            use_split_objects=False,
            use_split_groups=False,
        )
    else:
        bpy.ops.import_scene.obj(
            filepath=path,
            axis_forward="Y",
            axis_up="Z",
            use_split_objects=False,
            use_split_groups=False,
        )


def main():
    args = _parse_args(sys.argv)
    import bpy
    import numpy as np

    with open(args.scene_json, "r", encoding="utf-8") as f:
        scene = json.load(f)
    with open(args.cameras_json, "r", encoding="utf-8") as f:
        cameras = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    split_dir = os.path.join(args.mesh_dir, "split")
    if not os.path.isdir(split_dir):
        # mesh_dir 可能直接是含 split 的路径的父目录；也兼容 mesh_dir 本身含 split
        alt = os.path.join(os.path.dirname(args.mesh_dir), "split")
        raise FileNotFoundError(f"找不到 split 目录: {split_dir}")

    _reset_scene(bpy)

    # World: no ambient
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0, 0, 0, 1)
    bg.inputs[1].default_value = 0.0

    for obj_key, obj_cfg in scene["objects"].items():
        obj_path = os.path.join(split_dir, f"{obj_key}.obj")
        if not os.path.isfile(obj_path):
            print(f"[WARN] missing mesh {obj_path}, skip")
            continue
        before = set(bpy.data.objects.keys())
        _import_obj(bpy, obj_path)
        after = [n for n in bpy.data.objects.keys() if n not in before]
        mat_cfg = obj_cfg["material"]
        mat = _principled_material(
            bpy,
            obj_key,
            mat_cfg.get("diffuse", [0.8, 0.8, 0.8]),
            mat_cfg.get("specular", [0.0, 0.0, 0.0]),
            mat_cfg.get("roughness", 0.5),
            [e * float(args.emission_scale) for e in mat_cfg.get("emissive", [0.0, 0.0, 0.0])],
        )
        for name in after:
            obj = bpy.data.objects[name]
            if obj.type != "MESH":
                continue
            obj.name = obj_key
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            if not mat_cfg.get("smooth_shading", True):
                for poly in obj.data.polygons:
                    poly.use_smooth = False
            else:
                for poly in obj.data.polygons:
                    poly.use_smooth = True

    # Cycles setup
    scene_b = bpy.context.scene
    scene_b.render.engine = "CYCLES"
    scene_b.cycles.samples = int(args.spp)
    scene_b.render.resolution_x = int(args.resolution)
    scene_b.render.resolution_y = int(args.resolution)
    scene_b.render.film_transparent = False
    scene_b.render.image_settings.file_format = "OPEN_EXR"
    scene_b.render.image_settings.color_mode = "RGB"
    scene_b.render.image_settings.color_depth = "32"
    # 线性 HDR，不做 Filmic 压缩
    try:
        scene_b.view_settings.view_transform = "Raw"
        scene_b.view_settings.look = "None"
    except Exception:
        pass
    scene_b.cycles.device = "CPU" if args.device == "CPU" else "GPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = args.device if args.device != "CPU" else "NONE"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
    except Exception as e:
        print(f"[WARN] GPU setup failed ({e}), fallback CPU")
        scene_b.cycles.device = "CPU"

    # Render each camera
    for i, cam in enumerate(cameras):
        c2w = _look_at_c2w(cam["position"], cam["look_at"], cam["up"])
        fov_deg = float(cam["fov"])
        # Blender camera sensor
        cam_data = bpy.data.cameras.new(name=f"Cam_{i}")
        # fov is vertical? RenderFormer JSON uses degrees; Blender lens via angle
        cam_data.angle = math.radians(fov_deg)
        cam_obj = bpy.data.objects.new(f"Cam_{i}", cam_data)
        bpy.context.collection.objects.link(cam_obj)
        from mathutils import Matrix

        cam_obj.matrix_world = Matrix(c2w.tolist())
        scene_b.camera = cam_obj

        out_path = os.path.join(args.output_dir, f"{i:04d}_gt_full.exr")
        scene_b.render.filepath = os.path.abspath(out_path)
        # 快速自检：首帧打印场景包围盒与相机
        if i == 0:
            from mathutils import Vector

            coords = []
            n_mesh = 0
            for ob in bpy.data.objects:
                if ob.type != "MESH":
                    continue
                n_mesh += 1
                for corner in ob.bound_box:
                    coords.append(ob.matrix_world @ Vector(corner))
            if coords:
                import numpy as _np

                pts = _np.array([[c.x, c.y, c.z] for c in coords], dtype=_np.float64)
                print(f"[SCENE] bbox min={pts.min(0)} max={pts.max(0)} n_mesh={n_mesh}")
            print(f"[CAM] pos={cam['position']} look={cam['look_at']} fov={fov_deg}")
        print(f"[RENDER] view {i} -> {out_path}")
        bpy.ops.render.render(write_still=True)

        # also save c2w/fov sidecar
        meta = {
            "view_id": i,
            "position": cam["position"],
            "look_at": cam["look_at"],
            "up": cam["up"],
            "fov": fov_deg,
            "c2w": c2w.tolist(),
        }
        with open(os.path.join(args.output_dir, f"{i:04d}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # remove camera object for next
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data)

    print("[DONE] all views rendered")


if __name__ == "__main__":
    main()

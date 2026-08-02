"""Hunyuan3D-2.1 paint adapter (Stage T). Generalizes pbr_compare/run_hunyuan.py to a single
pair or a --pairs-file. Runs in the `hunyuan3d` env with cwd = Hunyuan3D-2.1/hy3dpaint/
(relative imports). Loads the pipeline once, iterates pairs.

Seed is hardcoded to 0 inside utils/multiview_utils.py, so we record seed:0 regardless of
request (PRD risk 6). Writes the obj/mtl/albedo set + textured_mesh.glb.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Launched with cwd = Hunyuan3D-2.1/hy3dpaint/; put it on sys.path for its relative imports.
sys.path.insert(0, os.getcwd())

from _adapter_common import add_common_args, load_pairs, finalize_output, emit_result


def main() -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--max-num-view", type=int, default=6)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--no-remesh", action="store_true",
                    help="disable Hunyuan's default ~40k-face remesh (keep original topology)")
    args = ap.parse_args()

    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    conf = Hunyuan3DPaintConfig(max_num_view=args.max_num_view, resolution=args.resolution)
    pipe = Hunyuan3DPaintPipeline(conf)

    for pair in load_pairs(args):
        out_dir = pair["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        out_obj = os.path.join(out_dir, "textured_mesh.obj")
        glb_path = os.path.join(out_dir, "textured_mesh.glb")
        use_remesh = not args.no_remesh
        try:
            pipe(mesh_path=pair["mesh"], image_path=pair["image"],
                 output_mesh_path=out_obj, use_remesh=use_remesh, save_glb=True)
            finalize_output(glb_path, pair.get("camera_json"))
            # Hunyuan hardcodes seed 0 in utils/multiview_utils.py.
            emit_result({"backend": "hunyuan", "glb_path": glb_path, "seed": 0, "ok": True})
        except Exception as e:  # noqa: BLE001
            emit_result({"backend": "hunyuan", "glb_path": None, "ok": False, "error": str(e)})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""TRELLIS.2 texturing adapter (Stage T). Generalizes pbr_compare/run_trellis2.py to a
single pair or a --pairs-file. Runs in the `trellis2` env with cwd = TRELLIS.2/ (so
texturing_pipeline.json resolves). Loads the pipeline once, iterates pairs.

The pipeline re-normalizes the mesh internally (preprocess_mesh), so we pass the mesh as-is
(original, or a pre-reposed original); RGBA input makes preprocess_image use our alpha cutout.
After texturing, restore original orientation if Stage R re-posed (finalize_output).
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# We are launched with cwd = TRELLIS.2/; put it on sys.path so `import trellis2` resolves
# (running `python /abs/script.py` only adds the script dir, not cwd).
sys.path.insert(0, os.getcwd())

import trimesh
from PIL import Image

from _adapter_common import add_common_args, emit_result, finalize_output, load_pairs


def _load_mesh(path: str) -> trimesh.Trimesh:
    m = trimesh.load(path)
    return m.to_mesh() if isinstance(m, trimesh.Scene) else m


def main() -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--resolution", type=int, default=1024, choices=[512, 1024])
    ap.add_argument("--texture-size", type=int, default=2048)
    args = ap.parse_args()

    from trellis2.pipelines import Trellis2TexturingPipeline

    pipe = Trellis2TexturingPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
    pipe.cuda()

    for pair in load_pairs(args):
        out_dir = pair["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        glb_path = os.path.join(out_dir, "textured.glb")
        try:
            mesh = _load_mesh(pair["mesh"])
            image = Image.open(pair["image"])  # RGBA -> preprocess_image uses our alpha
            out = pipe.run(mesh, image, seed=int(pair.get("seed", args.seed)),
                           resolution=args.resolution, texture_size=args.texture_size,
                           preprocess_image=True)
            out.export(glb_path, extension_webp=True)
            finalize_output(glb_path, pair.get("camera_json"))
            emit_result({"backend": "trellis2", "glb_path": glb_path, "ok": True})
        except Exception as e:  # noqa: BLE001
            emit_result({"backend": "trellis2", "glb_path": None, "ok": False, "error": str(e)})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

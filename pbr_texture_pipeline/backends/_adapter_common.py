"""Shared helpers for backend adapter scripts (Stage T).

Adapter scripts run INSIDE a backend's conda env (trellis2) with cwd = the backend repo.
They must not import the heavy pbr_texture_pipeline.rendering module. Re-pose un-rotation
here is therefore trimesh+numpy only.

CLI contract (single pair or --pairs-file):
  --mesh <path> --image <rgba> --out-dir <dir> --seed N --camera-json <path>
Emits one machine-readable result line per pair:  [PBR_RESULT] {json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

RESULT_MARKER = "[PBR_RESULT]"


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--mesh", help="mesh to texture (already re-posed if repose is active)")
    ap.add_argument("--image", help="RGBA reference (chosen_rgba.png)")
    ap.add_argument("--out-dir", help="output dir for this pair")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--camera-json", default=None, help="control/camera.json (for repose R)")
    ap.add_argument("--pairs-file", default=None,
                    help="JSON list of {mesh,image,out_dir,seed,camera_json} to load backend once")


def load_pairs(args: argparse.Namespace) -> list[dict]:
    """One pair from CLI args, or many from --pairs-file (backend loads once)."""
    if args.pairs_file:
        with open(args.pairs_file) as f:
            pairs = json.load(f)
        return pairs if isinstance(pairs, list) else [pairs]
    return [{
        "mesh": args.mesh, "image": args.image, "out_dir": args.out_dir,
        "seed": args.seed, "camera_json": args.camera_json,
    }]


def norm_params(vertices) -> tuple:
    """TRELLIS preprocess_mesh normalization params (numpy-only; adapters avoid
    importing pbr_texture_pipeline): v' = (v - center) * scale. TRELLIS swaps Y/Z then undoes it on
    output, so inverting center+scale maps its output back to the input (link) frame."""
    v = np.asarray(vertices)
    vmin, vmax = v.min(axis=0), v.max(axis=0)
    center = (vmin + vmax) / 2.0
    scale = 0.99999 / (vmax - vmin).max()
    return center, float(scale)


def denormalize_mesh(mesh, center, scale):
    """Invert norm_params on an output mesh in place (articulated per-group texturing)."""
    mesh.vertices = mesh.vertices / scale + center
    return mesh


def unrepose_glb(glb_path: str, R_mat) -> None:
    """Apply R^-1 to a textured GLB's vertices in place (lossless; textures are UV-space)."""
    Rinv = np.linalg.inv(np.asarray(R_mat, dtype=np.float64))
    scene = trimesh.load(glb_path)
    geoms = scene.geometry.values() if isinstance(scene, trimesh.Scene) else [scene]
    for g in geoms:
        g.vertices = np.asarray(g.vertices) @ Rinv.T
    scene.export(glb_path)


def finalize_output(glb_path: str, camera_json: str | None) -> None:
    """If the mesh was re-posed in Stage R, restore original orientation on the output GLB."""
    if not camera_json or not Path(camera_json).exists():
        return
    with open(camera_json) as f:
        cam = json.load(f)
    if cam.get("repose_applied") and cam.get("R") is not None:
        unrepose_glb(glb_path, cam["R"])


def emit_result(result: dict) -> None:
    print(f"{RESULT_MARKER} {json.dumps(result)}", flush=True)

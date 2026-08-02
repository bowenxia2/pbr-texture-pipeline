"""Gate 1: verify the canonical-front camera and depth polarity conventions (PRD section 11).

This is the single most bug-prone claim in the project (PRD risk 1) and must be
confirmed empirically on day one before any Phase 1 render code is trusted.

What it does on pbr_compare/inputs/hunyuan_case1 (which ships mesh.glb + the reference
image.png shot from the object's true front):
  1. Load mesh.glb, apply the TRELLIS.2 preprocess_mesh normalization (copied verbatim).
  2. Render the canonical-front (yaw=pi, pitch=0, r=2, fov=40) shaded / depth / normal.
  3. Also render the OTHER three cardinal yaws so we can see which one actually matches
     image.png (falsifies the yaw=pi claim if wrong).
  4. Write ControlNet-convention depth (normalized inverse depth, near=white, far=black,
     bg black) and print polarity stats.
Outputs land in <out_dir> for visual comparison against image.png.

Run:
  cd pbr-texture-pipeline
  conda run -n trellis2 python scripts/verify_conventions.py \
      --out jobs/_gate1_conventions
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

# Make TRELLIS.2 importable, then pin the shared HF cache.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pbr_texture_pipeline.config import load_config  # noqa: E402

_CFG = load_config()
for _k, _v in _CFG.hf_env().items():
    os.environ.setdefault(_k, _v)
sys.path.insert(0, str(_CFG.repo("trellis2")))

from trellis2.renderers import MeshRenderer  # noqa: E402
from trellis2.representations.mesh import Mesh  # noqa: E402
from trellis2.utils.render_utils import (  # noqa: E402
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
)


# --- copied verbatim from Trellis2TexturingPipeline.preprocess_mesh (do not import) ---
def preprocess_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    vertices = mesh.vertices.copy()
    vertices_min = vertices.min(axis=0)
    vertices_max = vertices.max(axis=0)
    center = (vertices_min + vertices_max) / 2
    scale = 0.99999 / (vertices_max - vertices_min).max()
    vertices = (vertices - center) * scale
    tmp = vertices[:, 1].copy()
    vertices[:, 1] = -vertices[:, 2]
    vertices[:, 2] = tmp
    assert np.all(vertices >= -0.5) and np.all(vertices <= 0.5), "vertices out of range"
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)


def load_norm_mesh(mesh_path: str) -> Mesh:
    m = trimesh.load(mesh_path, force="scene")
    m = m.to_mesh() if isinstance(m, trimesh.Scene) else m
    m = preprocess_mesh(m)
    verts = torch.tensor(m.vertices, dtype=torch.float32, device="cuda")
    faces = torch.tensor(m.faces, dtype=torch.int32, device="cuda")
    return Mesh(verts, faces)


def render(mesh: Mesh, yaw: float, pitch: float, resolution: int, ssaa: int):
    r, fov = _CFG.get("render.r"), _CFG.get("render.fov_deg")
    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    renderer = MeshRenderer()
    renderer.rendering_options.resolution = resolution
    renderer.rendering_options.near = 1.0
    renderer.rendering_options.far = 10.0
    renderer.rendering_options.ssaa = ssaa
    out = renderer.render(mesh, extr, intr, return_types=["mask", "depth", "normal"])
    mask = out["mask"].detach().cpu().numpy()          # [H,W] 0/1
    depth = out["depth"].detach().cpu().numpy()         # [H,W] camera-space z
    normal = out["normal"].detach().cpu().numpy()       # [3,H,W] in (n+1)/2
    return mask, depth, normal


def headlight_shade(normal_enc: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Clay render: |n . view| grayscale. Normals are camera-space, encoded (n+1)/2."""
    n = normal_enc * 2.0 - 1.0                 # decode to [-1,1], shape [3,H,W]
    nz = n[2]                                   # camera-space z component
    shade = np.clip(np.abs(nz), 0, 1)
    shade = (shade * mask * 255).astype(np.uint8)
    return Image.fromarray(shade, mode="L")


def depth_vis_controlnet(depth: np.ndarray, mask: np.ndarray):
    """Normalized inverse depth, near=white far=black, bg black (ControlNet/MiDaS convention)."""
    m = mask > 0.5
    if not m.any():
        return Image.fromarray(np.zeros_like(depth, dtype=np.uint8), mode="L"), (0.0, 0.0)
    z = depth[m]
    near_hit, far_hit = float(z.min()), float(z.max())  # near=closest z, far=farthest z
    d_vis = np.zeros_like(depth, dtype=np.float32)
    denom = max(far_hit - near_hit, 1e-6)
    d_vis[m] = (far_hit - depth[m]) / denom             # closest -> 1 (white), farthest -> 0 (black)
    img = (np.clip(d_vis, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img, mode="L"), (near_hit, far_hit)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case",
        default=str(Path(_CFG.path).parent.parent / "pbr_compare/inputs/hunyuan_case1"),
        help="dir containing mesh.glb + image.png",
    )
    ap.add_argument("--out", default="jobs/_gate1_conventions")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--ssaa", type=int, default=2)
    args = ap.parse_args()

    case = Path(args.case)
    mesh_path = case / "mesh.glb"
    ref_path = case / "image.png"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gate1] case: {case}")
    print(f"[gate1] mesh: {mesh_path}")
    mesh = load_norm_mesh(str(mesh_path))
    print(f"[gate1] normalized mesh: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces")

    # Copy reference for side-by-side.
    if ref_path.exists():
        Image.open(ref_path).convert("RGB").save(out_dir / "reference_image.png")

    # Render the 4 cardinal yaws so we can SEE which matches image.png.
    cardinals = {
        "yaw0_front-candidate": 0.0,
        "yaw_half_pi": math.pi / 2,
        "yaw_pi_CANONICAL": math.pi,
        "yaw_3half_pi": 3 * math.pi / 2,
    }
    for name, yaw in cardinals.items():
        mask, depth, normal = render(mesh, yaw, 0.0, args.resolution, args.ssaa)
        cov = float((mask > 0.5).mean())
        headlight_shade(normal, mask).save(out_dir / f"shaded_{name}.png")
        print(f"[gate1] {name:26s} yaw={yaw:.4f} mask_coverage={cov:.3f}")

    # Full control-map set at the canonical front.
    mask, depth, normal = render(mesh, math.pi, 0.0, args.resolution, args.ssaa)
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(out_dir / "mask.png")
    n_img = (np.clip(normal.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(n_img, mode="RGB").save(out_dir / "normal.png")
    dvis, (near_hit, far_hit) = depth_vis_controlnet(depth, mask)
    dvis.save(out_dir / "depth_controlnet.png")

    print(f"[gate1] canonical depth: near_hit(z_min)={near_hit:.4f} far_hit(z_max)={far_hit:.4f}")
    print("[gate1] depth polarity: closest surface -> WHITE(255), farthest -> BLACK(0), bg black.")
    print(f"[gate1] outputs -> {out_dir.resolve()}")
    print("[gate1] NEXT: visually confirm shaded_yaw_pi_CANONICAL.png matches reference_image.png's viewpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

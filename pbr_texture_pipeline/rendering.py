"""Stage R: mesh normalization, cameras, VLM contact sheet, and control maps (PRD section 3).

All rendering uses TRELLIS.2's nvdiffrast `MeshRenderer` (no Blender, no pyrender). This
module MUST run in the `trellis2` env (renderer stack: cumesh/o_voxel/nvdiffrast/flex_gemm).

Conventions are the ones empirically confirmed at Gate 1 (see README "Conventions (verified)"):
  - canonical front camera = yaw=pi, pitch=0, r=2, fov=40
  - depth = normalized inverse depth, near=white far=black, bg black
"""
from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from pbr_texture_pipeline.config import load_config

_CFG = load_config()

# TRELLIS.2 is not pip-installed; put its repo on sys.path from config (never hardcode).
_TRELLIS_REPO = str(_CFG.repo("trellis2"))
if _TRELLIS_REPO not in sys.path:
    sys.path.insert(0, _TRELLIS_REPO)

# TRELLIS.2 renderer stack (env-gated import; needs the `trellis2` conda env).
from trellis2.renderers import MeshRenderer  # noqa: E402
from trellis2.representations.mesh import Mesh  # noqa: E402
from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics  # noqa: E402
CANONICAL_YAW = float(_CFG.get("render.canonical_yaw"))   # pi
CANONICAL_PITCH = float(_CFG.get("render.canonical_pitch"))
R = float(_CFG.get("render.r"))
FOV_DEG = float(_CFG.get("render.fov_deg"))

_RENDER_NEAR = 1.0
_RENDER_FAR = 10.0

# One renderer/context reused across calls (nvdiffrast context is expensive to build).
_RENDERER: Optional[MeshRenderer] = None


def _renderer(resolution: int, ssaa: int) -> MeshRenderer:
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = MeshRenderer()
    _RENDERER.rendering_options.resolution = resolution
    _RENDERER.rendering_options.near = _RENDER_NEAR
    _RENDERER.rendering_options.far = _RENDER_FAR
    _RENDERER.rendering_options.ssaa = ssaa
    return _RENDERER


# --- normalization (Task 1.2) ------------------------------------------------
# up="y" is copied verbatim from Trellis2TexturingPipeline.preprocess_mesh; do NOT import
# the full pipeline just for this (PRD section 3). Vertices -> [-0.5,0.5], axis swap
# y'=-z, z'=y (glTF Y-up -> internal Z-up).
# up="z" is for geometry whose source frame is already Z-up (the PartNet-Mobility URDF
# world frame): center+scale only, no axis swap. Feeding Z-up geometry through the Y-up
# swap tips the object onto its side, which turns every yaw orbit into a tumble.
def preprocess_mesh(mesh: trimesh.Trimesh, up: str = "y") -> trimesh.Trimesh:
    vertices = mesh.vertices.copy()
    vertices_min = vertices.min(axis=0)
    vertices_max = vertices.max(axis=0)
    center = (vertices_min + vertices_max) / 2
    scale = 0.99999 / (vertices_max - vertices_min).max()
    vertices = (vertices - center) * scale
    if up == "y":
        tmp = vertices[:, 1].copy()
        vertices[:, 1] = -vertices[:, 2]
        vertices[:, 2] = tmp
    elif up != "z":
        raise ValueError(f"unknown up axis {up!r} (expected 'y' or 'z')")
    assert np.all(vertices >= -0.5) and np.all(vertices <= 0.5), "vertices out of range"
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)


def export_yup(mesh: trimesh.Trimesh, path) -> None:
    """Export a Z-up internal-frame mesh to a Y-up glTF file.

    The internal frame (after preprocess_mesh with up="z") is Z-up; glTF convention is Y-up.
    Applying (x, y, z) -> (x, z, -y) before saving means standard viewers show the object
    upright and TRELLIS.2's preprocess_mesh (which assumes Y-up input) correctly round-trips
    back to Z-up internal.
    """
    v = np.asarray(mesh.vertices)
    yup = trimesh.Trimesh(
        vertices=np.column_stack([v[:, 0], v[:, 2], -v[:, 1]]),
        faces=mesh.faces, process=False)
    yup.export(path)


def _flatten(mesh) -> trimesh.Trimesh:
    """Load result -> single Trimesh (pattern from pbr_compare/run_trellis2.py)."""
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    return mesh


def load_and_normalize(mesh_path: str, jobdir) -> trimesh.Trimesh:
    """Save input/original.<ext>, normalize, save input/mesh_norm.glb; return normalized mesh."""
    src = Path(mesh_path)
    orig = jobdir.original(src.suffix)
    orig.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != orig.resolve():
        shutil.copyfile(src, orig)

    raw = _flatten(trimesh.load(mesh_path))
    norm = preprocess_mesh(raw)
    norm.export(jobdir.mesh_norm())  # single Trimesh, no scene
    return norm


def to_mesh_repr(mesh: trimesh.Trimesh) -> Mesh:
    """trimesh -> TRELLIS.2 Mesh representation on cuda."""
    return to_mesh_repr_arrays(mesh.vertices, mesh.faces)


def to_mesh_repr_arrays(vertices: np.ndarray, faces: np.ndarray) -> Mesh:
    """Bare (vertices, faces) arrays -> TRELLIS.2 Mesh on cuda (articulated group renders)."""
    verts = torch.tensor(np.asarray(vertices), dtype=torch.float32, device="cuda")
    f = torch.tensor(np.asarray(faces), dtype=torch.int32, device="cuda")
    return Mesh(verts, f)


def to_internal_frame(vertices: np.ndarray, up: str = "y") -> np.ndarray:
    """The `preprocess_mesh` transform (center+scale to [-0.5,0.5], axis swap y'=-z, z'=y
    when up="y") applied to a bare vertex array, without the range assert or trimesh
    wrapping. up="z" (already-Z-up source frames: the URDF world frame) centers and scales
    without the swap.

    Used by Stage E to bring the backend's textured output (assembled in the Z-up URDF world
    frame) back into the internal Z-up frame the condition camera (yaw=pi) is defined in.
    """
    v = np.asarray(vertices, dtype=np.float64).copy()
    vmin, vmax = v.min(axis=0), v.max(axis=0)
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    v = (v - center) * scale
    if up == "y":
        tmp = v[:, 1].copy()
        v[:, 1] = -v[:, 2]
        v[:, 2] = tmp
    elif up != "z":
        raise ValueError(f"unknown up axis {up!r} (expected 'y' or 'z')")
    return v


def colored_mesh_repr(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> Mesh:
    """trimesh geometry + per-vertex RGB in [0,1] -> TRELLIS.2 Mesh with vertex_attrs (cuda).

    The MeshRenderer "attr" path interpolates `vertex_attrs` per pixel, so this renders the
    output's appearance without reconstructing full PBR materials (adequate for the semantic /
    transfer eval metrics and preview renders).
    """
    verts = torch.tensor(np.asarray(vertices), dtype=torch.float32, device="cuda")
    f = torch.tensor(np.asarray(faces), dtype=torch.int32, device="cuda")
    attrs = torch.tensor(np.asarray(colors), dtype=torch.float32, device="cuda")
    return Mesh(verts, f, attrs)


def render_appearance(mesh_repr: Mesh, yaw: float, pitch: float,
                      resolution: int, ssaa: int) -> dict:
    """Render vertex-colored appearance + mask from (yaw,pitch). rgb uint8 [H,W,3], mask [H,W]."""
    extr, intr = get_camera(yaw, pitch)
    out = _renderer(resolution, ssaa).render(mesh_repr, extr, intr, return_types=["mask", "attr"])
    mask = out["mask"].detach().cpu().numpy()
    rgb = out["attr"].detach().cpu().numpy()          # [3,H,W] in [0,1]
    rgb = np.clip(rgb.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    return {"rgb": rgb, "mask": mask}


# --- camera (Task 1.2) -------------------------------------------------------
def get_camera(yaw: float, pitch: float, r: float = R, fov: float = FOV_DEG):
    """(yaw,pitch,r,fov) -> (extrinsics, intrinsics), exactly as render_utils defines."""
    return yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)


# --- render primitive --------------------------------------------------------
def render_view(mesh_repr: Mesh, yaw: float, pitch: float, resolution: int, ssaa: int,
                return_types=("mask", "depth", "normal")) -> dict:
    """Render one view; returns numpy arrays: mask [H,W], depth [H,W], normal [3,H,W]."""
    extr, intr = get_camera(yaw, pitch)
    out = _renderer(resolution, ssaa).render(mesh_repr, extr, intr, return_types=list(return_types))
    res = {}
    if "mask" in return_types:
        res["mask"] = out["mask"].detach().cpu().numpy()
    if "depth" in return_types:
        res["depth"] = out["depth"].detach().cpu().numpy()
    if "normal" in return_types:
        res["normal"] = out["normal"].detach().cpu().numpy()
    return res


def headlight_shade(normal_enc: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Clay render: |n . view| grayscale (normals camera-space, encoded (n+1)/2). uint8 [H,W]."""
    n = normal_enc * 2.0 - 1.0
    shade = np.clip(np.abs(n[2]), 0.0, 1.0)
    return (shade * (mask > 0.5) * 255).astype(np.uint8)


def depth_to_controlnet(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalized inverse depth (near=white, far=black), bg black - the Gate-1 confirmed polarity."""
    m = mask > 0.5
    out = np.zeros_like(depth, dtype=np.float32)
    if m.any():
        z = depth[m]
        near_hit, far_hit = float(z.min()), float(z.max())
        denom = max(far_hit - near_hit, 1e-6)
        out[m] = (far_hit - depth[m]) / denom
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# --- contact sheet + re-pose (Task 1.3) --------------------------------------
CONTACT_N = 8                         # 8 azimuth panels, front + k*45deg
CONTACT_STEP = math.pi / 4            # 45 deg
CONTACT_PITCH = math.radians(float(_CFG.get("render.contact_pitch_deg")))  # 15 deg


def contact_yaw(k: int) -> float:
    """Camera yaw for contact panel k = canonical front + k*45deg."""
    return CANONICAL_YAW + k * CONTACT_STEP


def render_contact_sheet(jobdir, mesh_repr: Mesh, resolution: int = 512, ssaa: int = 1,
                         also_depth: bool = False,
                         depth_resolution: Optional[int] = None) -> dict:
    """Render 8 shaded azimuth views and assemble the 2x4 contact sheet.

    When also_depth is True, depth buffers and camera matrices are captured at
    depth_resolution (defaults to resolution) for each of the 8 viewpoints.
    Returned dict always has 'path'; with also_depth it also has 'depths' [8,H,W],
    'extrinsics' [8,4,4], 'intrinsics' [8,3,3].
    """
    render_res = depth_resolution if (also_depth and depth_resolution) else resolution
    rt = ("mask", "depth", "normal") if also_depth else ("mask", "normal")
    panels = []
    depths, extrs, intrs = [], [], []
    for k in range(CONTACT_N):
        yaw = contact_yaw(k)
        r = render_view(mesh_repr, yaw, CONTACT_PITCH, render_res, ssaa, return_types=rt)
        shade = headlight_shade(r["normal"], r["mask"])
        if render_res != resolution:
            shade = np.array(Image.fromarray(shade, mode="L").resize(
                (resolution, resolution), Image.LANCZOS))
        Image.fromarray(shade, mode="L").save(jobdir.view(k))
        panels.append(shade)
        if also_depth:
            depths.append(r["depth"])
            extr, intr = get_camera(yaw, CONTACT_PITCH)
            extrs.append(extr.detach().cpu().numpy())
            intrs.append(intr.detach().cpu().numpy())

    # 2x4 grid.
    rows = [np.concatenate(panels[r * 4:(r + 1) * 4], axis=1) for r in range(2)]
    sheet = np.concatenate(rows, axis=0)
    Image.fromarray(sheet, mode="L").convert("RGB").save(jobdir.contact_sheet())

    result: dict = {"path": jobdir.contact_sheet()}
    if also_depth:
        result["depths"] = np.stack(depths)
        result["extrinsics"] = np.stack(extrs)
        result["intrinsics"] = np.stack(intrs)
    return result


def repose_matrix(front_index: int) -> np.ndarray:
    """Rotation about +Z that maps chosen-front panel `front_index` to the canonical front.

    Panel k is viewed at camera yaw = pi + k*45deg. Rotating the mesh by Rz(a_j) with
    a_j = j*45deg makes that panel's face point at the canonical (yaw=pi) camera.
    Returns a 3x3 rotation matrix (identity when front_index == 0).
    """
    a = front_index * CONTACT_STEP
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def apply_repose(mesh: trimesh.Trimesh, R_mat: np.ndarray) -> trimesh.Trimesh:
    """Rotate a normalized mesh about +Z (re-pose so chosen front becomes canonical)."""
    v = np.asarray(mesh.vertices) @ R_mat.T
    return trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)


def unrepose_glb(glb_path: str, R_mat: np.ndarray) -> None:
    """Apply R^-1 to a textured output GLB's vertices in place (lossless: textures are UV-space)."""
    Rinv = np.linalg.inv(R_mat)
    scene = trimesh.load(glb_path)
    meshes = scene.geometry.values() if isinstance(scene, trimesh.Scene) else [scene]
    for g in meshes:
        g.vertices = np.asarray(g.vertices) @ Rinv.T
    scene.export(glb_path)


# --- control maps (Task 1.4) -------------------------------------------------
def render_control_maps(
    jobdir,
    mesh_repr: Mesh,
    yaw: float = CANONICAL_YAW,
    pitch: float = CANONICAL_PITCH,
    resolution: Optional[int] = None,
    ssaa: Optional[int] = None,
    repose_applied: bool = False,
    R_mat: Optional[np.ndarray] = None,
) -> dict:
    """Render depth/normal/canny/mask at the condition camera and write control/ + camera.json."""
    resolution = resolution or int(_CFG.get("render.resolution"))
    ssaa = ssaa or int(_CFG.get("render.ssaa"))
    r = render_view(mesh_repr, yaw, pitch, resolution, ssaa,
                    return_types=("mask", "depth", "normal"))
    mask, depth, normal = r["mask"], r["depth"], r["normal"]

    # depth.png: Gate-1 confirmed inverse-depth polarity.
    Image.fromarray(depth_to_controlnet(depth, mask), mode="L").save(jobdir.control("depth"))

    # normal.png: (n+1)/2 encoding (source for canny edge detection).
    normal_img = (np.clip(normal.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(normal_img, mode="RGB").save(jobdir.control("normal"))

    # canny.png: cv2.Canny over the NORMAL render (clean part edges), (100,200), dilate 1px.
    lo, hi = _CFG.get("render.canny_thresholds")
    edges = cv2.Canny(normal_img, int(lo), int(hi))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    Image.fromarray(edges, mode="L").save(jobdir.control("canny"))

    # mask.png: silhouette.
    mask_u8 = ((mask > 0.5) * 255).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(jobdir.control("mask"))

    coverage = float((mask > 0.5).mean())
    cam = {
        "yaw": yaw, "pitch": pitch, "r": R, "fov_deg": FOV_DEG,
        "resolution": resolution, "repose_applied": bool(repose_applied),
        "R": R_mat.tolist() if R_mat is not None else None,
        "mask_coverage": coverage,
    }
    jobdir.write_json(jobdir.camera_json(), cam)
    return {"mask_coverage": coverage, "camera": cam}

"""Existing-appearance rendering + per-group visibility for articulated assets (WS1/WS2).

PartNet-Mobility OBJs carry MTL materials (map_Kd textures or flat Kd colors) but often very
few vertices (as few as 14), so per-vertex color sampling needs subdivision first:
per submesh, iterate `mesh.subdivide()` (preserves UVs) until the max edge <= a fraction of
the asset extent or a vertex cap, then `visual.to_color()`; flat-Kd OBJs throw in `to_color()`
and fall back to `material.main_color`.

All CUDA-touching work (rendering) stays inside functions; importing this module is cheap.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from pbr_texture_pipeline.config import load_config

_CFG = load_config()

_MAX_VERTICES = 60_000


# --- per-mesh color extraction ------------------------------------------------
def _mesh_color_arrays(tm, max_edge: Optional[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One trimesh (with visual/material) -> (vertices, faces, colors[0..1]).

    Subdivides textured meshes until max edge <= max_edge (or the vertex cap) so texture
    detail survives per-vertex sampling; falls back to the material's flat color when UV/image
    sampling is unavailable.
    """
    import trimesh

    mesh = tm
    if max_edge is not None and max_edge > 0:
        try:
            while (len(mesh.vertices) < _MAX_VERTICES
                   and len(mesh.edges_unique) > 0
                   and float(mesh.edges_unique_length.max()) > max_edge):
                mesh = mesh.subdivide()
        except Exception:  # noqa: BLE001 - subdivision is best-effort refinement
            mesh = tm if len(tm.vertices) else mesh

    colors = None
    try:
        vc = mesh.visual.to_color().vertex_colors
        colors = np.asarray(vc)[:, :3].astype(np.float32) / 255.0
        if colors.shape[0] != len(mesh.vertices):
            colors = None
    except Exception:  # noqa: BLE001 - flat-Kd OBJs throw in to_color()
        colors = None
    if colors is None:
        colors = np.tile(_flat_color(mesh)[None, :], (len(mesh.vertices), 1))
    return (np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64), colors)


def _flat_color(mesh) -> np.ndarray:
    """Best-effort flat RGB in [0,1] for a mesh whose visual cannot be sampled per vertex."""
    try:
        mat = getattr(mesh.visual, "material", None)
        if mat is not None and getattr(mat, "main_color", None) is not None:
            return np.asarray(mat.main_color[:3], dtype=np.float32) / 255.0
    except Exception:  # noqa: BLE001
        pass
    return np.array([0.6, 0.6, 0.6], dtype=np.float32)


def _iter_geoms(loaded):
    """Yield (mesh, world_transform) for a trimesh load result (Scene or single mesh)."""
    import trimesh

    if isinstance(loaded, trimesh.Scene):
        for node_name in loaded.graph.nodes_geometry:
            T, geom_name = loaded.graph[node_name]
            yield loaded.geometry[geom_name], np.asarray(T, dtype=np.float64)
    else:
        yield loaded, np.eye(4)


def load_colored_arrays(path: str, max_edge_frac: Optional[float] = None) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust colored-arrays loader for any mesh/scene file (also used by Stage E when a scene
    has >1 geometry or `to_color()` throws). Returns (vertices, faces, colors[0..1]) with all
    node transforms baked."""
    import trimesh

    loaded = trimesh.load(path)
    geoms = list(_iter_geoms(loaded))
    if not geoms:
        raise ValueError(f"no geometry in {path}")

    max_edge = None
    if max_edge_frac:
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for tm, T in geoms:
            v = np.asarray(tm.vertices, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
            if len(v):
                lo = np.minimum(lo, v.min(axis=0))
                hi = np.maximum(hi, v.max(axis=0))
        max_edge = float((hi - lo).max()) * float(max_edge_frac)

    return _concat_colored([(tm, T) for tm, T in geoms], max_edge)


def _concat_colored(geoms: list, max_edge: Optional[float]) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_v, all_f, all_c = [], [], []
    offset = 0
    for tm, T in geoms:
        if len(tm.vertices) == 0:
            continue
        v, f, c = _mesh_color_arrays(tm, max_edge)
        v = v @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]
        all_v.append(v)
        all_f.append(f + offset)
        all_c.append(c)
        offset += len(v)
    if not all_v:
        raise ValueError("no non-empty geometry")
    return (np.concatenate(all_v, axis=0), np.concatenate(all_f, axis=0),
            np.concatenate(all_c, axis=0))


# --- asset-level appearance ---------------------------------------------------
def _load_visual_with_materials(asset, vis):
    """Load one URDF visual's OBJ with its MTL materials (may be a Scene for multi-material)."""
    import trimesh

    return trimesh.load(str(asset.asset_dir / vis.obj), process=False)


def load_colored_asset_arrays(asset, groups, fk: dict) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whole-asset existing appearance at rest pose: every visual with materials, per-visual
    origin + link FK baked, colors sampled per vertex. Returns (vertices, faces, colors)."""
    max_edge_frac = float(_CFG.get("articulated.appearance_max_edge_frac", 0.02))
    pending = []  # (mesh, world_T)
    for g in groups:
        link_T = fk.get(g.link, np.eye(4))
        for vis in g.visuals:
            loaded = _load_visual_with_materials(asset, vis)
            for tm, T in _iter_geoms(loaded):
                pending.append((tm, link_T @ vis.origin @ T))

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for tm, T in pending:
        v = np.asarray(tm.vertices, dtype=np.float64)
        if len(v):
            v = v @ T[:3, :3].T + T[:3, 3]
            lo = np.minimum(lo, v.min(axis=0))
            hi = np.maximum(hi, v.max(axis=0))
    max_edge = float((hi - lo).max()) * max_edge_frac
    return _concat_colored(pending, max_edge)


def group_has_real_texture(asset, group) -> bool:
    """True when any of the group's OBJs binds an image texture (map_Kd)."""
    for vis in group.visuals:
        try:
            loaded = _load_visual_with_materials(asset, vis)
        except Exception:  # noqa: BLE001
            continue
        for tm, _T in _iter_geoms(loaded):
            mat = getattr(tm.visual, "material", None)
            if mat is not None and getattr(mat, "image", None) is not None:
                return True
    return False


def dominant_color(asset, group) -> list[int]:
    """Area-agnostic mean RGB (0-255) of a group's existing appearance (textures + flat Kd)."""
    samples = []
    for vis in group.visuals:
        try:
            loaded = _load_visual_with_materials(asset, vis)
        except Exception:  # noqa: BLE001
            continue
        for tm, _T in _iter_geoms(loaded):
            if len(tm.vertices) == 0:
                continue
            _v, _f, c = _mesh_color_arrays(tm, None)
            if len(c):
                samples.append(c.mean(axis=0))
    if not samples:
        return [153, 153, 153]
    mean = np.stack(samples).mean(axis=0)
    return [int(round(float(x) * 255)) for x in mean]


def is_chromatic(rgb: list[int]) -> bool:
    """True when a flat color carries real appearance information (visibly non-gray).

    Blank PartNet assets (e.g. 19179) use achromatic gray Kd values, which must NOT count as
    existing appearance; painted assets (19898's brown wood) are clearly chromatic.
    """
    c = np.asarray(rgb, dtype=np.float32) / 255.0
    return float(c.max() - c.min()) > 0.06


def render_appearance_sheet(job, vertices: np.ndarray, faces: np.ndarray,
                            colors: np.ndarray, resolution: int = 512, ssaa: int = 2,
                            R_mat: Optional[np.ndarray] = None):
    """Render the existing appearance from the 8 contact-sheet yaws -> views/appearance_sheet.png.

    `vertices` are in the assembled world frame; they are re-normalized into the internal
    frame (+ the Stage R front-panel repose `R_mat`, when given), so panels line up 1:1 with
    the clay contact sheet.
    """
    from PIL import Image

    from pbr_texture_pipeline import rendering as R

    verts = R.to_internal_frame(vertices, up="z")  # URDF world frame is Z-up
    if R_mat is not None:
        verts = verts @ np.asarray(R_mat, dtype=np.float64).T
    mesh = R.colored_mesh_repr(verts, faces, colors)
    panels = []
    for k in range(R.CONTACT_N):
        out = R.render_appearance(mesh, R.contact_yaw(k), R.CONTACT_PITCH, resolution, ssaa)
        rgb = out["rgb"].copy()
        rgb[out["mask"] <= 0.5] = 0
        panels.append(rgb)
    rows = [np.concatenate(panels[r * 4:(r + 1) * 4], axis=1) for r in range(2)]
    sheet = np.concatenate(rows, axis=0)
    Image.fromarray(sheet, mode="RGB").save(job.appearance_sheet())
    return job.appearance_sheet()


# --- per-group visibility at the condition camera (crop-ref policy) -----------
_DEPTH_EPS = 0.01   # depth-consistency tolerance (camera r=2, object in [-0.5,0.5])


def group_visibility(job, group_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
                     center: np.ndarray, scale: float,
                     resolution: Optional[int] = None) -> dict:
    """Visible-pixel share per group at the condition camera.

    `group_arrays` maps group_id -> (world-frame vertices, faces) of the FK-assembled group;
    `center`/`scale` are the assembly normalization used for input/mesh_norm.glb, so every
    group lands in the exact frame the control maps were rendered from. Writes
    control/groups/coverage.json + a visible-mask PNG per group; returns the coverage dict.
    """
    from PIL import Image

    from pbr_texture_pipeline import rendering as R

    resolution = resolution or int(_CFG.get("render.resolution"))
    ssaa = int(_CFG.get("render.ssaa"))
    camera = job.read_json(job.camera_json()) if job.camera_json().is_file() else {}
    yaw = float(camera.get("yaw", R.CANONICAL_YAW))
    pitch = float(camera.get("pitch", R.CANONICAL_PITCH))
    # Stage R's front-panel repose (camera.json R) must apply here too, or the group masks
    # would not line up with the control maps / chosen.png.
    R_mat = (np.asarray(camera["R"], dtype=np.float64)
             if camera.get("repose_applied") and camera.get("R") is not None else None)

    def _internal(v: np.ndarray) -> np.ndarray:
        # World (Z-up) -> internal: center+scale, no axis swap; then the front-panel repose.
        w = (np.asarray(v, dtype=np.float64) - center) * scale
        if R_mat is not None:
            w = w @ R_mat.T
        return w

    # Full-assembly depth once.
    all_v = np.concatenate([_internal(v) for v, _f in group_arrays.values()], axis=0)
    all_f = []
    off = 0
    for v, f in group_arrays.values():
        all_f.append(np.asarray(f) + off)
        off += len(v)
    all_f = np.concatenate(all_f, axis=0)
    full = R.render_view(R.to_mesh_repr_arrays(all_v, all_f), yaw, pitch, resolution, ssaa,
                         return_types=("mask", "depth"))
    full_mask = full["mask"] > 0.5
    full_depth = full["depth"]

    coverage: dict[str, dict] = {}
    (job.group_coverage().parent).mkdir(parents=True, exist_ok=True)
    for gid, (v, f) in group_arrays.items():
        alone = R.render_view(R.to_mesh_repr_arrays(_internal(v), np.asarray(f)), yaw, pitch,
                              resolution, ssaa, return_types=("mask", "depth"))
        alone_mask = alone["mask"] > 0.5
        visible = alone_mask & full_mask & (np.abs(alone["depth"] - full_depth) < _DEPTH_EPS)
        alone_px = int(alone_mask.sum())
        px = int(visible.sum())
        coverage[gid] = {"px": px, "alone_px": alone_px,
                         "share": (px / alone_px) if alone_px else 0.0}
        Image.fromarray((visible * 255).astype(np.uint8), mode="L").save(job.group_mask(gid))

    job.write_json(job.group_coverage(), coverage)
    return coverage

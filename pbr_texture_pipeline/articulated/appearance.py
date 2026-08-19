"""Per-group visibility + colored-mesh loading for articulated assets.

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

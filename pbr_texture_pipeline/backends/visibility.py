"""Point visibility against saved multi-view depth buffers (PRD_articulated_v2).

One implementation serves both sides of global texture mode so they cannot diverge:
Stage R computes per-group occlusion statistics from surface samples, and the global
TRELLIS adapter computes the per-texel rest-pose visibility mask from texel surface points.
The orchestrator imports this module as ``pbr_texture_pipeline.backends.visibility``; the adapter, which
must not import ``pbr_texture_pipeline.*``, imports it by script-directory path exactly like
``_adapter_common``.

The npz produced by :func:`save_views` stores the depth buffers together with the camera
extrinsics/intrinsics they were rendered with (utils3d conventions: extrinsics are 4x4
world-to-camera, OpenCV axes with +z forward; intrinsics are 3x3 normalized, uv in [0, 1]).
Storing the matrices keeps this module free of camera math that would have to track the
renderer. Numpy is the only dependency.
"""
from __future__ import annotations

import numpy as np

# Depth agreement tolerance, in render-frame units (the normalized frame has extent 1 and
# camera distance r=2, so depths are ~1.5-2.5). Matches the depth-consistency epsilon the
# per-part pipeline uses for group visibility masks.
DEPTH_EPS = 0.01


def save_views(path, depths, extrinsics, intrinsics) -> None:
    """Write a visibility npz: depth [V,H,W], extr [V,4,4], intr [V,3,3] (float32)."""
    depths = np.asarray(depths, dtype=np.float32)
    extr = np.asarray(extrinsics, dtype=np.float32)
    intr = np.asarray(intrinsics, dtype=np.float32)
    if depths.ndim != 3 or extr.shape != (len(depths), 4, 4) or intr.shape != (len(depths), 3, 3):
        raise ValueError(f"inconsistent view arrays: depth {depths.shape}, "
                         f"extr {extr.shape}, intr {intr.shape}")
    np.savez_compressed(path, depth=depths, extr=extr, intr=intr)


def load_views(path) -> dict:
    """Load a visibility npz into {'depth': [V,H,W], 'extr': [V,4,4], 'intr': [V,3,3]}."""
    with np.load(path) as z:
        return {"depth": z["depth"], "extr": z["extr"], "intr": z["intr"]}


def visible_from_view(points: np.ndarray, depth: np.ndarray, extr: np.ndarray,
                      intr: np.ndarray, eps: float = DEPTH_EPS) -> np.ndarray:
    """Bool mask [N]: point visible in one view.

    A point is visible when its projection lands inside the image and no rendered geometry
    sits more than `eps` in front of it: either the depth sample matches the point's camera
    depth, or the sample is background (0, nothing occludes the ray; happens for sub-pixel
    silhouette slivers of the object itself).
    """
    p = np.asarray(points, dtype=np.float64)
    cam = p @ np.asarray(extr[:3, :3], dtype=np.float64).T + np.asarray(extr[:3, 3],
                                                                        dtype=np.float64)
    z = cam[:, 2]
    ok = z > 1e-6
    zs = np.where(ok, z, 1.0)
    u = intr[0, 0] * cam[:, 0] / zs + intr[0, 2]
    v = intr[1, 1] * cam[:, 1] / zs + intr[1, 2]
    h, w = depth.shape
    px = np.floor(u * w).astype(np.int64)
    py = np.floor(v * h).astype(np.int64)
    inside = ok & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    d = np.zeros(len(p), dtype=np.float64)
    d[inside] = depth[py[inside], px[inside]]
    return inside & ((d <= 0.0) | (z <= d + eps))


def visible_from_any(points: np.ndarray, views: dict, eps: float = DEPTH_EPS) -> np.ndarray:
    """Bool mask [N]: point visible in at least one of the saved views."""
    points = np.asarray(points, dtype=np.float64)
    out = np.zeros(len(points), dtype=bool)
    for depth, extr, intr in zip(views["depth"], views["extr"], views["intr"]):
        todo = ~out
        if not todo.any():
            break
        out[todo] = visible_from_view(points[todo], depth, extr, intr, eps)
    return out

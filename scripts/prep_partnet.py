"""Combine PartNet-Mobility part-OBJs into a single untextured mesh per object.

PartNet-Mobility objects ship as many `textured_objs/*.obj` part files. pbr-texture-pipeline's Stage R
expects ONE mesh file per job, so this merges all parts of an object into a single geometry
(materials stripped: pbr-texture-pipeline textures from scratch, so any source color is irrelevant and
would only confuse the clay contact sheet). Output name encodes the model category so the
Gate 7 review can read it at a glance.

    conda run -n trellis2 python scripts/prep_partnet.py \
        --root partnet_mobility \
        --out  jobs_gate7_inputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def _zup_to_yup(vertices: np.ndarray) -> np.ndarray:
    """Rotate vertices from Z-up (PartNet-Mobility convention) to Y-up (glTF convention).

    (x, y, z) -> (x, z, -y): old Z (up) becomes new Y (up)."""
    out = np.empty_like(vertices)
    out[:, 0] = vertices[:, 0]
    out[:, 1] = vertices[:, 2]
    out[:, 2] = -vertices[:, 1]
    return out


def combine_object(obj_dir: Path) -> trimesh.Trimesh:
    """Merge every part OBJ under textured_objs/ into one Trimesh (geometry only).

    PartNet-Mobility OBJs are Z-up; the merged GLB is rotated to Y-up so the
    pipeline's standard normalization (preprocess_mesh with up='y') and TRELLIS.2's
    own preprocess_mesh both produce an upright object."""
    part_dir = obj_dir / "textured_objs"
    parts = sorted(part_dir.glob("*.obj"))
    if not parts:
        raise FileNotFoundError(f"no part OBJs in {part_dir}")
    geoms: list[trimesh.Trimesh] = []
    for p in parts:
        loaded = trimesh.load(p, process=False, force="mesh")
        if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
            # Drop visuals; pbr-texture-pipeline re-textures, so source color is noise.
            loaded.visual = trimesh.visual.ColorVisuals(loaded)
            geoms.append(loaded)
    if not geoms:
        raise ValueError(f"all parts empty in {part_dir}")
    merged = trimesh.util.concatenate(geoms)
    merged.vertices = _zup_to_yup(merged.vertices)
    return merged


def main() -> int:
    ap = argparse.ArgumentParser("prep_partnet")
    ap.add_argument("--root", required=True, help="partnet_mobility root (dirs are object ids)")
    ap.add_argument("--out", required=True, help="output dir for combined <id>_<cat>.glb")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written = []
    for obj_dir in sorted(root.iterdir()):
        if not obj_dir.is_dir():
            continue
        meta_p = obj_dir / "meta.json"
        cat = "object"
        if meta_p.is_file():
            cat = json.load(open(meta_p)).get("model_cat", "object")
        cat_slug = cat.lower().replace(" ", "_")
        try:
            mesh = combine_object(obj_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[prep] SKIP {obj_dir.name} ({cat}): {e}")
            continue
        dst = out / f"{obj_dir.name}_{cat_slug}.glb"
        mesh.export(dst)
        ext = mesh.bounds[1] - mesh.bounds[0]
        print(f"[prep] {dst.name}: {len(mesh.vertices)} verts, "
              f"{len(mesh.faces)} faces, extent={np.round(ext, 3)}")
        written.append(str(dst))
    print(f"[prep] wrote {len(written)} meshes -> {out}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Stage a run directory for the batch pipeline.

Symlinks the light meshes from ``meshes/`` into ``meshes_run/`` unchanged, and writes
decimated copies of the few very heavy meshes (which risk OOM / very slow texturing at the
mandated render resolution). Originals in ``meshes/`` are never touched.

Run in the trellis2 env:
    conda run -n trellis2 python scripts/stage_run_meshes.py
"""
import os
import sys
import glob
import warnings

warnings.filterwarnings("ignore")
import trimesh

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "meshes")
DST = os.path.join(HERE, "meshes_run")

# Meshes above this face count get decimated to TARGET_FACES.
HEAVY_FACES = 500_000
TARGET_FACES = 250_000


def as_single_mesh(path):
    s = trimesh.load(path, process=False)
    if isinstance(s, trimesh.Scene):
        return s.to_mesh()
    return s


def main():
    os.makedirs(DST, exist_ok=True)
    meshes = sorted(glob.glob(os.path.join(SRC, "*.glb")))
    if not meshes:
        print(f"no meshes in {SRC}", file=sys.stderr)
        return 1
    n_link = n_decim = 0
    for f in meshes:
        name = os.path.basename(f)
        out = os.path.join(DST, name)
        m = as_single_mesh(f)
        nf = len(m.faces)
        if nf > HEAVY_FACES:
            # Decimate a clean single mesh and export as glb. Call fast_simplification
            # directly (trimesh's wrapper mis-forwards target_count on this version).
            import fast_simplification
            v, f2 = fast_simplification.simplify(
                m.vertices, m.faces, target_count=TARGET_FACES)
            dec = trimesh.Trimesh(vertices=v, faces=f2, process=False)
            dec.export(out)
            print(f"  decimated {name}: {nf:,} -> {len(dec.faces):,} faces "
                  f"({os.path.getsize(out)//1024} KB)")
            n_decim += 1
        else:
            # Fresh symlink to the pristine original.
            if os.path.islink(out) or os.path.exists(out):
                os.remove(out)
            os.symlink(os.path.abspath(f), out)
            n_link += 1
    print(f"\nstaged {n_link} symlinks + {n_decim} decimated -> {DST} "
          f"({len(os.listdir(DST))} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

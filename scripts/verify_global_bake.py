"""Gate A10 (PRD.md section 15): per-group bake correctness.

On one articulated job whose global-mode Stage T has run (textured/trellis2/groups/*.glb +
textured/trellis2/global/pass_a.json exist), this script:

1. re-decodes the pass A field on the same (merged mesh, reference, seed) and bakes the
   WHOLE merged mesh in one atlas - the baseline the per-group bakes must visually match
   (decode is deterministic at fixed seed; the split spike measured max texel diff 0);
2. renders the assembled per-group result and the whole-mesh bake side by side from four
   yaws into <job>/eval/a10/ for seam inspection;
3. reports per-group texel density (texels per face from pass_a.json, against the share of
   one shared 2048 atlas the group would have gotten) - the "handles measurably sharper"
   number.

Run in the trellis2 env with a GPU:
    conda run -n trellis2 python scripts/verify_global_bake.py jobs_v2/<job_id>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

SHARED_ATLAS = 2048  # the single-atlas size a non-split bake would have used


def _render_strip(glb_path: Path, out_png: Path, resolution: int = 512) -> None:
    from PIL import Image

    from pbr_texture_pipeline import rendering as R
    from pbr_texture_pipeline.articulated.appearance import load_colored_arrays

    v, f, c = load_colored_arrays(str(glb_path))
    # Both A10 strips render articulated geometry (Z-up frames): no glTF axis swap.
    v = R.to_internal_frame(np.asarray(v, dtype=np.float64), up="z")
    mesh = R.colored_mesh_repr(v, f, c)
    pitch = math.radians(15.0)
    tiles = [R.render_appearance(mesh, R.CANONICAL_YAW + math.radians(d), pitch,
                                 resolution, 2)["rgb"]
             for d in (0.0, 90.0, 180.0, 270.0)]
    Image.fromarray(np.concatenate(tiles, axis=1), mode="RGB").save(out_png)


def _whole_mesh_bake(job_root: Path, out_glb: Path, resolution: int, texture_size: int,
                     seed: int) -> None:
    """One unsplit bake of the merged rest-pose mesh from a fresh decode of the same field
    (same calls as the global adapter's pass A; proven identical to pipe.run by the 4.1
    spike)."""
    import trimesh
    from PIL import Image

    from pbr_texture_pipeline.config import load_config

    cfg = load_config()
    trellis_root = str(cfg.repo("trellis2"))
    os.chdir(trellis_root)
    sys.path.insert(0, trellis_root)
    import torch
    from trellis2.pipelines import Trellis2TexturingPipeline

    pipe = Trellis2TexturingPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
    pipe.cuda()
    mesh = trimesh.load(str(job_root / "input" / "mesh_norm.glb"), force="mesh",
                        process=False)
    image = Image.open(job_root / "ref" / "chosen_rgba.png")
    with torch.no_grad():
        out = pipe.run(mesh, image, seed=seed, resolution=resolution,
                       texture_size=texture_size, preprocess_image=True)
    out.export(str(out_glb))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job_root", type=Path)
    ap.add_argument("--resolution", type=int, default=1024)
    args = ap.parse_args()
    job_root = args.job_root.resolve()

    pass_a_path = job_root / "textured" / "trellis2" / "global" / "pass_a.json"
    assembled = job_root / "textured" / "trellis2" / "assembled.glb"
    if not pass_a_path.is_file() or not assembled.is_file():
        print(f"[A10] FAIL: run global-mode Stage T first ({pass_a_path} / {assembled})")
        return 1
    pass_a = json.loads(pass_a_path.read_text())
    seed = int(pass_a.get("seed", 42))

    out_dir = job_root / "eval" / "a10"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Whole-mesh baseline bake from the same field.
    whole_glb = out_dir / "whole_mesh_bake.glb"
    if not whole_glb.is_file():
        _whole_mesh_bake(job_root, whole_glb, args.resolution, SHARED_ATLAS, seed)

    # 2. Side-by-side render strips.
    _render_strip(assembled, out_dir / "assembled_groups.png")
    _render_strip(whole_glb, out_dir / "whole_mesh.png")
    print(f"[A10] renders -> {out_dir}/assembled_groups.png vs whole_mesh.png "
          "(inspect: no seams beyond contact edges)")

    # 3. Texel density report: texels per unit surface area, per group, against the uniform
    #    density one shared SHARED_ATLAS bake would spread over the whole object. Areas come
    #    from the Stage R group meshes (rigid FK preserves area).
    import trimesh

    job = json.loads((job_root / "job.json").read_text())
    groups = {g["group_id"]: g for g in (job.get("asset") or {}).get("groups", [])}
    meta = pass_a.get("groups", {})
    areas: dict[str, float] = {}
    for gid in meta:
        mesh_path = job_root / "groups" / gid / "mesh.glb"
        if mesh_path.is_file():
            m = trimesh.load(str(mesh_path), force="mesh", process=False)
            areas[gid] = float(m.area)
    total_area = sum(areas.values()) or 1.0
    fill = (sum(m.get("texels", 0) for m in meta.values())
            / (sum(int(m.get("texture_size", 1024)) ** 2 for m in meta.values()) or 1))
    shared_density = SHARED_ATLAS * SHARED_ATLAS * fill / total_area

    print(f"[A10] shared-atlas baseline density: {shared_density:,.0f} texels/area "
          f"(fill {fill:.2f})")
    print(f"[A10] {'group':<40} {'faces':>7} {'texels':>9} {'density':>11} {'gain':>7}")
    small_gains = []
    for gid, m in sorted(meta.items(), key=lambda kv: kv[1].get("n_faces", 0)):
        if m.get("constant_pbr") or gid not in areas or areas[gid] <= 0:
            continue
        texels, faces = m.get("texels", 0), max(m.get("n_faces", 1), 1)
        density = texels / areas[gid]
        gain = density / shared_density
        is_tiny = bool(groups.get(gid, {}).get("tiny"))
        print(f"[A10] {gid:<40} {faces:>7} {texels:>9} {density:>11,.0f} "
              f"{gain:>6.1f}x{' (tiny)' if is_tiny else ''}")
        # Tiny groups bake at tiny_texture_size by design, so their density is
        # intentionally low; the sharpness claim is about regular small groups.
        if faces < 2000 and not is_tiny:
            small_gains.append(gain)
    if small_gains:
        ok = min(small_gains) > 1.0
        print(f"[A10] small-group (<2000 faces) density gain: min {min(small_gains):.1f}x "
              f"-> {'PASS' if ok else 'FAIL'} (must exceed 1x)")
    print("[A10] final PASS additionally requires visually consistent strips "
          "(no seams beyond contact edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

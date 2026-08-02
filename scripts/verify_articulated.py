#!/usr/bin/env python
"""Gates A1, A9, A11, A12a/b, A16: articulated correctness checks that need no GPU.

A1 (PRD.md section 11), per asset under partnet_mobility/:
  1. every URDF visual has a name attribute; the semantic group partition covers every visual
     exactly once;
  2. assembled rest-pose vertices == R_base @ raw_concat (max err < 1e-6), where R_base is
     the common FK world transform of every link (PartNet assets are authored so the joint
     chains cancel at rest);
  3. assembled bbox == R_base @ bounding_box.json (when present);
  4. metadata fallback: re-parse a copy with semantics.txt/result.json/meta.json/
     bounding_box.json deleted; grouping and labels identical; required_files closure exists
     on disk.

A12a (PRD_articulated_v2 section 12): the opened joint configuration keeps every joint value
inside its limits and produces finite assembled bounds; runs from the asset alone.

A9 (PRD_articulated_v2 section 12): slicing the persisted input/mesh_norm.glb by
input/face_ranges.json reproduces each group's geometry: undoing re-pose and
normalization on the sliced triangles matches merge_group_mesh output pushed through
rest-pose FK (the Z-up URDF world assembly is normalized with no axis swap). Needs job dirs produced by Stage R; pass --jobs-root to enable (the newest job
per asset with a face_ranges.json is checked; A9 is reported as SKIP without one).
The re-pose rotation is read from the job's camera.json so this script never imports
pbr_texture_pipeline.rendering (torch + renderer at import time).

A16 (PRD.md section 4.6): for every urdf job under --jobs-root that has both a metadata
category (job.json asset.category) and a Stage V spec (vlm/spec.json), the spec's
category matches the metadata category after normalization and its category_source is
"metadata". No GPU; detects Stage V mis-classifications that contradict the dataset.

    conda run -n trellis2 python scripts/verify_articulated.py [--jobs-root jobs/]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbr_texture_pipeline.articulated import urdf as U  # noqa: E402
from pbr_texture_pipeline.config import load_config  # noqa: E402

ASSETS_ROOT = Path(__file__).resolve().parent.parent / "partnet_mobility"
_META_FILES = ("semantics.txt", "result.json", "meta.json", "bounding_box.json")
_CFG = load_config()

FK_TOL = 1e-6
BBOX_TOL = 1e-5
# mesh_norm.glb stores float32 vertices; undoing normalization scales the rounding error up
# to object extent, so A9 tolerates 1e-5 of the extent (float32 eps is ~6e-8 at |v|<=0.5).
A9_REL_TOL = 1e-5


def _load_obj_vertices(asset_dir: Path, rel: str) -> np.ndarray:
    import trimesh

    m = trimesh.load(str(asset_dir / rel), force="mesh", process=False, skip_materials=True)
    return np.asarray(m.vertices, dtype=np.float64)


def _apply(T: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v @ T[:3, :3].T + T[:3, 3]


def check_asset(asset_dir: Path) -> list[str]:
    errors: list[str] = []
    asset = U.parse_asset(asset_dir)

    # 1. Visual names + partition.
    n_visuals = sum(len(vs) for vs in asset.links.values())
    if not asset.all_visuals_named:
        errors.append("some visuals lack a name attribute")
    groups = U.build_groups(asset, "semantic")
    n_grouped = sum(len(g.visuals) for g in groups)
    if n_grouped != n_visuals:
        errors.append(f"group partition covers {n_grouped}/{n_visuals} visuals")
    seen = set()
    for g in groups:
        for vis in g.visuals:
            key = (g.link, vis.obj, id(vis))
            if key in seen:
                errors.append(f"visual {vis.obj} in {g.link} grouped twice")
            seen.add(key)

    # 2. FK: PartNet authors OBJs in one global frame, so the COMPOSED per-visual world
    #    transform fk[link] @ visual_origin equals the base rotation R_base for every visual
    #    at rest pose (per-link FK translations are cancelled by the visual origins).
    fk = U.link_world_transforms(asset)
    links_with_visuals = [ln for ln, vs in asset.links.items() if vs]
    R_base = fk[links_with_visuals[0]] @ asset.links[links_with_visuals[0]][0].origin
    spread = max(float(np.abs(fk[ln] @ vis.origin - R_base).max())
                 for ln in links_with_visuals for vis in asset.links[ln])
    if spread > FK_TOL:
        errors.append(f"composed visual world transforms differ (spread {spread:.2e})")

    assembled = []
    raw = []
    for ln in links_with_visuals:
        for vis in asset.links[ln]:
            v = _load_obj_vertices(asset_dir, vis.obj)
            assembled.append(_apply(fk[ln] @ vis.origin, v))
            raw.append(v)
    assembled = np.concatenate(assembled, axis=0)
    raw = np.concatenate(raw, axis=0)
    err = float(np.abs(assembled - _apply(R_base, raw)).max())
    if err > FK_TOL:
        errors.append(f"assembled != R_base @ raw_concat (max err {err:.2e})")

    # 3. bbox cross-check (skipped when bounding_box.json is absent).
    if asset.bbox is not None:
        lo = np.asarray(asset.bbox["min"], dtype=np.float64)
        hi = np.asarray(asset.bbox["max"], dtype=np.float64)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        tc = _apply(R_base, corners)
        ref_lo, ref_hi = tc.min(axis=0), tc.max(axis=0)
        got_lo, got_hi = assembled.min(axis=0), assembled.max(axis=0)
        extent = float((ref_hi - ref_lo).max()) or 1.0
        bbox_err = float(max(np.abs(got_lo - ref_lo).max(), np.abs(got_hi - ref_hi).max()))
        if bbox_err > max(BBOX_TOL, 1e-5) * extent:
            errors.append(f"assembled bbox != R_base @ bounding_box.json "
                          f"(err {bbox_err:.2e}, extent {extent:.3f})")

    # 4. Metadata fallback: strip optional metadata, expect identical grouping + labels.
    with tempfile.TemporaryDirectory(prefix="bt_a1_") as tmp:
        stripped = Path(tmp) / asset_dir.name
        shutil.copytree(asset_dir, stripped,
                        ignore=shutil.ignore_patterns(*_META_FILES))
        asset2 = U.parse_asset(stripped)
        if asset2.category is not None:
            errors.append("stripped asset still reports a category")
        groups2 = U.build_groups(asset2, "semantic")
        sig1 = sorted((g.group_id, g.label, len(g.visuals)) for g in groups)
        sig2 = sorted((g.group_id, g.label, len(g.visuals)) for g in groups2)
        if sig1 != sig2:
            errors.append("grouping/labels change when metadata is deleted")
        closure = U.required_files(stripped / "mobility.urdf")
        missing = [r for r in closure if not (stripped / r).is_file()]
        if missing:
            errors.append(f"required_files closure missing on disk: {missing[:5]}")

    return errors


def check_opened_pose(asset_dir: Path) -> list[str]:
    """Gate A12a: opened configuration inside limits, finite assembled bounds, and at least
    one movable joint actually moved (otherwise the open-pose pass is vacuous)."""
    errors: list[str] = []
    asset = U.parse_asset(asset_dir)
    frac = float(_CFG.get("articulated.global.open_pose_frac", 0.8))

    moved = 0
    for j in asset.joints.values():
        q = U.opened_pose_q(j, frac, asset)
        rest = U.rest_pose_q(j)
        if j.lower is not None and j.upper is not None:
            if not (j.lower - 1e-9 <= q <= j.upper + 1e-9):
                errors.append(f"joint {j.name}: opened q {q:.4f} outside "
                              f"[{j.lower:.4f}, {j.upper:.4f}]")
        if not np.isfinite(q):
            errors.append(f"joint {j.name}: opened q not finite")
        if abs(q - rest) > 1e-9:
            moved += 1
    movable = sum(1 for j in asset.joints.values()
                  if j.type in ("revolute", "prismatic", "continuous")
                  and j.lower is not None and j.upper is not None and j.upper != j.lower)
    if movable and not moved:
        errors.append(f"{movable} movable joint(s) but none moved at frac {frac}")

    fk_open = U.link_world_transforms_at(asset, frac)
    pts = []
    for ln, visuals in asset.links.items():
        for vis in visuals:
            v = _load_obj_vertices(asset_dir, vis.obj)
            pts.append(_apply(fk_open[ln] @ vis.origin, v))
    allp = np.concatenate(pts, axis=0)
    if not np.isfinite(allp).all():
        errors.append("opened assembled vertices contain non-finite values")
    return errors


def _find_job_dir(jobs_root: Path, asset_name: str) -> Path | None:
    """Newest job dir under jobs_root for the asset (job ids are <stem>_<timestamp>) that has
    a persisted face_ranges.json."""
    candidates = sorted(d for d in jobs_root.glob(f"{asset_name}_*")
                        if (d / "input" / "face_ranges.json").is_file())
    return candidates[-1] if candidates else None


def check_merge_split(asset_dir: Path, job_dir: Path) -> list[str]:
    """Gate A9: per-group slices of the persisted merged mesh reproduce merge_group_mesh
    output after undoing normalization, re-pose, and rest-pose FK (the Z-up world
    assembly enters the internal frame with no axis swap)."""
    import trimesh

    errors: list[str] = []
    meta = json.loads((job_dir / "input" / "face_ranges.json").read_text())
    merged = trimesh.load(str(job_dir / "input" / "mesh_norm.glb"), force="mesh",
                          process=False)
    mv = np.asarray(merged.vertices, dtype=np.float64)
    mf = np.asarray(merged.faces)

    n_faces = sum(e - s for s, e in meta["face_ranges"].values())
    if len(mf) != n_faces:
        return [f"mesh_norm.glb has {len(mf)} faces, face_ranges cover {n_faces}"]

    # Undo, in reverse order of Stage R: re-pose (the R recorded in the job's camera.json;
    # orientation detection may have picked a panel other than the config fallback), then
    # center+scale. Stage R normalizes the Z-up world assembly with up="z": no axis swap.
    cam_path = job_dir / "control" / "camera.json"
    cam = json.loads(cam_path.read_text()) if cam_path.is_file() else {}
    if cam.get("repose_applied") and cam.get("R") is not None:
        Rz = np.asarray(cam["R"], dtype=np.float64)
    else:
        Rz = np.eye(3)
    v = mv @ Rz  # (Rz^-1 @ v^T)^T == v @ Rz (Rz orthonormal)
    center = np.asarray(meta["norm"]["center"], dtype=np.float64)
    scale = float(meta["norm"]["scale"])
    v = v / scale + center

    asset = U.parse_asset(asset_dir)
    groups = {g.group_id: g for g in U.build_groups(asset, "semantic")}
    fk = U.link_world_transforms(asset)
    extent = 1.0 / scale
    tol = A9_REL_TOL * extent

    for gid in meta["group_order"]:
        g = groups.get(gid)
        if g is None:
            errors.append(f"group {gid} in face_ranges.json but not in asset grouping")
            continue
        start, stop = meta["face_ranges"][gid]
        m = U.merge_group_mesh(asset, g)
        T = fk.get(g.link, np.eye(4))
        expect_v = _apply(T, np.asarray(m.vertices, dtype=np.float64))
        got_tri = v[mf[start:stop]]
        expect_tri = expect_v[np.asarray(m.faces)]
        if got_tri.shape != expect_tri.shape:
            errors.append(f"group {gid}: slice has {got_tri.shape[0]} faces, "
                          f"expected {expect_tri.shape[0]}")
            continue
        err = float(np.abs(got_tri - expect_tri).max())
        if err > tol:
            errors.append(f"group {gid}: slice mismatch {err:.2e} > {tol:.2e}")
    return errors


def check_orient(job_dir: Path) -> list[str]:
    """Gate A11: the Orient-Anything-V2 front decision recorded in camera.json matches the
    verified PartNet front (articulated.front_panel fallback); an unconfident model must
    have fallen back with a logged reason rather than repose to a different panel."""
    cam_path = job_dir / "control" / "camera.json"
    if not cam_path.is_file():
        return ["control/camera.json missing"]
    cam = json.loads(cam_path.read_text())
    orient = cam.get("orient")
    if orient is None:
        return ["no orient block in camera.json (articulated.orient.enabled off during "
                "Stage R?)"]
    errors: list[str] = []
    fallback = int(orient.get("fallback_panel", 2))
    chosen = orient.get("front_panel")
    if chosen != fallback:
        errors.append(f"effective front panel {chosen} != verified panel {fallback} "
                      f"(method={orient.get('method')})")
    if orient.get("method") == "fallback":
        # Correct degraded behavior, but the discrepancy must be visible.
        print(f"        note: fell back ({orient.get('reason')}); "
              f"predicted={orient.get('predicted_panel')}")
    return errors


def _mask_coverage(path: Path) -> float:
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("L"))
    return float((arr > 127).mean())


def check_open_pose_artifacts(job_dir: Path) -> list[str]:
    """Gate A12b (post-hoc over Stage R artifacts): the open-pose control maps exist and are
    not degenerate, opening does not HIDE surface area overall (the occlusion statistics are
    the direct measure; a wrong open direction - dishwasher 12085's shelf sliding inward -
    or a projection/polarity bug in the visibility npz shows up here), and the recorded
    occlusion fractions are sane.

    Raw open-vs-rest mask coverage is deliberately NOT compared: the open pose is normalized
    by its own (larger) bounds and doors legitimately swing away from the camera, so the
    silhouette can shrink while the pass still reveals interiors.
    """
    errors: list[str] = []
    control = job_dir / "control"
    needed = [control / "open" / n for n in ("depth.png", "normal.png", "canny.png",
                                             "mask.png")]
    needed += [control / "camera_open.json", control / "visibility_rest.npz",
               control / "visibility_open.npz"]
    missing = [str(p.relative_to(job_dir)) for p in needed if not p.is_file()]
    if missing:
        return [f"missing artifacts: {', '.join(missing)}"]

    cov_open = _mask_coverage(control / "open" / "mask.png")
    if cov_open < 0.02:
        errors.append(f"open-pose mask coverage {cov_open:.3f} is degenerate")

    job = json.loads((job_dir / "job.json").read_text())
    groups = (job.get("asset") or {}).get("groups", [])
    occs = [g.get("occluded_frac_rest") for g in groups]
    if not groups or any(o is None for o in occs):
        errors.append("occluded_frac_rest missing from asset.groups records")
        return errors
    bad = [g["group_id"] for g in groups
           for o in (g["occluded_frac_rest"], g["occluded_frac_open"])
           if not (0.0 <= o <= 1.0)]
    if bad:
        errors.append(f"occlusion fractions outside [0,1] for {sorted(set(bad))}")
    if min(occs) >= 0.5:
        errors.append(f"most-visible group is {min(occs):.2f} occluded; the visibility "
                      "test likely disagrees with the renderer")
    weight = sum(g.get("n_faces", 1) for g in groups)
    mean_rest = sum(g["occluded_frac_rest"] * g.get("n_faces", 1) for g in groups) / weight
    mean_open = sum(g["occluded_frac_open"] * g.get("n_faces", 1) for g in groups) / weight
    if mean_open > mean_rest + 0.05:
        errors.append(f"opening HIDES surface area (mean occluded {mean_rest:.3f} rest -> "
                      f"{mean_open:.3f} open); wrong open direction?")
    print(f"        occluded rest={mean_rest:.3f} open={mean_open:.3f} "
          f"open-mask={cov_open:.3f}")
    return errors


def check_spec_category(jobs_root: Path) -> int:
    """Gate A16: every urdf job with a metadata category and a Stage V spec has
    spec.category matching the metadata category (normalized, substring in either
    direction counts) and spec.category_source == "metadata". Returns the fail count."""
    from pbr_texture_pipeline.vlm import normalize_category

    checked = failed = 0
    for job_dir in sorted(d for d in jobs_root.iterdir()
                          if (d / "job.json").is_file()):
        job = json.loads((job_dir / "job.json").read_text())
        if job.get("kind") != "urdf":
            continue
        cat = (job.get("asset") or {}).get("category")
        spec_path = job_dir / "vlm" / "spec.json"
        if not cat or not spec_path.is_file():
            continue
        checked += 1
        spec = json.loads(spec_path.read_text())
        errors = []
        want = normalize_category(cat).lower()
        got = normalize_category(str(spec.get("category") or "")).lower()
        if not got or (got not in want and want not in got):
            errors.append(f"spec.category {spec.get('category')!r} does not match "
                          f"metadata category {cat!r}")
        if spec.get("category_source") != "metadata":
            errors.append(f"spec.category_source is {spec.get('category_source')!r}, "
                          "expected 'metadata'")
        status = "PASS" if not errors else "FAIL"
        print(f"[A16][{status}] {job_dir.name}: metadata category {cat!r}")
        for e in errors:
            print(f"        - {e}")
        failed += bool(errors)
    if checked == 0:
        print("[A16] SKIP: no urdf job with a metadata category and vlm/spec.json\n")
    else:
        print(f"[A16] {'FAIL' if failed else 'PASS'}: {checked - failed}/{checked} "
              "jobs clean\n")
    return failed


def _run_gate(tag: str, asset_dirs: list[Path], check, describe=None) -> int:
    failed = checked = 0
    for d in asset_dirs:
        errs = check(d)
        if errs is None:
            print(f"[{tag}][SKIP] {d.name}: no Stage R job found")
            continue
        checked += 1
        status = "PASS" if not errs else "FAIL"
        line = f"[{tag}][{status}] {d.name}"
        if describe:
            line += f": {describe(d)}"
        print(line)
        for e in errs:
            print(f"        - {e}")
        failed += bool(errs)
    verdict = "FAIL" if failed else ("PASS" if checked == len(asset_dirs) else "INCOMPLETE")
    print(f"[{tag}] {verdict}: {checked - failed}/{len(asset_dirs)} assets clean\n")
    return failed if checked == len(asset_dirs) else max(failed, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs-root", type=Path, default=None,
                    help="jobs root with Stage R output; enables gate A9")
    args = ap.parse_args()

    asset_dirs = sorted(d for d in ASSETS_ROOT.iterdir()
                        if d.is_dir() and (d / "mobility.urdf").is_file())
    if not asset_dirs:
        print(f"no assets under {ASSETS_ROOT}")
        return 1

    def describe(d: Path) -> str:
        asset = U.parse_asset(d)
        groups = U.build_groups(asset, "semantic")
        return (f"{len(asset.links)} links, "
                f"{sum(len(v) for v in asset.links.values())} visuals, {len(groups)} groups, "
                f"category={asset.category}")

    failed = _run_gate("A1", asset_dirs, check_asset, describe)
    failed += _run_gate("A12a", asset_dirs, check_opened_pose)

    if args.jobs_root is None:
        print("[A9] SKIP: no --jobs-root given (also skips A11, A12b, A16)")
    else:
        def with_job(check_fn):
            def check(d: Path):
                job_dir = _find_job_dir(args.jobs_root, d.name)
                if job_dir is None:
                    return None
                return check_fn(d, job_dir)
            return check

        failed += _run_gate("A9", asset_dirs, with_job(check_merge_split))
        failed += _run_gate("A11", asset_dirs, with_job(lambda d, j: check_orient(j)))
        failed += _run_gate("A12b", asset_dirs,
                            with_job(lambda d, j: check_open_pose_artifacts(j)))
        failed += check_spec_category(args.jobs_root)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

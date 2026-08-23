"""Batch CLI: many assets, unattended, stage-major.

    conda run -n trellis2 python -m pbr_texture_pipeline.batch \
      --assets 'partnet_mobility/*/mobility.urdf' --jobs-root jobs/ \
      --stages render,texture --seed 42 --resume

Execution is stage-major for model-load efficiency: the backend loads once over all its
approved pairs via `--pairs-file`.

Heavy imports (torch, the renderer) are deferred into each stage so `--help` and
argument parsing never touch CUDA.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pbr_texture_pipeline.config import load_config
from pbr_texture_pipeline.jobdir import JobDir, make_job_id, resolve_stages

_CFG = load_config()


# --- CLI ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("pbr_texture_pipeline.batch", description="Unattended batch texturing.")
    ap.add_argument("--assets", "--meshes", required=True, dest="assets",
                    help="glob of URDF files (mobility.urdf or model.urdf), "
                         "e.g. 'partnet_mobility/*/mobility.urdf' or "
                         "'articraft_extracted/*/model.urdf'")
    ap.add_argument("--jobs-root", default="jobs")
    ap.add_argument("--backends", default="trellis2",
                    help="comma list: trellis2")
    ap.add_argument("--stages", default="render,vlm,imageedit,texture",
                    help="comma list / aliases R,V,E,T")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-mode", default=_CFG.get("gpu_mode", "dual"),
                    choices=["dual", "single"])
    ap.add_argument("--limit", type=int, default=None, help="cap number of meshes processed")
    ap.add_argument("--resume", action="store_true",
                    help="reuse the latest matching job per mesh; skip stages marked done")
    ap.add_argument("--group-by", default=str(_CFG.get("articulated.group_by", "semantic")),
                    choices=["semantic", "link"],
                    help="texture-group granularity for urdf assets")
    return ap


def expand_meshes(pattern: str, limit: Optional[int]) -> list[str]:
    paths = sorted(glob.glob(pattern, recursive=True))
    paths = [p for p in paths if Path(p).is_file()]
    if limit is not None:
        paths = paths[:limit]
    return paths


# --- job resolution (--resume) ----------------------------------------------
def find_or_create_job(jobs_root: str, mesh_path: str, resume: bool,
                       params: dict) -> JobDir:
    """On --resume, reuse the most recent existing job for this asset; else create fresh."""
    if resume:
        target = str(Path(mesh_path).resolve())
        matches = [j for j in JobDir.load_all(jobs_root)
                   if j.state.get("mesh_source") == target]
        if matches:
            job = max(matches, key=lambda j: j.state.get("created", ""))
            print(f"[batch] resume: reusing {job.job_id}")
            return job
    category = ""
    asset_dir = Path(mesh_path).resolve().parent
    meta = asset_dir / "meta.json"
    if meta.is_file():
        try:
            category = json.loads(meta.read_text()).get("model_cat") or ""
        except (OSError, json.JSONDecodeError):
            category = ""
    if not category:
        import xml.etree.ElementTree as ET
        try:
            robot_name = ET.parse(str(mesh_path)).getroot().get("name")
            if robot_name:
                category = robot_name.replace("_", " ").strip()
        except Exception:  # noqa: BLE001
            pass
    return JobDir.create(jobs_root, mesh_path, params=params,
                         job_id=make_job_id(mesh_path, category))


# --- Stage R -----------------------------------------------------------------
def stage_render(jobs: list[JobDir]) -> None:
    from pbr_texture_pipeline.articulated import stages as AS
    for job in jobs:
        if job.is_done("render"):
            print(f"[batch][R] skip (done) {job.job_id}")
            continue
        try:
            job.start("render")
            info = AS.render(job)
            job.finish("render", params={"n_groups": info["n_groups"],
                                         "n_tiny": info["n_tiny"]})
            print(f"[batch][R] {job.job_id} "
                  f"groups={info['n_groups']} (tiny {info['n_tiny']})")
        except Exception as e:  # noqa: BLE001
            job.fail("render", traceback.format_exc())
            print(f"[batch][R] FAIL {job.job_id}: {e}")


# --- Stage V -----------------------------------------------------------------
def stage_vlm(jobs: list[JobDir]) -> None:
    """Stage V: VLM material analysis. Model loads once for all assets."""
    from pbr_texture_pipeline.backends import registry

    approved = [j for j in jobs if j.is_done("render") and not j.is_done("vlm")]
    if not approved:
        print("[batch][V] no approved jobs")
        return

    items = []
    for job in approved:
        job.start("vlm")
        front = job.render_front_white()
        if not front.is_file():
            job.fail("vlm", "front_white.png missing")
            continue
        items.append({
            "image": str(front),
            "output": str(job.vlm_materials()),
            "job_id": job.job_id,
        })

    if items:
        items_path = Path(approved[0].root.parent) / "_items_vlm.json"
        with open(items_path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"[batch][V] vlm: {len(items)} items")
        registry.vlm_infer_batch(str(items_path))
        items_path.unlink(missing_ok=True)

    for job in approved:
        if job.vlm_materials().is_file():
            materials = job.vlm_materials().read_text().strip()
            job.finish("vlm", params={"materials_preview": materials[:200]})
            print(f"[batch][V] {job.job_id}: {materials[:80]}")
        else:
            job.fail("vlm", "materials.txt not produced")
            print(f"[batch][V] FAIL {job.job_id}")


# --- Stage E -----------------------------------------------------------------
def stage_imageedit(jobs: list[JobDir]) -> None:
    """Stage E: image enhancement. Model loads once for all assets."""
    from pbr_texture_pipeline.backends import registry

    approved = [j for j in jobs if j.is_done("vlm") and not j.is_done("imageedit")]
    if not approved:
        print("[batch][E] no approved jobs")
        return

    items = []
    for job in approved:
        job.start("imageedit")
        front = job.render_front_white()
        depth = job.render_depth(0)
        mat_path = job.vlm_materials()
        if not front.is_file() or not depth.is_file() or not mat_path.is_file():
            job.fail("imageedit", "missing front_white, depth, or materials")
            continue
        materials = mat_path.read_text().strip()
        items.append({
            "source": str(front),
            "depth": str(depth),
            "materials": materials,
            "output": str(job.enhanced_view(0)),
            "job_id": job.job_id,
        })

    if items:
        items_path = Path(approved[0].root.parent) / "_items_imageedit.json"
        with open(items_path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"[batch][E] imageedit: {len(items)} items")
        registry.imageedit_infer_batch(str(items_path))
        items_path.unlink(missing_ok=True)

    for job in approved:
        if job.enhanced_view(0).is_file():
            job.finish("imageedit", params={"enhanced": str(job.enhanced_view(0))})
            print(f"[batch][E] {job.job_id}: ok")
        else:
            job.fail("imageedit", "enhanced_0.png not produced")
            print(f"[batch][E] FAIL {job.job_id}")


# --- Stage T -----------------------------------------------------------------
def _original_mesh(job: JobDir) -> str:
    """Untouched upload (backends normalize it themselves; never feed mesh_norm.glb)."""
    cands = sorted(job.path("input").glob("original.*"))
    if not cands:
        raise FileNotFoundError(f"no input/original.* in {job.root}")
    return str(cands[0])


def stage_texture(jobs: list[JobDir], backends: list[str], seed: int) -> None:
    """One trellis2 global pair per job, per-group bakes inside the adapter."""
    stage_texture_urdf(jobs, backends, seed)


def _grade_urdf_backend(job: JobDir, backend: str) -> None:
    """Assemble one backend's group outputs and record the done/needs_review/error status."""
    from pbr_texture_pipeline.articulated import stages as AS

    try:
        res = AS.assemble_backend(job, backend)
        non_tiny = [g for g in job.state["asset"]["groups"] if not g.get("tiny")]
        ok_non_tiny = sum(1 for g in non_tiny
                          if res["groups"].get(g["group_id"]) == "ok")
        if ok_non_tiny == len(non_tiny):
            status = "done"
        elif res["n_ok"] > 0:
            status = "needs_review"
        else:
            status = "error"
        prev = job.stage("texture").get("params", {}).get("backends", {})
        prev[backend] = {"ok": status == "done", "n_ok": res["n_ok"],
                         "n_total": res["n_total"],
                         "glb_path": res["assembled_glb"]}
        job.finish("texture", status, params={"backends": prev})
        print(f"[batch][T] {job.job_id} {backend}: {res['n_ok']}/{res['n_total']} "
              f"groups -> {status}")
    except Exception as e:  # noqa: BLE001
        job.fail("texture", traceback.format_exc())
        print(f"[batch][T] FAIL {job.job_id} {backend}: {e}")


def stage_texture_urdf(jobs: list[JobDir], backends: list[str], seed: int) -> None:
    """Stage T: one trellis2 global pair per job (one field decode on the merged mesh,
    per-group bakes inside the adapter, tiny groups included), then URDF + assembled.glb.
    Resume is group-granular (the pair lists only missing group GLBs).
    """
    from pbr_texture_pipeline.articulated import stages as AS
    from pbr_texture_pipeline.backends import registry

    approved = []
    for j in jobs:
        if not j.is_done("render"):
            continue
        approved.append(j)
    if not approved:
        print("[batch][T] no approved jobs")
        return

    global_pairs = []
    for job in approved:
        job.start("texture", seed=seed)
        pair = AS.global_texture_pair(job, seed)
        if pair is not None:
            global_pairs.append(pair)
    if global_pairs:
        pairs_path = approved[0].root.parent / "_pairs_trellis2_urdf.json"
        with open(pairs_path, "w") as f:
            json.dump(global_pairs, f, indent=2)
        print(f"[batch][T] trellis2: {len(global_pairs)} job pairs (urdf)")
        registry.texture_pairs("trellis2", str(pairs_path))
        pairs_path.unlink(missing_ok=True)
    for job in approved:
        _grade_urdf_backend(job, "trellis2")


# --- driver ------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    stages = resolve_stages([s.strip() for s in args.stages.split(",") if s.strip()])
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    assets = expand_meshes(args.assets, args.limit)
    if not assets:
        print(f"[batch] no assets matched {args.assets!r}")
        return 1
    print(f"[batch] {len(assets)} assets | stages={stages} | backends={backends} "
          f"| gpu_mode={args.gpu_mode}")

    job_params = {"batch": True, "backends": backends,
                  "gpu_mode": args.gpu_mode, "group_by": args.group_by}
    jobs = [find_or_create_job(args.jobs_root, m, args.resume, job_params) for m in assets]

    if "render" in stages:
        stage_render(jobs)
    if "vlm" in stages:
        stage_vlm(jobs)
    if "imageedit" in stages:
        stage_imageedit(jobs)
    if "texture" in stages:
        stage_texture(jobs, backends, args.seed)

    # Summary.
    ok = 0
    for job in jobs:
        statuses = {s: job.status(s) for s in stages}
        flagged = any(v == "needs_review" for v in statuses.values())
        errored = any(v == "error" for v in statuses.values())
        if not errored and not flagged:
            ok += 1
        print(f"[batch] {job.job_id}: {statuses}")
    print(f"[batch] done: {ok}/{len(jobs)} clean (no error / needs_review)")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

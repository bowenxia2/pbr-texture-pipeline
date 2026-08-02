"""Batch CLI: many meshes, unattended, stage-major (PRD section 8).

    conda run -n trellis2 python -m pbr_texture_pipeline.batch \
      --meshes 'partnet_mobility/**/*.obj' --jobs-root jobs/ \
      --backends trellis2,hunyuan --stages render,vlm,diffuse,texture,eval \
      --candidates 4 --seed 42 --material-hint "clean, factory-new" \
      --select iou+clip --gpu-mode dual --limit 100 --resume

Execution is stage-major for model-load efficiency (mirrors the pbr_compare sweeps): the VLM
loads once for all meshes, Qwen-Image loads once for all, and each backend loads once over all
its approved pairs via `--pairs-file`. Between GPU stages the previous model family is unloaded
so Stage T (texturing subprocess) has VRAM even on a single GPU (PRD section 7 stage-exclusion).

Heavy imports (torch, the renderer, Qwen-Image) are deferred into each stage so `--help` and
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

# The workers and adapters set this in their spawn env (ipc.py / registry.py); the batch
# process runs Stage D's 20B model in-process, where fragmentation on a 46 GB card is the
# difference between fitting and OOM (seen 2026-07-21: job 2 of a 5-job diffuse run).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pbr_texture_pipeline.config import load_config
from pbr_texture_pipeline.jobdir import JobDir, make_urdf_job_id, resolve_stages

_CFG = load_config()


# --- CLI ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("pbr_texture_pipeline.batch", description="Unattended batch texturing.")
    ap.add_argument("--meshes", required=True,
                    help="glob of input meshes (.glb/.obj/...) and/or mobility.urdf files, "
                         "e.g. 'partnet_mobility/*/mobility.urdf'")
    ap.add_argument("--jobs-root", default="jobs")
    ap.add_argument("--backends", default="trellis2",
                    help="comma list: trellis2,hunyuan")
    ap.add_argument("--stages", default="render,vlm,diffuse,plan,texture,eval,judge",
                    help="comma list / aliases R,V,D,P,T,E,J")
    ap.add_argument("--candidates", type=int, default=int(_CFG.get("diffusion.candidates", 4)))
    ap.add_argument("--seed", type=int, default=int(_CFG.get("diffusion.seed", 42)))
    ap.add_argument("--material-hint", default="",
                    help="batch-level appearance steer injected into the VLM prompt")
    ap.add_argument("--select", default="iou+clip", choices=["iou+clip", "iou"],
                    help="candidate auto-selection policy (Stage D)")
    ap.add_argument("--gpu-mode", default=_CFG.get("gpu_mode", "dual"),
                    choices=["dual", "single"])
    ap.add_argument("--limit", type=int, default=None, help="cap number of meshes processed")
    ap.add_argument("--resume", action="store_true",
                    help="reuse the latest matching job per mesh; skip stages marked done")
    ap.add_argument("--strict", action="store_true",
                    help="do NOT texture jobs whose Stage D was flagged needs_review")
    ap.add_argument("--spec-cache-by-category", action="store_true",
                    help="(deferred, PRD open question 10) reuse one spec per category")
    ap.add_argument("--diffusion-kind", default="depth", choices=["depth", "depth+canny"])
    # Articulated (urdf) options (PRD.md, articulated extension).
    ap.add_argument("--keep-appearance", action="store_true",
                    default=bool(_CFG.get("articulated.keep_appearance", False)),
                    help="steer the VLM to match the asset's existing colors/materials")
    ap.add_argument("--group-by", default=str(_CFG.get("articulated.group_by", "semantic")),
                    choices=["semantic", "link"],
                    help="texture-group granularity for urdf assets")
    ap.add_argument("--refine", action="store_true",
                    help="execute targeted retexturing of VLM-confirmed bad groups "
                         "(candidate selection and confirmation always run with Stage E)")
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
    """On --resume, reuse the most recent existing job for this mesh; else create fresh.

    A .urdf path creates an articulated (kind="urdf") job with an asset-id-based job id
    (every asset's file is `mobility.urdf`, so stem-based ids would collide).
    """
    if resume:
        target = str(Path(mesh_path).resolve())
        matches = [j for j in JobDir.load_all(jobs_root)
                   if j.state.get("mesh_source") == target]
        if matches:
            job = max(matches, key=lambda j: j.state.get("created", ""))
            print(f"[batch] resume: reusing {job.job_id}")
            return job
    if Path(mesh_path).suffix.lower() == ".urdf":
        category = ""
        meta = Path(mesh_path).resolve().parent / "meta.json"
        if meta.is_file():
            try:
                category = json.loads(meta.read_text()).get("model_cat") or ""
            except (OSError, json.JSONDecodeError):
                category = ""
        return JobDir.create(jobs_root, mesh_path, params=params, kind="urdf",
                             job_id=make_urdf_job_id(mesh_path, category))
    return JobDir.create(jobs_root, mesh_path, params=params)


# --- Stage R -----------------------------------------------------------------
def stage_render(jobs: list[JobDir]) -> None:
    from pbr_texture_pipeline import rendering as R
    for job in jobs:
        if job.is_done("render"):
            print(f"[batch][R] skip (done) {job.job_id}")
            continue
        try:
            job.start("render")
            if job.kind == "urdf":
                from pbr_texture_pipeline.articulated import stages as AS
                info = AS.render(job)
                job.finish("render", params={"mask_coverage": info["mask_coverage"],
                                             "n_groups": info["n_groups"],
                                             "n_tiny": info["n_tiny"],
                                             "has_appearance": info["has_appearance"]})
                print(f"[batch][R] {job.job_id} coverage={info['mask_coverage']:.3f} "
                      f"groups={info['n_groups']} (tiny {info['n_tiny']}) "
                      f"appearance={info['has_appearance']}")
                continue
            norm = R.load_and_normalize(job.state["mesh_source"], job)
            mesh_repr = R.to_mesh_repr(norm)
            R.render_contact_sheet(job, mesh_repr)
            info = R.render_control_maps(job, mesh_repr)
            job.finish("render", params={"mask_coverage": info["mask_coverage"]})
            print(f"[batch][R] {job.job_id} coverage={info['mask_coverage']:.3f}")
        except Exception as e:  # noqa: BLE001
            job.fail("render", traceback.format_exc())
            print(f"[batch][R] FAIL {job.job_id}: {e}")


# --- Stage V -----------------------------------------------------------------
def stage_vlm(jobs: list[JobDir], material_hint: str, spec_cache_by_category: bool) -> None:
    """Automated caption -> spec per mesh, via the VLM worker subprocess.

    The Qwen3.6 AWQ checkpoint runs under vLLM in the dedicated `vlm` env (torch >= 2.8), so
    unlike the other stages it cannot be imported in-process; the JSON-lines worker gives the
    same load-once-over-all-meshes behavior.
    """
    from pbr_texture_pipeline.backends import registry
    from pbr_texture_pipeline.workers.ipc import WorkerClient

    todo = [j for j in jobs if not j.is_done("vlm") and j.is_done("render")]
    if not todo:
        return
    model_id = _CFG.model("vlm")
    worker = WorkerClient("pbr_texture_pipeline.workers.vlm_worker", _CFG.vlm_env_name,
                          cuda_visible="0", conda_bin=registry._conda_bin(),
                          extra_env=_CFG.hf_env(), name="vlm")
    worker.start()
    cache: dict[str, dict] = {}
    try:
        for job in todo:
            try:
                job.start("vlm")
                res = worker.call("batch", job_root=str(job.root),
                                  contact_sheet=str(job.contact_sheet()),
                                  material_hint=material_hint)
                spec, fb = res["spec"], res["fallback"]
                if spec_cache_by_category:
                    cache.setdefault(spec.get("category", "object"), spec)
                job.finish("vlm", "needs_review" if fb else "done",
                           params={"spec_fallback": fb, "model": model_id})
                print(f"[batch][V] {job.job_id} category={spec.get('category')} fallback={fb}")
            except Exception as e:  # noqa: BLE001
                job.fail("vlm", traceback.format_exc())
                print(f"[batch][V] FAIL {job.job_id}: {e}")
    finally:
        worker.shutdown()    # frees the whole vLLM pool before Stage D / T


# --- Stage D -----------------------------------------------------------------
def stage_diffuse(jobs: list[JobDir], n: int, seed: int, select: str,
                  kind: str, strict: bool) -> None:
    from pbr_texture_pipeline import diffusion as D
    todo = [j for j in jobs if not j.is_done("diffuse") and j.is_done("vlm")]
    if not todo:
        return
    try:
        for job in todo:
            try:
                spec = job.read_json(job.spec())
                prompt, negative = spec["ref_prompt"], spec.get("negative_prompt", "")
                job.start("diffuse", seed=seed)
                # Free the previous job's RMBG/CLIP before the 20B forward (VRAM headroom).
                D.unload_scoring_models()
                D.generate_candidates(job, prompt, negative, base_seed=seed, n=n, kind=kind)
                scores = D.score_candidates(job, prompt, n=n)
                sel = D.select_batch(scores)
                D.cutout(job, sel["index"])
                # Articulated: a second generation from the open-pose depth with the
                # selected candidate's seed, while the pipe is still resident.
                if (job.kind == "urdf"
                        and bool(_CFG.get("articulated.global.open_pose_pass", True))
                        and job.path("control", "open", "depth.png").is_file()):
                    D.generate_open_reference(job, seed, sel["index"])
                status = "needs_review" if sel["needs_review"] else "done"
                job.finish("diffuse", status,
                           params={"selected": sel["index"], "reason": sel["reason"],
                                   "ious": [round(s["iou"], 3) for s in scores],
                                   "select_policy": select})
                print(f"[batch][D] {job.job_id} sel={sel['index']} "
                      f"needs_review={sel['needs_review']}")
            except Exception as e:  # noqa: BLE001
                job.fail("diffuse", traceback.format_exc())
                print(f"[batch][D] FAIL {job.job_id}: {e}")
                # A mid-generation failure (e.g. OOM) can leave the offload hooks with
                # modules stranded on CPU; drop the pipe so the next job reloads cleanly.
                D.unload_pipe()
    finally:
        D.unload_pipe()      # free Qwen-Image+ControlNet before Stage T subprocesses


# --- Stage P (articulated material plan) -------------------------------------
def stage_plan(jobs: list[JobDir]) -> None:
    """Per-part material plan for urdf jobs via the vlm worker (one load over all jobs).

    Flat mesh jobs mark the stage done with {"skipped": true} so downstream gating and the
    jobs browser stay uniform.
    """
    from pbr_texture_pipeline.backends import registry
    from pbr_texture_pipeline.workers.ipc import WorkerClient

    for job in jobs:
        if job.kind == "mesh" and not job.is_done("plan"):
            job.start("plan")
            job.finish("plan", params={"skipped": True})

    todo = [j for j in jobs if j.kind == "urdf" and not j.is_done("plan")
            and j.is_done("render")]
    if not todo:
        return
    model_id = _CFG.model("vlm")
    worker = WorkerClient("pbr_texture_pipeline.workers.vlm_worker", _CFG.vlm_env_name,
                          cuda_visible="0", conda_bin=registry._conda_bin(),
                          extra_env=_CFG.hf_env(), name="vlm")
    worker.start()
    try:
        for job in todo:
            try:
                job.start("plan")
                res = worker.call("plan", job_root=str(job.root), timeout=3600)
                fb = res["fallback"]
                n_fb = sum(1 for g in res["plan"]["groups"].values() if g.get("fallback"))
                job.finish("plan", "needs_review" if fb else "done",
                           params={"plan_fallback": fb, "n_group_fallbacks": n_fb,
                                   "model": model_id})
                print(f"[batch][P] {job.job_id} materials="
                      f"{len(res['plan']['materials'])} fallback={fb} "
                      f"group_fallbacks={n_fb}")
            except Exception as e:  # noqa: BLE001
                job.fail("plan", traceback.format_exc())
                print(f"[batch][P] FAIL {job.job_id}: {e}")
    finally:
        worker.shutdown()


# --- Stage T -----------------------------------------------------------------
def _original_mesh(job: JobDir) -> str:
    """Untouched upload (backends normalize it themselves; never feed mesh_norm.glb)."""
    cands = sorted(job.path("input").glob("original.*"))
    if not cands:
        raise FileNotFoundError(f"no input/original.* in {job.root}")
    return str(cands[0])


def stage_texture(jobs: list[JobDir], backends: list[str], seed: int, strict: bool) -> None:
    """Group by backend: load each backend once over all approved pairs (--pairs-file)."""
    from pbr_texture_pipeline.backends import registry

    urdf_jobs = [j for j in jobs if j.kind == "urdf"]
    jobs = [j for j in jobs if j.kind == "mesh"]
    if urdf_jobs:
        stage_texture_urdf(urdf_jobs, backends, seed, strict)

    # Approved = Stage D produced a cutout. needs_review is textured too unless --strict.
    approved = []
    for j in jobs:
        st = j.status("diffuse")
        if st == "done" or (st == "needs_review" and not strict):
            if j.chosen_rgba().is_file():
                approved.append(j)
    if not approved:
        if not urdf_jobs:
            print("[batch][T] no approved jobs")
        return

    for backend in backends:
        pairs = []
        job_by_outdir = {}
        for job in approved:
            if job.output_glb(backend) is not None and job.is_done("texture"):
                continue  # already textured this backend (resume)
            out_dir = str(job.textured_dir(backend))
            pairs.append({
                "mesh": _original_mesh(job), "image": str(job.chosen_rgba()),
                "out_dir": out_dir, "seed": seed, "camera_json": str(job.camera_json()),
            })
            job_by_outdir[out_dir] = job
        if not pairs:
            continue

        pairs_path = approved[0].root.parent / f"_pairs_{backend}.json"
        with open(pairs_path, "w") as f:
            json.dump(pairs, f, indent=2)

        print(f"[batch][T] {backend}: {len(pairs)} pairs")
        for job in job_by_outdir.values():
            job.start("texture", seed=seed)
        res = registry.texture_pairs(backend, str(pairs_path))

        # Map each emitted result back to a job by its out_dir / glb path.
        done_dirs = set()
        for r in res.get("results", []):
            glb = r.get("glb_path") or ""
            for out_dir, job in job_by_outdir.items():
                if glb.startswith(out_dir):
                    ok = bool(r.get("ok") and Path(glb).exists())
                    prev = job.stage("texture").get("params", {}).get("backends", {})
                    prev[backend] = {"ok": ok, "glb_path": glb if ok else None}
                    job.finish("texture", "done" if ok else "error",
                               params={"backends": prev})
                    done_dirs.add(out_dir)
        # Jobs that emitted no result line for this backend -> mark error.
        for out_dir, job in job_by_outdir.items():
            if out_dir not in done_dirs:
                prev = job.stage("texture").get("params", {}).get("backends", {})
                prev[backend] = {"ok": False, "glb_path": None}
                job.finish("texture", "error", params={"backends": prev})
        pairs_path.unlink(missing_ok=True)


def _grade_urdf_backend(job: JobDir, backend: str) -> None:
    """Assemble one backend's group outputs and record the done/needs_review/error status
    (shared by the per_part and global texture modes; the rules are identical)."""
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


def stage_texture_urdf(jobs: list[JobDir], backends: list[str], seed: int,
                       strict: bool) -> None:
    """Articulated Stage T: one trellis2_global pair per job (one field decode on the merged
    mesh, per-group bakes inside the adapter, tiny groups included), then per-backend URDF +
    assembled.glb. Resume is group-granular (the pair lists only missing group GLBs).

    Hunyuan has no articulated path (its whole-object conditioning does not fit per-group
    bakes); urdf jobs are trellis2-only and Stage J records a walkover.
    """
    from pbr_texture_pipeline.articulated import stages as AS
    from pbr_texture_pipeline.backends import registry

    approved = []
    for j in jobs:
        st = j.status("diffuse")
        if not (st == "done" or (st == "needs_review" and not strict)):
            continue
        if j.status("plan") not in ("done", "needs_review"):
            print(f"[batch][T] skip {j.job_id}: no material plan (run Stage P)")
            continue
        approved.append(j)
    if not approved:
        print("[batch][T] no approved urdf jobs")
        return

    if "hunyuan" in backends:
        print("[batch][T] hunyuan skipped for articulated jobs (no articulated path)")
    if "trellis2" not in backends:
        return

    global_pairs = []
    for job in approved:
        job.start("texture", seed=seed)
        pair = AS.global_texture_pair(job, seed)
        if pair is not None:
            global_pairs.append(pair)
    if global_pairs:
        pairs_path = approved[0].root.parent / "_pairs_trellis2_global_urdf.json"
        with open(pairs_path, "w") as f:
            json.dump(global_pairs, f, indent=2)
        print(f"[batch][T] trellis2_global: {len(global_pairs)} job pairs (urdf)")
        registry.texture_pairs("trellis2_global", str(pairs_path))
        pairs_path.unlink(missing_ok=True)
    for job in approved:
        _grade_urdf_backend(job, "trellis2")


# --- Stage E -----------------------------------------------------------------
def stage_eval(jobs: list[JobDir], backends: list[str]) -> None:
    from pbr_texture_pipeline import eval as E
    for job in jobs:
        if job.is_done("eval"):
            continue
        # Only eval jobs that produced at least one textured output.
        if not any(job.output_glb(b) is not None for b in backends):
            continue
        try:
            job.start("eval")
            E.run_eval(job, backends)
            job.finish("eval")
            print(f"[batch][E] {job.job_id} metrics written")
        except Exception as e:  # noqa: BLE001
            job.fail("eval", traceback.format_exc())
            print(f"[batch][E] FAIL {job.job_id}: {e}")
    if jobs:
        idx = E.write_index(str(jobs[0].root.parent))
        print(f"[batch][E] index -> {idx['html']} ({idx['n_rows']} rows)")


# --- targeted retexturing (PRD_articulated_v2 section 7) ----------------------
def stage_refine(jobs: list[JobDir], backends: list[str], seed: int,
                 force: bool) -> None:
    """Measure-then-repair pass between eval and judge for articulated jobs.

    Candidate selection (heuristics over eval/diagnostics.json) and VLM confirmation always
    run, so the trigger rate is a tracked metric; the retexture execution itself runs only
    when articulated.refine.enabled is set or --refine was passed. Confirmed-bad groups are
    moved aside and re-baked through the global Stage T path with a shifted seed (the
    missing-GLB resume rule re-emits exactly that set; the same seed would deterministically
    reproduce the rejected texels), then the job is reassembled and its eval/judge results
    are invalidated so the following stages re-run on the new textures.
    """
    from pbr_texture_pipeline import eval as E
    from pbr_texture_pipeline.articulated import stages as AS
    from pbr_texture_pipeline.backends import registry
    from pbr_texture_pipeline.workers.ipc import WorkerClient

    backend = "trellis2"  # diagnostics (field voxels, blur, pass-B fractions) are trellis2's
    if backend not in backends:
        return
    todo = [j for j in jobs if j.kind == "urdf"
            and j.status("eval") in ("done", "needs_review")]
    if not todo:
        return

    # 1. Heuristic selection + confirmation crops (renderer in this process, before the
    #    vlm worker spawns; same ordering rule as Stage J's sheet prep).
    flagged: list[tuple[JobDir, list[dict], dict]] = []
    for job in todo:
        try:
            cands = AS.select_retexture_candidates(job)
            if not cands:
                print(f"[batch][F] {job.job_id}: no retexture candidates")
                continue
            crops = E.render_group_crops(job, backend, [c["group_id"] for c in cands])
            print(f"[batch][F] {job.job_id}: {len(cands)} candidate(s): "
                  + ", ".join(c["group_id"] for c in cands))
            flagged.append((job, cands, crops))
        except Exception as e:  # noqa: BLE001
            print(f"[batch][F] candidate selection FAIL {job.job_id}: {e}")
    if not flagged:
        return

    # 2. VLM confirmation (one worker over all flagged jobs) -> vlm/refine.json.
    confirmed: list[tuple[JobDir, list[str]]] = []
    worker = WorkerClient("pbr_texture_pipeline.workers.vlm_worker", _CFG.vlm_env_name,
                          cuda_visible="0", conda_bin=registry._conda_bin(),
                          extra_env=_CFG.hf_env(), name="vlm")
    worker.start()
    try:
        for job, cands, crops in flagged:
            try:
                res = worker.call("refine", job_root=str(job.root),
                                  candidates=cands, crops=crops, timeout=3600)
                bad = [gid for gid, r in res["results"].items()
                       if r.get("grade") in ("blurry", "wrong_material", "missing_texture")]
                grades = {gid: r.get("grade") for gid, r in res["results"].items()}
                print(f"[batch][F] {job.job_id} grades={grades}")
                if bad:
                    confirmed.append((job, bad))
            except Exception as e:  # noqa: BLE001
                print(f"[batch][F] confirmation FAIL {job.job_id}: {e}")
    finally:
        worker.shutdown()

    enabled = force or bool(_CFG.get("articulated.refine.enabled", False))
    if not confirmed:
        print("[batch][F] no groups confirmed bad")
        return
    if not enabled:
        print(f"[batch][F] {sum(len(g) for _, g in confirmed)} group(s) confirmed bad "
              f"across {len(confirmed)} job(s); retexture execution disabled "
              f"(articulated.refine.enabled / --refine)")
        return

    # 3. Retexture through the global path: move the bad GLBs aside, then re-bake exactly
    #    that set from a fresh-seed field decode (same-seed would reproduce the bad texels).
    refine_seed = seed + 101
    affected: list[JobDir] = []
    for job, gids in confirmed:
        moved = AS.retexture_groups(job, backend, gids)
        print(f"[batch][F] {job.job_id}: retexturing {moved} (seed {refine_seed})")
        if moved:
            affected.append(job)
    if not affected:
        return

    global_pairs = []
    for job in affected:
        job.start("texture", seed=refine_seed)
        pair = AS.global_texture_pair(job, refine_seed)
        if pair is not None:
            global_pairs.append(pair)
    if global_pairs:
        pairs_path = affected[0].root.parent / "_pairs_trellis2_global_refine.json"
        with open(pairs_path, "w") as f:
            json.dump(global_pairs, f, indent=2)
        print(f"[batch][F] trellis2_global: {len(global_pairs)} retexture pair(s)")
        registry.texture_pairs("trellis2_global", str(pairs_path))
        pairs_path.unlink(missing_ok=True)
    for job in affected:
        _grade_urdf_backend(job, backend)

    # 4. The recorded eval metrics and any judge verdict now describe the old textures.
    for job in affected:
        job.judge_verdict().unlink(missing_ok=True)
        job.reset("judge")
        job.reset("eval")
    stage_eval(affected, backends)


# --- Stage J -----------------------------------------------------------------
def stage_judge(jobs: list[JobDir], backends: list[str]) -> None:
    """VLM picks the best textured output per job; both outputs stay on disk.

    Phase 1 (judge sheets, TRELLIS.2 renderer in this process) runs for ALL jobs before the
    vlm worker spawns, so renderer allocations never race the vLLM pool grab. Walkovers
    (one GLB) finalize without a worker; only contested jobs (two GLBs) pay a VLM call.
    """
    from pbr_texture_pipeline import judge as J
    from pbr_texture_pipeline.backends import registry
    from pbr_texture_pipeline.workers.ipc import WorkerClient

    todo = [j for j in jobs if not j.is_done("judge") and j.status("texture") != "pending"]
    if not todo:
        return

    contested: list[tuple[JobDir, dict]] = []
    for job in todo:
        try:
            # Crash recovery: verdict.json exists but judge never reached done -> re-finalize
            # (idempotent: just rewrites job.json), no VLM call.
            if job.judge_verdict().is_file():
                verdict = job.read_json(job.judge_verdict())
                out = J.finalize(job, verdict)
                print(f"[batch][J] {job.job_id} recovered winner={out['winner']} "
                      f"method={verdict.get('method')}")
                continue
            prep = J.prepare(job, backends)
            available = prep["available"]
            if not available:
                job.fail("judge", "no textured output to judge")
                print(f"[batch][J] FAIL {job.job_id}: no textured output")
            elif len(available) == 1:
                job.start("judge")
                out = J.finalize(job, J.walkover_verdict(
                    available[0], "only one textured output available"))
                print(f"[batch][J] {job.job_id} winner={out['winner']} method=walkover")
            else:
                contested.append((job, prep))
        except Exception as e:  # noqa: BLE001
            job.fail("judge", traceback.format_exc())
            print(f"[batch][J] FAIL {job.job_id}: {e}")

    if not contested:
        return
    model_id = _CFG.model("vlm")
    worker = WorkerClient("pbr_texture_pipeline.workers.vlm_worker", _CFG.vlm_env_name,
                          cuda_visible="0", conda_bin=registry._conda_bin(),
                          extra_env=_CFG.hf_env(), name="vlm")
    worker.start()
    try:
        for job, prep in contested:
            try:
                job.start("judge")
                labels = J.label_assignment(job.job_id, prep["available"])
                spec = job.read_json(job.spec()) if job.spec().is_file() else {}
                res = worker.call(
                    "judge",
                    sheet_a=prep["sheets"][labels["A"]],
                    sheet_b=prep["sheets"][labels["B"]],
                    ref=str(job.chosen()) if job.chosen().is_file() else None,
                    category=spec.get("category", "object"))
                if res["verdict"] is None:
                    verdict = J.fallback_verdict(prep["available"],
                                                 raw_response=res.get("raw", ""),
                                                 model=model_id)
                else:
                    verdict = J.vlm_verdict(res["verdict"], labels, res.get("raw", ""),
                                            model_id)
                out = J.finalize(job, verdict)
                print(f"[batch][J] {job.job_id} winner={out['winner']} "
                      f"method={verdict['method']} confidence={verdict['confidence']}")
            except Exception as e:  # noqa: BLE001
                job.fail("judge", traceback.format_exc())
                print(f"[batch][J] FAIL {job.job_id}: {e}")
    finally:
        worker.shutdown()


# --- driver ------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    stages = resolve_stages([s.strip() for s in args.stages.split(",") if s.strip()])
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    meshes = expand_meshes(args.meshes, args.limit)
    if not meshes:
        print(f"[batch] no meshes matched {args.meshes!r}")
        return 1
    print(f"[batch] {len(meshes)} meshes | stages={stages} | backends={backends} "
          f"| gpu_mode={args.gpu_mode}")

    job_params = {"batch": True, "backends": backends, "select": args.select,
                  "material_hint": args.material_hint, "gpu_mode": args.gpu_mode,
                  "keep_appearance": bool(args.keep_appearance),
                  "group_by": args.group_by}
    jobs = [find_or_create_job(args.jobs_root, m, args.resume, job_params) for m in meshes]

    if "render" in stages:
        stage_render(jobs)
    if "vlm" in stages:
        stage_vlm(jobs, args.material_hint, args.spec_cache_by_category)
    if "diffuse" in stages:
        stage_diffuse(jobs, args.candidates, args.seed, args.select,
                      args.diffusion_kind, args.strict)
    if "plan" in stages:
        stage_plan(jobs)
    if "texture" in stages:
        stage_texture(jobs, backends, args.seed, args.strict)
    if "eval" in stages:
        stage_eval(jobs, backends)
        stage_refine(jobs, backends, args.seed, force=args.refine)
    if "judge" in stages:
        stage_judge(jobs, backends)
        if jobs:
            # Refresh the index so its winner column reflects the verdicts.
            from pbr_texture_pipeline import eval as E
            idx = E.write_index(str(jobs[0].root.parent))
            print(f"[batch][J] index refreshed -> {idx['html']}")

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

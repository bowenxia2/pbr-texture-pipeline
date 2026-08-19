"""Batch CLI: many assets, unattended, stage-major (PRD section 8).

    conda run -n trellis2 python -m pbr_texture_pipeline.batch \\
      --assets 'partnet_mobility/*/mobility.urdf' --jobs-root jobs/ \\
      --stages render,vlm,diffuse,plan,texture,eval,judge \\
      --candidates 4 --seed 42 --material-hint "clean, factory-new" \\
      --select iou+clip --gpu-mode dual --limit 100 --resume

Execution is stage-major for model-load efficiency: the VLM loads once for all assets,
Qwen-Image loads once for all, and the backend loads once over all its approved pairs via
`--pairs-file`. Between GPU stages the previous model family is unloaded so Stage T (texturing
subprocess) has VRAM even on a single GPU (PRD section 7 stage-exclusion).

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
    ap.add_argument("--canny-scale", type=float, default=None,
                    help="canny ControlNet scale (default from config.yaml)")
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
            job.finish("render", params={"mask_coverage": info["mask_coverage"],
                                         "n_groups": info["n_groups"],
                                         "n_tiny": info["n_tiny"]})
            print(f"[batch][R] {job.job_id} coverage={info['mask_coverage']:.3f} "
                  f"groups={info['n_groups']} (tiny {info['n_tiny']})")
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
                  canny_scale, strict: bool) -> None:
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
                D.generate_candidates(job, prompt, negative, base_seed=seed, n=n,
                                      canny_scale=canny_scale)
                scores = D.score_candidates(job, prompt, n=n)
                sel = D.select_batch(scores)
                D.cutout(job, sel["index"])
                if (bool(_CFG.get("articulated.global.open_pose_pass", True))
                        and job.path("control", "open", "depth.png").is_file()
                        and job.path("control", "open", "canny.png").is_file()):
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


# --- Stage P (material plan) -------------------------------------------------
def stage_plan(jobs: list[JobDir]) -> None:
    """Per-part material plan via the VLM worker (one load over all jobs)."""
    from pbr_texture_pipeline.backends import registry
    from pbr_texture_pipeline.workers.ipc import WorkerClient

    todo = [j for j in jobs if not j.is_done("plan") and j.is_done("render")]
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
    """One trellis2 global pair per job, per-group bakes inside the adapter."""
    stage_texture_urdf(jobs, backends, seed, strict)


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
    """Stage T: one trellis2 global pair per job (one field decode on the merged mesh,
    per-group bakes inside the adapter, tiny groups included), then URDF + assembled.glb.
    Resume is group-granular (the pair lists only missing group GLBs).
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


# --- targeted retexturing (PRD section 7) ------------------------------------
def stage_refine(jobs: list[JobDir], backends: list[str], seed: int,
                 force: bool) -> None:
    """Measure-then-repair pass between eval and judge.

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
    todo = [j for j in jobs if j.status("eval") in ("done", "needs_review")]
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
        pairs_path = affected[0].root.parent / "_pairs_trellis2_refine.json"
        with open(pairs_path, "w") as f:
            json.dump(global_pairs, f, indent=2)
        print(f"[batch][F] trellis2: {len(global_pairs)} retexture pair(s)")
        registry.texture_pairs("trellis2", str(pairs_path))
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
    assets = expand_meshes(args.assets, args.limit)
    if not assets:
        print(f"[batch] no assets matched {args.assets!r}")
        return 1
    print(f"[batch] {len(assets)} assets | stages={stages} | backends={backends} "
          f"| gpu_mode={args.gpu_mode}")

    job_params = {"batch": True, "backends": backends, "select": args.select,
                  "material_hint": args.material_hint, "gpu_mode": args.gpu_mode,
                  "group_by": args.group_by}
    jobs = [find_or_create_job(args.jobs_root, m, args.resume, job_params) for m in assets]

    if "render" in stages:
        stage_render(jobs)
    if "vlm" in stages:
        stage_vlm(jobs, args.material_hint, args.spec_cache_by_category)
    if "diffuse" in stages:
        stage_diffuse(jobs, args.candidates, args.seed, args.select,
                      args.canny_scale, args.strict)
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

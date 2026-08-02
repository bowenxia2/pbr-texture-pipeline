"""Drive the entire Gradio pipeline through app.py's real tab callbacks.

This is the closest E2E to a browser user without a browser: it calls the exact
module-level callback functions each Tab wires up, in order, against a real mesh,
using the real WorkerManager (workers + GPU + subprocess backends).

  conda run -n trellis2 python scripts/drive_pipeline.py --mesh meshes/mug_e5e87ddb.glb \
      --backends trellis2 hunyuan
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pbr_texture_pipeline.app as app  # noqa: E402


class FakeSelect:
    """Stand-in for gr.SelectData; choose_candidate only reads .index."""
    def __init__(self, index):
        self.index = index


def _t(label, fn):
    t0 = time.time()
    print(f"\n=== {label} ===", flush=True)
    out = fn()
    print(f"[{label}] done in {time.time()-t0:.1f}s", flush=True)
    return out


def _urdf_upload_files(asset_dir: Path) -> list[str]:
    """The complete upload set for an asset: mobility.urdf + reference closure + optional
    metadata (mirrors what a user multi-selects in Tab 1)."""
    from pbr_texture_pipeline.articulated import urdf as U

    files = [asset_dir / "mobility.urdf"]
    files += [asset_dir / r for r in U.required_files(asset_dir / "mobility.urdf")]
    for name in ("meta.json", "semantics.txt", "result.json", "bounding_box.json"):
        if (asset_dir / name).is_file():
            files.append(asset_dir / name)
    return [str(f) for f in files]


def run_urdf(args) -> int:
    """Gate A8: drive the articulated pipeline through app.py's real tab callbacks."""
    import json as _json

    from pbr_texture_pipeline.jobdir import JobDir

    asset_dir = Path(args.urdf).resolve()
    files = _urdf_upload_files(asset_dir)
    print(f"[A8] asset {asset_dir.name}: {len(files)} upload files")

    # --- negative case: drop one OBJ; expect exact missing relpath, no job created ----------
    objs = [f for f in files if f.endswith(".obj")]
    incomplete = [f for f in files if f != objs[0]]
    before = {d.name for d in Path(args.jobs_root).iterdir()} \
        if Path(args.jobs_root).is_dir() else set()
    neg = _t("Tab1 upload_urdf (incomplete)", lambda: app.upload_urdf(incomplete))
    assert neg[0] is None, "incomplete upload must not create a job"
    missing_rel = Path(objs[0]).name
    assert missing_rel in neg[-1], f"status must name the missing file: {neg[-1]}"
    after = {d.name for d in Path(args.jobs_root).iterdir()} \
        if Path(args.jobs_root).is_dir() else set()
    assert before == after, "incomplete upload created a job dir"
    print(f"[A8] negative case OK: reported missing {missing_rel}, no job created")

    # --- Tab 1: complete upload -> articulated Stage R --------------------------------------
    r = _t("Tab1 upload_urdf", lambda: app.upload_urdf(files))
    job_id, model3d, contact, front_upd, depth, canny, appearance, status = r
    assert job_id, f"no job_id from upload_urdf: {status}"
    assert contact and Path(contact).is_file(), "no contact sheet"
    assert depth and Path(depth).is_file(), "no depth control"
    job = JobDir.load(Path(args.jobs_root) / job_id)
    assert job.kind == "urdf" and job.state.get("asset", {}).get("groups"), "no asset section"
    print(f"[A8] job={job_id} groups={len(job.state['asset']['groups'])} "
          f"appearance={appearance}")

    # --- Tabs 2-3 --------------------------------------------------------------------------
    c1 = _t("Tab2 chat_open", lambda: app.chat_open(job_id, ""))
    assert c1[0], "chat_open produced no history"
    c3 = _t("Tab2 chat_finalize", lambda: app.chat_finalize(job_id))
    assert c3[1] and "ref_prompt" in c3[1], "no finalized spec"
    prompt, negative, _ctrl = app.load_ref_defaults(job_id)
    g = _t("Tab3 generate_ref",
           lambda: app.generate_ref(job_id, prompt, negative, 42, 0.9, 4.0, "depth", False))
    assert g[0] and len(g[0]) == 4, "expected 4 candidates"
    ch = _t("Tab3 choose_candidate", lambda: app.choose_candidate(job_id, FakeSelect(0)))
    assert ch[1] and Path(ch[1]).is_file(), "no cutout"
    print(f"[A8] {app.approve_ref(job_id)}")

    # --- Tab 4: texture (Stage P runs automatically inside) ---------------------------------
    def _run_texture():
        last = None
        for update in app.run_texture(job_id, args.backends, 1024, 2048, False, 42):
            logs = update[0]
            tail = logs.strip().splitlines()[-1] if logs.strip() else ""
            print(f"[A8][T] {tail}", flush=True)
            last = update
        return last

    tex = _t("Tab4 run_texture (urdf)", _run_texture)
    rows = tex[1]
    assert rows, "per-group table is empty"
    job = JobDir.load(Path(args.jobs_root) / job_id)
    assert job.plan().is_file(), "Stage P plan.json missing"
    produced = [b for b in args.backends if job.output_glb(b) is not None]
    print(f"[A8] assembled outputs: {produced}; group rows={len(rows)}")
    assert produced, "no assembled.glb produced"
    for b in produced:
        assert job.textured_urdf(b).is_file(), f"no textured URDF for {b}"

    # --- Tab 4: judge ----------------------------------------------------------------------
    jr = _t("Tab4 run_judge", lambda: app.run_judge(job_id))
    verdict = _json.loads(jr[1])
    assert verdict.get("winner") in produced, f"bad judge verdict: {jr[0]}"
    print(f"[A8] winner={verdict['winner']} method={verdict.get('method')}")

    # --- Tab 5 + viewer routes -------------------------------------------------------------
    rows5 = app.refresh_jobs()
    this = [row for row in rows5 if row[0] == job_id]
    assert this and this[0][1] == "urdf", f"Tab 5 row wrong: {this}"

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    fapp = FastAPI()
    app.add_viewer_routes(fapp)
    client = TestClient(fapp)
    for b in produced:
        page = client.get(f"/viewer/{job_id}/{b}")
        assert page.status_code == 200 and "Joint Controls" in page.text, \
            f"viewer page failed for {b}: {page.status_code}"
        scene_glbs = [v for link in job.state["asset"]["groups"]
                      if (glb := job.textured_group_glb(b, link["group_id"])).is_file()
                      for v in [glb]]
        sub = client.get(f"/viewer/{job_id}/job/textured/{b}/groups/"
                         f"{job.state['asset']['groups'][0]['group_id']}.glb")
        assert sub.status_code in (200, 404), "job file route broken"
        print(f"[A8] /viewer/{job_id}/{b} OK ({len(scene_glbs)} group GLBs)")

    job = JobDir.load(Path(args.jobs_root) / job_id)  # run_judge wrote job.json since our load
    print(f"\n==== A8 PASS (backends {produced}, judge={job.status('judge')}) ====")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", help="flat mesh input (original pipeline drive)")
    ap.add_argument("--urdf", help="articulated asset dir (gate A8), e.g. partnet_mobility/19179")
    ap.add_argument("--jobs-root", default="jobs_pipetest")
    ap.add_argument("--backends", nargs="+", default=["trellis2"])
    args = ap.parse_args()
    if bool(args.mesh) == bool(args.urdf):
        ap.error("pass exactly one of --mesh or --urdf")

    app.JOBS_ROOT = args.jobs_root
    Path(args.jobs_root).mkdir(parents=True, exist_ok=True)

    if args.urdf:
        return run_urdf(args)

    # --- Sanity: the Blocks graph constructs (Tab wiring, gradio-6 compat) ---
    _t("build()", lambda: app.build())
    print("[build] Blocks constructed OK (all 5 tabs wired)", flush=True)

    # === Tab 1: Mesh -> Stage R ===============================================
    r = _t("Tab1 upload_mesh", lambda: app.upload_mesh(args.mesh))
    job_id, model3d, contact, front_upd, depth, canny, status = r
    assert job_id, "no job_id from upload_mesh"
    assert contact and Path(contact).is_file(), f"no contact sheet: {contact}"
    assert depth and Path(depth).is_file(), f"no depth control: {depth}"
    assert canny and Path(canny).is_file(), f"no canny control: {canny}"
    print(f"[Tab1] job={job_id}\n       {status}", flush=True)

    # re-render control maps (front panel 0 = canonical, no re-pose)
    d2 = _t("Tab1 rerender_controls",
            lambda: app.rerender_controls(job_id, "0", 0.0, 0.0))
    assert d2[0] and Path(d2[0]).is_file(), "rerender produced no depth"
    print(f"[Tab1] {d2[2]}", flush=True)

    # === Tab 2: Appearance chat -> Stage V ====================================
    c1 = _t("Tab2 chat_open", lambda: app.chat_open(job_id, ""))
    history, caption, spec_str, cstatus = c1
    assert history, "chat_open produced no history"
    assert caption, "chat_open produced no mesh caption"
    print(f"[Tab2] {cstatus}\n       caption={caption[:120]}\n       spec={spec_str[:200]}",
          flush=True)

    c2 = _t("Tab2 chat_send",
            lambda: app.chat_send(job_id, "Make it look a bit more worn and realistic.", history))
    print(f"[Tab2] send: {c2[3]}", flush=True)

    c3 = _t("Tab2 chat_finalize", lambda: app.chat_finalize(job_id))
    final_spec = c3[1]
    assert final_spec and "ref_prompt" in final_spec, "no finalized spec"
    print(f"[Tab2] {c3[2]}", flush=True)

    # === Tab 3: Reference image -> Stage D ====================================
    ld = _t("Tab3 load_ref_defaults", lambda: app.load_ref_defaults(job_id))
    prompt, negative, ctrl = ld
    assert prompt, "spec had no ref_prompt for the UI"
    print(f"[Tab3] prompt={prompt[:80]}", flush=True)

    g = _t("Tab3 generate_ref",
           lambda: app.generate_ref(job_id, prompt, negative, 42, 0.8, 6.0, "depth", False))
    gallery, gstatus, seed_upd = g
    assert gallery and len(gallery) == 4, f"expected 4 candidates, got {gallery}"
    print(f"[Tab3] {gstatus}", flush=True)

    # choose candidate #0 (simulates clicking the gallery)
    ch = _t("Tab3 choose_candidate",
            lambda: app.choose_candidate(job_id, FakeSelect(0)))
    chosen, cutout, chstatus = ch
    assert cutout and Path(cutout).is_file(), f"no cutout: {cutout}"
    print(f"[Tab3] {chstatus}", flush=True)

    ap_ = _t("Tab3 approve_ref", lambda: app.approve_ref(job_id))
    print(f"[Tab3] {ap_}", flush=True)

    # === Tab 4: Texture & review -> Stage T ===================================
    def _run_texture():
        last = None
        for update in app.run_texture(job_id, args.backends, 1024, 2048, False, 42):
            logs = update[0]
            # print only the newest tail line to keep output readable
            tail = logs.strip().splitlines()[-1] if logs.strip() else ""
            print(f"[Tab4] {tail}", flush=True)
            last = update
        return last

    tex = _t("Tab4 run_texture", _run_texture)
    models = tex[1:]
    produced = [m for m in models if m]
    produced_backends = [b for b, m in zip(app.BACKENDS, models) if m]
    print(f"[Tab4] produced {len(produced)}/{len(args.backends)} GLBs: {produced}", flush=True)

    # === Tab 4: Judge -> Stage J ==============================================
    import json as _json
    from pbr_texture_pipeline.jobdir import JobDir

    jr = _t("Tab4 run_judge", lambda: app.run_judge(job_id))
    jstatus, verdict_str = jr[0], jr[1]
    verdict = _json.loads(verdict_str)
    assert verdict.get("winner"), f"judge produced no winner: {jstatus}"
    assert verdict.get("method") in ("vlm", "walkover", "fallback_default"), verdict
    job = JobDir.load(Path(args.jobs_root) / job_id)
    surviving = [b for b in app.BACKENDS if job.output_glb(b) is not None]
    assert set(surviving) == set(produced_backends), \
        (f"expected every produced backend's GLB to still be on disk, got {surviving} "
         f"(produced {produced_backends})")
    assert verdict["winner"] in surviving, \
        f"winner {verdict['winner']} has no surviving GLB, got {surviving}"
    assert job.final_glb() is not None and job.final_glb().is_file(), "final_glb missing"
    judged_ok = job.status("judge") in ("done", "needs_review")
    print(f"[Tab4] {jstatus}\n       method={verdict['method']} "
          f"final_glb={job.final_glb()}", flush=True)

    # === Tab 5: Jobs browser ==================================================
    rows = _t("Tab5 refresh_jobs", lambda: app.refresh_jobs())
    this = [row for row in rows if row[0] == job_id]
    assert this, "job not listed in Tab 5"
    assert this[0][3] == verdict["winner"], f"Tab 5 winner column mismatch: {this[0]}"
    print(f"[Tab5] row={this[0]}", flush=True)

    ok = len(produced) == len(args.backends) and bool(this) and judged_ok
    print(f"\n==== PIPELINE {'PASS' if ok else 'PARTIAL/FAIL'} "
          f"(backends {len(produced)}/{len(args.backends)}, "
          f"judge={job.status('judge')} winner={verdict['winner']}) ====", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 2
    finally:
        if app._MGR is not None:
            app._MGR.shutdown()
    raise SystemExit(rc)

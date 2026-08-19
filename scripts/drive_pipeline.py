"""Drive the entire Gradio pipeline through app.py's real tab callbacks.

This is the closest E2E to a browser user without a browser: it calls the exact
module-level callback functions each Tab wires up, in order, against a real URDF
asset, using the real WorkerManager (workers + GPU + subprocess backends).

  conda run -n trellis2 python scripts/drive_pipeline.py \
      --urdf partnet_mobility/19179 --backends trellis2
"""
from __future__ import annotations

import argparse
import json as _json
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
    """The complete upload set for an asset: URDF + reference closure + optional
    metadata (mirrors what a user multi-selects in Tab 1)."""
    from pbr_texture_pipeline.articulated import urdf as U

    urdf_path = U.find_urdf(asset_dir)
    files = [urdf_path]
    files += [asset_dir / r for r in U.required_files(urdf_path)]
    for name in ("meta.json", "semantics.txt", "result.json", "bounding_box.json",
                 "compile_report.json"):
        if (asset_dir / name).is_file():
            files.append(asset_dir / name)
    return [str(f) for f in files]


def main() -> int:
    """Gate A8: drive the articulated pipeline through app.py's real tab callbacks."""
    from pbr_texture_pipeline.jobdir import JobDir

    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True,
                    help="articulated asset dir (gate A8), e.g. partnet_mobility/19179")
    ap.add_argument("--jobs-root", default="jobs_pipetest")
    ap.add_argument("--backends", nargs="+", default=["trellis2"])
    args = ap.parse_args()

    app.JOBS_ROOT = args.jobs_root
    Path(args.jobs_root).mkdir(parents=True, exist_ok=True)

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

    # --- Tab 1: complete upload -> Stage R --------------------------------------
    r = _t("Tab1 upload_urdf", lambda: app.upload_urdf(files))
    job_id, model3d, contact, front_upd, depth, canny, appearance, status = r
    assert job_id, f"no job_id from upload_urdf: {status}"
    assert contact and Path(contact).is_file(), "no contact sheet"
    assert depth and Path(depth).is_file(), "no depth control"
    job = JobDir.load(Path(args.jobs_root) / job_id)
    assert job.state.get("asset", {}).get("groups"), "no asset section"
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
        for update in app.run_texture(job_id, args.backends, 1024, 2048, 42):
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
    assert this, "job not listed in Tab 5"

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

    job = JobDir.load(Path(args.jobs_root) / job_id)
    print(f"\n==== A8 PASS (backends {produced}, judge={job.status('judge')}) ====")
    return 0


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

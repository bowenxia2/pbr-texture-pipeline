"""Gradio wizard orchestrator. 3-tab workflow: Upload/Render/Enhance -> Texture -> Jobs.

One `gr.Blocks` wizard, one job per session. Session state is only `gr.State(job_id)`;
everything else lives on disk in the job dir, so sessions survive restarts and batch jobs are
browsable identically. Stage R uses pyrender (CPU-only), Stage T uses GPU.

    conda run -n trellis2 python -m pbr_texture_pipeline.app
"""
from __future__ import annotations

import json
import queue
import shutil
import threading
from pathlib import Path
from typing import Optional

import gradio as gr

from pbr_texture_pipeline.config import load_config
from pbr_texture_pipeline.jobdir import JobDir

_CFG = load_config()
JOBS_ROOT = "jobs"
BACKENDS = ["trellis2"]

_STREAM_DONE = object()


def _job(job_id: str) -> JobDir:
    return JobDir.load(Path(JOBS_ROOT) / job_id)


# --- Tab 1: Upload & Render --------------------------------------------------
def upload_urdf(file_paths: Optional[list]):
    """Create an articulated job from a multi-file URDF upload, then run Stage R
    (pyrender textured views). Returns (job_id, mesh_norm, gallery, status)."""
    import tempfile

    from pbr_texture_pipeline.articulated import stages as AS
    from pbr_texture_pipeline.articulated import urdf as U
    from pbr_texture_pipeline.jobdir import make_job_id

    empty = (None, None, None)
    if not file_paths:
        return (*empty, "Upload a .urdf file plus every file it references.")
    files = [Path(f) for f in file_paths]
    urdfs = [f for f in files if f.suffix.lower() == ".urdf"]
    if len(urdfs) != 1:
        return (*empty, f"Expected exactly one .urdf in the upload, got {len(urdfs)}.")

    urdf_name = urdfs[0].name
    by_basename: dict[str, Path] = {}
    dupes = set()
    for f in files:
        if f.name in by_basename:
            dupes.add(f.name)
        by_basename[f.name] = f

    _OPTIONAL_META = ("meta.json", "semantics.txt", "result.json", "bounding_box.json",
                      "compile_report.json")
    tmp = Path(tempfile.mkdtemp(prefix="pbr_texture_pipeline_urdf_"))
    try:
        shutil.copyfile(urdfs[0], tmp / urdf_name)
        while True:
            closure = U.required_files(tmp / urdf_name)
            missing = [r for r in closure if not (tmp / r).is_file()]
            progress = False
            for rel in missing:
                base = Path(rel).name
                if base in dupes:
                    return (*empty, f"Ambiguous upload: multiple files named '{base}' "
                                    f"(needed for {rel}). Rename or re-upload.")
                src = by_basename.get(base)
                if src is not None:
                    (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, tmp / rel)
                    progress = True
            if not progress:
                break
        missing = [r for r in U.required_files(tmp / urdf_name)
                   if not (tmp / r).is_file()]
        if missing:
            listing = "\n".join(f"- {r}" for r in missing[:20])
            more = f"\n... and {len(missing) - 20} more" if len(missing) > 20 else ""
            return (*empty, f"Missing {len(missing)} referenced file(s); no job created. "
                            f"Re-upload the complete set:\n{listing}{more}")
        for name in _OPTIONAL_META:
            if name in by_basename and not (tmp / name).is_file():
                shutil.copyfile(by_basename[name], tmp / name)

        category = ""
        if (tmp / "meta.json").is_file():
            try:
                category = json.loads((tmp / "meta.json").read_text()).get("model_cat") or ""
            except (OSError, json.JSONDecodeError):
                category = ""
        if not category:
            import xml.etree.ElementTree as ET
            try:
                robot_name = ET.parse(str(tmp / urdf_name)).getroot().get("name")
                if robot_name:
                    category = robot_name.replace("_", " ").strip()
            except Exception:  # noqa: BLE001
                pass
        urdf_in_tmp = tmp / urdf_name
        job = JobDir.create(JOBS_ROOT, urdf_in_tmp,
                            job_id=make_job_id(urdf_in_tmp, category))
        shutil.copytree(tmp, job.asset_dir())
        job.state["mesh_source"] = str(job.asset_dir() / urdf_name)
        job._touch_and_write()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    job.start("render")
    info = AS.render(job)
    job.finish("render", params={"n_groups": info["n_groups"],
                                  "n_tiny": info["n_tiny"]})

    gallery = []
    if job.render_view(0).is_file():
        gallery.append((str(job.render_view(0)), "Front view"))
    if job.render_front_white().is_file():
        gallery.append((str(job.render_front_white()), "Front (white bg)"))
    if job.render_canny(0).is_file():
        gallery.append((str(job.render_canny(0)), "Canny map"))
    status = (f"Job {job.job_id} - Stage R done. "
              f"{info['n_groups']} groups ({info['n_tiny']} tiny). "
              f"Click 'Enhance' to run VLM + ImageEdit, then proceed to texturing.")
    return job.job_id, str(job.mesh_norm()), gallery, status


# --- Tab 1b: Enhance (VLM + ImageEdit) ---------------------------------------
def run_enhance(job_id: str):
    """Run Stage V (VLM) then Stage E (ImageEdit or ImageGen) on the front panel.
    Returns (materials_text, enhanced_gallery, status)."""
    import traceback

    from pbr_texture_pipeline.articulated import stages as AS

    if not job_id:
        return "", [], "No job yet. Upload a URDF in Tab 1 first."
    job = _job(job_id)
    if not job.is_done("render"):
        return "", [], "Run render first."

    try:
        job.start("vlm")
        vlm_info = AS.vlm(job)
        classification = vlm_info.get("classification", "edit")
        job.finish("vlm", params={
            "materials_preview": vlm_info["materials"][:200],
            "classification": classification,
        })
        materials = vlm_info["materials"]
    except Exception as e:  # noqa: BLE001
        job.fail("vlm", traceback.format_exc())
        return "", [], f"VLM failed: {e}"

    try:
        job.start("imageedit")
        edit_info = AS.imageedit(job)
        path_taken = edit_info.get("path", classification)
        job.finish("imageedit", params={
            "enhanced": edit_info["enhanced"],
            "path": path_taken,
        })
    except Exception as e:  # noqa: BLE001
        job.fail("imageedit", traceback.format_exc())
        return materials, [], f"VLM done, but Stage E failed: {e}"

    gallery = []
    if job.render_front_white().is_file():
        gallery.append((str(job.render_front_white()), "Original"))
    if job.enhanced_view(0).is_file():
        label = "Generated" if path_taken == "generate" else "Enhanced"
        gallery.append((str(job.enhanced_view(0)), label))
    return materials, gallery, (
        f"Done ({path_taken} path). Materials: {materials[:100]}..."
    )


# --- Tab 2: Texture & Assembly -----------------------------------------------
def _group_rows(job: JobDir, backends: list) -> list:
    """Per-group status rows for the Tab 3 dataframe."""
    rows = []
    for g in job.state.get("asset", {}).get("groups", []):
        gid = g["group_id"]
        row = [gid, g["label"]]
        for b in backends:
            row.append("done" if job.textured_group_glb(b, gid).is_file() else "pending")
        rows.append(row)
    return rows


def _viewer_frames(job: JobDir, backends: list) -> str:
    """Joint-viewer iframes for every backend with a textured URDF."""
    frames = []
    for b in backends:
        if job.textured_urdf(b).is_file():
            frames.append(
                f'<div style="flex:1;min-width:420px"><div style="font-size:12px;'
                f'opacity:.7;padding:2px 0">{b} - joint viewer</div>'
                f'<iframe src="/viewer/{job.job_id}/{b}" '
                f'style="width:100%;height:480px;border:1px solid #333;border-radius:8px">'
                f'</iframe></div>')
    if not frames:
        return ""
    return '<div style="display:flex;gap:12px;flex-wrap:wrap">' + "".join(frames) + "</div>"


def _model_slots(glbs: dict) -> tuple:
    vals = list(glbs.values())
    return tuple((vals[i] if i < len(vals) else None) for i in range(len(BACKENDS)))


def _run_texture_urdf(job: JobDir, backends: list, seed: int):
    """Tab 3 texturing flow: one trellis2 global pair (one field decode, per-group bakes
    in the adapter, group-granular resume) -> assembly.
    Yields (logs, group_rows, iframes, *model_slots)."""
    from pbr_texture_pipeline.articulated import stages as AS
    from pbr_texture_pipeline.backends import registry

    logs = ""
    glbs: dict[str, str] = {}

    def _yield():
        return (logs, _group_rows(job, BACKENDS), _viewer_frames(job, BACKENDS),
                *_model_slots(glbs))

    runs = [("trellis2", "trellis2")] if "trellis2" in backends else []
    for backend, adapter in runs:
        job.start("texture", seed=int(seed))
        pair = AS.global_texture_pair(job, int(seed))
        pairs = [pair] if pair is not None else []
        logs += ("\n=== trellis2 (global): 1 job pair (one field decode, per-group "
                 "bakes) ===\n" if pairs
                 else "\n=== trellis2 (global): all groups already textured ===\n")
        yield _yield()

        if pairs:
            pairs_path = job.path("textured", backend, "_pairs_urdf.json")
            pairs_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pairs_path, "w") as f:
                json.dump(pairs, f, indent=2)

            q: "queue.Queue" = queue.Queue()
            holder: dict = {}

            def _work(b=adapter, p=str(pairs_path)):
                try:
                    holder["res"] = registry.texture_pairs(
                        b, p, on_line=lambda text, cr: q.put((text, cr)))
                except Exception as exc:  # noqa: BLE001
                    holder["exc"] = exc
                finally:
                    q.put(_STREAM_DONE)

            th = threading.Thread(target=_work, daemon=True)
            th.start()
            live = ""
            while True:
                item = q.get()
                if item is _STREAM_DONE:
                    break
                text, cr = item
                if cr:
                    live = text
                else:
                    logs += text + "\n"
                    live = ""
                yield logs + live, _group_rows(job, BACKENDS), \
                    _viewer_frames(job, BACKENDS), *_model_slots(glbs)
            th.join()
            pairs_path.unlink(missing_ok=True)
            if "exc" in holder:
                logs += f"{backend}: ERROR launching adapter - {holder['exc']}\n"
                job.finish("texture", "error")
                yield _yield()
                continue

        try:
            res = AS.assemble_backend(job, backend)
            non_tiny = [g for g in job.state["asset"]["groups"] if not g.get("tiny")]
            ok_non_tiny = sum(1 for g in non_tiny
                              if res["groups"].get(g["group_id"]) == "ok")
            status = ("done" if ok_non_tiny == len(non_tiny)
                      else "needs_review" if res["n_ok"] > 0 else "error")
            prev = job.stage("texture").get("params", {}).get("backends", {})
            prev[backend] = {"ok": status == "done", "n_ok": res["n_ok"],
                             "n_total": res["n_total"], "glb_path": res["assembled_glb"]}
            job.finish("texture", status, params={"backends": prev})
            logs += (f"{backend}: {res['n_ok']}/{res['n_total']} groups ok -> {status}; "
                     f"assembled={res['assembled_glb']}\n")
            if res["assembled_glb"]:
                glbs[backend] = res["assembled_glb"]
        except Exception as exc:  # noqa: BLE001
            job.finish("texture", "error")
            logs += f"{backend}: assembly FAILED - {exc}\n"
        yield _yield()


def run_texture(job_id: str, backends: list, resolution: int, texture_size: int,
                seed: int):
    """Texture on selected backends; stream logs and reveal one Model3D viewer per backend."""
    if not job_id:
        yield "No job yet.", [], "", *_model_slots({})
        return
    job = _job(job_id)
    if not (job.enhanced_view(0).is_file() or job.render_view(0).is_file()):
        yield "Run render (and optionally enhance) in Tab 1 first.", [], "", *_model_slots({})
        return
    yield from _run_texture_urdf(job, backends, seed)


# --- Tab 4: Jobs browser -----------------------------------------------------
def refresh_jobs():
    rows = []
    for job in JobDir.load_all(JOBS_ROOT):
        statuses = {s: job.status(s) for s in ("render", "vlm", "imageedit", "texture")}
        flagged = any(v in ("needs_review", "error") for v in statuses.values())
        textured = [b for b in BACKENDS if job.output_glb(b) is not None]
        category = (job.state.get("asset") or {}).get("category", "")
        rows.append([job.job_id, category, ",".join(textured),
                     json.dumps(statuses), "yes" if flagged else ""])
    return rows


def toggle_flag(job_id: str, flag: bool):
    if not job_id:
        return "No job selected."
    job = _job(job_id)
    job.set_param("reviewed", not flag)
    job.set_param("flagged", bool(flag))
    return f"{job.job_id}: flagged={flag} written to job.json."


# --- Blocks ------------------------------------------------------------------
def build() -> gr.Blocks:
    with gr.Blocks(title="pbr-texture-pipeline") as demo:
        gr.Markdown("# pbr-texture-pipeline - multi-ref TRELLIS.2")
        job_state = gr.State(None)

        with gr.Tabs():
            # Tab 1: Upload, Render & Enhance
            with gr.Tab("1. Upload, Render & Enhance"):
                with gr.Row():
                    with gr.Column():
                        urdf_up = gr.Files(
                            label="Upload asset: .urdf + every referenced file",
                            file_count="multiple")
                        model3d = gr.Model3D(label="Normalized mesh")
                    with gr.Column():
                        render_gallery = gr.Gallery(
                            label="Front view + depth", columns=2, height=420)
                mesh_status = gr.Markdown("Upload a URDF asset to begin.")

                urdf_up.change(
                    upload_urdf, [urdf_up],
                    [job_state, model3d, render_gallery, mesh_status])

                gr.Markdown("### Enhance (VLM + ImageEdit)")
                enhance_btn = gr.Button("Enhance front panel")
                materials_out = gr.Textbox(label="Material description", lines=2)
                enhance_gallery = gr.Gallery(
                    label="Original vs Enhanced", columns=2, height=420)
                enhance_status = gr.Markdown("")
                enhance_btn.click(
                    run_enhance, [job_state],
                    [materials_out, enhance_gallery, enhance_status])

            # Tab 2: Texture & Assembly
            with gr.Tab("2. Texture & Assembly"):
                backends_cb = gr.CheckboxGroup(BACKENDS, value=["trellis2"], label="Backends")
                with gr.Row():
                    res_sl = gr.Slider(512, 2048, 1024, step=256, label="Resolution")
                    tex_sl = gr.Slider(1024, 2048, 2048, step=1024, label="Texture size")
                    tseed = gr.Number(label="Seed", value=42, precision=0)
                run_btn = gr.Button("Run texturing")
                tex_logs = gr.Textbox(label="Logs", lines=12, max_lines=12)
                group_table = gr.Dataframe(
                    headers=["group", "label", *BACKENDS],
                    label="Per-group status", interactive=False, wrap=True)
                viewer_frames = gr.HTML(label="Joint viewers")
                with gr.Row():
                    viewers = [gr.Model3D(label=f"Output {chr(65 + i)}")
                               for i in range(len(BACKENDS))]
                run_btn.click(run_texture,
                              [job_state, backends_cb, res_sl, tex_sl, tseed],
                              [tex_logs, group_table, viewer_frames, *viewers])

            # Tab 3: Jobs
            with gr.Tab("3. Jobs"):
                refresh_btn = gr.Button("Refresh")
                jobs_table = gr.Dataframe(
                    headers=["job_id", "category", "textured", "statuses", "flagged"],
                    label="Jobs", interactive=False, wrap=True)
                with gr.Row():
                    flag_job = gr.Textbox(label="job_id to flag")
                    flag_val = gr.Checkbox(False, label="flagged")
                    flag_btn = gr.Button("Write flag")
                flag_status = gr.Markdown("")
                refresh_btn.click(refresh_jobs, [], [jobs_table])
                flag_btn.click(toggle_flag, [flag_job, flag_val], [flag_status])

    return demo


# --- joint-viewer routes: mounted on Gradio's underlying FastAPI app ----------
def add_viewer_routes(app) -> None:
    """Mount /viewer/{job_id}/{backend} (+ asset/job file subroutes) on a FastAPI app."""
    from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

    from pbr_texture_pipeline.articulated import viewer as V

    def _viewer_job(job_id: str) -> Optional[JobDir]:
        root = Path(JOBS_ROOT) / Path(job_id).name
        if not (root / "job.json").is_file():
            return None
        return JobDir.load(root)

    @app.get("/viewer/{job_id}/{backend}")
    def viewer_page(job_id: str, backend: str):
        job = _viewer_job(job_id)
        if job is None:
            return PlainTextResponse("no such job", status_code=404)
        if backend not in BACKENDS or not job.textured_urdf(backend).is_file():
            return PlainTextResponse(f"no textured URDF for {backend}", status_code=404)
        return HTMLResponse(V.make_html(V.build_scene_data(job, backend)))

    @app.get("/viewer/{job_id}/asset/{relpath:path}")
    def viewer_asset(job_id: str, relpath: str):
        return _viewer_file(job_id, "asset", relpath)

    @app.get("/viewer/{job_id}/job/{relpath:path}")
    def viewer_jobfile(job_id: str, relpath: str):
        return _viewer_file(job_id, "job", relpath)

    def _viewer_file(job_id: str, kind: str, relpath: str):
        job = _viewer_job(job_id)
        if job is None:
            return PlainTextResponse("no such job", status_code=404)
        p = V.resolve_file(job, kind, relpath)
        if p is None:
            return PlainTextResponse("not found", status_code=404)
        return FileResponse(str(p), media_type=V.mime_for(p))


def main() -> None:
    import uvicorn
    from fastapi import FastAPI

    demo = build()
    demo.queue()
    fastapi_app = FastAPI()
    add_viewer_routes(fastapi_app)
    gr.mount_gradio_app(fastapi_app, demo, path="/")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=int(_CFG.get("app.port", 7860)))


if __name__ == "__main__":
    main()

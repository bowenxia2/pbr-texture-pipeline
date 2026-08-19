"""Gradio wizard orchestrator (PRD section 7). Thin: zero CUDA in-process.

One `gr.Blocks` wizard, one job per session. Session state is only `gr.State(job_id)`;
everything else lives on disk in the job dir, so sessions survive restarts and batch jobs are
browsable identically. All GPU work goes through the persistent workers via `WorkerManager`.

The tab callbacks are plain module-level functions of (job_id, ...) returning updates, so they
can be driven headlessly (see scripts/drive_pipeline.py).

    conda run -n trellis2 python -m pbr_texture_pipeline.app            # launches on 0.0.0.0:7860
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
from pbr_texture_pipeline.workers.manager import WorkerManager

_CFG = load_config()
JOBS_ROOT = "jobs"
BACKENDS = ["trellis2"]

# Appearance-prompt defaults, mirrored from pbr_texture_pipeline.vlm so the orchestrator can build a usable
# fallback prompt without importing the CUDA/torch-heavy vlm module in-process.
_PROMPT_BOILERPLATE = "single object, centered, plain neutral gray background, soft even studio lighting"
_DEFAULT_NEGATIVE = ("cartoon, painting, illustration, text, watermark, cluttered background, "
                     "harsh shadows, strong reflections, people, multiple objects")

# Sentinel marking end-of-stream on the texture log queue (Tab 4).
_STREAM_DONE = object()

# One manager for the whole app; workers start lazily on first GPU op and stay resident.
_MGR: Optional[WorkerManager] = None


def mgr() -> WorkerManager:
    global _MGR
    if _MGR is None:
        _MGR = WorkerManager(gpu_mode=_CFG.get("gpu_mode", "dual"))
    return _MGR


def _job(job_id: str) -> JobDir:
    return JobDir.load(Path(JOBS_ROOT) / job_id)


def _spec_str(spec: dict) -> str:
    return json.dumps(spec, indent=2)


# --- Tab 1: Upload -----------------------------------------------------------
def upload_urdf(file_paths: Optional[list]):
    """Create an articulated job from a multi-file URDF upload (WS7).

    Validates the URDF's transitive reference closure against the uploaded set (matched by
    basename), rebuilds the asset's relative layout under input/asset/, then runs articulated
    Stage R via the imaging worker. Missing files are reported exactly; no job is created.
    Returns (job_id, mesh_norm, contact, front_radio, depth, canny, status).
    """
    import shutil
    import tempfile

    from pbr_texture_pipeline.articulated import urdf as U
    from pbr_texture_pipeline.jobdir import make_job_id

    empty = (None, None, None, gr.update(choices=[], value=None), None, None)
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
    r = mgr().render(str(job.root))
    # The worker process wrote the asset section into job.json (AS.render -> set_asset);
    # finish() rewrites the whole in-memory state, so reload before it clobbers that.
    job = JobDir.load(job.root)
    job.finish("render", params={"mask_coverage": r["mask_coverage"],
                                 "n_groups": r.get("n_groups"), "n_tiny": r.get("n_tiny")})
    choices = [str(i) for i in range(8)]
    status = (f"Articulated job {job.job_id} - Stage R done. "
              f"{r.get('n_groups')} groups ({r.get('n_tiny')} tiny), "
              f"mask coverage {r['mask_coverage']:.2f}. "
              "Proceed to Tab 2; the material plan runs automatically at texture time.")
    return (job.job_id, str(job.mesh_norm()), r["contact_sheet"],
            gr.update(choices=choices, value="0"),
            r["control"]["depth"], r["control"]["canny"], status)


def rerender_controls(job_id: str, front_index: str, yaw_nudge: float,
                      pitch_nudge: float):
    if not job_id:
        return None, None, "No job yet."
    fi = int(front_index) if front_index is not None else 0
    r = mgr().rerender(str(_job(job_id).root), front_index=fi, yaw_nudge=yaw_nudge,
                       pitch_nudge=pitch_nudge)
    reposed = fi != 0
    status = (f"Control maps re-rendered (front={fi}, reposed={reposed}, "
              f"nudge=({yaw_nudge:.0f},{pitch_nudge:.0f})). coverage {r['mask_coverage']:.2f}.")
    return r["control"]["depth"], r["control"]["canny"], status


# --- Tab 2: Appearance chat --------------------------------------------------
def chat_open(job_id: str, material_hint: str):
    if not job_id:
        return [], "", "{}", "No job yet - upload a mesh in Tab 1 first."
    job = _job(job_id)
    job.start("vlm")
    res = mgr().vlm_open(str(job.contact_sheet()), material_hint=material_hint or "",
                         job_root=str(job.root))
    spec = res["spec"] or {}
    caption = res.get("caption", "")
    mgr().vlm_save(str(job.root), spec, res["messages"], caption=caption)
    history = _messages_to_chat(res["messages"])
    return (history, caption, _spec_str(spec),
            "Auto-generated caption and spec. Chat to refine, or Finalize.")


def _ensure_chat_open(job: JobDir) -> list:
    """Return the transcript, seeding a VLM session (system prompt + contact sheet) if none
    exists yet. Without this, clicking Send/Finalize before 'Start chat' would run the VLM with
    no role instruction and no mesh image, so it answers as a generic assistant and never emits
    a [SPEC] block - leaving spec.json empty (which is why 'Load prompt from spec' loaded nothing)."""
    prev = _load_messages(job)
    if prev:
        return prev
    job.start("vlm")
    res = mgr().vlm_open(str(job.contact_sheet()), material_hint="", job_root=str(job.root))
    mgr().vlm_save(str(job.root), res["spec"] or {}, res["messages"],
                   caption=res.get("caption", ""))
    return res["messages"]


def chat_send(job_id: str, user_text: str, history: list):
    if not job_id or not user_text:
        return history, gr.update(), "", gr.update()
    job = _job(job_id)
    prev = _ensure_chat_open(job)
    res = mgr().vlm_regenerate(prev, user_text)
    spec = res["spec"]
    mgr().vlm_save(str(job.root), spec or _load_spec(job), res["messages"])
    return (_messages_to_chat(res["messages"]),
            _spec_str(spec) if spec else gr.update(), "",
            "Spec updated." if spec else "No spec block this turn; keep chatting or Finalize.")


def chat_finalize(job_id: str):
    if not job_id:
        return [], "{}", "No job yet."
    job = _job(job_id)
    prev = _ensure_chat_open(job)
    res = mgr().vlm_finalize(prev)
    spec = res["spec"] or _load_spec(job)
    mgr().vlm_save(str(job.root), spec, res["messages"])
    job.finish("vlm")
    return _messages_to_chat(res["messages"]), _spec_str(spec), "Spec finalized and saved."


def save_spec_edits(job_id: str, spec_text: str):
    if not job_id:
        return "No job yet."
    try:
        spec = json.loads(spec_text)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    job = _job(job_id)
    job.write_json(job.spec(), spec)
    return "Edited spec saved to disk (human has the last word)."


def _messages_to_chat(messages: list) -> list:
    """Worker message list -> gr.Chatbot(type='messages') entries (drop system, strip images)."""
    out = []
    for m in messages:
        if m.get("role") == "system":
            continue
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(c.get("text", "[image]") if isinstance(c, dict) else str(c)
                               for c in content)
        out.append({"role": m["role"], "content": content})
    return out


def _load_messages(job: JobDir) -> list:
    """Read the saved transcript, unwrapping the {"messages": [...]} envelope written by
    vlm.save_transcript (falling back gracefully if a bare list was stored)."""
    if job.transcript().is_file():
        data = job.read_json(job.transcript())
        if isinstance(data, dict):
            return data.get("messages", [])
        return data or []
    return []


def _load_spec(job: JobDir) -> dict:
    return job.read_json(job.spec()) if job.spec().is_file() else {}


# --- Tab 3: Reference image --------------------------------------------------
def load_ref_defaults(job_id: str):
    if not job_id:
        return "", "", None
    job = _job(job_id)
    spec = _load_spec(job)
    depth = str(job.control("depth")) if job.control("depth").is_file() else None
    prompt = spec.get("ref_prompt", "")
    negative = spec.get("negative_prompt", "") or _DEFAULT_NEGATIVE
    if not prompt:
        # No finalized spec (Tab 2 skipped or VLM produced no [SPEC]): still hand the user a
        # usable, prompt-discipline-compliant starting point instead of an empty box.
        cat = spec.get("category") or "object"
        prompt = (f"a photograph of a {cat} made of typical realistic materials, "
                  f"product photography, {_PROMPT_BOILERPLATE}")
    return prompt, negative, depth


def generate_ref(job_id: str, prompt: str, negative: str, seed: int, cn_scale: float,
                 canny_scale: float, guidance: float, reroll: bool):
    if not job_id:
        return None, "No job yet.", gr.update()
    job = _job(job_id)
    base = int(seed)
    if reroll:
        base = int(job.stage("diffuse").get("seed") or seed) + 4
    job.start("diffuse", seed=base)
    mgr().diffuse(str(job.root), prompt, negative, base_seed=base, n=4,
                  cn_scale=cn_scale, canny_scale=canny_scale, guidance=guidance)
    s = mgr().score(str(job.root), prompt, n=4)
    scores = s["scores"]
    gallery = []
    for sc in scores:
        clip = f"{sc['clip']:.3f}" if sc["clip"] is not None else "n/a"
        gallery.append((str(job.candidate(sc["index"])),
                        f"#{sc['index']} IoU {sc['iou']:.2f} CLIP {clip}"))
    sel = s["selection"]
    status = (f"Generated 4 candidates (base_seed={base}). Auto-pick #{sel['index']} "
              f"({sel['reason']}). Click a candidate to choose + cut out.")
    return gallery, status, gr.update(value=base)


def _global_open_pass(job: JobDir) -> bool:
    """True when this job textures with the open-pose pass on."""
    return bool(_CFG.get("articulated.global.open_pose_pass", True))


def choose_candidate(job_id: str, evt: gr.SelectData):
    if not job_id:
        return None, None, None, "No job yet."
    idx = evt.index if isinstance(evt.index, int) else evt.index[0]
    job = _job(job_id)
    mgr().cutout(str(job.root), int(idx))
    status = f"Chosen candidate #{idx}; RMBG cutout ready."
    open_rgba = None
    if (_global_open_pass(job)
            and job.path("control", "open", "depth.png").is_file()
            and job.path("control", "open", "canny.png").is_file()):
        # Pass B reference at the same seed/index, while the pipe is resident (Stage D
        # addition, PRD.md section 5.6).
        base = int(job.stage("diffuse").get("seed") or _CFG.get("diffusion.seed", 42))
        mgr().open_reference(str(job.root), base, int(idx))
        open_rgba = str(job.path("ref", "chosen_rgba_open.png"))
        status += " Open-pose (pass B) reference generated."
    return str(job.chosen()), str(job.chosen_rgba()), open_rgba, status


def approve_ref(job_id: str):
    if not job_id:
        return "No job yet."
    job = _job(job_id)
    if not job.chosen_rgba().is_file():
        return "No cutout yet - click a candidate first."
    job.finish("diffuse")
    return "Reference approved. Proceed to Tab 4 to texture."


# --- Tab 4: Texture & review -------------------------------------------------
def _original_mesh(job: JobDir) -> str:
    # If a non-zero front was chosen in Tab 1, Stage R wrote the re-posed mesh; texture that
    # orientation so it matches the control maps / reference (the backend then un-reposes the
    # output via camera.json R). Otherwise use the untouched original.
    if job.mesh_reposed().is_file():
        return str(job.mesh_reposed())
    cands = sorted(job.path("input").glob("original.*"))
    if not cands:
        raise FileNotFoundError("no input/original.* - re-upload the mesh")
    return str(cands[0])


def _group_rows(job: JobDir, backends: list) -> list:
    """Read-only per-group status rows for the Tab 4 dataframe (urdf jobs). The last column
    is the fraction of texels a global-mode bake took from the open-pose field (pass B),
    read from the trellis2 adapter metadata."""
    plan = job.read_json(job.plan()) if job.plan().is_file() else {}
    pass_b: dict = {}
    pa = job.path("textured", "trellis2", "global", "pass_a.json")
    if pa.is_file():
        pass_b = {gid: m.get("pass_b_frac")
                  for gid, m in job.read_json(pa).get("groups", {}).items()}
    rows = []
    for g in job.state.get("asset", {}).get("groups", []):
        gid = g["group_id"]
        material = plan.get("groups", {}).get(gid, {}).get("share_key", "")
        row = [gid, g["label"], material + (" (tiny->pbr)" if g.get("tiny") else "")]
        for b in backends:
            row.append("done" if job.textured_group_glb(b, gid).is_file() else "pending")
        frac = pass_b.get(gid)
        row.append(f"{frac * 100.0:.0f}%" if isinstance(frac, (int, float)) else "")
        rows.append(row)
    return rows


def _ref_slots(job: Optional[JobDir]) -> tuple:
    """(pass A, pass B) reference paths for the Tab 4 display."""
    if job is None:
        return None, None
    a = job.chosen_rgba()
    b = job.path("ref", "chosen_rgba_open.png")
    return (str(a) if a.is_file() else None), (str(b) if b.is_file() else None)


def _viewer_frames(job: JobDir, backends: list) -> str:
    """Joint-viewer iframes for every backend with a textured URDF (urdf jobs)."""
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


def _run_texture_urdf(job: JobDir, backends: list, seed: int):
    """Tab 4 texturing flow: Stage P (automatic) -> one trellis2 global pair (one field
    decode, per-group bakes in the adapter, group-granular resume) -> assembly.
    Yields (logs, group_rows, iframes, ref_a, ref_b, *slots)."""
    from pbr_texture_pipeline.articulated import stages as AS

    logs = ""
    glbs: dict[str, str] = {}

    def _yield():
        return (logs, _group_rows(job, BACKENDS), _viewer_frames(job, BACKENDS),
                *_ref_slots(job), *_model_slots(glbs))

    # Stage P runs invisibly on first need (decision 5); catalog fallback on failure.
    if not job.plan().is_file():
        logs += "=== material plan (Stage P, automatic) ===\n"
        yield _yield()
        try:
            job.start("plan")
            res = mgr().vlm_plan(str(job.root))
            job.finish("plan", "needs_review" if res["fallback"] else "done",
                       params={"plan_fallback": res["fallback"]})
            logs += (f"plan: {len(res['plan']['materials'])} materials, "
                     f"fallback={res['fallback']}\n")
        except Exception as exc:  # noqa: BLE001
            logs += f"plan: VLM failed ({exc}); using catalog fallback\n"
            # pbr_texture_pipeline.vlm is CUDA-free at import; only its model loaders touch torch.
            from pbr_texture_pipeline.vlm import catalog_fallback_plan
            cat = AS.effective_category(job)
            plan = catalog_fallback_plan(job.state["asset"], cat)
            plan.update({"fallback": True, "category": cat, "model": None})
            job.write_json(job.plan(), plan)
            job.finish("plan", "needs_review", params={"plan_fallback": True})
        yield _yield()

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
                    holder["res"] = mgr().texture_pairs(
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
                    _viewer_frames(job, BACKENDS), *_ref_slots(job), *_model_slots(glbs)
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
    """Texture on selected backends; stream logs and reveal one Model3D viewer per backend.
    Runs the articulated global flow (Stage P happens automatically)."""
    if not job_id:
        yield "No job yet.", [], "", None, None, *_model_slots({})
        return
    job = _job(job_id)
    if not job.chosen_rgba().is_file():
        yield "Approve a reference in Tab 3 first.", [], "", None, None, *_model_slots({})
        return
    yield from _run_texture_urdf(job, backends, seed)


def _model_slots(glbs: dict) -> tuple:
    vals = list(glbs.values())
    return tuple((vals[i] if i < len(vals) else None) for i in range(len(BACKENDS)))


def run_judge(job_id: str):
    """Stage J: judge the textured outputs and pick a winner. Both outputs are kept on disk.

    Sheets render in the imaging worker; the verdict call runs in the vlm worker; label
    assignment and finalize run here (CUDA-free, judge.py keeps its torch imports lazy).
    Returns (status_md, verdict_json_str, viewer_iframes, *model_slots) with only the winner's
    viewer filled by default; the runner-up's GLB is still on disk under textured/<backend>/.
    """
    if not job_id:
        return "No job yet.", "{}", "", *_model_slots({})
    from pbr_texture_pipeline import judge as J
    job = _job(job_id)

    if job.judge_verdict().is_file() and not job.is_done("judge"):
        # Crash recovery: verdict was written but finalize never completed. Re-finalize
        # (idempotent deletion), no sheets or VLM call needed.
        verdict = job.read_json(job.judge_verdict())
    else:
        prep = mgr().judge_sheets(str(job.root), BACKENDS)
        available = prep["available"]
        if not available:
            return ("No textured output to judge - run texturing first.", "{}", "",
                    *_model_slots({}))
        job.start("judge")
        if len(available) == 1:
            verdict = J.walkover_verdict(available[0],
                                         "only one textured output available")
        else:
            labels = J.label_assignment(job.job_id, available)
            spec = _load_spec(job)
            res = mgr().vlm_judge(prep["sheets"][labels["A"]], prep["sheets"][labels["B"]],
                                  ref=str(job.chosen()) if job.chosen().is_file() else None,
                                  category=spec.get("category", "object"))
            model_id = _CFG.get("models.vlm")
            if res["verdict"] is None:
                verdict = J.fallback_verdict(available, raw_response=res.get("raw", ""),
                                             model=model_id)
            else:
                verdict = J.vlm_verdict(res["verdict"], labels, res.get("raw", ""), model_id)

    out = J.finalize(job, verdict)
    winner = out["winner"]
    status = f"Winner: {winner} ({verdict.get('method')}, {verdict.get('confidence')} confidence)."
    if verdict.get("method") != "walkover":
        status += " Both outputs kept on disk."
    if verdict.get("reasoning"):
        status += f" Reasoning: {verdict['reasoning']}"
    if out["status"] == "needs_review":
        status += " NEEDS REVIEW: the VLM gave no valid verdict, config default_winner applied."
    glb = job.output_glb(winner)
    frames = _viewer_frames(job, [winner])
    return (status, json.dumps(verdict, indent=2), frames,
            *_model_slots({winner: str(glb)} if glb else {}))


# --- Tab 5: Jobs browser -----------------------------------------------------
def refresh_jobs():
    rows = []
    for job in JobDir.load_all(JOBS_ROOT):
        spec = _load_spec(job)
        statuses = {s: job.status(s)
                    for s in ("render", "vlm", "diffuse", "plan", "texture", "eval", "judge")}
        flagged = "needs_review" in statuses.values()
        textured = [b for b in BACKENDS if job.output_glb(b) is not None]
        category = spec.get("category") or (job.state.get("asset") or {}).get("category") or ""
        rows.append([job.job_id, category, ",".join(textured),
                     job.winner() or "", json.dumps(statuses), "yes" if flagged else ""])
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
        gr.Markdown("# pbr-texture-pipeline - VLM-guided texturing of articulated assets")
        job_state = gr.State(None)

        with gr.Tabs():
            # Tab 1
            with gr.Tab("1. Upload"):
                with gr.Row():
                    with gr.Column():
                        urdf_up = gr.Files(
                            label="Upload asset: .urdf + every referenced file",
                            file_count="multiple")
                        model3d = gr.Model3D(label="Blank mesh (normalized)")
                        front_radio = gr.Radio(
                            label="Front-facing panel (0-7)", choices=[],
                            info="Pick which contact-sheet panel faces front; the mesh is "
                                 "re-posed and re-rendered immediately.")
                        with gr.Row():
                            yaw_nudge = gr.Slider(-30, 30, 0, step=5, label="Yaw nudge (deg)")
                            pitch_nudge = gr.Slider(-20, 20, 0, step=5, label="Pitch nudge (deg)")
                        rerender_btn = gr.Button("Re-render (apply nudges)")
                    with gr.Column():
                        contact = gr.Image(label="Contact sheet (VLM view)")
                        with gr.Row():
                            depth_img = gr.Image(label="Depth control")
                            canny_img = gr.Image(label="Canny control")
                mesh_status = gr.Markdown("Upload a URDF asset to begin.")

                urdf_up.change(
                    upload_urdf, [urdf_up],
                    [job_state, model3d, contact, front_radio, depth_img, canny_img,
                     mesh_status])
                rerender_inputs = [job_state, front_radio, yaw_nudge, pitch_nudge]
                rerender_outputs = [depth_img, canny_img, mesh_status]
                # Selecting a front panel re-poses + re-renders immediately (.input fires only on
                # user selection, not the programmatic value set by upload_urdf).
                front_radio.input(rerender_controls, rerender_inputs, rerender_outputs)
                rerender_btn.click(rerender_controls, rerender_inputs, rerender_outputs)

            # Tab 2
            with gr.Tab("2. Appearance chat"):
                material_hint = gr.Textbox(label="Material hint (optional)", value="")
                open_btn = gr.Button("Auto-generate caption + spec (VLM sees the mesh)")
                caption_box = gr.Textbox(label="Mesh caption (auto-generated)",
                                         interactive=False, lines=4)
                chatbot = gr.Chatbot(label="Appearance chat", height=380)
                with gr.Row():
                    msg = gr.Textbox(label="Message", scale=4)
                    send_btn = gr.Button("Send", scale=1)
                finalize_btn = gr.Button("Finalize spec")
                spec_box = gr.Code(label="spec.json (editable - you have the last word)",
                                   language="json", value="{}")
                save_spec_btn = gr.Button("Save edited spec")
                chat_status = gr.Markdown("")

                open_btn.click(chat_open, [job_state, material_hint],
                               [chatbot, caption_box, spec_box, chat_status])
                send_btn.click(chat_send, [job_state, msg, chatbot],
                               [chatbot, spec_box, msg, chat_status])
                finalize_btn.click(chat_finalize, [job_state], [chatbot, spec_box, chat_status])
                save_spec_btn.click(save_spec_edits, [job_state, spec_box], [chat_status])

            # Tab 3
            with gr.Tab("3. Reference image"):
                with gr.Row():
                    with gr.Column():
                        ctrl_thumb = gr.Image(label="Depth control")
                        prompt_box = gr.Textbox(label="Prompt", lines=3)
                        negative_box = gr.Textbox(label="Negative prompt", lines=2)
                        with gr.Row():
                            seed_sl = gr.Number(label="Base seed", value=42, precision=0)
                            cn_sl = gr.Slider(0.5, 1.0, float(_CFG.get("diffusion.cn_scale")),
                                              step=0.05, label="Depth CN scale")
                            canny_sl = gr.Slider(0.0, 1.0, float(_CFG.get("diffusion.canny_scale")),
                                                 step=0.05, label="Canny CN scale")
                            guid_sl = gr.Slider(1.0, 10.0, float(_CFG.get("diffusion.guidance")),
                                                step=0.5, label="True CFG")
                        with gr.Row():
                            gen_btn = gr.Button("Generate")
                            reroll_btn = gr.Button("Reroll (+4 seeds)")
                        load_defaults_btn = gr.Button("Load prompt from spec")
                    with gr.Column():
                        cand_gallery = gr.Gallery(label="Candidates (click to choose)",
                                                  columns=2, height=420)
                        with gr.Row():
                            chosen_img = gr.Image(label="Chosen (RGB)")
                            cutout_img = gr.Image(label="Cutout (RGBA)")
                        open_cutout_img = gr.Image(
                            label="Open-pose reference (global mode, pass B)")
                        approve_btn = gr.Button("Approve reference")
                ref_status = gr.Markdown("")

                load_defaults_btn.click(load_ref_defaults, [job_state],
                                        [prompt_box, negative_box, ctrl_thumb])
                gen_btn.click(
                    lambda j, p, n, s, c, y, g: generate_ref(j, p, n, s, c, y, g, False),
                    [job_state, prompt_box, negative_box, seed_sl, cn_sl, canny_sl, guid_sl],
                    [cand_gallery, ref_status, seed_sl])
                reroll_btn.click(
                    lambda j, p, n, s, c, y, g: generate_ref(j, p, n, s, c, y, g, True),
                    [job_state, prompt_box, negative_box, seed_sl, cn_sl, canny_sl, guid_sl],
                    [cand_gallery, ref_status, seed_sl])
                cand_gallery.select(choose_candidate, [job_state],
                                    [chosen_img, cutout_img, open_cutout_img, ref_status])
                approve_btn.click(approve_ref, [job_state], [ref_status])

            # Tab 4
            with gr.Tab("4. Texture & review"):
                backends_cb = gr.CheckboxGroup(BACKENDS, value=["trellis2"], label="Backends")
                with gr.Row():
                    res_sl = gr.Slider(512, 2048, 1024, step=256, label="Resolution")
                    tex_sl = gr.Slider(1024, 2048, 2048, step=1024, label="Texture size")
                    tseed = gr.Number(label="Seed", value=42, precision=0)
                run_btn = gr.Button("Run texturing")
                tex_logs = gr.Textbox(label="Logs", lines=12, max_lines=12)
                with gr.Row():
                    ref_a_img = gr.Image(label="Pass A reference (rest pose)")
                    ref_b_img = gr.Image(label="Pass B reference (open pose)")
                group_table = gr.Dataframe(
                    headers=["group", "label", "material", *BACKENDS, "pass B texels"],
                    label="Per-group status", interactive=False, wrap=True)
                viewer_frames = gr.HTML(label="Joint viewers")
                with gr.Row():
                    viewers = [gr.Model3D(label=f"Output {chr(65 + i)}")
                               for i in range(len(BACKENDS))]
                run_btn.click(run_texture,
                              [job_state, backends_cb, res_sl, tex_sl, tseed],
                              [tex_logs, group_table, viewer_frames, ref_a_img, ref_b_img,
                               *viewers])
                judge_btn = gr.Button("Judge outputs (VLM picks a winner; both outputs kept)")
                judge_status = gr.Markdown("")
                judge_verdict_box = gr.Code(label="judge verdict", language="json", value="{}")
                judge_btn.click(run_judge, [job_state],
                                [judge_status, judge_verdict_box, viewer_frames, *viewers])

            # Tab 5
            with gr.Tab("5. Jobs"):
                refresh_btn = gr.Button("Refresh")
                jobs_table = gr.Dataframe(
                    headers=["job_id", "category", "textured", "winner", "statuses",
                             "flagged"],
                    label="Jobs", interactive=False, wrap=True)
                with gr.Row():
                    flag_job = gr.Textbox(label="job_id to flag")
                    flag_val = gr.Checkbox(False, label="flagged")
                    flag_btn = gr.Button("Write flag")
                flag_status = gr.Markdown("")
                refresh_btn.click(refresh_jobs, [], [jobs_table])
                flag_btn.click(toggle_flag, [flag_job, flag_val], [flag_status])

    return demo


# --- joint-viewer routes (WS7): mounted on Gradio's underlying FastAPI app -------
def add_viewer_routes(app) -> None:
    """Mount /viewer/{job_id}/{backend} (+ asset/job file subroutes) on a FastAPI app.

    The page requests `asset/...` and `job/...` RELATIVE to its own URL, which resolves to
    the subroutes below; pbr_texture_pipeline.articulated.viewer owns scene data + path resolution."""
    from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

    from pbr_texture_pipeline.articulated import viewer as V

    def _viewer_job(job_id: str) -> Optional[JobDir]:
        root = Path(JOBS_ROOT) / Path(job_id).name  # basename only: no traversal
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

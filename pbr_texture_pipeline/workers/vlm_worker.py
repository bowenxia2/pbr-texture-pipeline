"""Persistent VLM worker (PRD section 7): Qwen3.6-35B-A3B caption -> spec (Stage V).

Runs in the dedicated `vlm` env (vLLM; needs torch >= 2.8, which the trellis2 env cannot
provide) on GPU 0, kept as a SEPARATE process from the imaging worker so either can be
evicted independently (~20 GB AWQ weights + vLLM's pre-allocated KV/state pool).

The chat message history lives on the orchestrator side and is passed in/out each turn (it is
plain JSON: role + text, with image content as file-path strings), so this worker is stateless
between calls apart from the resident model.

Ops:
  open       {contact_sheet, material_hint}          -> {reply, spec, caption, messages}
  regenerate {messages, user_text}                   -> {reply, spec, messages}
  finalize   {messages}                              -> {reply, spec, messages}
  save       {job_root, spec, messages[, caption]}   -> persist spec/transcript(/caption) json
  batch      {job_root, contact_sheet, material_hint}-> run the auto caption->spec flow
  plan       {job_root}                              -> Stage P material plan (urdf jobs)
  judge      {sheet_a, sheet_b[, ref, category]}     -> {verdict: A/B dict or None, raw}
  refine     {job_root, candidates, crops}           -> grade retexture candidates (urdf)
  unload     {}                                      -> drop the engine (stage exclusion)
"""
from __future__ import annotations

from pbr_texture_pipeline.jobdir import JobDir
from pbr_texture_pipeline.workers.ipc import serve


def _open(args: dict) -> dict:
    from pbr_texture_pipeline import vlm
    ctx = {"appearance_sheet": None, "articulation_note": "", "keep_appearance": False,
           "metadata_note": ""}
    if args.get("job_root"):
        # urdf jobs: the interactive opening turn gets the same articulated context as batch
        # (appearance sheet + articulation summary + metadata note, WS3/WS7).
        ctx = vlm.articulated_context(JobDir.load(args["job_root"]))
    messages, reply, spec, caption = vlm.opening_turn(
        args["contact_sheet"], args.get("material_hint", ""),
        appearance_sheet=ctx["appearance_sheet"], articulation_note=ctx["articulation_note"],
        keep_appearance=ctx["keep_appearance"], metadata_note=ctx["metadata_note"])
    return {"reply": reply, "spec": spec, "caption": caption, "messages": messages}


def _regenerate(args: dict) -> dict:
    from pbr_texture_pipeline import vlm
    messages, reply, spec = vlm.regenerate(args["messages"], args["user_text"])
    return {"reply": reply, "spec": spec, "messages": messages}


def _finalize(args: dict) -> dict:
    from pbr_texture_pipeline import vlm
    messages, reply, spec = vlm.force_finalize(args["messages"])
    return {"reply": reply, "spec": spec, "messages": messages}


def _save(args: dict) -> dict:
    from pbr_texture_pipeline import vlm
    job = JobDir.load(args["job_root"])
    spec = args["spec"]
    if job.kind == "urdf" and isinstance(spec, dict):
        # The metadata category is authoritative: an interactive session can refine
        # materials, but cannot persist a spec whose category contradicts the dataset.
        cat = (job.state.get("asset") or {}).get("category")
        if cat:
            spec, mismatch = vlm.apply_category_metadata(spec, cat)
            if mismatch:
                print(f"[vlm] saved spec category {spec.get('vlm_category')!r} contradicts "
                      f"metadata category {cat!r}; forced to metadata")
    vlm.save_spec(job, spec)
    vlm.save_transcript(job, args["messages"])
    if args.get("caption"):
        vlm.save_caption(job, args["caption"])
    return {"spec": str(job.spec()), "transcript": str(job.transcript())}


def _batch(args: dict) -> dict:
    from pbr_texture_pipeline import vlm
    job = JobDir.load(args["job_root"])
    res = vlm.run_auto(job, args["contact_sheet"], material_hint=args.get("material_hint", ""))
    return {"spec": res["spec"], "caption": res["caption"], "fallback": res["fallback"]}


def _plan(args: dict) -> dict:
    from pbr_texture_pipeline import vlm
    job = JobDir.load(args["job_root"])
    res = vlm.run_plan(job)
    return {"plan": res["plan"], "fallback": res["fallback"]}


def _judge(args: dict) -> dict:
    from pbr_texture_pipeline import vlm
    return vlm.run_judge(args["sheet_a"], args["sheet_b"], args.get("ref"),
                         category=args.get("category", "object"))


def _refine(args: dict) -> dict:
    from pbr_texture_pipeline import vlm
    job = JobDir.load(args["job_root"])
    return vlm.run_refine(job, args["candidates"], args["crops"])


def _unload(_args: dict) -> dict:
    from pbr_texture_pipeline import vlm
    vlm.unload_model()
    return {"unloaded": True}


HANDLERS = {
    "open": _open,
    "regenerate": _regenerate,
    "finalize": _finalize,
    "save": _save,
    "batch": _batch,
    "plan": _plan,
    "judge": _judge,
    "refine": _refine,
    "unload": _unload,
}


if __name__ == "__main__":
    serve(HANDLERS, name="vlm")

"""Persistent imaging worker (PRD section 7): Stage R rendering + Stage D Qwen-Image/ControlNet + RMBG.

Runs in the `trellis2` env (one env now covers the renderer, Qwen-Image, and RMBG - see README
"Environment reality"), on GPU 1 in dual mode so the 20B Stage D model does not collide with the
VLM on GPU 0. The renderer (~1 GB) and RMBG stay resident; the Qwen-Image ControlNet pipeline
uses model CPU offload (peaks ~40 GB during a diffuse, then releases). Zero CUDA lives in the
Gradio process; it talks to this worker over the JSON-lines IPC in `ipc.py`.

Ops (args are plain JSON; every op operates on a job dir on disk):
  render        {job_root}                     -> Stage R: normalize + contact sheet + control maps
  rerender      {job_root, front_index, yaw_nudge, pitch_nudge}
                                               -> re-pose (front != 0) / nudge, re-render control maps
  diffuse       {job_root, prompt, negative, base_seed, n, cn_scale, guidance, kind}
                                               -> N candidates into ref/
  score         {job_root, prompt, n}          -> [{index, iou, clip}]
  cutout        {job_root, index}              -> chosen.png + chosen_rgba.png
  open_ref      {job_root, base_seed, index}   -> pass B reference for global-mode urdf jobs
  judge_sheets  {job_root, backends}           -> Stage J phase 1: 2x2 judge sheets per textured GLB
  unload        {}                             -> drop the Qwen-Image Stage D model (frees the texturing GPU)
"""
from __future__ import annotations

from pbr_texture_pipeline.jobdir import JobDir
from pbr_texture_pipeline.workers.ipc import serve


def _render(args: dict) -> dict:
    from pbr_texture_pipeline import rendering as R
    job = JobDir.load(args["job_root"])
    if job.kind == "urdf":
        from pbr_texture_pipeline.articulated import stages as AS
        info = AS.render(job)
        return {"contact_sheet": str(job.contact_sheet()),
                "mask_coverage": info["mask_coverage"],
                "control": {n: str(job.control(n)) for n in ("depth", "normal", "canny", "mask")},
                "camera_json": str(job.camera_json()),
                "appearance_sheet": (str(job.appearance_sheet())
                                     if job.appearance_sheet().is_file() else None),
                "n_groups": info.get("n_groups"), "n_tiny": info.get("n_tiny")}
    norm = R.load_and_normalize(job.state["mesh_source"], job)
    mesh_repr = R.to_mesh_repr(norm)
    R.render_contact_sheet(job, mesh_repr)
    info = R.render_control_maps(job, mesh_repr)
    return {"contact_sheet": str(job.contact_sheet()),
            "mask_coverage": info["mask_coverage"],
            "control": {n: str(job.control(n)) for n in ("depth", "normal", "canny", "mask")},
            "camera_json": str(job.camera_json())}


def _rerender(args: dict) -> dict:
    """Re-render control maps after a front-view choice / re-pose / nudge (Tab 1)."""
    import math
    import numpy as np
    from pbr_texture_pipeline import rendering as R

    job = JobDir.load(args["job_root"])
    norm = R.load_and_normalize(job.state["mesh_source"], job)

    # Choosing a non-zero front panel IS the request to re-pose it to canonical; there is no
    # useful "front != 0 without re-pose" (the control camera is fixed, so the choice would be a
    # silent no-op). front == 0 is already canonical, so re-pose is skipped.
    front_index = int(args.get("front_index", 0))
    repose = front_index != 0
    R_mat = None
    mesh = norm
    if repose:
        R_mat = R.repose_matrix(front_index)
        mesh = R.apply_repose(norm, R_mat)
        mesh.export(job.mesh_reposed())  # Stage T textures this reposed orientation
    else:
        # Drop a stale reposed mesh from a previous non-zero choice so Stage T falls back to
        # the original when the user returns to front 0.
        job.mesh_reposed().unlink(missing_ok=True)

    mesh_repr = R.to_mesh_repr(mesh)
    yaw = R.CANONICAL_YAW + math.radians(float(args.get("yaw_nudge", 0.0)))
    pitch = R.CANONICAL_PITCH + math.radians(float(args.get("pitch_nudge", 0.0)))
    info = R.render_control_maps(job, mesh_repr, yaw=yaw, pitch=pitch,
                                repose_applied=repose,
                                R_mat=np.asarray(R_mat) if R_mat is not None else None)
    return {"mask_coverage": info["mask_coverage"],
            "control": {n: str(job.control(n)) for n in ("depth", "normal", "canny", "mask")},
            "camera_json": str(job.camera_json())}


def _diffuse(args: dict) -> dict:
    from pbr_texture_pipeline import diffusion as D
    job = JobDir.load(args["job_root"])
    sidecars = D.generate_candidates(
        job, args["prompt"], args.get("negative", ""),
        base_seed=int(args["base_seed"]), n=int(args.get("n", 4)),
        cn_scale=args.get("cn_scale"), guidance=args.get("guidance"),
        kind=args.get("kind", "depth"), use_union=bool(args.get("use_union", False)))
    return {"candidates": [str(job.candidate(s["index"])) for s in sidecars],
            "sidecars": sidecars}


def _score(args: dict) -> dict:
    from pbr_texture_pipeline import diffusion as D
    job = JobDir.load(args["job_root"])
    scores = D.score_candidates(job, args["prompt"], n=int(args.get("n", 4)))
    sel = D.select_batch(scores)
    return {"scores": scores, "selection": sel}


def _cutout(args: dict) -> dict:
    from pbr_texture_pipeline import diffusion as D
    job = JobDir.load(args["job_root"])
    D.cutout(job, int(args["index"]))
    return {"chosen": str(job.chosen()), "chosen_rgba": str(job.chosen_rgba())}


def _open_ref(args: dict) -> dict:
    """Pass B reference (PRD_articulated_v2 Stage D addition): one Qwen-Image generation from
    control/open/depth.png at the same seed/index the rest-pose selection chose, plus its
    RMBG cutout -> ref/chosen_rgba_open.png."""
    from pbr_texture_pipeline import diffusion as D
    job = JobDir.load(args["job_root"])
    path = D.generate_open_reference(job, int(args["base_seed"]), int(args["index"]))
    return {"chosen_rgba_open": str(path)}


def _judge_sheets(args: dict) -> dict:
    """Stage J phase 1: render the multi-view judge sheets (the renderer lives here, not in
    the vlm worker)."""
    from pbr_texture_pipeline import judge as J
    job = JobDir.load(args["job_root"])
    return J.prepare(job, args["backends"])


def _unload(_args: dict) -> dict:
    """Drop the Qwen-Image Stage D pipeline (frees the texturing GPU before Stage T)."""
    from pbr_texture_pipeline import diffusion as D
    D.unload_pipe()
    return {"unloaded": True}


HANDLERS = {
    "render": _render,
    "rerender": _rerender,
    "diffuse": _diffuse,
    "score": _score,
    "cutout": _cutout,
    "open_ref": _open_ref,
    "judge_sheets": _judge_sheets,
    "unload": _unload,
}


if __name__ == "__main__":
    serve(HANDLERS, name="imaging")

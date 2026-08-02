# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

pbr-texture-pipeline textures untextured 3D meshes: render the blank mesh, have a VLM (Qwen3.6-35B-A3B) caption it and propose a category-aware appearance prompt, generate a pose-matched reference image (Qwen-Image 20B + depth ControlNet), drive two existing texturing backends (TRELLIS.2, Hunyuan3D-2.1) with the (mesh, reference) pair, then have the same VLM judge the two textured outputs and pick a winner with reasoning; both outputs are kept.
There are two job kinds: flat meshes (`kind: "mesh"`) and articulated PartNet-Mobility URDF assets (`kind: "urdf"`, added 2026-07-21).
Articulated jobs texture in global mode only (sole path since 2026-07-28): one TRELLIS.2 field decode over the merged rest-pose assembly (plus an open-pose decode so interiors get evidence), then a per-group atlas bake from the shared field; hunyuan has no articulated path, so urdf jobs are trellis2-only and Stage J records a walkover.
`PRD.md` is the single authoritative spec for the whole pipeline, both job kinds, the app, the batch CLI, and the gates; read it before making design changes.
There is no separate implementation-plan document and no second PRD; `PRD.md` is the only design doc.
`config.yaml` is the single source of truth for paths, model ids, envs, and stage defaults (including the `articulated:` section); never hardcode them.
When writing docs in this repo (PRDs, plans, reports), do not invent jargon or use niche tech slang: use standard technical terms where they are the correct terms and plain English otherwise, without dumbing anything down.

## Environment

`config.yaml` is the source of truth; PRD.md section 1.6 documents the same reality in full:

- `trellis2` env (Python 3.10) runs Stages R/D, the Gradio app, the batch CLI, and the trellis2 backend adapter.
- `vlm` env runs only the Stage V worker: Qwen3.6-35B-A3B AWQ under vLLM (torch >= 2.8, incompatible with trellis2's torch 2.6). Stage V is an automated caption->spec flow; the app chat remains as an override.
- `hunyuan3d` env runs only the Hunyuan adapter (custom_rasterizer, torch 2.5.1).
- `orianyv2` env (Python 3.11) runs only `scripts/orient_infer.py` (Orient-Anything-V2 front detection inside articulated Stage R).
- TRELLIS.2, Hunyuan3D-2.1, and Orient-Anything-V2 are git submodules of this repo (see README.md "Setup"); `config.yaml`'s `repos:` section resolves them by relative path, with `config.local.yaml` (gitignored) available to point at different checkouts.
- HF cache defaults to `.cache/huggingface` inside the repo (override via `config.local.yaml`'s `env.hf_cache`, e.g. to a shared/larger disk); all model weights resolve there. Never write anything under `/home`.
- HPC cluster with no SLURM; 2x A40 (46 GB each). `gpu_mode: dual|single` in config.yaml controls the VRAM strategy.

## Commands

There is no test suite, linter, or build step.
E2E verification is the `e2e-test` skill (`.claude/skills/e2e-test/SKILL.md`); everything else is the pipeline scripts below.

```bash
# Gradio app (5-tab wizard, 0.0.0.0:7860)
conda run -n trellis2 python -m pbr_texture_pipeline.app

# Batch CLI (stage-major: each model loads once over all meshes)
conda run -n trellis2 python -m pbr_texture_pipeline.batch \
  --meshes 'meshes_run/*.glb' --jobs-root jobs/ \
  --backends trellis2,hunyuan --stages render,vlm,diffuse,plan,texture,eval,judge --resume

# Batch CLI, articulated: a mobility.urdf glob creates urdf jobs (flags: --keep-appearance,
# --group-by {semantic,link}, --refine to execute targeted retexturing after eval)
conda run -n trellis2 python -m pbr_texture_pipeline.batch \
  --meshes 'partnet_mobility/*/mobility.urdf' --jobs-root jobs/ \
  --backends trellis2 --stages render,vlm,diffuse,plan,texture,eval,judge --resume

# Gate 1 conventions check (canonical front camera + depth polarity)
conda run -n trellis2 python scripts/verify_conventions.py --out jobs/_gate1_conventions

# Articulated gates A1/A12a (no GPU); with --jobs-root also A9/A11/A12b over Stage R output
conda run -n trellis2 python scripts/verify_articulated.py --jobs-root jobs/

# Articulated gate A10: per-group bake vs whole-mesh bake on one global-mode job (GPU)
conda run -n trellis2 python scripts/verify_global_bake.py jobs/<job_id>

# Joint-slider viewer for one articulated job (CUDA-free; the app embeds the same core)
conda run -n trellis2 python scripts/articulated_viewer.py jobs/<job_id> --backend trellis2 --port 8090

# One-time Stage D weight fetch (or --check to only report)
python -m scripts.download_controlnet

# Stage run inputs: symlink light meshes, decimate heavy ones into meshes_run/
conda run -n trellis2 python scripts/stage_run_meshes.py

# Browsable HTML gallery of textured results for a jobs root
python scripts/build_viewer.py <jobs-root>   # then: cd <jobs-root> && python -m http.server 8080
```

Stage names accept aliases R,V,D,P,T,E,J for render,vlm,diffuse,plan,texture,eval,judge.

## Architecture

Seven idempotent stages, each reading/writing only its per-mesh job directory under `jobs/<job_id>/`:
render (R) -> vlm (V) -> diffuse (D) -> plan (P) -> texture (T) -> eval (E) -> judge (J).
Stage P is the articulated per-group material plan (`vlm/plan.json`, `[PLAN]` block with catalog fallback), kept as material metadata (PBR hints, app display, VLM context); flat mesh jobs mark it done with `{"skipped": true}`.
Stage J (`pbr_texture_pipeline/judge.py` + `vlm.run_judge`) renders a 2x2 multi-view judge sheet per textured backend, has the Stage V VLM pick a winner with reasoning (A/B labels randomized per job to fight position bias), and writes `judge/verdict.json`; both backends' `textured/<backend>/` dirs are kept, and `JobDir.final_glb()` resolves to the winner's GLB for consumers that want a single best pick.
One textured output resolves as a walkover without a VLM call; a VLM failure/refusal after one retry falls back to `judge.default_winner` (config.yaml) with status `needs_review`.
`pbr_texture_pipeline/jobdir.py` owns the `job.json` state machine (per-stage status/params/seeds, schema v2 with top-level `kind` and an `asset` section for urdf jobs); it makes every stage resumable and lets interactive (Gradio) and batch modes share one code path.
The full job-dir layout is in the `jobdir.py` docstring and PRD section 2.

Articulated jobs (`pbr_texture_pipeline/articulated/`): `urdf.py` parses `mobility.urdf` into per-link visuals and joints, groups them by URDF `<visual name>` (semantic groups), does rest-pose FK (joints at q=0 clamped into limits) plus opened-pose FK (`link_world_transforms_at`), and rewrites the textured URDF + `assembled.glb`; only the URDF plus its transitive file closure is required - `semantics.txt`/`result.json`/`meta.json`/`bounding_box.json` are optional with fallbacks.
Stage R additionally records `input/face_ranges.json` (per-group face ranges into the merged mesh + norm params), exports the open-pose merged mesh, renders open-pose control maps, and writes per-view depth/visibility npz files; Orient-Anything-V2 picks the front panel when cross-panel agreement holds, else `articulated.front_panel`.
Stage T is one `trellis2_global` adapter run per job (`pbr_texture_pipeline/backends/trellis2_global_adapter.py`): decode the TRELLIS.2 field once for the merged rest-pose mesh and once for the open pose, bake every group its own UV atlas from the shared field (texels hidden at rest are re-sampled from the open-pose field with a blend band), map each bake to the link frame via the orchestrator-computed `to_link`, and bake tiny groups (below `articulated.min_group_faces`/`min_group_area_frac`) at `tiny_texture_size`.
Resume is group-granular (the pair lists only groups whose `textured/trellis2/groups/<gid>.glb` is absent); `output_glb` prefers `assembled.glb`, which routes Stage E/J, `final_glb`, and the viewers to the assembled result with no kind-branching.
Stage E writes articulated-state renders and `eval/diagnostics.json`; targeted retexturing (`--refine` or `articulated.refine.enabled`) moves VLM-confirmed bad groups aside and re-bakes them from a fresh-seed global decode.
`articulated/stages.py` holds the stage drivers shared by `batch.py` and `app.py`; the app's Tab 1 has a multi-file URDF upload that validates the reference closure, its Texture button runs Stage P automatically before the global bake, and Tab 4 embeds the joint-slider viewer (`articulated/viewer.py`) as FastAPI iframe routes.

Process model (`pbr_texture_pipeline/workers/`): the Gradio app (`app.py`) is a thin orchestrator with zero CUDA in-process.
GPU work happens in persistent worker subprocesses speaking JSON-lines over stdin/stdout (`ipc.py`): a VLM worker (GPU 0) and an imaging worker for rendering + Qwen-Image + RMBG (GPU 1).
`manager.py` owns GPU placement, VRAM polling via nvidia-smi (queue rather than OOM; shared node), and the unload/reload dance: Qwen-Image and texturing share GPU 1, so the manager drops the imaging worker's Qwen model before each texture run.

Backends (`pbr_texture_pipeline/backends/`): each adapter runs via subprocess (`conda run -n <env> ...`) with a hard cwd requirement per backend repo; `registry.py` maps backend name to `{script, env, cwd}`.
The adapter contract is `texture(mesh_path, image_rgba_path, out_dir, seed, camera_json, params) -> {glb_path, logs}`; adapters also accept `--pairs-file` so a batch loads each backend once.

Tab callbacks in `app.py` are plain module-level functions of `(job_id, ...)` so they can be driven headlessly (see `scripts/drive_pipeline.py`).
Heavy imports (torch, renderer, diffusers) are deferred inside functions so importing `pbr_texture_pipeline.app`/`pbr_texture_pipeline.batch` and `--help` never touch CUDA; preserve this when editing.

## Critical conventions and gotchas

- Canonical front camera is `yaw = pi, pitch = 0` (verified in Gate 1); after TRELLIS.2 normalization, glTF forward maps to internal -Y. Camera math lives in `pbr_texture_pipeline/rendering.py` and mirrors TRELLIS.2's `render_utils` (`r=2, fov=40` defaults).
- Depth control maps use MiDaS-style normalized inverse depth (near = white, far = black, background black); wrong polarity makes ControlNet produce inside-out objects.
- Hunyuan's rasterizer segfaults (exit 139) at CUDA teardown AFTER writing all outputs. Adapter success is judged by the emitted `[PBR_RESULT] ok:true` line plus the GLB existing on disk, never by subprocess exit code. Do not try to fix the segfault.
- Hunyuan's seed is hardcoded to 0 in its own code; seed sweeps only vary TRELLIS.2.
- Backends receive `chosen_rgba.png` (RMBG-2.0 cutout done by pbr-texture-pipeline); the alpha channel guarantees the same cutout everywhere, so backends never re-run rembg.
- `meshes/` holds pristine test meshes (never modify); `meshes_run/` is the staged/decimated run set produced by `stage_run_meshes.py`; `partnet_mobility/` holds the pristine articulated test assets (never modify).
- The PartNet-Mobility URDF world frame is Z-up (wheels/feet at minimum z, verified on all 5 test assets); every articulated world-frame path uses the internal-frame transforms with `up="z"` (center+scale, no glTF Y-up axis swap). Feeding that geometry through the default Y-up swap tips assets onto their side and turns every yaw orbit (contact sheet, turntable, judge sheet) into a tumble.
- PartNet-Mobility assets face contact-sheet panel 2 in the assembled URDF world frame (re-verified in the Z-up frame on all 5 test assets, 2026-07-28); articulated Stage R pre-poses by `articulated.front_panel` so the canonical camera sees the front, recorded in `camera.json` (`repose_applied`/`R`) like any re-pose.
- Articulated frame handling: the global adapter bakes in the TRELLIS field frame and maps each group to its link frame with the pair's `to_link` matrix (composed orchestrator-side; gate A9 pins the frame chain it inverts). A `to_link` bounds mismatch fails the group loudly; bbox shrinkage up to 5% of extent is tolerated because `cumesh.uv_unwrap` welds sliver faces on low-face-count groups (seen on 48-face PartNet parts).
- The adapter merges `pass_a.json`/`pass_b.json` metadata on partial (resume) runs instead of rewriting them; keep that when editing, or resumed jobs lose the earlier groups' bake stats that Stage E diagnostics read.
- Hunyuan has no articulated path: urdf jobs are trellis2-only by design and Stage J records a walkover; do not "fix" this by feeding hunyuan per-part inputs (that path was removed 2026-07-28).
- TRELLIS.2 forces `alphaMode='OPAQUE'` on its output GLBs, so glass parts (windows, cabinet panes) cannot be textured as transparent; an opaque frosted look is the accepted compromise, not a bug to fix.
- For urdf jobs, the dataset category (`meta.json` `model_cat`) is authoritative ground truth for Stage V, never something the VLM re-classifies: `articulated_context` feeds a metadata note to both the caption and spec turns, and `apply_category_metadata` force-sets `spec.category` afterward (VLM disagreement is kept only as `spec.vlm_category`, `spec.category_source` is always `"metadata"`). Gate A16 in `scripts/verify_articulated.py` checks this invariant.

## Key external references

- `TRELLIS.2/trellis2/` (submodule) - renderer (`renderers/mesh_renderer.py`) and normalization (`pipelines/trellis2_texturing.py`) that pbr-texture-pipeline reuses.
- Prior internal R&D (`pbr_compare`, `trellis_pbr`; not part of this repo) originated the verified backend APIs, env/cwd incantations, and the VLM chat patterns `pbr_texture_pipeline/vlm.py` ports.
## Maintaining this file

Update CLAUDE.md only when a change introduces a durable
repository-wide convention, architectural constraint, validation command
or product invariant.

Do not add temporary implementation notes, feature progress or
information that can be inferred directly from the code.

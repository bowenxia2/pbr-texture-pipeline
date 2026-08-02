---
name: e2e-test
description: Run pbr-texture-pipeline's full end-to-end pipeline test (all stages, real GPUs, real backends) via scripts/drive_pipeline.py and judge pass/fail, in flat-mesh mode (--mesh) or articulated URDF mode (--urdf). Use after any change to the pipeline, app callbacks, workers, backend adapters, or the articulated package.
---

# pbr-texture-pipeline E2E pipeline test

Drives `app.py`'s real tab callbacks in order (Tab 1 render -> Tab 2 VLM chat -> Tab 3 reference image -> Tab 4 texturing -> Tab 4 judge -> Tab 5 jobs browser) against a real input, using the real WorkerManager, GPU workers, and backend subprocesses.
This is the closest thing to a browser user without a browser, and the project's primary verification (there is no unit-test suite).
Two modes: `--mesh` for the original flat-mesh pipeline (6 stages; `plan` is auto-skipped) and `--urdf` for an articulated PartNet-Mobility asset (7 stages including Stage P, per-group texturing, textured URDF + assembled.glb, joint-viewer routes).
Pick the mode that matches the change: articulated changes need `--urdf`; shared-pipeline changes are usually covered faster by `--mesh`.

## Pre-checks

1. Confirm GPU headroom before launching; the run needs roughly 17 GB on GPU 0 (VLM) and up to ~40 GB on GPU 1 (Qwen-Image, then texturing):

   ```bash
   nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
   ```

   The manager queues rather than OOMs on a busy node, so a contended GPU means a slower run, not a broken one.
2. Cheap wiring smoke (seconds, no GPU) - catches import/signature/Blocks-graph breakage before committing to the long run. Run from the repo root:

   ```bash
   conda run -n trellis2 python -c "import pbr_texture_pipeline.app as app; app.build(); print('build ok')"
   ```

## Run (flat mesh)

Run from the repo root:

```bash
conda run -n trellis2 python scripts/drive_pipeline.py \
  --mesh meshes/mug_e5e87ddb.glb \
  --jobs-root jobs_pipetest \
  --backends trellis2 hunyuan
```

- Run it in the background and poll the log; a cold run takes roughly 20-45 minutes (VLM load ~2 min, 4 diffusion candidates at ~1-2 min each, then several minutes per texture backend).
- `--backends` accepts any subset; use `--backends trellis2` for a faster check when Hunyuan is not in question.
- Any small mesh in `meshes/` works; the mug is the conventional choice.

## Run (articulated URDF)

Run from the repo root:

```bash
conda run -n trellis2 python scripts/drive_pipeline.py \
  --urdf partnet_mobility/19179 \
  --jobs-root jobs_pipetest \
  --backends trellis2 hunyuan
```

- Exercises the app's URDF path end to end: the multi-file `upload_urdf` callback (including a negative case that must report the exact missing relpath and create no job), articulated Stage R, chat/spec/reference as usual, `run_texture` with automatic Stage P + the trellis2_global bake (one field decode, per-group atlases) + assembly, judge, Tab 5, and the `/viewer/<job_id>/<backend>` FastAPI joint-viewer routes.
- `partnet_mobility/19179` (blank table) is the conventional choice; `8930` (textured door) exercises the appearance-sheet path.
  Texturing is one trellis2_global run per job (one field decode plus a bake per group; tens of minutes on a many-group asset), so group count still drives runtime.
- Before this (or after any change to `pbr_texture_pipeline/articulated/`), run the free no-GPU gate first: `conda run -n trellis2 python scripts/verify_articulated.py` must report all assets OK.
- Hunyuan has no articulated path: urdf jobs are trellis2-only and the judge records a walkover for them by design; do not report that as a failure.

## Pass criteria (flat mesh)

- The script prints `==== PIPELINE PASS (backends N/N, judge=done winner=<backend>) ====` and exits 0.
- After the judge step, BOTH backends' GLBs remain under `jobs_pipetest/<job_id>/textured/` (`textured.glb` for trellis2, `textured_mesh.glb` for hunyuan) - Stage J no longer deletes the runner-up, it only records a winner with reasoning in `judge/verdict.json`.
- `judge/verdict.json` exists with `winner`, `method` (`vlm` for a dual-backend run, `walkover` for a single-backend run), and non-empty `reasoning` when method is `vlm`.
- `judge=needs_review` with `method: fallback_default` means the VLM never emitted a valid A/B verdict - the pipeline still passes structurally but investigate the raw response in `verdict.json`.
- Anything else (PARTIAL/FAIL, nonzero exit, a traceback) is a real failure; read the streamed `[Tab4]` log lines to find which stage broke.

## Pass criteria (articulated)

- The script prints `==== A8 PASS (backends [...], judge=done) ====` and exits 0; any assertion failure or traceback is a real failure.
- The incomplete-upload negative case must have been rejected with the missing relpath listed and no job created (the script asserts this before the real upload).
- Per backend that ran: `textured/<backend>/groups/<gid>.glb` for the non-tiny groups, `textured/<backend>/mobility_textured.urdf`, and `textured/<backend>/assembled.glb` all exist; `vlm/plan.json` exists and the per-group table is non-empty.
- The `/viewer/<job_id>/<backend>` route returns a page containing "Joint Controls"; spot-check joints slide with textures attached by opening it (or `scripts/articulated_viewer.py`) when the change touched FK/assembly.
- Tiny groups showing a flat constant-PBR material is expected behavior, not a texturing failure.

## Known-benign noise

- Hunyuan's rasterizer segfaults (exit 139) at CUDA teardown AFTER writing all outputs; success is judged by the `[PBR_RESULT] ok:true` line plus the GLB on disk, and the log says "benign teardown segfault, output intact". Do not chase this.
- First-ever run downloads nothing; all weights are pre-cached in the shared HF cache. A "loading model" pause of a few minutes per stage is normal.

## Cleanup

`jobs_pipetest/` is a throwaway jobs root; leave the latest run for inspection but feel free to delete older `jobs_pipetest/<job_id>` dirs.

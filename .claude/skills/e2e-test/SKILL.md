---
name: e2e-test
description: Run pbr-texture-pipeline's full end-to-end pipeline test (all stages, real GPUs, real backend) via scripts/drive_pipeline.py and judge pass/fail for articulated URDF assets. Use after any change to the pipeline, app callbacks, workers, backend adapter, or the articulated package.
---

# pbr-texture-pipeline E2E pipeline test

Drives `app.py`'s real tab callbacks in order (Tab 1 render -> Tab 2 VLM chat -> Tab 3 reference image -> Tab 4 texturing -> Tab 4 judge -> Tab 5 jobs browser) against a real input, using the real WorkerManager, GPU workers, and backend subprocesses.
This is the closest thing to a browser user without a browser, and the project's primary verification (there is no unit-test suite).
Runs the full 7-stage articulated pipeline (including Stage P, per-group texturing, textured URDF + assembled.glb, joint-viewer routes) on a PartNet-Mobility URDF asset.

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

## Run

Run from the repo root:

```bash
conda run -n trellis2 python scripts/drive_pipeline.py \
  --urdf partnet_mobility/19179 \
  --jobs-root jobs_pipetest \
  --backends trellis2
```

- Exercises the app's URDF path end to end: the multi-file `upload_urdf` callback (including a negative case that must report the exact missing relpath and create no job), articulated Stage R, chat/spec/reference as usual, `run_texture` with automatic Stage P + the trellis2 bake (one field decode, per-group atlases) + assembly, judge, Tab 5, and the `/viewer/<job_id>/<backend>` FastAPI joint-viewer routes.
- `partnet_mobility/19179` (blank table) is the conventional choice; `8930` (textured door) exercises the appearance-sheet path.
  Texturing is one trellis2 run per job (one field decode plus a bake per group; tens of minutes on a many-group asset), so group count still drives runtime.
- Before this (or after any change to `pbr_texture_pipeline/articulated/`), run the free no-GPU gate first: `conda run -n trellis2 python scripts/verify_articulated.py` must report all assets OK.

## Pass criteria

- The script prints `==== A8 PASS (backends [...], judge=done) ====` and exits 0; any assertion failure or traceback is a real failure.
- The incomplete-upload negative case must have been rejected with the missing relpath listed and no job created (the script asserts this before the real upload).
- Per backend that ran: `textured/<backend>/groups/<gid>.glb` for the non-tiny groups, `textured/<backend>/mobility_textured.urdf`, and `textured/<backend>/assembled.glb` all exist; `vlm/plan.json` exists and the per-group table is non-empty.
- The `/viewer/<job_id>/<backend>` route returns a page containing "Joint Controls"; spot-check joints slide with textures attached by opening it (or `scripts/articulated_viewer.py`) when the change touched FK/assembly.
- Tiny groups showing a flat constant-PBR material is expected behavior, not a texturing failure.
- `judge/verdict.json` exists with `winner`, `method` (`walkover` for a single-backend run), and `status: done`.
- Anything else (PARTIAL/FAIL, nonzero exit, a traceback) is a real failure; read the streamed `[Tab4]` log lines to find which stage broke.

## Known-benign noise

- First-ever run downloads nothing; all weights are pre-cached in the shared HF cache. A "loading model" pause of a few minutes per stage is normal.

## Cleanup

`jobs_pipetest/` is a throwaway jobs root; leave the latest run for inspection but feel free to delete older `jobs_pipetest/<job_id>` dirs.

# pbr-texture-pipeline

This pipeline enhances textures on articulated 3D assets (URDF-format objects with moving joints, such as a cabinet whose doors open, drawn from the PartNet-Mobility and Articraft-10K datasets).
It keeps the assets' original textures as a starting point: it renders the textured mesh from the front via pyrender, runs a VLM to describe the visible materials, enhances the rendered view with an image-editing model conditioned on those materials and a depth map, and feeds the enhanced image to a forked TRELLIS.2 that supports multi-reference-image conditioning.
TRELLIS.2 decodes a PBR texture field for the whole object, cleans each part's mesh (merging coincident vertices, splitting at hard edges for correct normals), and bakes a separate UV atlas for each part.
The parts are reassembled into a textured URDF and a combined GLB.

See `PRD.md` for the full design spec.
This README only covers setup and running it.

## Requirements

- Linux with an NVIDIA GPU.
  Developed and verified on 2x NVIDIA A40 (46 GB of GPU memory, also called VRAM, each).
  A single GPU also works, using a reduced-memory strategy (`gpu_mode: single` in `config.yaml`), but a GPU with less than ~40 GB of VRAM will struggle to fit the TRELLIS.2 texturing backend.
- ~50 GB free disk for the conda environments, plus space for model weights.
- `conda` (or `mamba`) and `git` with submodule support.

## Setup

### 1. Clone with submodules

The texturing tool and the front-detection model each live in their own separate repository.
Rather than copying their code into this repo, we reference them as git submodules - a git feature that links a repo to a specific commit of another repo - pinned to commits verified to work with this pipeline.
This keeps this repo small:

```bash
git clone --recurse-submodules <this-repo-url>
cd pbr-texture-pipeline
```

If you already cloned without `--recurse-submodules`, run `git submodule update --init`.

The TRELLIS.2 submodule points to a fork that includes multi-reference-image conditioning support.
A few small patches on top of the pinned fork are still required.
Apply them once after checkout:

```bash
git apply patches/trellis2.patch --directory=TRELLIS.2
```

### 2. Create the three conda environments

Each environment exists because a different part of the pipeline needs a Python or library version that conflicts with the others.
The pipeline always reads which environment to use from `config.yaml`; the names are never hardcoded elsewhere.

**`trellis2`** (Python 3.10) - runs stages R (render), E (imageedit), and T (texture), the Gradio web app, the batch command-line tool, and the TRELLIS.2 texturing backend:

```bash
cd TRELLIS.2
conda create -n trellis2 python=3.10 -y
conda activate trellis2
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
pip install -r ../environments/trellis2-extra.txt
cd ..
```

**`vlm`** (Python 3.12) - runs only `scripts/vlm_infer.py`, a subprocess that uses Qwen2.5-VL-7B-Instruct via transformers to analyze the materials visible in a rendered view (Stage V):

```bash
conda create -n vlm python=3.12 -y
conda activate vlm
pip install -r environments/vlm-requirements.txt
```

**`orianyv2`** (Python 3.11) - runs only `scripts/orient_infer.py`, a short-lived helper process that detects which side of an articulated object faces forward, used while rendering articulated assets:

```bash
conda create -n orianyv2 python=3.11 -y
conda activate orianyv2
pip install $(grep -vE '^(bpy|gradio)' Orient-Anything-V2/requirements.txt)
```

These installs are sensitive to your CUDA driver/toolkit version.
If a pinned package fails to install, check the corresponding upstream repo's own install docs (`TRELLIS.2/README.md`, `Orient-Anything-V2/README.md`) for current guidance.

### 3. Point config at your machine (if needed)

`config.yaml` ships with defaults that work out of the box: caches under `.cache/` inside the repo, and backend repos resolved from the submodule paths above.
If you want the Hugging Face cache, torch cache, or backend repos to live somewhere else (a shared/larger disk, or checkouts you already have elsewhere), copy `config.local.yaml.example` to `config.local.yaml` (gitignored) and override only the keys you need.
It's merged on top of `config.yaml` at load time, so any key you don't set keeps its default.

### 4. Download model weights

```bash
conda run -n trellis2 python -m scripts.download_orient_anything  # Orient-Anything-V2 checkpoint, ~5 GB
```

Every other model (TRELLIS.2, CLIP, Qwen2.5-VL for Stage V, Qwen-Image-Edit for Stage E) is fetched automatically on first use via `huggingface_hub`, into the same cache.

### 5. Get test data

**PartNet-Mobility**: `partnet_mobility/` (articulated URDF test assets) is not included in this repo: it's a subset of the SAPIEN PartNet-Mobility dataset, which requires agreeing to its own terms before you can download it.
Get it from https://sapien.ucsd.edu/downloads and place object folders (e.g. `partnet_mobility/8930/`) directly under `partnet_mobility/`.

**Articraft-10K**: extract assets from the Articraft-10K tar.gz archives:

```bash
bash scripts/extract_articraft.sh --limit 100  # extract up to 100 assets
```

This places object folders under `articraft_extracted/`.

## Running

**Gradio app** (interactive, 3-tab wizard):

```bash
conda run -n trellis2 python -m pbr_texture_pipeline.app
```

Opens on `0.0.0.0:7860`.

**Batch CLI** (runs one stage at a time across all assets, so each model is loaded into memory once instead of once per asset):

```bash
# PartNet-Mobility assets
conda run -n trellis2 python -m pbr_texture_pipeline.batch \
  --assets 'partnet_mobility/*/mobility.urdf' --jobs-root jobs/ \
  --stages render,vlm,imageedit,texture --resume

# Articraft-10K assets
conda run -n trellis2 python -m pbr_texture_pipeline.batch \
  --assets 'articraft_extracted/*/model.urdf' --jobs-root jobs_v2/ \
  --stages render,vlm,imageedit,texture --resume
```

Stage names accept aliases R, V, E, T.
Stages V (vlm) and E (imageedit) are optional: Stage T falls back to the raw rendered front view if the enhanced image is absent.
Job outputs land under `jobs/<job_id>/`, one directory per asset.
If a run is interrupted, rerunning with `--resume` picks up from the last completed stage instead of starting over.

## Verification

There is no automated test suite.
Verify results by inspecting the per-job outputs (rendered views, textured GLBs) and the HTML gallery:

```bash
python scripts/build_viewer.py jobs/
cd jobs/ && python -m http.server 8080
```

## Repo layout

- `pbr_texture_pipeline/` - the pipeline package: the pipeline stages, the Gradio app, the batch CLI, the backend adapter (code that connects to the TRELLIS.2 texturing tool), and articulated-asset support.
- `scripts/` - one-off tools: model downloads, mesh prep, viewers, dataset extraction, VLM inference (`vlm_infer.py`), and image-edit inference (`imageedit_infer.py`).
- `TRELLIS.2/`, `Orient-Anything-V2/` - git submodules for the texturing tool and the front-detection model.
- `patches/` - small patches applied to the TRELLIS.2 submodule (see Setup step 1).
- `environments/` - pip requirements layered on top of each backend's own install steps.
- `config.yaml` / `config.local.yaml` - single source of truth for paths, model ids, and stage defaults; see `config.local.yaml.example`.
- `PRD.md` - the full design spec for the pipeline and the app.

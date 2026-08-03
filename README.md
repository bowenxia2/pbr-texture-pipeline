# pbr-texture-pipeline

This pipeline adds textures and materials to 3D meshes that don't have any.
Given a blank mesh, it renders an image of the mesh, then a vision-language model (a VLM - an AI model that can look at an image and generate text about it) writes a caption and a text prompt describing what the object's surface should look like, based on what kind of object it is.
It generates a reference image from that prompt, showing the described surface from the same camera angle as the rendered mesh.
It feeds the mesh and the reference image into two different texturing tools (TRELLIS.2 and Hunyuan3D-2.1, referred to in this repo as "backends"), each of which produces its own textured version of the mesh.
Finally, the same VLM compares the two textured results and picks the one it judges better; both are kept.

It supports two kinds of input.
The first is simple, single-piece meshes.
The second is articulated assets: multi-part objects with moving joints (for example, a cabinet whose doors open), described using the URDF format (a standard file format for objects and robots with movable parts) and drawn from the PartNet-Mobility dataset.
For articulated assets, all the parts are textured together in one pass and then split back out into separate per-part textures.

See `PRD.md` for the full design spec.
This README only covers setup and running it.

## Requirements

- Linux with an NVIDIA GPU.
  Developed and verified on 2x NVIDIA A40 (46 GB of GPU memory, also called VRAM, each).
  A single GPU also works, using a reduced-memory strategy (`gpu_mode: single` in `config.yaml`), but a GPU with less than ~40 GB of VRAM will struggle to fit the VLM and the image-generation (diffusion) models at the same time.
- ~50 GB free disk for the conda environments, plus space for model weights.
  The VLM alone is ~24 GB; the full set of cached weights is well over 100 GB.
- `conda` (or `mamba`) and `git` with submodule support.

## Setup

### 1. Clone with submodules

The two texturing tools and the front-detection model each live in their own separate repository.
Rather than copying their code into this repo, we reference them as git submodules - a git feature that links a repo to a specific commit of another repo - pinned to commits verified to work with this pipeline.
This keeps this repo small:

```bash
git clone --recurse-submodules <this-repo-url>
cd pbr-texture-pipeline
```

If you already cloned without `--recurse-submodules`, run `git submodule update --init`.

A few small patches on top of the pinned upstream commits are required: a fix for a broken attribute path, a fix for a config path that only resolved correctly when run from one specific working directory, and a couple of pinned dependency version bumps.
Apply them once after checkout:

```bash
git apply patches/trellis2.patch --directory=TRELLIS.2
git apply patches/hunyuan3d-2.1.patch --directory=Hunyuan3D-2.1
```

### 2. Create the four conda environments

Each of the four environments exists because a different part of the pipeline needs a Python or library version that conflicts with the others.
The pipeline always reads which environment to use from `config.yaml`; the names are never hardcoded elsewhere.

**`trellis2`** (Python 3.10) - runs the rendering step, the reference-image generation step (which uses a diffusion model), the Gradio web app, the batch command-line tool, and the TRELLIS.2 texturing tool itself:

```bash
cd TRELLIS.2
conda create -n trellis2 python=3.10 -y
conda activate trellis2
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
pip install -r ../environments/trellis2-extra.txt
cd ..
```

**`vlm`** (Python 3.12) - runs only the captioning and prompt-writing step: the Qwen3.6-35B-A3B model, compressed with a technique called AWQ and served with vLLM (a fast engine for running large language models).
This needs torch >= 2.8, which conflicts with the `trellis2` environment's pinned torch 2.6:

```bash
conda create -n vlm python=3.12 -y
conda activate vlm
pip install -r environments/vlm-requirements.txt
```

**`hunyuan3d`** (Python 3.11) - runs only the Hunyuan3D-2.1 texturing tool, which needs its own rasterizer (the component that converts 3D geometry into 2D pixels) and torch 2.5.1:

```bash
conda create -n hunyuan3d python=3.11 -y
conda activate hunyuan3d
cd Hunyuan3D-2.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cd hy3dpaint/custom_rasterizer && pip install -e . && cd ../..
cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh && cd ../../..
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P Hunyuan3D-2.1/hy3dpaint/ckpt
```

**`orianyv2`** (Python 3.11) - runs only `scripts/orient_infer.py`, a short-lived helper process that detects which side of an articulated object faces forward, used while rendering articulated assets:

```bash
conda create -n orianyv2 python=3.11 -y
conda activate orianyv2
pip install $(grep -vE '^(bpy|gradio)' Orient-Anything-V2/requirements.txt)
```

These installs are sensitive to your CUDA driver/toolkit version.
If a pinned package fails to install, check the corresponding upstream repo's own install docs (`TRELLIS.2/README.md`, `Hunyuan3D-2.1/README.md`, `Orient-Anything-V2/README.md`) for current guidance.

### 3. Point config at your machine (if needed)

`config.yaml` ships with defaults that work out of the box: caches under `.cache/` inside the repo, and backend repos resolved from the submodule paths above.
If you want the Hugging Face cache, torch cache, or backend repos to live somewhere else (a shared/larger disk, or checkouts you already have elsewhere), copy `config.local.yaml.example` to `config.local.yaml` (gitignored) and override only the keys you need.
It's merged on top of `config.yaml` at load time, so any key you don't set keeps its default.

### 4. Download model weights

```bash
conda run -n trellis2 python -m scripts.download_vlm            # VLM, ~24 GB
conda run -n trellis2 python -m scripts.download_controlnet     # Qwen-Image + ControlNet
conda run -n trellis2 python -m scripts.download_orient_anything  # Orient-Anything-V2 checkpoint, ~5 GB
```

Every other model (TRELLIS.2, RMBG-2.0, CLIP) is fetched automatically on first use via `huggingface_hub`, into the same cache.

### 5. Get test data (optional)

`partnet_mobility/` (articulated URDF test assets) is not included in this repo: it's a subset of the SAPIEN PartNet-Mobility dataset, which requires agreeing to its own terms before you can download it.
Get it from https://sapien.ucsd.edu/downloads and place object folders (e.g. `partnet_mobility/8930/`) directly under `partnet_mobility/`.
This is only needed for articulated (URDF) jobs; flat-mesh jobs just need your own `.glb`/`.obj` files.

## Running

**Gradio app** (interactive, 5-tab wizard):

```bash
conda run -n trellis2 python -m pbr_texture_pipeline.app
```

Opens on `0.0.0.0:7860`.

**Batch CLI** (runs one stage at a time across all meshes, so each model is loaded into memory once instead of once per mesh):

```bash
conda run -n trellis2 python -m pbr_texture_pipeline.batch \
  --meshes 'meshes_run/*.glb' --jobs-root jobs/ \
  --backends trellis2,hunyuan --stages render,vlm,diffuse,plan,texture,eval,judge --resume
```

For articulated assets, point `--meshes` at a `mobility.urdf` glob instead (Hunyuan has no articulated support, so those jobs only use the TRELLIS.2 backend):

```bash
conda run -n trellis2 python -m pbr_texture_pipeline.batch \
  --meshes 'partnet_mobility/*/mobility.urdf' --jobs-root jobs/ \
  --backends trellis2 --stages render,vlm,diffuse,plan,texture,eval,judge --resume
```

Job outputs land under `jobs/<job_id>/`, one directory per mesh.
If a run is interrupted, rerunning with `--resume` picks up from the last completed stage instead of starting over.

## Verification

There's no automated test suite.
Verification instead relies on a set of standalone checking scripts (referred to in this repo as "gates") plus a script that drives the pipeline end-to-end (see `CLAUDE.md`'s "Commands" section for the full list).
For example:

```bash
conda run -n trellis2 python scripts/verify_conventions.py --out jobs/_gate1_conventions
conda run -n trellis2 python scripts/verify_articulated.py --jobs-root jobs/
```

## Repo layout

- `pbr_texture_pipeline/` - the pipeline package: the pipeline stages, the Gradio app, the batch CLI, the backend adapters (code that connects to each texturing tool), and articulated-asset support.
- `scripts/` - one-off tools: model downloads, verification/gate scripts, mesh prep, viewers.
- `TRELLIS.2/`, `Hunyuan3D-2.1/`, `Orient-Anything-V2/` - git submodules for the texturing tools and the front-detection model.
- `patches/` - small patches applied to the submodules (see Setup step 1).
- `environments/` - pip requirements layered on top of each backend's own install steps.
- `config.yaml` / `config.local.yaml` - single source of truth for paths, model ids, and stage defaults; see `config.local.yaml.example`.
- `PRD.md` - the full design spec for the pipeline, both job kinds, and the app.

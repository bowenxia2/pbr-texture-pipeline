# pbr-texture-pipeline

Textures untextured 3D meshes automatically. Given a blank mesh, it renders the mesh, has a
vision-language model (VLM) caption it and propose a category-aware appearance prompt,
generates a pose-matched reference image, drives two texturing backends (TRELLIS.2 and
Hunyuan3D-2.1) with the (mesh, reference) pair, then has the VLM judge the two textured
outputs and pick a winner.

It handles two kinds of input: flat meshes, and articulated PartNet-Mobility URDF assets
(multi-part objects with joints, textured part-by-part from one shared texture field).

See `PRD.md` for the full design spec. This README only covers setup and running it.

## Requirements

- Linux with an NVIDIA GPU. Developed and verified on 2x NVIDIA A40 (46 GB each); a single
  GPU works with a reduced VRAM strategy (`gpu_mode: single` in `config.yaml`), but a GPU
  with less than ~40 GB will struggle to fit the VLM and diffusion models.
- ~50 GB free disk for the conda environments, plus space for model weights (the VLM alone
  is ~24 GB; the full set of cached weights is well over 100 GB).
- `conda` (or `mamba`) and `git` with submodule support.

## Setup

### 1. Clone with submodules

The two texturing backends and the front-detection model are vendored as git submodules
(pinned to a commit verified to work with this pipeline) rather than copied in, to keep this
repo small:

```bash
git clone --recurse-submodules <this-repo-url>
cd pbr-texture-pipeline
```

If you already cloned without `--recurse-submodules`, run `git submodule update --init`.

A few small patches on top of the pinned upstream commits are required (fixes to a broken
attribute path, a cwd-relative config path, and a couple of pinned dependency versions).
Apply them once after checkout:

```bash
git apply patches/trellis2.patch --directory=TRELLIS.2
git apply patches/hunyuan3d-2.1.patch --directory=Hunyuan3D-2.1
```

### 2. Create the four conda environments

Each environment exists because some part of the stack pins an incompatible dependency
version; adapters and workers always read env names from `config.yaml`, never hardcode them.

**`trellis2`** (Python 3.10) - runs Stage R (render), Stage D (diffusion), the Gradio app,
the batch CLI, and the TRELLIS.2 backend adapters:

```bash
cd TRELLIS.2
conda create -n trellis2 python=3.10 -y
conda activate trellis2
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
pip install -r ../environments/trellis2-extra.txt
cd ..
```

**`vlm`** (Python 3.12) - runs only the Stage V worker (Qwen3.6-35B-A3B AWQ under vLLM,
which needs torch >= 2.8, incompatible with the `trellis2` env's pinned torch 2.6):

```bash
conda create -n vlm python=3.12 -y
conda activate vlm
pip install -r environments/vlm-requirements.txt
```

**`hunyuan3d`** (Python 3.11) - runs only the Hunyuan3D-2.1 backend adapter
(`custom_rasterizer`, torch 2.5.1):

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

**`orianyv2`** (Python 3.11) - runs only `scripts/orient_infer.py`, a short-lived subprocess
for front-panel detection during articulated Stage R:

```bash
conda create -n orianyv2 python=3.11 -y
conda activate orianyv2
pip install $(grep -vE '^(bpy|gradio)' Orient-Anything-V2/requirements.txt)
```

These installs are sensitive to your CUDA driver/toolkit version; if a pinned wheel fails to
resolve, check the corresponding upstream repo's own install docs (`TRELLIS.2/README.md`,
`Hunyuan3D-2.1/README.md`, `Orient-Anything-V2/README.md`) for the current guidance.

### 3. Point config at your machine (if needed)

`config.yaml` ships with defaults that work out of the box: caches under `.cache/` inside the
repo, backend repos resolved from the submodule paths above. If you want the HF cache, torch
cache, or backend repos to live somewhere else (a shared/larger disk, or checkouts you
already have elsewhere), copy `config.local.yaml.example` to `config.local.yaml` (gitignored)
and override only the keys you need; it's deep-merged on top of `config.yaml` at load time.

### 4. Download model weights

```bash
conda run -n trellis2 python -m scripts.download_vlm            # VLM, ~24 GB
conda run -n trellis2 python -m scripts.download_controlnet     # Qwen-Image + ControlNet
conda run -n trellis2 python -m scripts.download_orient_anything  # Orient-Anything-V2 ckpt, ~5 GB
```

Every other model (TRELLIS.2, RMBG-2.0, CLIP) is fetched automatically on first use via
`huggingface_hub`, into the same cache.

### 5. Get test data (optional)

`partnet_mobility/` (articulated URDF test assets) is not included in this repo: it's a
subset of the SAPIEN PartNet-Mobility dataset, which requires agreeing to its own terms to
download. Get it from https://sapien.ucsd.edu/downloads and place object folders (e.g.
`partnet_mobility/8930/`) directly under `partnet_mobility/`. This is only needed for
articulated (URDF) jobs; flat-mesh jobs just need your own `.glb`/`.obj` files.

## Running

**Gradio app** (interactive, 5-tab wizard):

```bash
conda run -n trellis2 python -m pbr_texture_pipeline.app
```

Opens on `0.0.0.0:7860`.

**Batch CLI** (stage-major: each model loads once over all meshes):

```bash
conda run -n trellis2 python -m pbr_texture_pipeline.batch \
  --meshes 'meshes_run/*.glb' --jobs-root jobs/ \
  --backends trellis2,hunyuan --stages render,vlm,diffuse,plan,texture,eval,judge --resume
```

For articulated assets, point `--meshes` at a `mobility.urdf` glob instead (hunyuan has no
articulated path, so those jobs are trellis2-only):

```bash
conda run -n trellis2 python -m pbr_texture_pipeline.batch \
  --meshes 'partnet_mobility/*/mobility.urdf' --jobs-root jobs/ \
  --backends trellis2 --stages render,vlm,diffuse,plan,texture,eval,judge --resume
```

Job outputs land under `jobs/<job_id>/`, one directory per mesh, resumable per stage.

## Verification

There's no unit test suite; verification is a set of gate scripts plus an end-to-end drive
script (see `CLAUDE.md` "Commands" for the full list), e.g.:

```bash
conda run -n trellis2 python scripts/verify_conventions.py --out jobs/_gate1_conventions
conda run -n trellis2 python scripts/verify_articulated.py --jobs-root jobs/
```

## Repo layout

- `pbr_texture_pipeline/` - the pipeline package (stages, app, batch CLI, backend adapters,
  articulated support).
- `scripts/` - one-off tools: model downloads, gate/verification scripts, mesh prep, viewers.
- `TRELLIS.2/`, `Hunyuan3D-2.1/`, `Orient-Anything-V2/` - git submodules for the texturing
  backends and front-detection model.
- `patches/` - small patches applied to the submodules (see Setup step 1).
- `environments/` - pip requirements layered on top of each backend's own install steps.
- `config.yaml` / `config.local.yaml` - single source of truth for paths, model ids, and
  stage defaults; see `config.local.yaml.example`.
- `PRD.md` - the full design spec for the pipeline, both job kinds, and the app.

# PRD: pbr-texture-pipeline - texturing of articulated 3D assets

Status: implemented.
This is the sole authoritative spec for pbr-texture-pipeline.
It covers the four-stage pipeline (R -> V -> E -> T) for articulated URDF assets (PartNet-Mobility and Articraft-10K).

Historical note: an earlier seven-stage pipeline (R -> V -> D -> P -> T -> E -> J) used VLM captioning, ControlNet diffusion, material planning, evaluation, and judging across two texturing backends (TRELLIS.2 and Hunyuan3D-2.1).
That pipeline treated assets as blank/untextured and generated reference images from scratch.
The current pipeline keeps the assets' original textures as a starting point and uses a VLM + image-editing model to enhance the front-panel view before feeding it to TRELLIS.2, which made the intermediate diffusion, selection, and judging stages unnecessary.

## 1. Overview and goals

### 1.1 Problem

One working texturing backend is available: the forked TRELLIS.2 (vendored as the `TRELLIS.2` git submodule, forked at bowenxia2/TRELLIS.2), which takes (mesh, reference image) and produces a PBR-textured mesh.
The assets already have basic textures (diffuse maps on OBJ meshes), but these are low-resolution and lack PBR detail.
The pipeline uses the original textures as conditioning to produce PBR materials via TRELLIS.2.

### 1.2 Solution shape

1. Render the textured URDF asset from the front via pyrender, preserving the original materials, plus a depth map and white-background composite.
2. Run a VLM (Qwen2.5-VL-7B-Instruct) on the white-background front view to produce a material description.
3. Run an image-editing model (Qwen-Image-Edit-2511) on the front view + depth map, guided by the material description, to produce an enhanced reference image with realistic PBR surface detail.
4. Feed the enhanced front view (or the raw front view as fallback) to the forked TRELLIS.2 to produce PBR-textured per-group atlas bakes from a single decoded voxel field.

### 1.3 Modes

- Interactive: one URDF asset at a time, human in the loop, via Gradio (3-tab wizard).
- Batch: many URDF assets, each model loads once over all assets (stage-major execution), results browsable via HTML gallery or jobs tab.

### 1.4 Scale target

Each of the two datasets has 100-2000 assets.
Batch mode processes them unattended with `--resume` for restartability.

## 2. Pipeline stages

### 2.1 Stage R (Render)

Parse the URDF, build semantic groups, compute rest-pose FK, and render the front textured RGBA view plus depth map.

Inputs:
- URDF file (`mobility.urdf` for PartNet-Mobility, `model.urdf` for Articraft-10K) plus its transitive file closure (OBJ meshes, MTL files, texture images).
  `find_urdf()` checks both filenames.
- `meta.json` category (optional; falls back to URDF `<robot name>` attribute).

Processing:
1. Parse URDF into per-link visuals and joints via `articulated/urdf.py`.
2. Build groups by URDF `<visual name>` (semantic grouping).
3. Compute rest-pose FK (joints at q=0 clamped into limits).
4. Merge groups into a single rest-pose mesh for TRELLIS.2 (geometry only, no materials).
5. Normalize the merged mesh (`preprocess_mesh(up="z")`), export to `mesh_norm.glb` (Y-up glTF convention via `export_yup`).
6. Record per-group face ranges into the merged mesh (`face_ranges.json`).
7. Orient-Anything-V2 front detection (optional): render a quick contact sheet via nvdiffrast, run Orient-V2 in the `orianyv2` subprocess env, determine the front panel.
   Falls back to `articulated.front_panel` config value.
8. Merge groups with materials (`merge_group_mesh(with_materials=True)`) for textured rendering.
9. Render the front RGBA view via pyrender (EGL backend, headless) at yaw=pi (front), at the configured resolution (default 1024x1024).
   Each group mesh is positioned by its FK transform, center+scale normalized, and reposed by the Orient-V2 result.
   Transparent background (alpha=0 where no geometry).
10. For the front view (view 0), also capture the depth buffer as an inverse-depth ControlNet map (`depth_0.png`) and composite the RGBA onto a white background (`front_white.png`).

Outputs (all under `jobs/<job_id>/`):
- `input/mesh_norm.glb` - normalized merged mesh for TRELLIS.2.
- `input/face_ranges.json` - per-group face ranges + normalization params.
- `render/view_0.png` - front textured RGBA view (pyrender).
- `render/depth_0.png` - inverse-depth ControlNet map of the front view.
- `render/front_white.png` - front view composited onto white background.
- `control/camera.json` - repose matrix and Orient-V2 decision.
- `groups/<gid>/mesh.glb` - per-group mesh (geometry only).
- `groups/<gid>/norm.json` - per-group normalization params and bounds.

### 2.2 Stage V (VLM)

Run a vision-language model on the white-background front-panel image to produce a material description and a texture quality classification.

Inputs:
- `render/front_white.png` from Stage R.
- Asset category from `meta.json` `model_cat` (or `<robot name>` URDF attribute as fallback).

Processing:
1. Load Qwen2.5-VL-7B-Instruct via transformers `AutoModelForImageTextToText` in the `vlm` conda env (`device_map="auto"`).
2. First classify texture quality: send the front view with a prompt asking whether the object has meaningful existing textures worth preserving (`edit`) or is blank/basic and should be generated from scratch (`generate`).
3. Then send the front view with a material analysis prompt (variant matched to the classification) that asks for a part-by-part description (material type, color, surface finish).
4. The classification routes the Stage E prompt: `edit` tells Qwen-Image-Edit to preserve existing textures while enhancing, `generate` tells it to replace blank surfaces with realistic textures.

Outputs:
- `vlm/materials.txt` - semicolon-separated material description text.
- `vlm/classification.txt` - texture quality classification (`edit` or `generate`).

Stage V is optional.
If skipped, Stage E cannot run, and Stage T falls back to the raw front-panel view.

### 2.3 Stage E (ImageEdit)

Enhance or generate textures for the front-panel image using Qwen-Image-Edit.
The VLM classification from Stage V selects the prompt template, not a separate pipeline.

Inputs:
- `render/front_white.png` from Stage R (source image).
- `render/canny_0.png` from Stage R (canny edge map for structural conditioning).
- `vlm/materials.txt` from Stage V (material description for the prompt).
- `vlm/classification.txt` from Stage V (`edit` or `generate`).

Processing:
1. Load Qwen-Image-Edit-2511 via diffusers `QwenImageEditPlusPipeline` in the `trellis2` conda env, with `device_map="balanced"` across both GPUs.
2. Select a prompt template based on classification:
   - `edit`: preserve the object's existing textures, colors, and patterns while enhancing materials with realistic surface properties.
   - `generate`: replace blank/flat surfaces with realistic material textures inferred by the VLM from shape and category.
3. Pass the source image and canny map to the edit pipeline in both cases.
4. Generate one enhanced image.

In batch mode, Stage E loads the model once for all items.

Outputs:
- `imageedit/enhanced_0.png` - enhanced or generated front view.

Stage E is optional.
If skipped, Stage T falls back to the raw front-panel view.

### 2.4 Stage T (Texture)

Feed the enhanced front view (or raw front view fallback) to TRELLIS.2's image conditioning, decode one PBR voxel field, and bake per-group atlases.

Pair construction (`stages.py:global_texture_pair`):
- `image` field is a list containing the enhanced front view (`imageedit/enhanced_0.png`) if it exists, otherwise the raw front view (`render/view_0.png`).
- `merged_mesh` is the normalized `mesh_norm.glb`.
- Per-group entries carry `to_link` (4x4 matrix mapping adapter output frame back to link frame), `texture_size`, and `bounds_link`.
- Resume is group-granular: the pair lists only groups whose `textured/trellis2/groups/<gid>.glb` is absent.

The `to_link` matrix chain (composed orchestrator-side):
`inv(FK) @ undo_Stage_R_norm @ undo_repose @ Y-up-to-Z-up @ undo_adapter_norm`

Adapter (`trellis2_adapter.py`):
1. Open and preprocess the reference image.
2. Pass it to `pipe.get_cond(images, ...)` - the forked TRELLIS.2 supports multi-image conditioning via token concatenation (`trellis2_texturing.py:159-181`), but Stage T currently sends a single image.
3. Decode the PBR voxel field once for the merged rest-pose mesh.
4. For each group: extract the group's face range, merge coincident vertices (`merge_vertices(merge_norm=True)`) for proper adjacency, split vertices at hard edges (>60 degrees) to preserve creases, UV-unwrap via `cumesh.uv_unwrap`, recompute vertex normals from the post-unwrap topology, bake a UV atlas from the shared field, and apply the `to_link` transform to map to link frame.
5. Tiny groups (below `min_group_faces` or `min_group_area_frac`) bake at `tiny_texture_size` (default 256) instead of `per_link_texture_size` (default 1024).

Assembly:
- Collect per-group GLBs, write the textured URDF, build `assembled.glb` for the joint-slider viewer.
- The adapter merges `pass_a.json` metadata on partial (resume) runs instead of rewriting; this preserves earlier groups' bake stats.

Outputs:
- `textured/trellis2/groups/<gid>.glb` - per-group textured mesh.
- `textured/trellis2/global/pass_a.json` - bake metadata.
- `textured/trellis2/mobility_textured.urdf` - textured URDF with collision symlinks.
- `textured/trellis2/assembled.glb` - rest-pose assembly for viewing.
- `textured/trellis2/original.urdf` - copy of the original URDF.
- `textured/trellis2/assembled_original.glb` - rest-pose assembly of the original (for comparison).

## 3. Job directory layout

```
jobs/<job_id>/
  job.json                              # state machine (schema v6)
  input/
    original.<ext>                      # copy of the URDF
    mesh_norm.glb                       # normalized merged mesh (Y-up glTF)
    face_ranges.json                    # per-group face ranges + norm params
  render/
    view_0.png                          # front textured RGBA view (pyrender)
    depth_0.png                         # inverse-depth ControlNet map
    front_white.png                     # front view on white background
  groups/<gid>/
    mesh.glb                            # per-group geometry
    norm.json                           # per-group norm params + bounds
  control/
    camera.json                         # repose matrix, Orient-V2 decision
  vlm/
    materials.txt                       # VLM material description (Stage V)
    classification.txt                  # texture quality: edit or generate (Stage V)
  imageedit/
    enhanced_0.png                      # enhanced front view (Stage E)
  textured/trellis2/
    global/pass_a.json                  # bake metadata
    groups/<gid>.glb                    # per-group textured mesh
    mobility_textured.urdf              # textured URDF
    assembled.glb                       # rest-pose assembly
    original.urdf                       # copy of original URDF
    assembled_original.glb              # original rest-pose assembly
```

## 4. Configuration

`config.yaml` is the single source of truth.
`config.local.yaml` (gitignored) overrides for local paths and preferences.

Key sections:

```yaml
render:
  num_views: 1
  canonical_yaw: 3.14159265
  canonical_pitch: 0.0
  r: 2.0
  fov_deg: 40.0
  resolution: 1024
  ssaa: 2
  contact_pitch_deg: 15
  canny_thresholds: [100, 200]

imageedit:
  num_inference_steps: 40
  guidance_scale: 1.0
  true_cfg_scale: 4.0

texture:
  trellis2_resolution: 1024
  trellis2_texture_size: 2048

articulated:
  group_by: semantic
  front_panel: 2
  rest_pose: zero_clamped
  min_group_faces: 20
  min_group_area_frac: 0.0005
  collision_mode: symlink
  orient:
    enabled: true
  global:
    per_link_texture_size: 1024
    tiny_texture_size: 256

models:
  trellis2: microsoft/TRELLIS.2-4B
  clip: openai/clip-vit-large-patch14
  orient: Viglong/OriAnyV2_ckpt
  vlm: Qwen/Qwen2.5-VL-7B-Instruct
  imageedit: Qwen/Qwen-Image-Edit-2511

gpu_mode: dual
```

## 5. Environments

| Env | Python | Purpose |
|---|---|---|
| `trellis2` | 3.10 | Stages R, E, and T, Gradio app, batch CLI, TRELLIS.2 adapter, ImageEdit adapter |
| `vlm` | 3.12 | Only `scripts/vlm_infer.py` (VLM material analysis in Stage V) |
| `orianyv2` | 3.11 | Only `scripts/orient_infer.py` (Orient-Anything-V2 front detection) |

Dependencies:
- pyrender (for Stage R textured rendering, EGL backend)
- TRELLIS.2 and Orient-Anything-V2 as git submodules

### 5.1 HPC environment setup

On HPC clusters where CUDA and GCC are provided as environment modules:
- Load `cuda/12.4.x` and `gcc/11.x` (or newer; GCC >= 9 required) before building the trellis2 env's compiled CUDA extensions.
- Set `CUDA_HOME` to the loaded CUDA toolkit path so build scripts can find `nvcc`.
- Add `--no-cache-dir` to pip install commands if the pip cache is on a different filesystem from the build directory (avoids `Invalid cross-device link` errors during wheel installation).

### 5.2 Hugging Face authentication

TRELLIS.2 loads a gated model (`facebook/dinov3-vitl16-pretrain-lvd1689m`) at Stage T startup.
Accept its license on Hugging Face and authenticate via `hf auth login --token <token>` before running Stage T.

## 6. CLI and Gradio

### 6.1 Batch CLI

```bash
conda run -n trellis2 python -m pbr_texture_pipeline.batch \
  --assets 'partnet_mobility/*/mobility.urdf' --jobs-root jobs/ \
  --stages render,vlm,imageedit,texture --resume
```

For Articraft-10K assets:

```bash
bash scripts/extract_articraft.sh --limit 100
conda run -n trellis2 python -m pbr_texture_pipeline.batch \
  --assets 'articraft_extracted/*/model.urdf' --jobs-root jobs_v2/ \
  --stages render,vlm,imageedit,texture --resume
```

Stage names accept aliases R, V, E, T.
Execution is stage-major: each model loads once for all approved pairs.

### 6.2 Gradio app

3-tab wizard at `0.0.0.0:7860`:
1. **Upload & Render & Enhance** - upload URDF + referenced files, run Stage R, optionally run Stages V and E to enhance the front view.
2. **Texture** - run Stage T, view per-group status, joint-slider viewer.
3. **Jobs** - browse all jobs with per-stage statuses.

### 6.3 Viewer and gallery

```bash
# Joint-slider viewer for one job
conda run -n trellis2 python scripts/articulated_viewer.py jobs/<job_id> --backend trellis2 --port 8090

# HTML gallery of all results
python scripts/build_viewer.py <jobs-root>
cd <jobs-root> && python -m http.server 8080
```

## 7. Frame conventions

- URDF world frame is Z-up.
- Articulated rendering normalizes with `up="z"` (center+scale only, no axis swap).
- Exported `mesh_norm.glb` is Y-up (glTF convention) via `rendering.export_yup`.
- TRELLIS.2's `preprocess_mesh` assumes Y-up input and round-trips to Z-up internal.
- The `to_link` matrix chain accounts for both normalization steps and the Y-up/Z-up swap.
- Canonical front camera: yaw=pi, pitch=0.
- PartNet-Mobility assets face panel 2 in their assembled URDF world frame; Stage R pre-poses by `articulated.front_panel`.
- A `to_link` bounds mismatch fails the group loudly; bbox shrinkage up to 5% of extent is tolerated.

## 8. Known compromises

- TRELLIS.2 forces `alphaMode='OPAQUE'` on its output GLBs; all glass parts get a frosted opaque look.
- URDF primitives (`<box>`, `<cylinder>`, `<sphere>`) are converted to trimesh objects; they render without texture in Stage R and receive texturing from TRELLIS.2 based on surrounding context.
- Stages V and E are optional; if skipped, Stage T uses the raw pyrender front view, which has lower surface detail than an enhanced view but still preserves the original colors and proportions.

# PRD: pbr-texture-pipeline - texturing of articulated 3D assets

Status: implemented.
This is the sole authoritative spec for pbr-texture-pipeline.
It covers the two-stage pipeline (R -> T) for articulated URDF assets (PartNet-Mobility and Articraft-10K).

Historical note: an earlier seven-stage pipeline (R -> V -> D -> P -> T -> E -> J) used VLM captioning, ControlNet diffusion, material planning, evaluation, and judging.
That pipeline treated assets as blank/untextured and generated reference images from scratch.
The current pipeline keeps the assets' original textures as a starting point, which made the intermediate stages unnecessary.

## 1. Overview and goals

### 1.1 Problem

One working texturing backend is available: TRELLIS.2 texturing (vendored as the `TRELLIS.2` git submodule), which takes (untextured mesh, reference images) and produces a PBR-textured mesh.
The assets already have basic textures (diffuse maps on OBJ meshes), but these are low-resolution and lack PBR detail.
The pipeline uses the original textures as conditioning to produce PBR materials via TRELLIS.2.

### 1.2 Solution shape

1. Render the textured URDF asset from 4 angles via pyrender, preserving the original materials.
2. Feed all 4 rendered views to the forked TRELLIS.2 (which supports multi-reference-image conditioning via token concatenation) to produce PBR-textured per-group atlas bakes from a single decoded voxel field.

### 1.3 Modes

- Interactive: one URDF asset at a time, human in the loop at every stage, via Gradio (3-tab wizard).
- Batch: many URDF assets, each model loads once over all assets (stage-major execution), results browsable via HTML gallery or jobs tab.

### 1.4 Scale target

Each of the two datasets has 100-2000 assets.
Batch mode processes them unattended with `--resume` for restartability.

## 2. Pipeline stages

### 2.1 Stage R (Render)

Parse the URDF, build semantic groups, compute rest-pose FK, and render 4 textured RGBA views.

Inputs:
- URDF file (`mobility.urdf` for PartNet-Mobility, `model.urdf` for Articraft-10K) plus its transitive file closure (OBJ meshes, MTL files, texture images).
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
9. Render 4 RGBA views via pyrender (EGL backend, headless) at 90-degree azimuth steps starting from yaw=pi (front), at the configured resolution (default 1024x1024).
   Each group mesh is positioned by its FK transform, center+scale normalized, and reposed by the Orient-V2 result.
   Transparent background (alpha=0 where no geometry).

Outputs (all under `jobs/<job_id>/`):
- `input/mesh_norm.glb` - normalized merged mesh for TRELLIS.2.
- `input/face_ranges.json` - per-group face ranges + normalization params.
- `render/view_{0,1,2,3}.png` - 4 textured RGBA views.
- `control/camera.json` - repose matrix and Orient-V2 decision.
- `groups/<gid>/mesh.glb` - per-group mesh (geometry only).
- `groups/<gid>/norm.json` - per-group normalization params and bounds.

### 2.2 Stage T (Texture)

Feed all 4 rendered views to TRELLIS.2's multi-image conditioning, decode one PBR voxel field, and bake per-group atlases.

Pair construction (`stages.py:global_texture_pair`):
- `image` field is a list of 4 paths: the rendered views.
- `merged_mesh` is the normalized `mesh_norm.glb`.
- Per-group entries carry `to_link` (4x4 matrix mapping adapter output frame back to link frame), `texture_size`, and `bounds_link`.
- Resume is group-granular: the pair lists only groups whose `textured/trellis2/groups/<gid>.glb` is absent.

The `to_link` matrix chain (composed orchestrator-side):
`inv(FK) @ undo_Stage_R_norm @ undo_repose @ Y-up-to-Z-up @ undo_adapter_norm`

Adapter (`trellis2_adapter.py`):
1. Open and preprocess each of the 4 images.
2. Pass the list to `pipe.get_cond(images, ...)` - the forked TRELLIS.2 concatenates multi-image patch tokens along the sequence dimension (`trellis2_texturing.py:159-181`).
3. Decode the PBR voxel field once for the merged rest-pose mesh.
4. For each group: extract the group's face range, UV-unwrap via `cumesh.uv_unwrap`, bake a UV atlas from the shared field, and apply the `to_link` transform to map to link frame.
5. Tiny groups (below `min_group_faces` or `min_group_area_frac`) bake at `tiny_texture_size` (default 256) instead of `per_link_texture_size` (default 1024).

Assembly:
- Collect per-group GLBs, write the textured URDF, build `assembled.glb` for the joint-slider viewer.
- The adapter merges `pass_a.json` metadata on partial (resume) runs instead of rewriting; this preserves earlier groups' bake stats.

Outputs:
- `textured/trellis2/groups/<gid>.glb` - per-group textured mesh.
- `textured/trellis2/global/pass_a.json` - bake metadata.
- `textured/trellis2/mobility_textured.urdf` - textured URDF with collision symlinks.
- `textured/trellis2/assembled.glb` - rest-pose assembly for viewing.

## 3. Job directory layout

```
jobs/<job_id>/
  job.json                              # state machine (schema v4)
  input/
    original.<ext>                      # copy of the URDF
    mesh_norm.glb                       # normalized merged mesh (Y-up glTF)
    face_ranges.json                    # per-group face ranges + norm params
  render/
    view_{0,1,2,3}.png                  # 4 textured RGBA views (pyrender)
  groups/<gid>/
    mesh.glb                            # per-group geometry
    norm.json                           # per-group norm params + bounds
  control/
    camera.json                         # repose matrix, Orient-V2 decision
  textured/trellis2/
    global/pass_a.json                  # bake metadata
    groups/<gid>.glb                    # per-group textured mesh
    mobility_textured.urdf              # textured URDF
    assembled.glb                       # rest-pose assembly
```

## 4. Configuration

`config.yaml` is the single source of truth.
`config.local.yaml` (gitignored) overrides for local paths and preferences.

Key sections:

```yaml
render:
  num_views: 4
  canonical_yaw: 3.14159265
  canonical_pitch: 0.0
  r: 2.0
  fov_deg: 40.0
  resolution: 1024
  ssaa: 2
  contact_pitch_deg: 15
  canny_thresholds: [100, 200]

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

gpu_mode: dual
```

## 5. Environments

| Env | Python | Purpose |
|---|---|---|
| `trellis2` | 3.10 | Both stages (R, T), Gradio app, batch CLI, TRELLIS.2 adapter |
| `orianyv2` | 3.11 | Only `scripts/orient_infer.py` (Orient-Anything-V2 front detection) |

Dependencies:
- pyrender (for Stage R textured rendering, EGL backend)
- TRELLIS.2 and Orient-Anything-V2 as git submodules

## 6. CLI and Gradio

### 6.1 Batch CLI

```bash
conda run -n trellis2 python -m pbr_texture_pipeline.batch \
  --assets 'partnet_mobility/*/mobility.urdf' --jobs-root jobs/ \
  --stages render,texture --resume
```

Stage names accept aliases R, T.
Execution is stage-major: the TRELLIS.2 backend loads once for all approved pairs.

### 6.2 Gradio app

3-tab wizard at `0.0.0.0:7860`:
1. **Upload & Render** - upload URDF + referenced files, run Stage R, view 4 rendered views.
2. **Texture & Assembly** - run Stage T, view per-group status, joint-slider viewer.
3. **Jobs** - browse all jobs with 2-stage statuses.

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

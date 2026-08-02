# PRD: pbr-texture-pipeline - VLM-guided texturing of untextured 3D meshes

Status: implemented and verified.
This is the sole authoritative spec for pbr-texture-pipeline, covering both job kinds (flat meshes and articulated PartNet-Mobility URDF assets) and the full seven-stage pipeline.
It replaces the earlier two-document split (`PRD.md` draft plus `PRD_articulated_v2.md`); that split existed because articulated texturing went through a full redesign (per-part texturing, shipped 2026-07-21, replaced wholesale by global texturing on 2026-07-28) and the two documents were merged back into one once the design stabilized.
Historical context that explains why a design was chosen is kept where it prevents a regression to a discarded approach; a full changelog of every intermediate update is not.

## 1. Overview and goals

### 1.1 Problem

Two working texturing backends are available, each taking (untextured mesh, reference image) and producing a PBR-textured mesh: TRELLIS.2 texturing (vendored as the `TRELLIS.2` git submodule) and Hunyuan3D-2.1 paint (vendored as the `Hunyuan3D-2.1/hy3dpaint` git submodule).
A third backend, Pixal3D `texture_inference`, was originally in scope but was the weakest of the three and was removed entirely (adapter, registry entry, config, UI) early in the project; it is not discussed further.
Historically the reference image had to be supplied by hand.
pbr-texture-pipeline adds the missing upstream stage: given only a blank mesh, generate a category-aware, realistic, pose-matched reference image, then drive the existing backends with it.

### 1.2 Solution shape

1. Render the blank mesh so a VLM can see it.
2. A local VLM (Qwen3.6-35B-A3B, AWQ 4-bit, served by vLLM) first captions the mesh from the rendered views, then emits a realistic appearance prompt from the contact sheet plus that caption.
   Category-awareness is enforced: a chair can be wood, plastic, or metal, but never brick.
   The Gradio chat survives as an interactive override on top of the auto-generated spec.
3. A local diffusion model (Qwen-Image + depth ControlNet) generates a reference image conditioned on a render of the mesh, so the image pose exactly matches the mesh from the chosen camera.
4. The (mesh, reference image) pair is fed to one or both backends (for articulated assets, this becomes one whole-object pair driving a single global texturing run; see section 7.2).
5. The same VLM judges the textured outputs and picks a winner with reasoning; every output is kept on disk.

### 1.3 Two job kinds

- `kind: "mesh"`: a single flat mesh, textured by both backends.
- `kind: "urdf"`: an articulated PartNet-Mobility asset (a `mobility.urdf` plus its linked visual/collision geometry), textured by TRELLIS.2 only.
  Articulated jobs texture in global mode (section 7.2): one TRELLIS.2 field decode over the merged rest-pose assembly, plus an open-pose decode so interior surfaces get visual evidence, then a per-group atlas bake from the shared field.
  Hunyuan has no articulated path, so urdf jobs are trellis2-only and Stage J records a walkover.

### 1.4 Modes

- Interactive: one mesh (or one URDF asset) at a time, human in the loop at every stage, via Gradio.
- Batch: many meshes or URDF assets, the VLM auto-proposes the prompt, candidates are auto-selected, backends run unattended, and a human reviews afterward in a jobs browser or the HTML gallery.

### 1.5 Scope

In scope and built: reference-image generation (render, VLM, diffusion); orchestration, per-job state, the Gradio app, the batch CLI, and the evaluation and judge stages; adapters wrapping the two backends; the articulated job kind end to end, including global texturing, targeted retexturing, and the joint-slider viewer.

Out of scope, by design, and not revisited: modifying the two texturing backends themselves; training or fine-tuning any model; a global Hunyuan articulated path (its bake back-projects six fixed views, so surfaces hidden in every view cannot receive real texture, and its mandatory xatlas re-unwrap reorders vertices in a way that complicates splitting a merged result); searching over articulated joint states or avoiding self-collision when opening joints; retexturing PartNet collision meshes; delivering transparent glass (TRELLIS.2 forces `alphaMode='OPAQUE'` on its output GLBs, so an opaque frosted look is the accepted compromise for windows and cabinet panes, not a bug to fix).

Deferred to a later phase, each gated separately if pursued (see section 16): a per-part detail pass that re-decodes the TRELLIS field for one flagged part alone; multi-reference conditioning (front and back images) for TRELLIS; Orient-Anything-V2 front detection for flat-mesh jobs (today it is articulated-only); enabling targeted retexturing by default once diagnostics show it is worth the cost; per-category VLM spec caching for homogeneous batches (all chairs in one batch share one call).

### 1.6 Environment and design constraints

- HPC cluster with no SLURM; jobs launch directly from bash. Current node: 2x NVIDIA A40 (46 GB each); `gpu_mode: dual|single` in `config.yaml` controls the VRAM strategy and the design degrades to a single GPU.
- All models are local open-weights, resolved from `config.yaml` model ids; the Hugging Face cache defaults to `.cache/huggingface` inside the repo and can be pointed at a shared/larger disk via `config.local.yaml` (`env.hf_cache`). Never write anything under `/home`.
- Four conda environments, each pinned by an incompatible dependency in one part of the stack, never hardcoded (adapters and workers read env names and paths from `config.yaml`):
  - `trellis2` (Python 3.10): the compiled TRELLIS.2 renderer stack (cumesh, o_voxel, nvdiffrast, flex_gemm, natten). Runs Stage R, Stage D, the Gradio app, the batch CLI, and the TRELLIS.2 backend adapters (flat and global).
  - `vlm`: the Stage V worker only. Qwen3.6-35B-A3B AWQ under vLLM needs torch >= 2.8, incompatible with the `trellis2` env's pinned torch 2.6.
  - `hunyuan3d`: the Hunyuan adapter only (`custom_rasterizer`, torch 2.5.1).
  - `orianyv2` (Python 3.11): only `scripts/orient_infer.py` (Orient-Anything-V2 front detection inside articulated Stage R), invoked as a short-lived subprocess.

## 2. Pipeline architecture and job directory

### 2.1 Stages

Seven idempotent stages, each reading and writing only its own per-mesh job directory under `jobs/<job_id>/`:

```
Stage R  (render)   mesh/URDF -> normalized geometry + VLM contact sheet + control maps + camera.json
Stage V  (vlm)      contact sheet -> appearance spec (category, materials, ref prompt/negative)
Stage D  (diffuse)  control map + spec -> N candidate reference images -> chosen RGBA reference
Stage P  (plan)     urdf jobs only: per-group material plan (vlm/plan.json); flat jobs skip
Stage T  (texture)  (mesh, chosen reference) -> textured GLB per backend
                     urdf jobs: one global TRELLIS.2 run -> per-group bake -> textured URDF + assembled.glb
Stage E  (eval)     turntable previews + alignment/consistency metrics + index row
                     urdf jobs also: articulated-state renders + per-group diagnostics
Stage J  (judge)    VLM compares the textured outputs and picks a winner + reasoning; every output is kept
```

Stage names accept the aliases R, V, D, P, T, E, J in the batch CLI and `job.json`.

### 2.2 `job.json` state machine

`pbr_texture_pipeline/jobdir.py` owns `job.json`: a schema-v2 document with a top-level `kind` (`"mesh"` or `"urdf"`), an `asset` section for urdf jobs, and per-stage `status` (`pending|running|done|error|needs_review`), `params`, `seed`, and timestamps for every stage in `STAGES = ("render", "vlm", "diffuse", "plan", "texture", "eval", "judge")`.
This makes every stage resumable and lets interactive (a human approves each transition) and batch (a policy approves it) modes share one code path.
Loading an old job dir backfills any stage added after it was created (`judge`, then `plan`) in memory, so old job dirs never `KeyError`; schema-v1 jobs predate articulated support and are treated as flat mesh jobs.

### 2.3 Job directory layout

```
jobs/<job_id>/                      # job_id = <mesh-stem>_<yyyymmdd-HHMMSS>, or for urdf jobs
                                     # <partnet-asset-id>_<category-slug>_urdf_<yyyymmdd-HHMMSS>
  job.json
  input/
    original.<ext>                  # untouched upload
    mesh_norm.glb                   # normalized single mesh (flat: Y-up axis swap; urdf: Z-up merged assembly)
    face_ranges.json                # urdf only: which merged-mesh face range belongs to which group
    mesh_norm_open.glb              # urdf only: merged mesh at the open-pose joint configuration
    asset/                          # urdf only: the rebuilt URDF file tree (app uploads); batch jobs
                                     # reference the source asset dir directly via job.json's asset.asset_dir
  views/
    view_{00..07}.png               # 8-azimuth shaded contact views for the VLM/user
    contact_sheet.png               # 2x4 grid actually sent to the VLM
    appearance_sheet.png            # urdf only: 2x4 sheet of the asset's existing appearance (skipped if blank)
  control/
    camera.json                     # yaw, pitch, r, fov_deg, resolution, repose_applied, R (3x3)
    depth.png  normal.png  canny.png  mask.png
    open/                           # urdf only: depth/normal/canny/mask + camera_open.json at open pose
    groups/                         # urdf only: coverage.json (per-group condition-camera visibility) + <gid>_mask.png
    visibility_{rest,open}.npz      # urdf only: per-view depth/visibility used for occlusion stats and per-texel masking
  vlm/
    transcript.json                 # full chat messages (system/user/assistant)
    spec.json                       # structured appearance spec (schema in section 4.2)
    caption.json                    # auto-generated mesh caption + captioning model id
    plan.json                       # urdf only: Stage P per-group material plan
    refine.json                     # urdf only: Stage E/T targeted-retexturing VLM grades, if run
  ref/
    candidate_{00..03}.png          # raw Qwen-Image outputs, each with a .json sidecar (seed, cn_scale, true_cfg_scale)
    chosen.png                      # RGB as generated
    chosen_rgba.png                 # after RMBG cutout; this is what backends receive
    generated_open/, chosen_rgba_open.png   # urdf only: the open-pose reference and its cutout
  groups/<gid>/
    mesh.glb                        # urdf only: merged per-group mesh in the link frame (Stage R)
    norm.json                       # urdf only: center/scale/face/area sidecar for verification
  textured/
    trellis2/textured.glb           # flat
    hunyuan/textured_mesh.obj|.mtl|*_albedo.jpg|...  textured_mesh.glb   # flat
    <backend>/groups/<gid>.glb      # urdf: per-group bake, mapped to the link frame
    <backend>/global/pass_a.json, pass_b.json   # urdf: field-decode bookkeeping
    <backend>/mobility_textured.urdf, assembled.glb   # urdf: textured URDF + rest-pose FK assembly
  previews/
    <backend>_turntable.mp4         # or 4-view PNG strip
    <backend>_condview.png          # render from the condition camera, for alignment eval
    <backend>_judgesheet.png        # 2x2 multi-view sheet (front/right/back/left) for Stage J
  eval/
    metrics.json
    states/<backend>/, diagnostics.json   # urdf only: articulated-state renders + per-group diagnostics
  judge/
    verdict.json                    # winner, method, label_map, criteria, reasoning
```

Only `mobility.urdf` plus its transitive file closure is required for urdf jobs; `semantics.txt`, `result.json`, `meta.json`, and `bounding_box.json` are optional, each with a fallback.
The authoritative layout is the `jobdir.py` docstring; this section mirrors it.

### 2.4 Resolving the textured output

`JobDir.output_glb(backend)` returns the textured GLB a backend wrote: `assembled.glb` if present (articulated jobs), else `textured.glb` (TRELLIS.2 flat) or `textured_mesh.glb` (Hunyuan flat), else any `*.glb` directly in the backend's `textured/<backend>/` directory (non-recursive, so an articulated job's per-group GLBs under `groups/` never pollute the fallback).
`JobDir.final_glb()` resolves to the Stage J winner's `output_glb`; before judging, it resolves only if exactly one backend produced a GLB.
Because articulated output is produced in this same shape, Stage E, Stage J, the joint viewer, and the gallery consume it with no branching on job kind.

## 3. Stage R: rendering

### 3.1 Reuse, do not rewrite

All rendering uses TRELLIS.2's nvdiffrast-based `MeshRenderer` (`TRELLIS.2/trellis2/renderers/mesh_renderer.py`), which returns `mask`, `depth`, and `normal` per camera at any resolution, plus the camera helpers in `TRELLIS.2/trellis2/utils/render_utils.py` (`yaw_pitch_r_fov_to_extrinsics_intrinsics`).
No Blender and no pyrender; headless GL is painful on this cluster and unnecessary.

### 3.2 Normalization

Flat meshes: load with `trimesh.load`, flatten any `Scene` via `.to_mesh()`, then apply the exact normalization from `Trellis2TexturingPipeline.preprocess_mesh` (vertices to `[-0.5, 0.5]`, axis swap `y' = -z, z' = y`, glTF Y-up to internal Z-up), saved as `input/mesh_norm.glb`.

Articulated assets: the PartNet-Mobility URDF world frame is already Z-up (wheels and feet sit at minimum z on all 5 test assets), so Stage R normalizes with `up="z"` (center and scale only, no glTF Y-up axis swap).
Feeding that geometry through the default Y-up swap tips assets onto their side and turns every yaw orbit (contact sheet, turntable, judge sheet) into a tumble; this is the single most consequential frame difference between the two job kinds.

Either way, all cameras are then defined in the same normalized frame the backends effectively assume.

### 3.3 Camera parameterization

`(yaw, pitch, r=2, fov=40)` exactly as `render_utils` defines: camera at `r * (sin y cos p, cos y cos p, sin p)`, look-at origin, up +Z.
`r=2, fov=40` matches TRELLIS.2's own render defaults, so framing statistics match what the backends saw in training.

### 3.4 Canonical front and the pose constraint

TRELLIS.2 and Hunyuan assume the condition image is shot from the mesh's canonical front.
For flat meshes, after the normalization axis swap, glTF forward (+Z toward viewer) maps to internal -Y, so the canonical front camera is `yaw = pi, pitch = 0` in `render_utils` coordinates; this is the default condition camera and was verified empirically (`scripts/verify_conventions.py`, gate 1 in section 15).

### 3.5 Front-view selection and re-posing

Flat meshes with an arbitrary file orientation (for example a raw `.ply`):

- Stage R renders 8 shaded views at azimuths `front + k*45 degrees`, pitch 15 degrees, plus a top view, into the contact sheet.
- The VLM is asked which panel shows the object's front (`front_view_index` in the spec); the user can override via the UI.
- If the chosen front differs from the canonical front, the interactive app re-poses the mesh by rotating about +Z so the chosen azimuth becomes yaw = pi, records the rotation `R` in `camera.json`, and textures the re-posed mesh; after texturing, `R^-1` is applied to the output GLB's vertices so the delivered asset keeps its original orientation. This is lossless because textures live in UV space, and one mechanism satisfies both backends.
- Small deviations are allowed without re-posing: the UI permits +/-30 degrees yaw and +/-20 degrees pitch nudges, which the backends tolerate.
- Batch mode never re-poses: `batch.py` always feeds the original mesh to Stage T and ignores the VLM's `front_view_index`, so an arbitrary-orientation mesh in a batch run is textured from whatever the file's front happens to be. This is a known, accepted limitation, not a design gap.

Articulated assets: PartNet-Mobility assets face contact-sheet panel 2 in the assembled URDF world frame (verified on all 5 test assets in the Z-up frame).
Stage R predicts the front panel with Orient-Anything-V2 (section 7.2.5): the model runs on the 8 contact-sheet panel renders, and the panel whose predicted azimuth is closest to 0 is picked as the front, gated by cross-panel agreement (predictions across panels 45 degrees apart should differ by the same steps; `articulated.orient.min_agreement_deg` bounds the allowed disagreement).
On a confident prediction, Stage R re-poses to that panel; otherwise it falls back to `articulated.front_panel` (2).
Either way the decision and the re-pose rotation are recorded in `camera.json` exactly like the flat-mesh re-pose bookkeeping, so Stage R pre-poses before rendering and the canonical camera sees the front.

### 3.6 Control maps

Rendered at 1024x1024 from the condition camera with `ssaa=2`:

- `depth.png`: the renderer returns camera-space z; converted to the ControlNet convention of normalized inverse depth (near = white, far = black), background black: `d_vis = (far_hit - z) / (far_hit - near_hit)` masked by `mask`. ControlNet-depth was trained on MiDaS-style relative inverse depth; wrong polarity produces inside-out objects.
- `normal.png`: `(n+1)/2` encoding as returned by the renderer; used for canny derivation and debugging, not as a ControlNet input.
- `canny.png`: `cv2.Canny` over the normal render (not depth), thresholds (100, 200), dilated 1 px. Normal discontinuities give clean part edges on smooth untextured geometry.
- `mask.png`: silhouette; used for alignment scoring and as an optional inpaint mask.

### 3.7 VLM contact views

Same renderer with simple headlight shading (`normal . view` grayscale) so the untextured mesh reads as a clay render, which VLMs classify reliably.

### 3.8 Articulated additions

- **Merged mesh and face ranges, written once**: Stage R persists the merged rest-pose mesh (`input/mesh_norm.glb`) together with `input/face_ranges.json`, recording which face range belongs to which group. The open-pose merged mesh is built with the identical group order, so the same face ranges apply to both meshes. Stage T loads these files and never rebuilds the merge, because face ranges are only valid against the exact face array they were recorded from; a rebuild that differed even slightly would silently assign textures to the wrong parts.
- **Open-pose geometry**: Stage R computes an opened joint configuration (section 7.2.6) and merges a second whole-object mesh at that configuration, rendering its own control maps (`input/mesh_norm_open.glb`, `control/open/*`, `camera_open.json`, same canonical camera and re-pose bookkeeping as the rest pose).
- **Per-group occlusion statistics**: for each group, Stage R estimates the fraction of surface area not visible from any of the 8 contact-sheet viewpoints, at rest pose and at open pose, written into the `asset.groups[]` records (`occluded_frac_rest`, `occluded_frac_open`). These values feed the Stage E diagnostics and the retexturing candidate selection (section 7.2.7).
- **Semantic labels persisted**: Stage R writes the sorted set of `semantics.txt` link labels into `asset.semantic_labels`; they often carry category evidence the URDF visual names lack (12085's `dishwasher_body` vs visual names like `frame`, `shelf`) and feed the Stage V metadata note (section 4.6). Jobs rendered before this field existed still work; the note builder tolerates its absence.

## 4. Stage V: VLM caption -> appearance spec

### 4.1 Model and env

Qwen3.6-35B-A3B (natively multimodal MoE, AWQ 4-bit) served by vLLM, in its own `vlm` conda env because the AWQ checkpoint needs torch >= 2.8, incompatible with the `trellis2` env's compiled renderer stack (pinned to torch 2.6).
The flow is caption then spec: turn 1 captions the mesh from the rendered views alone (category, parts, geometry; no invented materials), saved to `vlm/caption.json`; turn 2 receives the contact sheet plus that caption and emits the detailed `[SPEC]` block below.
The tag-regex structured extraction, the force-finalize turn, and the fallback-after-N-attempts philosophy are ported from `trellis_pbr/vlm_chat.py`.

### 4.2 Structured output: `spec.json`

Emitted inside a `[SPEC] ... [/SPEC]` block, regex-extracted, parsed with `json.loads`.
One auto-retry with "your JSON was invalid, re-emit" on parse failure, then a template fallback built from the category plus the last user message.

```json
{
  "category": "chair",
  "category_confidence": "high|medium|low",
  "front_view_index": 4,
  "materials": [
    {"region": "seat/back", "material": "walnut wood, satin finish"},
    {"region": "legs", "material": "brushed steel"}
  ],
  "style": "realistic, contemporary",
  "ref_prompt": "a photograph of a mid-century walnut wood chair with brushed steel legs, satin finish, product photography, centered, plain white background, soft even studio lighting, 8k, photorealistic",
  "negative_prompt": "cartoon, painting, illustration, text, watermark, cluttered background, harsh shadows, strong reflections, people, multiple objects"
}
```

urdf jobs with a metadata category carry two extra fields (section 4.6): `category_source: "metadata"` always, and `vlm_category` recording the VLM's own answer when it disagreed with the metadata.

### 4.3 System prompt: three enforced behaviors

1. Category inference: "You are shown clay renders of an untextured 3D object from 8 angles. First state what the object is and which panel shows its front."
2. Realism gate: propose only materials the object category is actually manufactured from. If the user requests an implausible material (a brick chair, a glass hammer handle), refuse in one sentence and offer plausible alternatives; stylized or fictional finishes are allowed only if the user explicitly insists after the warning.
3. Prompt discipline: the emitted `ref_prompt` must end with fixed boilerplate (single object, centered, plain white background, soft even studio lighting), shown to the model as a literal example. Background and lighting control is what makes RMBG cutouts and PBR decomposition work downstream.

### 4.4 Interactive chat loop

The opening turn is automatic: the VLM receives the contact sheet and emits a description plus an initial proposed spec, rendered beside the chatbot.
Each user message may regenerate the spec.
A "Finalize" button sends the force-finalize message.
The spec is editable as raw JSON before Stage D, so the human always has the last word.

### 4.5 Batch mode

Replaces the human with a fixed two-turn script: contact sheet plus "propose the final spec now, realistic materials only, then emit [SPEC]"; if no valid spec, one force-finalize retry, then the template fallback: `"a photograph of a {category} made of typical realistic materials, product photography, plain white background, soft even studio lighting"`.
`--material-hint "<text>"` optionally injects a user-level steer for the whole batch (for example "medieval, weathered" for a batch of props).
The spec fallback rate is a tracked batch metric.

### 4.6 Articulated: whole-object flow plus authoritative dataset metadata

Caption and appearance spec are produced from the whole-object renders exactly as for flat meshes; the prompt already describes the whole object, which is what global texturing needs.

On top of that, the dataset category of a urdf asset is treated as authoritative ground truth (2026-07-29; motivated by a dishwasher whose Stage V caption re-classified it as a sideboard and produced sideboard textures).
The metadata already on disk per asset is `meta.json` `model_cat` (parsed into `asset.category` by Stage R) and the `semantics.txt` per-link labels (persisted by Stage R as `asset.semantic_labels`).

The mechanism, all in `pbr_texture_pipeline/vlm.py`:

- **Metadata note**: `articulated_context(job)` builds one authoritative paragraph per job from the normalized category (`normalize_category` splits CamelCase and lowercases, so "StorageFurniture" becomes "storage furniture") plus the union of `asset.semantic_labels` and the group labels.
  The note states the category as ground truth and instructs the model to describe, not re-classify.
  It is given to BOTH turns: the caption turn's user message opens with it (the caption system prompt says dataset metadata is authoritative and never to contradict it), and the spec turn's opening user message repeats it.
  `articulated_context` also carries the appearance sheet, articulation summary, and keep_appearance flag, and is shared by `run_auto` (batch) and the VLM worker's `open` op (interactive), so both paths see identical context.
  An asset without a metadata category gets no note and behaves exactly as before.
- **Authoritative spec-turn phrasing**: the articulation summary states the category as dataset ground truth and instructs the model to set the spec's `category` field to exactly that value; the spec system prompt's category rule says a stated metadata category must not be re-classified.
- **Enforcement**: after spec extraction (and after any force-finalize or template fallback), `apply_category_metadata` force-sets `spec.category` to the normalized metadata category and stamps `spec.category_source = "metadata"`.
  A VLM answer that disagrees (substring match in either direction counts as agreement, so "built-in dishwasher" matches "dishwasher") is kept as `spec.vlm_category` and logged, never kept as the category.
  In batch mode, when the mismatching spec's `ref_prompt` also fails to mention the category, one corrective regenerate turn ("dataset metadata says this object is a {category}, not a {vlm_category}; re-emit the [SPEC] block") is spent before the force-set result is saved.
  The worker's `save` op applies the same force-set (without the retry, since the user is in the loop) for urdf jobs, so an interactive session cannot persist a spec whose category contradicts metadata.
- **Gate A16** (section 15.2) checks the invariant over existing job dirs.

Flat mesh jobs (`kind: "mesh"`) are unaffected.

## 5. Stage D: reference-image generation

### 5.1 Model and ControlNet choice

Base model: `Qwen/Qwen-Image`, the 20B MMDiT text-to-image foundation model, driven through `QwenImageControlNetPipeline` (diffusers >= 0.35).
Depth is the sole control signal, via the InstantX `Qwen-Image-ControlNet-Union` adapter (canny, soft-edge, depth, and pose in one checkpoint) run in depth mode: it locks silhouette and coarse 3D structure while leaving surface appearance free.
Unlike SDXL multi-ControlNet, the Qwen union conditions on a single control image at a time, so canny stacking does not apply; `canny.png` is rendered only for debugging.
Weights are fetched once by `scripts/download_controlnet.py` into the shared HF cache.
`batch.py` exposes `--diffusion-kind` with a `depth+canny` choice, but it is inert: `diffusion.py` keeps a single depth pipeline and logs "using depth only" regardless of the flag.

### 5.2 Generation settings

1024x1024 (matches the control render), bf16, `enable_model_cpu_offload()` (the 20B DiT plus its 7B Qwen2.5-VL text encoder exceed a single A40, so modules stream on demand and only one is GPU-resident at a time), 30 steps, `true_cfg_scale=4.0` (real classifier-free guidance; the negative prompt only takes effect above 1), `controlnet_conditioning_scale=0.9` default (UI slider 0.5-1.0; higher means tighter pose, flatter appearance).

### 5.3 Background handling: three-layer defense

The goal is that backend RMBG preprocessing becomes trivial and deterministic:

1. Prompt and negative boilerplate enforce a plain white background and soft even lighting; baked shadows are the main enemy of PBR decomposition, so "harsh shadows, strong reflections" are explicitly in the negative prompt.
2. The control depth map has a black (empty) background, which strongly biases the generator toward clean backdrops.
3. pbr-texture-pipeline runs RMBG-2.0 itself to produce `chosen_rgba.png` and passes RGBA to the backends; `Trellis2TexturingPipeline.preprocess_image` uses the alpha channel directly when present and never re-runs rembg, guaranteeing the same cutout everywhere. The UI shows the cutout for approval.

### 5.4 Candidate grid and reroll

Each roll produces 4 candidates (seeds `base_seed + i`, generated sequentially with batch size 1 to bound VRAM), shown as a 2x2 gallery.
The user clicks to choose; "Reroll" bumps `base_seed += 4`; sliders for conditioning scale and guidance persist per job.
Every candidate gets a JSON sidecar recording `{seed, prompt_hash, cn_scale, guidance, steps, controlnet}` for reproducibility.

### 5.5 Batch auto-selection policy

Generate 4 candidates, filter by silhouette IoU(RMBG cutout mask, `control/mask.png`) >= 0.85, then pick max CLIP(prompt, image).
If all fail the IoU gate, take the max-IoU candidate and flag the job `needs_review` (still textured unless `--strict`).

### 5.6 Articulated: open-pose reference

A second Qwen-Image plus depth-ControlNet generation from `control/open/depth.png`, same appearance prompt and seed, written to `ref/generated_open/` with its own RMBG cutout `chosen_rgba_open.png`.
Same VLM/manual selection flow as the rest-pose reference in the app; batch mode picks the same candidate index as the rest-pose choice.
Skipped entirely if `articulated.global.open_pose_pass: false`.

## 6. Stage P: material plan (articulated only)

A single inexpensive VLM call produces a per-group material plan (`vlm/plan.json`, extracted from a `[PLAN]` block with a catalog fallback on parse failure).
It is kept as material metadata: PBR hints, the app's per-group table, and context for the Stage E/T refine grading (section 7.2.7).
It does not drive texturing directly; global texturing (section 7.2) samples one shared decoded field regardless of the plan.
Flat mesh jobs mark this stage done with `{"skipped": true}`.

## 7. Stage T: texturing backends

### 7.1 Flat mesh adapters

The proven `pbr_compare` drivers (`run_trellis2.py`, `run_hunyuan.py`) are generalized into single-pair adapters with one contract:

```
adapter.texture(mesh_path, image_rgba_path, out_dir, seed, camera_json, params) -> {glb_path, logs}
```

Both run via subprocess (`conda run -n <env> python <adapter_script> ...`), never in the orchestrator process, because each backend has hard cwd requirements and subprocesses give free VRAM isolation and crash containment.

| Backend | Invocation | Pose handling |
|---|---|---|
| TRELLIS.2 | cwd = `TRELLIS.2/`; `pipe.run(mesh, image, seed=..., resolution=1024, texture_size=2048, preprocess_image=True)`; RGBA input makes preprocessing use the alpha cutout | None needed: the reference was rendered from the canonical front, or the mesh was re-posed. Pass the re-posed mesh when re-posing is active; un-rotate the output afterward. |
| Hunyuan3D-2.1 | cwd = `hy3dpaint/`; `Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))(mesh_path, image_path, output_mesh_path, use_remesh, save_glb=True)`; `--no-remesh` exposed | Same as TRELLIS.2. Seed is hardcoded to 0 in Hunyuan's own code; the adapter records `seed: 0` regardless of the requested seed, so seed sweeps only vary TRELLIS.2. |

Adapters also accept `--pairs-file` with many jobs, so a batch run loads each backend once (the `pbr_compare` sweep pattern).
A thin `backends/registry.py` maps backend name to `{script, env, cwd}`.

Hunyuan's `custom_rasterizer` segfaults (exit 139) at CUDA teardown, after writing all outputs and emitting its result line.
Stage T therefore judges adapter success by the emitted `[PBR_RESULT] ok:true` line plus the GLB existing on disk, never by subprocess exit code; the segfault is post-success process teardown, not a texturing failure, and is not fixed.

### 7.2 Articulated: global texturing

#### 7.2.1 Why global mode replaced per-part texturing

The original articulated pipeline textured each semantic part group independently: every group got its own reference image (a Qwen-Image material swatch or a crop of the global reference) and its own backend run, and the results were reassembled with forward kinematics.
In practice it produced poor results, with these observed failure modes:

1. Part textures were mutually independent: nothing enforced that a drawer front matched the cabinet body in wood grain, tone, or wear.
2. Part references were close-ups (swatches, crops) that did not show the part in its actual pose, so backends textured parts without knowing their orientation or role in the whole object.
3. All rendering (conditioning, eval, judge) happened at rest pose only, so surfaces revealed by articulation (drawer interiors, inside faces of doors) were never seen by any model, were textured without visual evidence, and were never evaluated.
4. Small parts (handles, knobs, hinges) each consumed a full backend run for a few hundred faces, and seams between adjacent parts were a persistent risk.

This per-part path was removed from the codebase entirely on 2026-07-28: no mode knob, no fallback.
The v1-only config keys it used (`texture_mode`, swatch settings, crop-ref thresholds, `hunyuan_part_ref`, `part_ref_mode`) and the `swatches.py` module were removed with it.
`catalog.py` (the material catalog and swatch prompts) survives only as the Stage P plan-fallback source.

#### 7.2.2 Core idea

Texture the object globally, but bake each part its own texture:

```
URDF
  -> parse links, forward kinematics to rest pose
  -> merge all visuals into ONE static mesh
     while recording which face ranges belong to which group  (input/face_ranges.json)
  -> ONE TRELLIS.2 run on the merged mesh
     conditioned on ONE reference image of the whole object in its actual pose
  -> keep the decoded 3D PBR field
     (do not bake a single shared texture atlas)
  -> bake EACH group its own full-resolution UV atlas
     by sampling the shared field
  -> map each baked group back to its link frame
  -> rewrite textured URDF + assembled.glb
```

The key enabler is a verified property of TRELLIS.2 texturing (`TRELLIS.2/trellis2/pipelines/trellis2_texturing.py`): it does not project the reference image onto the surface, it decodes a 3D voxel field of PBR values indexed by position, then samples that field at surface points to fill a UV atlas (`postprocess_mesh`).
Because color is a function of 3D position, any subset of the merged surface can be baked independently from the same field.

This directly addresses the failure modes above: global consistency comes automatically, because every part samples the same field conditioned on one whole-object reference; small parts get full-resolution atlases (a handle gets its own `per_link_texture_size` atlas instead of a small region of one shared atlas), which fixes the atlas side of small-part blur while field resolution remains a residual limit (below); texture bleeding in UV space is impossible, because parts never share an atlas; hidden surfaces get plausible values, because the field is defined everywhere in the volume, not only on surfaces that appear in a rendered view; and the expensive diffusion step runs once per object (twice with the open-pose pass), not once per group.

Residual limitations, accepted with mitigations: color mixing at contact surfaces (a drawer side and the cabinet wall it touches occupy nearly the same region of the field and receive near-identical colors, which matches real furniture and is usually acceptable; targeted retexturing, section 7.2.7, covers the worst cases); field resolution bounds detail (the field has a fixed voxel count for the whole object, so a handle spanning a few voxels gets a high-resolution but smooth, detail-free texture; Stage E records how many field voxels each group spans, which predicts this failure directly, and the phase-2 per-part detail pass, section 16, is the planned recovery).

#### 7.2.3 Scope decisions

1. One global TRELLIS.2 run, per-group atlases baked from the shared decoded field. This requires calling into TRELLIS.2 internals from the adapter (splitting the run into a field-decode step and a bake step, then calling the bake once per group); the backend is no longer an opaque subprocess that returns one finished GLB.
2. The open-pose pass ships as a first-class part of the pipeline: a second global pass with joints opened, so the reference image and shape conditioning actually see interiors (section 7.2.6).
3. TRELLIS.2 only. Urdf jobs are trellis2-only and Stage J records a walkover (see the out-of-scope note in section 1.5 for why a global Hunyuan path is not built).

#### 7.2.4 Stage T driver

Articulated Stage T is one `trellis2_global` adapter run per job (`pbr_texture_pipeline/backends/trellis2_global_adapter.py`):

1. **Load the merged mesh and face ranges**: Stage T consumes `input/mesh_norm.glb` and `input/face_ranges.json` exactly as Stage R wrote them; it never rebuilds the merge (section 3.8 explains why).
2. **Global pass A (rest pose)**: the adapter runs TRELLIS.2 through shape encoding, image conditioning, and field decoding on the merged mesh, using `chosen_rgba.png` and the real `camera.json` (with re-pose applied, like flat-mesh jobs), then stops before baking. The decoded field stays in memory.
3. **Global pass B (open pose)**: the same, on the open-pose merged mesh with `chosen_rgba_open.png` and `camera_open.json`. Skipped if `articulated.global.open_pose_pass: false`.
4. **Per-group bake**: for each group, inside the adapter process: slice the group's faces from the merged mesh using the recorded face ranges, and UV-unwrap the group submesh (`cumesh.uv_unwrap`, the same routine TRELLIS uses for input without UVs); bake the atlas (size `per_link_texture_size`) from the pass A field, sampling at the group's surface points expressed in pass A's normalized frame; if pass B ran, compute a per-texel rest-pose visibility mask (the same visibility test as the Stage R occlusion statistics, applied to each texel's surface point), re-sample texels hidden at rest from the pass B field in pass B's frame, blended over a band of `blend_band_texels` at the visibility boundary so no hard line crosses the part; fill empty texels by inpainting (`cv2.inpaint`, as TRELLIS does), emit `textured/trellis2/groups/<gid>.glb`, mapped back to the link frame.
5. **Tiny groups**: baked from the field like every other group, with a smaller atlas (`tiny_texture_size`). A full backend run for 20 faces was wasteful under the old per-part path, but a group bake costs almost nothing once the field exists, so that shortcut lost its rationale; the adapter writes its own neutral constant-PBR GLB only as a fallback when UV unwrapping fails or the geometry is degenerate.
6. **Assembly**: `assemble_backend` collects `groups/<gid>.glb` (already bbox-checked per bake), writes the textured URDF with collision symlinks (`articulated.collision_mode: symlink`), and assembles the rest-pose `assembled.glb`.

Adapter contract: the pairs-file schema gains a `global` pair kind carrying `{merged_mesh, merged_mesh_open, image, image_open, camera_json, camera_json_open, face_ranges, out_dir, groups: [{group_id, out_glb, texture_size}]}`.
`registry.py` treats it like any other pair; success is still the `[PBR_RESULT]` marker plus every expected group GLB existing on disk.

Resume granularity: one global field decode is the smallest unit of work; if any group bake is missing, the pass re-runs and re-bakes only the missing groups.
Persisting the decoded field to disk for finer-grained resume is off by default (`persist_field: false`); the field is several gigabytes and re-decoding takes minutes.
The adapter merges `pass_a.json`/`pass_b.json` metadata on partial (resume) runs instead of rewriting them, so a resumed job does not lose earlier groups' bake stats that Stage E diagnostics read.

TRELLIS.2 forces `alphaMode='OPAQUE'` on its output GLBs, so glass parts (windows, cabinet panes) cannot be textured as transparent; an opaque frosted look is the accepted compromise.

#### 7.2.5 Orient-Anything-V2 integration

Repo: vendored as the `Orient-Anything-V2` git submodule (NeurIPS 2025; built on VGGT).
API: `VGGT_OriAny_Ref(out_dim=900)` plus `utils/app_utils.inf_single_case(model, pil_ref, pil_tgt=None)`; input is a single image, output is `{ref_az_pred, ref_el_pred, ref_ro_pred, ref_alpha_pred}` (azimuth, elevation, in-plane rotation, symmetry class).
Checkpoint `rotmod_realrotaug_best.pt` (5.05 GB) from Hugging Face repo `Viglong/OriAnyV2_ckpt`.

Usage in Stage R (see also section 3.5): run the model on the 8 contact-sheet panel renders (backgrounds are already clean, so no background removal is needed); the predicted azimuth per panel identifies which panel faces the camera, and because panels are 45 degrees apart, cross-panel agreement is the confidence check; on a confident prediction, use that panel for the re-pose, otherwise fall back to `articulated.front_panel` (2); record the decision and the per-panel predictions in `camera.json`.
The symmetry class (`ref_alpha_pred`) is recorded for possible future multi-reference view-selection use; no current behavior depends on it.

Execution model: a dedicated `orianyv2` conda env (Python 3.11 per its README; the `bpy` dependency is only used by the demo's axis-overlay renderer and is not installed).
Invoked as a batch subprocess (`conda run -n orianyv2 python scripts/orient_infer.py --images ... --out json`), following the backend-adapter pattern: load the model once, process many images, print JSON.
Not a persistent worker; it runs for seconds per job during Stage R, on whichever GPU is free.
One-time setup script `scripts/download_orient_anything.py` places the weights in the shared HF cache.

The same front-view selection could also serve flat-mesh jobs, where the front view is currently picked by the VLM from contact-sheet panels; that is deferred (section 1.5, section 16).
Orient-Anything-V2 is enabled for urdf jobs only.

#### 7.2.6 Open-pose pass design

- **Opened configuration**: every movable joint is set to `articulated.global.open_pose_frac` (default 0.8) of its clamped range, prismatic and revolute alike. Fixed joints and continuous joints without limits stay at rest. One shared configuration for the whole object; no per-joint search.
- **Self-collision is not solved**: PartNet-Mobility assets open cleanly at fractions below the maximum, and the fraction is configurable per run; gate A12 (section 15.2) verifies the opened configuration is valid.
- **Combining the two passes is per-texel, not per-part**: every group's atlas is baked from the rest-pose field, and only texels whose surface points were hidden at rest are overwritten from the open-pose field, blended over `blend_band_texels` at the boundary. A whole-part rule was rejected because many groups mix visible and hidden surfaces in a single part (a drawer is one group containing both its front face and its interior); any per-part rule either recolors the drawer front from the open-pose field (risking a visible mismatch when closed) or leaves the interior with rest-pose guesses. Per-texel masking guarantees that every surface visible at rest keeps exactly the pass A result.
- **Consistency between passes**: same appearance prompt, same seed, same category. The two reference images can still differ in detail; this is acceptable because pass B contributes only texels hidden at rest, so any style difference is confined to interior surfaces and the narrow blend band.

#### 7.2.7 Evaluation and targeted retexturing

Texture globally first, measure, then repair only what is broken:

1. **Select candidates** (heuristics, no model calls): after Stage E, a group is a retexturing candidate if any of the following hold: its `occluded_frac_rest` exceeds `refine.occluded_frac_threshold` and no open-pose pass ran (its hidden surfaces were textured without visual evidence); it spans fewer than `refine.min_field_voxels` field voxels, or its blur estimate is below `refine.blur_flag_ratio` times the object median (the field resolution was too coarse for the part); its bake emitted warnings.
2. **Confirm** (one VLM call per candidate): per-group crops from the state renders for the candidates only are graded by the Stage V VLM (good / blurry / wrong material / missing texture) with reasoning, written to `vlm/refine.json`.
3. **Retexture** (targeted, through the same global path): the confirmed-bad `groups/<gid>.glb` files are moved aside, and the missing-GLB resume rule re-bakes exactly that set from a fresh-seed global decode (the same seed would deterministically reproduce the rejected texels). Reassembly and a fresh eval/judge follow.

Candidate selection and VLM confirmation (steps 1-2) always run, so how often retexturing would trigger is measured continuously.
Actual retexturing execution (step 3) is off by default (`articulated.refine.enabled: false`); the per-group field bake is expected to make it rarely necessary.
It can be forced per run with `--refine` and is exercised by gate A14.
Phase 2 (section 16) upgrades step 3 to the per-part detail pass, which restores full field resolution for the flagged part while keeping consistency through the shared reference image.

## 8. Stage E: evaluation

Flat mesh jobs: turntable previews (or a 4-view PNG strip), a condition-view render for alignment scoring, `eval/metrics.json` (silhouette IoU for pose fidelity; CLIP(prompt, chosen reference) and CLIP(prompt, condition-view render of the textured output) for semantic fidelity; LPIPS(chosen reference, condition-view render) for transfer fidelity; an albedo shadow-bake spot check on Hunyuan's exported `*_albedo.jpg`), and a top-level index row.

Articulated jobs additionally:

- **Articulated-state renders**: for each movable joint, render the assembled textured object at 0%, 50%, and 100% of the joint's range (canonical camera plus one three-quarter view), written to `eval/states/<backend>/`, using the same forward-kinematics code the joint viewer uses in the browser.
- **Per-group diagnostics** (`eval/diagnostics.json`): the occlusion statistics carried from Stage R, the number of field voxels each group spans (the direct predictor of field-resolution blur), the per-group mean texel gradient measured from the state renders (a render-based blur estimate), and any bounding-box or bake warnings. This is the input for retexturing candidate selection (section 7.2.7).

## 9. Stage J: judge

The pipeline does not stop at two textured meshes with the winner choice left to a human.
Stage J reuses the Stage V VLM (no new model) to compare the textured outputs and declare a winner per job, with reasoning; every textured output is kept on disk.

Three phases per job, split across envs like the rest of the pipeline:

1. **Sheet prep** (`trellis2` env, `pbr_texture_pipeline/judge.py`): render `previews/<backend>_judgesheet.png` from each textured GLB, a 2x2 grid (front/right/back/left, `judge.sheet_yaws_deg`, 512 px tiles composed to 1024x1024), reusing `eval.load_output_colored` and `rendering.render_appearance`. In batch this runs in the batch process before the VLM worker spawns; in the app it is an imaging-worker op (`judge_sheets`), never inside the VLM worker, which has no renderer.
2. **Verdict** (`vlm` env worker, `vlm.run_judge`): one deterministic VLM call with the two sheets plus `ref/chosen.png` as context only, `[VERDICT]{json}[/VERDICT]` extraction mirroring the `[SPEC]` machinery, one force-finalize retry, then fallback. Criteria in strict priority order: seams/projection artifacts, sharpness, lighting neutrality, cross-view coherence, material plausibility, reference resemblance (tiebreaker only). A/B labels are assigned by a deterministic per-job md5 swap (`judge.label_assignment`) to fight position bias; a tie is not a valid answer.
3. **Materialization** (orchestrator, `judge.finalize`): write `judge/verdict.json` (schema: winner, method `vlm|walkover|fallback_default`, label_map, confidence, criteria, reasoning, model, raw_response) and record the winner in `job.json`. Both `textured/<backend>/` directories stay in place; `JobDir.final_glb()` resolves to the winner's GLB. The verdict is a recommendation with reasoning, not an elimination; the runner-up remains a fully valid output.

Degenerate cases: one GLB produces a walkover verdict without a VLM call (status `done`); zero GLBs is stage `error`; a VLM failure or refusal after one retry falls back to `judge.default_winner` with status `needs_review` and method `fallback_default`.
Re-running judge re-finalizes idempotently with the same winner (crash recovery when `verdict.json` exists but judge status is not `done`, with no VLM call).
Judge does not require Stage E to be `done`: it renders its own sheets from texture outputs only, preserving stage independence.
`vlm.limit_mm_per_prompt_images: 3` covers the three judge images; the sheets fit comfortably inside `max_model_len: 16384`.

Articulated jobs: the judge sheet gains one row, the object at open state (all movable joints at 100% of their clamped range), so the VLM sees interiors and moving-part boundaries when picking a winner.
Because urdf jobs are trellis2-only (section 7.2.1, section 7.2.3), Stage J always records a walkover for them; mechanics are otherwise unchanged.

## 10. Gradio app and GPU memory strategy

### 10.1 Process architecture

The Gradio app (`app.py`) is a thin orchestrator with zero CUDA in-process.
Three worker types, IPC via JSON-lines over stdin/stdout (`workers/ipc.py`; request `{op, args}` -> response `{ok, paths}`):

1. **Imaging worker** (persistent subprocess, GPU 1 in dual mode): nvdiffrast rendering (Stage R), Qwen-Image + ControlNet (20B, bf16, model CPU offload, ~40 GB peak during a diffuse), RMBG-2.0. One process because all are needed in the same interactive phase and rendering itself is cheap (~1 GB).
2. **VLM worker** (persistent subprocess, GPU 0): Qwen3.6-35B-A3B AWQ under vLLM in the dedicated `vlm` env (~20 GB weights plus a bounded pre-allocated pool), kept separate so either worker can be dropped independently; it stays resident so caption, spec, chat, and judge calls are always live.
3. **Texturing workers** (ephemeral subprocesses, GPU 1): the backend adapters, launched per texture request, exit when done (~21-24 GB peak). They share GPU 1 with the imaging worker, but Stage D and Stage T never overlap for a job, so `manager.py` drops the imaging worker's Qwen-Image model (`unload`) before a texture run and it reloads lazily on the next diffuse.

`manager.py` owns GPU placement (via `CUDA_VISIBLE_DEVICES` per worker environment), VRAM polling via `nvidia-smi` (queue rather than OOM; the node is shared with no SLURM arbitration), and this unload/reload dance.

### 10.2 VRAM budgets

- Dual A40 (current node): GPU 0 holds the VLM (~20 GB AWQ weights plus vLLM's pre-allocated pool) plus the renderer, resident so caption/spec/chat/judge stay live; GPU 1 runs Qwen-Image Stage D (model CPU offload, ~40 GB peak) and, on demand, texturing. The manager drops the imaging worker's Qwen-Image model before each texture run; the ~40-60 s reload cost is paid once per stage switch.
- Single GPU fallback (`gpu_mode: single`): the orchestrator enforces a stage-exclusion rule. Before launching a texturing worker it suspends both persistent workers (via `unload_model()` or termination) and restarts them lazily on the next chat or reroll request. The reload cost is paid once per stage switch.
- `manager._wait_for_vram` polls every 5 s but proceeds anyway after a 1800 s (30-minute) timeout, at OOM risk, and skips the check entirely if `nvidia-smi` is unavailable.

### 10.3 UI layout

A single `gr.Blocks` wizard, one job per session, state is `gr.State(job_id)` only; everything else lives on disk in the job dir, so sessions survive app restarts and batch jobs are browsable identically.

- **Tab 1, Mesh**: single-file upload for flat meshes, or a multi-file URDF upload that validates the reference closure for articulated assets; `gr.Model3D` of the blank mesh; contact sheet gallery; front-view radio (pre-filled from the VLM's `front_view_index`); yaw/pitch nudge sliders; "Render control maps" button with depth/canny previews. Re-posing is automatic whenever a non-front panel is chosen; there is no separate re-pose checkbox.
- **Tab 2, Appearance chat**: `gr.Chatbot` plus message box; live spec panel (editable raw JSON); "Finalize spec" button.
- **Tab 3, Reference image**: control map thumbnail; editable prompt/negative pre-filled from the spec; seed, CN scale, guidance sliders; 2x2 candidate gallery; "Reroll"; chosen image plus cutout preview; "Approve".
- **Tab 4, Texture and review**: backend checkboxes (trellis2 / hunyuan); per-backend params (resolution, texture size, no-remesh); Stage P (for urdf jobs) runs automatically before the global bake when the Texture button is pressed; the joint-slider viewer core (`articulated/viewer.py`) is embedded as FastAPI iframe routes for urdf jobs, alongside per-group tables, pass A/B reference images, and per-backend `gr.Model3D` viewers; "Judge outputs" runs Stage J and displays the winner's viewer by default, while both backends' GLBs remain on disk and downloadable from the job directory. There are no in-app download buttons and no "send back to Tab 3" control; rich visual review and downloads live in `scripts/build_viewer.py` and the job directory itself.
- **Tab 5, Jobs browser**: a text-only table over `jobs/*/job.json` for reviewing batch output (winner column from Stage J, approve/flag toggle written back to `job.json`); reference thumbnails and preview strips are not built here, they live in `scripts/build_viewer.py`.

## 11. Batch CLI

```bash
conda run -n trellis2 python -m pbr_texture_pipeline.batch \
  --meshes 'meshes_run/*.glb' --jobs-root jobs/ \
  --backends trellis2,hunyuan \
  --stages render,vlm,diffuse,plan,texture,eval,judge \
  --candidates 4 --seed 42 \
  --material-hint "clean, factory-new" \
  --select iou+clip \
  --gpu-mode dual \
  --limit 100 --resume
```

`--meshes` accepts flat meshes (`.glb`/`.obj`/...) and `mobility.urdf` files in the same glob; a `.urdf` path creates a urdf-kind job.

Full flag list: `--meshes` (required), `--jobs-root` (default `jobs`), `--backends` (default `trellis2`), `--stages` (default `render,vlm,diffuse,plan,texture,eval,judge`; accepts the R/V/D/P/T/E/J aliases), `--candidates`, `--seed`, `--material-hint`, `--select {iou+clip,iou}`, `--gpu-mode {dual,single}`, `--limit`, `--resume`, `--strict`, `--spec-cache-by-category`, `--diffusion-kind {depth,depth+canny}` (inert, section 5.1), and the articulated-only `--keep-appearance` (steer the VLM toward observed existing materials), `--group-by {semantic,link}`, `--refine` (force targeted retexturing on for this run).
There is no `--part-ref-mode` flag; it was removed with the per-part texturing path.

Execution is stage-major for model-load efficiency, mirroring the `pbr_compare` sweeps:

1. Stage R for all meshes (renderer only).
2. Stage V for all (VLM loaded once; two-turn auto script per section 4.5).
3. Stage D for all (Qwen-Image loaded once; auto-select per section 5.5; IoU failures flagged `needs_review` but still textured unless `--strict`).
4. Stage P for articulated jobs (one VLM worker over all urdf jobs; flat jobs are marked skipped).
5. Stage T grouped by backend (each backend loaded once, iterating over all approved pairs via `--pairs-file`); for urdf jobs this is the one global `trellis2_global` run per job (section 7.2.4).
6. Stage E: previews, `metrics.json`, articulated-state renders and diagnostics for urdf jobs, and a top-level index (HTML). Targeted retexturing (section 7.2.7) runs as a sub-step inside this stage, gated by `--refine` or `articulated.refine.enabled`; it is not a separate `--stages` token.
7. Stage J: judge sheets for all jobs first (renderer in the batch process), then one VLM worker over all contested jobs; walkovers skip the VLM; the index is rewritten so its winner column is fresh.

`--resume` skips any stage marked done in `job.json`.
On dual GPU, Stage D (GPU 0-ish workload) and Stage T for completed jobs (GPU 1) could overlap in principle; execution stays sequential for simplicity.

## 12. Configuration reference (`config.yaml`)

Every module reads paths and defaults from `config.yaml`; nothing is hardcoded.

- `gpu_mode`: `dual` or `single` (section 10.2).
- `env`: env names and interpreter paths for `trellis2` (primary: render, diffuse, app, batch, and the TRELLIS.2 adapters), `vlm_name` (Stage V/J worker), `orient_name` (Orient-Anything-V2 subprocess); `hf_cache` and `torch_cache` paths (both outside `/home`).
- `backend_envs`: `trellis2` and `hunyuan` (`hunyuan3d`) subprocess env names for Stage T adapters.
- `models`: HF repo ids for `vlm`, `qwen_image`, `rmbg`, `trellis2`, `clip`, `controlnet`, `orient`, all resolved inside `hf_cache`.
- `repos`: local paths to the `trellis2`, `hunyuan`, and `orient` backend/tool repos (adapters' hard cwd requirements).
- `render`: resolution, ssaa, `r`/`fov_deg` camera defaults, canonical yaw/pitch, contact-view pitch, canny thresholds.
- `vlm`: vLLM engine knobs (`max_model_len`, `gpu_memory_utilization`), per-call token budgets (`caption_max_new_tokens`, `spec_max_new_tokens`), and `limit_mm_per_prompt_images` (3, sized for Stage J's three images).
- `diffusion`: candidate count, seed, `cn_scale`, `guidance` (`true_cfg_scale`), steps, resolution.
- `select`: `iou_threshold` for batch auto-selection.
- `texture`: `trellis2_resolution`, `trellis2_texture_size`, `hunyuan_max_num_view`, `hunyuan_resolution`.
- `articulated`: the URDF job configuration, all keys top-level under `articulated:` unless noted:
  - `global`: `open_pose_pass`, `open_pose_frac`, `per_link_texture_size`, `tiny_texture_size`, `blend_band_texels`, `persist_field`, `multi_ref` (phase-2 experiment flag, unused today).
  - `orient`: `enabled`, `env`, `ckpt_file`, `min_agreement_deg`.
  - `refine`: `enabled` (execution; selection and confirmation always run), `blur_flag_ratio`, `occluded_frac_threshold`, `min_field_voxels`.
  - `group_by` (`semantic` default, or `link`), `front_panel` (fallback front panel, 2), `keep_appearance`, `rest_pose` (`zero_clamped`, the only supported policy), `min_group_faces` / `min_group_area_frac` (tiny-group thresholds), `appearance_max_edge_frac`, `plan_max_new_tokens`, `collision_mode` (`symlink`).
- `judge`: `default_winner`, `max_new_tokens`, `sheet_yaws_deg`, `sheet_tile_res`, `sheet_pitch_deg`.

## 13. Risks, invariants, and gotchas

1. **Canonical-front convention** (flat meshes): `yaw = pi` after the axis swap, verified empirically in gate 1 and load-bearing for every flat-mesh render; a change to the normalization axis swap must be re-verified against this gate.
2. **Depth encoding polarity**: analytic depth must be normalized to MiDaS-style inverse depth (near = white, far = black) or ControlNet produces inside-out objects; verified visually in gate 1.
3. **Baked lighting in generated references**: even with prompt discipline, the diffusion model bakes shadows and highlights, which PBR backends may transfer into albedo. Mitigations: negative-prompt terms, lower CN-scale trials, albedo inspection on Hunyuan's exported `*_albedo.jpg`.
4. **VLM JSON reliability**: the VLM occasionally emits malformed JSON; mitigated by retry plus template fallback, with the fallback rate tracked as a batch metric.
5. **Hunyuan determinism**: seed fixed at 0 in Hunyuan's own code, so seed sweeps only vary TRELLIS.2; document, do not fight.
6. **Hunyuan teardown segfault**: `custom_rasterizer` segfaults (exit 139) at CUDA teardown after every output is written and the result line is emitted; judge success by the `[PBR_RESULT] ok:true` marker plus the GLB on disk, never by subprocess exit code, and do not try to fix the segfault.
7. **Non-manifold or multi-part flat input meshes**: `Scene.to_mesh()` flattening can merge parts oddly; not a concern for articulated assets, which are parsed per-link instead.
8. **Env drift**: adapters and workers must read paths and env names from `config.yaml`, never hardcode them, because the env-to-stage mapping has changed twice already (Pixal3D removal, the Stage V vLLM upgrade) and will likely change again.
9. **Shared-node GPU contention without SLURM**: check free VRAM and queue rather than OOM (section 10.1); the 1800 s wait timeout is a real risk window, not a guarantee.
10. **PartNet-Mobility Z-up world frame**: articulated normalization uses `up="z"` with no glTF Y-up axis swap; using the flat-mesh swap on articulated geometry tips assets onto their side and turns every yaw orbit into a tumble (section 3.2). This is the most consequential single divergence between the two job kinds and the first thing to check if an articulated render looks wrong.
11. **`to_link` frame chain**: the global adapter bakes in the TRELLIS field frame and maps each group to its link frame with the pair's `to_link` matrix, composed orchestrator-side; a `to_link` bounds mismatch fails the group loudly. Bbox shrinkage up to 5% of extent is tolerated, because `cumesh.uv_unwrap` welds sliver faces on low-face-count groups (seen on 48-face PartNet parts); gate A9 pins the frame chain it inverts.
12. **`face_ranges.json` is a recording, not a re-derivable fact**: it is only valid against the exact merged-mesh face array Stage R wrote it from; Stage T must load it, never rebuild the merge (section 3.8).
13. **Resumed articulated jobs**: the adapter merges `pass_a.json`/`pass_b.json` metadata on partial runs instead of rewriting them; losing this would silently drop earlier groups' bake stats that Stage E diagnostics depend on.
14. **Color mixing at contact surfaces** is inherent to a shared field (section 7.2.2); accepted and documented, with targeted retexturing covering the worst cases.
15. **Open pose can reveal geometry PartNet never modeled** (missing interior faces render as holes); this affects the depth maps, not the bake. Gate A12 catches assets where the open-pose pass is not worth running, and the judge sees the open state as a backstop.
16. **Open-pose reference quality**: Qwen-Image with depth ControlNet has likely never seen a depth map of, say, a cabinet with every drawer and door open at once, so the pass B reference may come out confused, and it is the sole visual evidence for interior surfaces. Mitigations: the judge sees the open state, targeted retexturing covers bad interiors, and gate A13 includes inspection of the pass B references.
17. **Transparency**: TRELLIS.2 forces `alphaMode='OPAQUE'` on output GLBs; an opaque frosted look for glass parts is the accepted compromise, not a bug to fix (section 1.5, section 7.2.4).
18. **Hours-long articulated Stage T runs** on large assets vs. Gradio session drops: all state lives in the job dir and resume is group-granular, so a dropped session loses no completed work; large batches are still better suited to the batch CLI than the app.

## 14. Project layout

```
pbr-texture-pipeline/
  PRD.md                       # this document
  CLAUDE.md                    # quick-reference for agents: commands, architecture, gotchas
  config.yaml                  # gpu_mode, model ids, defaults, env names, repo paths, articulated: section
  pbr_texture_pipeline/
    __init__.py
    config.py                  # config.yaml loader
    jobdir.py                  # job.json schema, state machine, path helpers
    rendering.py                # normalization, cameras, contact sheet, control maps (TRELLIS.2 renderer)
    vlm.py                      # Qwen3.6 via vLLM: caption -> spec, chat override, extraction, plan, judge verdict, refine grading
    diffusion.py                 # Qwen-Image + ControlNet candidate generation, RMBG cutout, scoring
    judge.py                    # Stage J: judge sheets, A/B label assignment, verdict finalize
    eval.py                      # previews, IoU/CLIP/LPIPS metrics, index, output loading
    articulated/                # URDF job kind
      urdf.py                   # URDF parse, grouping, FK (rest + open pose), textured-URDF rewrite, assembly
      appearance.py              # existing-appearance rendering, group visibility
      catalog.py                 # material catalog (Stage P plan fallback)
      orient.py                  # Orient-Anything-V2 subprocess wrapper, cross-panel agreement
      stages.py                  # articulated stage drivers shared by batch.py and app.py
      viewer.py                  # joint-slider viewer core (standalone script + app iframe)
    workers/
      imaging_worker.py          # persistent: rendering + Qwen-Image + RMBG (GPU 1 in dual mode)
      vlm_worker.py               # persistent: Qwen3.6 (GPU 0)
      manager.py                  # GPU placement, VRAM polling, unload/reload orchestration
      ipc.py                       # JSON-lines protocol
    backends/
      registry.py                 # name -> {script, env, cwd}
      _adapter_common.py           # shared adapter helpers
      trellis2_adapter.py           # flat mesh: generalizes pbr_compare/run_trellis2.py
      trellis2_global_adapter.py    # articulated: field decode + per-group bake
      hunyuan_adapter.py
      visibility.py                 # occlusion/visibility helpers shared by Stage R and the global adapter
    app.py                        # Gradio orchestrator (Tabs 1-5)
    batch.py                      # CLI (section 11)
  scripts/
    verify_conventions.py         # gate 1: front-view / depth-polarity checks
    verify_articulated.py          # gates A1, A9, A11, A12a/b, A16
    verify_global_bake.py           # gate A10: per-group vs whole-mesh bake
    articulated_viewer.py           # standalone joint-slider viewer server for one job
    drive_pipeline.py                # headless driver for app callbacks (e2e-test skill)
    orient_infer.py                  # Orient-Anything-V2 subprocess entry point
    download_controlnet.py
    download_orient_anything.py
    download_vlm.py
    stage_run_meshes.py               # symlink/decimate meshes into meshes_run/
    build_viewer.py                    # browsable HTML gallery of textured results
    prep_partnet.py                     # PartNet-Mobility test asset staging
  meshes/                        # pristine test meshes (never modify)
  meshes_run/                    # staged/decimated run set produced by stage_run_meshes.py
  partnet_mobility/              # pristine articulated test assets (never modify)
  jobs/                          # runtime output (gitignored)
```

## 15. Verification and gates

### 15.1 Flat-mesh gates, in order

1. Conventions check (`verify_conventions.py`): on `pbr_compare/inputs/hunyuan_case1/mesh.glb`, render the canonical-front depth/normal and confirm the render visually matches `image.png`'s viewpoint; validate depth polarity visually against a MiDaS sample.
2. Rendering smoke: knight `.ply` produces an 8-view contact sheet and non-degenerate control maps (mask coverage 20-80% of frame).
3. VLM smoke: contact sheets for knight (expect "knight statue / armor", metal palette) and hunyuan_case1 yield a valid `spec.json` on first-or-retry in >= 9 of 10 runs.
4. Diffusion smoke: 4 candidates for knight; silhouette IoU(candidate cutout, control mask) >= 0.85 for >= 2 of 4; pose lock confirmed visually.
5. End-to-end vs. reference baseline: texture knight and hunyuan_case1 with generated references through both backends and compare side by side against the existing `pbr_compare/outputs/*` (ground-truth photo references) using `pbr_compare/EVALUATION_GUIDE.md`'s 10 criteria; this isolates reference-generation quality from backend quality.
6. Automated per-job metrics (`eval/metrics.json`): silhouette IoU, CLIP(prompt, reference), CLIP(prompt, condition-view render), LPIPS(reference, condition-view render), albedo shadow-bake spot check.
7. Batch dry run: 5-8 meshes spanning categories with `--backends trellis2`, reviewed via Tab 5; success is >= 80% of jobs needing no manual intervention.

### 15.2 Articulated gates

Gate A1 (FK, grouping, and metadata-optional-fallback correctness) is still live and automated (`scripts/verify_articulated.py` with no `--jobs-root`), because URDF parsing and grouping did not change when the per-part texturing mechanism was replaced.
The remaining original gates (A2-A8: per-part Stage R dry run, denorm/swatch round trip, material-plan-at-scale, the Hunyuan swatch out-of-distribution gate, the per-part end-to-end and app gates) tested the removed per-part swatch/crop-ref mechanism directly and are retired; they do not apply to global texturing and are not run.

Current gates, extending from A9 (`scripts/verify_articulated.py --jobs-root jobs/` covers A9/A11/A12a/A12b/A16; `verify_global_bake.py` covers A10; A13/A14 run through the `e2e-test` skill in `--urdf` mode):

- **A9 merge/split round trip** (no GPU): slicing `input/mesh_norm.glb` by `input/face_ranges.json` (the persisted files, not a rebuild) reproduces per-group meshes that match direct group-mesh construction after undoing normalization, re-pose, and forward kinematics, on all 5 PartNet test assets.
- **A10 per-group bake correctness** (GPU): on one cabinet, per-group bakes from a single pass A produce an assembled result visually consistent with one whole-mesh bake from the same field (no seams beyond contact edges), and handles are measurably sharper (per-group texel density reported).
- **A11 orientation agreement**: Orient-Anything-V2 front selection matches `front_panel: 2` on all 5 PartNet assets; on disagreement it falls back and logs the discrepancy.
- **A12 open-pose validity**, two parts: (a) no GPU, the opened configuration produces finite bounds and every joint value stays inside its limits; (b) GPU, condition-camera mask coverage at open pose is at least that of rest pose on drawer/door assets, and interiors are visible in `control/open/depth.png`.
- **A13 end to end**: the `e2e-test` skill in `--urdf` mode passes, with a judge verdict written (walkover to trellis2 by design). The pass B reference images are inspected as part of this gate.
- **A14 targeted retexturing**: with retexturing forced on (`--refine`), at least one group completes the full sequence: selected by heuristics, confirmed by the VLM, re-baked from a fresh-seed global decode, and reassembled.
- **A15 (blind comparison against the removed per-part pipeline)**: dropped by decision; global shipped without this comparison.
- **A16 spec category matches metadata** (no GPU): for every urdf job with a metadata category and a Stage V `vlm/spec.json`, `normalize_category(spec.category)` matches the metadata category and `spec.category_source == "metadata"` (section 4.6). Run against pre-fix job dirs it fails, confirming it detects the original mis-classification bug.

All currently-applicable gates (the flat gates, A1, and A9-A14) have passed on the 5 PartNet-Mobility test assets; A16 has passed on Stage V runs made after the metadata-authority change (12085, 8930), and pre-fix job dirs fail it by design.

## 16. Phases and deferred work

- **Shipped**: global articulated texturing with per-group field bake and per-texel pass masking, the open-pose pass, Orient-Anything-V2 front detection, articulated-state renders in Stage E/J, retexturing diagnostics with execution off by default.
- **Deferred, each to be gated separately if pursued**:
  - **Per-part detail pass**: for a part flagged by the field-resolution diagnostics, re-run the TRELLIS field decode on that part alone, so it gets the entire field to itself, conditioned on a crop of the whole-object reference image rather than a material swatch. Detail returns without losing consistency, because the part's reference comes from the same picture as everyone else's.
  - Multi-reference conditioning for TRELLIS (`get_cond` already accepts a list of images; feed front and back references).
  - Orient-Anything-V2 for flat-mesh front selection.
  - Enabling targeted retexturing by default, if the diagnostics show it triggers often enough to matter.
  - Per-category VLM spec caching for homogeneous batches.

## 17. Key reference files

- `TRELLIS.2/trellis2/renderers/mesh_renderer.py` (submodule) - control-map rendering API (mask/depth/normal).
- `TRELLIS.2/trellis2/pipelines/trellis2_texturing.py` (submodule) - mesh normalization and alpha-path image preprocessing that Stage R replicates and the global adapter's field decode/bake split builds on.
- Prior internal R&D (`pbr_compare`, `trellis_pbr`, `Qwen-Image` usage notes; not part of this repo) originated the verified backend APIs, the VLM chat patterns `pbr_texture_pipeline/vlm.py` ports, and the Qwen-Image diffusers usage Stage D ports.

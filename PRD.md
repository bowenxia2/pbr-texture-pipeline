# PRD: pbr-texture-pipeline - VLM-guided texturing of articulated 3D assets

Status: implemented and verified.
This is the sole authoritative spec for pbr-texture-pipeline, covering the full seven-stage pipeline for articulated URDF assets (PartNet-Mobility and Articraft-10K).
It replaces the earlier two-document split (`PRD.md` draft plus `PRD_articulated_v2.md`); that split existed because articulated texturing went through a full redesign (per-part texturing, shipped 2026-07-21, replaced wholesale by global texturing on 2026-07-28) and the two documents were merged back into one once the design stabilized.
Historical context that explains why a design was chosen is kept where it prevents a regression to a discarded approach; a full changelog of every intermediate update is not.

## 1. Overview and goals

### 1.1 Problem

One working texturing backend is available, taking (untextured mesh, reference image) and producing a PBR-textured mesh: TRELLIS.2 texturing (vendored as the `TRELLIS.2` git submodule).
Two earlier backends (Hunyuan3D-2.1 paint and Pixal3D `texture_inference`) were removed during development: Pixal3D was the weakest and went first; Hunyuan had no viable articulated path (its bake back-projects six fixed views, so surfaces hidden in every view cannot receive real texture, and its mandatory xatlas re-unwrap reorders vertices in a way that breaks splitting a merged result) and was removed when the pipeline became articulated-only.
Historically the reference image had to be supplied by hand.
pbr-texture-pipeline adds the missing upstream stage: given only a blank URDF asset, generate a category-aware, realistic, pose-matched reference image, then drive the texturing backend with it.

### 1.2 Solution shape

1. Render the blank URDF asset so a VLM can see it.
2. A local VLM (Qwen3.6-35B-A3B, AWQ 4-bit, served by vLLM) first captions the asset from the rendered views, then emits a realistic appearance prompt from the contact sheet plus that caption.
   Category-awareness is enforced: a chair can be wood, plastic, or metal, but never brick.
   The Gradio chat survives as an interactive override on top of the auto-generated spec.
3. A local diffusion model (Qwen-Image + depth+canny multi-ControlNet) generates a reference image conditioned on renders of the asset, so the image pose and edge detail exactly match the asset from the chosen camera.
4. The (merged mesh, reference image) pair drives a single global TRELLIS.2 texturing run (section 7), producing per-group atlases baked from the shared decoded field.
5. The same VLM judges the textured output; since there is only one backend, Stage J records a walkover. Every output is kept on disk.

### 1.3 Modes

- Interactive: one URDF asset at a time, human in the loop at every stage, via Gradio.
- Batch: many URDF assets, the VLM auto-proposes the prompt, candidates are auto-selected, the backend runs unattended, and a human reviews afterward in a jobs browser or the HTML gallery.

### 1.4 Scope

In scope and built: reference-image generation (render, VLM, diffusion); orchestration, per-job state, the Gradio app, the batch CLI, and the evaluation and judge stages; the TRELLIS.2 backend adapter; the full articulated pipeline end to end, including global texturing, targeted retexturing, and the joint-slider viewer.

Out of scope, by design, and not revisited: modifying the TRELLIS.2 texturing backend itself; training or fine-tuning any model; searching over articulated joint states or avoiding self-collision when opening joints; retexturing PartNet collision meshes; delivering transparent glass (TRELLIS.2 forces `alphaMode='OPAQUE'` on its output GLBs, so all glass parts get a frosted opaque look, which is the accepted compromise).

Deferred to a later phase, each gated separately if pursued (see section 16): a per-part detail pass that re-decodes the TRELLIS field for one flagged part alone; multi-reference conditioning (front and back images) for TRELLIS; enabling targeted retexturing by default once diagnostics show it is worth the cost; per-category VLM spec caching for homogeneous batches (all chairs in one batch share one call).

### 1.5 Environment and design constraints

- HPC cluster with no SLURM; jobs launch directly from bash. Current node: 2x NVIDIA A40 (46 GB each); `gpu_mode: dual|single` in `config.yaml` controls the VRAM strategy and the design degrades to a single GPU.
- All models are local open-weights, resolved from `config.yaml` model ids; the Hugging Face cache defaults to `.cache/huggingface` inside the repo and can be pointed at a shared/larger disk via `config.local.yaml` (`env.hf_cache`). Never write anything under `/home`.
- Three conda environments, each pinned by an incompatible dependency in one part of the stack, never hardcoded (adapters and workers read env names and paths from `config.yaml`):
  - `trellis2` (Python 3.10): the compiled TRELLIS.2 renderer stack (cumesh, o_voxel, nvdiffrast, flex_gemm, natten). Runs Stage R, Stage D, the Gradio app, the batch CLI, and the TRELLIS.2 backend adapter.
  - `vlm`: the Stage V worker only. Qwen3.6-35B-A3B AWQ under vLLM needs torch >= 2.8, incompatible with the `trellis2` env's pinned torch 2.6.
  - `orianyv2` (Python 3.11): only `scripts/orient_infer.py` (Orient-Anything-V2 front detection inside Stage R), invoked as a short-lived subprocess.

## 2. Pipeline architecture and job directory

### 2.1 Stages

Seven idempotent stages, each reading and writing only its own per-job directory under `jobs/<job_id>/`:

```
Stage R  (render)   URDF -> normalized merged geometry + VLM contact sheet + control maps + camera.json
Stage V  (vlm)      contact sheet -> appearance spec (category, materials, ref prompt/negative)
Stage D  (diffuse)  control map + spec -> N candidate reference images -> chosen RGBA reference
Stage P  (plan)     per-group material plan (vlm/plan.json)
Stage T  (texture)  one global TRELLIS.2 run -> per-group bake -> textured URDF + assembled.glb
Stage E  (eval)     turntable previews + alignment/consistency metrics + articulated-state renders + per-group diagnostics
Stage J  (judge)    single backend -> walkover verdict; every output is kept
```

Stage names accept the aliases R, V, D, P, T, E, J in the batch CLI and `job.json`.

### 2.2 `job.json` state machine

`pbr_texture_pipeline/jobdir.py` owns `job.json`: a schema-v3 document with an `asset` section and per-stage `status` (`pending|running|done|error|needs_review`), `params`, `seed`, and timestamps for every stage in `STAGES = ("render", "vlm", "diffuse", "plan", "texture", "eval", "judge")`.
This makes every stage resumable and lets interactive (a human approves each transition) and batch (a policy approves it) modes share one code path.
Loading an old job dir backfills any stage added after it was created (`judge`, then `plan`) in memory, so old job dirs never `KeyError`.

### 2.3 Job directory layout

```
jobs/<job_id>/                      # job_id = <asset-dir-name>_<category-slug>_urdf_<yyyymmdd-HHMMSS>
  job.json
  input/
    original.<ext>                  # untouched upload
    mesh_norm.glb                   # normalized merged rest-pose assembly (Y-up glTF convention)
    face_ranges.json                # which merged-mesh face range belongs to which group
    mesh_norm_open.glb              # merged mesh at the open-pose joint configuration
    asset/                          # the rebuilt URDF file tree (app uploads); batch jobs
                                     # reference the source asset dir directly via job.json's asset.asset_dir
  views/
    view_{00..07}.png               # 8-azimuth shaded contact views for the VLM/user
    contact_sheet.png               # 2x4 grid actually sent to the VLM
  control/
    camera.json                     # yaw, pitch, r, fov_deg, resolution, repose_applied, R (3x3)
    depth.png  normal.png  canny.png  mask.png
    camera_open.json                # open-pose camera (same fields as camera.json)
    open/                           # depth/normal/canny/mask at open pose
    visibility_rest.npz             # 8-view depth buffers (captured during the contact sheet render)
                                    #   used for rest-pose occlusion stats and the adapter's per-texel masking
  vlm/
    transcript.json                 # full chat messages (system/user/assistant)
    spec.json                       # structured appearance spec (schema in section 4.2)
    caption.json                    # auto-generated mesh caption + captioning model id
    plan.json                       # Stage P per-group material plan
    refine.json                     # Stage E/T targeted-retexturing VLM grades, if run
  ref/
    candidate_{00..03}.png          # raw Qwen-Image outputs, each with a .json sidecar (seed, cn_scale, true_cfg_scale)
    chosen.png                      # RGB as generated
    chosen_rgba.png                 # after RMBG cutout; this is what the backend receives
    generated_open/, chosen_rgba_open.png   # the open-pose reference and its cutout
  groups/<gid>/
    mesh.glb                        # merged per-group mesh in the link frame (Stage R)
    norm.json                       # center/scale/face/area sidecar for verification
  textured/
    trellis2/groups/<gid>.glb       # per-group bake, mapped to the link frame
    trellis2/global/pass_a.json, pass_b.json   # field-decode bookkeeping
    trellis2/mobility_textured.urdf, assembled.glb   # textured URDF + rest-pose FK assembly
  previews/
    trellis2_turntable.mp4          # or 4-view PNG strip
    trellis2_condview.png           # render from the condition camera, for alignment eval
    trellis2_judgesheet.png         # 2x2 rest-pose grid (front/right/back/left) + open-state row for Stage J
  eval/
    metrics.json
    states/trellis2/, diagnostics.json   # articulated-state renders + per-group diagnostics
  judge/
    verdict.json                    # winner, method, label_map, criteria, reasoning
```

Only the URDF file (`mobility.urdf` for PartNet-Mobility, `model.urdf` for Articraft-10K) plus its transitive file closure is required; `semantics.txt`, `result.json`, `meta.json`, and `bounding_box.json` are optional, each with a fallback.
When `meta.json` is absent, the `<robot name>` attribute in the URDF provides the category.
The authoritative layout is the `jobdir.py` docstring; this section mirrors it.

### 2.4 Resolving the textured output

`JobDir.output_glb(backend)` returns the textured GLB: `assembled.glb` if present, else any `*.glb` directly in the backend's `textured/<backend>/` directory (non-recursive, so per-group GLBs under `groups/` never pollute the fallback).
`JobDir.final_glb()` resolves to the Stage J winner's `output_glb`; since there is only one backend, Stage J always records a walkover.

## 3. Stage R: rendering

### 3.1 Reuse, do not rewrite

All rendering uses TRELLIS.2's nvdiffrast-based `MeshRenderer` (`TRELLIS.2/trellis2/renderers/mesh_renderer.py`), which returns `mask`, `depth`, and `normal` per camera at any resolution, plus the camera helpers in `TRELLIS.2/trellis2/utils/render_utils.py` (`yaw_pitch_r_fov_to_extrinsics_intrinsics`).
No Blender and no pyrender; headless GL is painful on this cluster and unnecessary.

### 3.2 Normalization

The PartNet-Mobility URDF world frame is Z-up (wheels and feet sit at minimum z on all 5 test assets), so Stage R normalizes with `up="z"` (center and scale only, no axis swap) for rendering - applying the Y-up swap on Z-up geometry would tip assets onto their side and turn every yaw orbit into a tumble.
The exported `input/mesh_norm.glb` and `input/mesh_norm_open.glb` are then converted to standard Y-up glTF convention (`rendering.export_yup`: `(x, y, z) -> (x, z, -y)`) so that TRELLIS.2's `preprocess_mesh` (which assumes Y-up input) correctly round-trips to its Z-up internal frame.
The `to_link` matrix and gate A9 both account for this Y-up export swap in their inverse chain.
All cameras are defined in the Z-up internal frame that the renderer uses.

### 3.3 Camera parameterization

`(yaw, pitch, r=2, fov=40)` exactly as `render_utils` defines: camera at `r * (sin y cos p, cos y cos p, sin p)`, look-at origin, up +Z.
`r=2, fov=40` matches TRELLIS.2's own render defaults, so framing statistics match what the backends saw in training.

### 3.4 Canonical front and the pose constraint

TRELLIS.2 assumes the condition image is shot from the mesh's canonical front.
The canonical front camera is `yaw = pi, pitch = 0` in `render_utils` coordinates.

### 3.5 Front-view selection and re-posing

PartNet-Mobility assets face contact-sheet panel 2 in the assembled URDF world frame (verified on all 5 test assets in the Z-up frame).
Stage R predicts the front panel with Orient-Anything-V2 (section 7.5): the model runs on the 8 contact-sheet panel renders, and the panel whose predicted azimuth is closest to 0 is picked as the front, gated by cross-panel agreement (predictions across panels 45 degrees apart should differ by the same steps; `articulated.orient.min_agreement_deg` bounds the allowed disagreement).
On a confident prediction, Stage R re-poses to that panel; otherwise it falls back to `articulated.front_panel` (2).
The decision and the re-pose rotation are recorded in `camera.json`, so Stage R pre-poses before rendering and the canonical camera sees the front.

### 3.6 Control maps

Rendered at 1024x1024 from the condition camera:

- `depth.png`: the renderer returns camera-space z; converted to the ControlNet convention of normalized inverse depth (near = white, far = black), background black: `d_vis = (far_hit - z) / (far_hit - near_hit)` masked by `mask`. ControlNet-depth was trained on MiDaS-style relative inverse depth; wrong polarity produces inside-out objects.
- `normal.png`: `(n+1)/2` encoding as returned by the renderer; the source for canny edge detection.
- `canny.png`: `cv2.Canny` over the normal render (not depth), thresholds (100, 200), dilated 1 px. Normal discontinuities give clean part edges on smooth untextured geometry. Fed to ControlNet alongside depth to preserve fine edge detail (panel seams, handles, surface features) in the generated reference.
- `mask.png`: silhouette; used for alignment scoring and as an optional inpaint mask.

### 3.7 VLM contact views

Same renderer with simple headlight shading (`normal . view` grayscale) so the untextured mesh reads as a clay render, which VLMs classify reliably.

### 3.8 Articulated additions

- **Merged mesh and face ranges, written once**: Stage R persists the merged rest-pose mesh (`input/mesh_norm.glb`) together with `input/face_ranges.json`, recording which face range belongs to which group. The open-pose merged mesh is built with the identical group order, so the same face ranges apply to both meshes. Stage T loads these files and never rebuilds the merge, because face ranges are only valid against the exact face array they were recorded from; a rebuild that differed even slightly would silently assign textures to the wrong parts.
- **Open-pose geometry**: Stage R computes an opened joint configuration (section 7.6) and merges a second whole-object mesh at that configuration, rendering its own control maps (`input/mesh_norm_open.glb`, `control/open/*`, `control/camera_open.json`, same canonical camera and re-pose bookkeeping as the rest pose).
- **Per-group occlusion statistics**: for each group, Stage R estimates the fraction of surface area not visible from any of the 8 contact-sheet viewpoints at rest pose, written into the `asset.groups[]` records (`occluded_frac_rest`). The depth buffers for this test are captured during the contact sheet render itself (no separate visibility pass). These values feed the Stage E diagnostics and the retexturing candidate selection (section 7.7).
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
  "ref_prompt": "a photograph of a mid-century walnut wood chair with brushed steel legs, satin finish, product photography, centered, plain neutral gray background, soft even studio lighting, 8k, photorealistic",
  "negative_prompt": "cartoon, painting, illustration, text, watermark, cluttered background, harsh shadows, strong reflections, people, multiple objects, extra items, accessories, props, loose parts, surrounding objects, contents, decorations not part of the object"
}
```

Jobs with a metadata category carry two extra fields (section 4.6): `category_source: "metadata"` always, and `vlm_category` recording the VLM's own answer when it disagreed with the metadata.

### 4.3 System prompt: four enforced behaviors

1. Category inference: "You are shown clay renders of an untextured 3D object from 8 angles. First state what the object is and which panel shows its front."
2. Realism gate: propose only materials the object category is actually manufactured from. If the user requests an implausible material (a brick chair, a glass hammer handle), refuse in one sentence and offer plausible alternatives; stylized or fictional finishes are allowed only if the user explicitly insists after the warning.
3. Prompt discipline: the emitted `ref_prompt` must end with fixed boilerplate (single object, centered, plain neutral gray background, soft even studio lighting), shown to the model as a literal example. Background and lighting control is what makes RMBG cutouts and PBR decomposition work downstream.
4. Object only: the `ref_prompt` must describe only the surface texture of the object's existing geometry - its materials, colors, and finishes. It must not add items, props, accessories, contents, or decorations that are not part of the mesh (no coffee beans on a coffee machine, no food in an oven, no clothes in a washing machine, no books on a shelf). The generated image is a texturing reference, not a lifestyle scene; the `negative_prompt` must include terms that suppress such additions. `DEFAULT_NEGATIVE` includes "extra items, accessories, props, loose parts, surrounding objects, contents, decorations not part of the object" by default.

### 4.4 Interactive chat loop

The opening turn is automatic: the VLM receives the contact sheet and emits a description plus an initial proposed spec, rendered beside the chatbot.
Each user message may regenerate the spec.
A "Finalize" button sends the force-finalize message.
The spec is editable as raw JSON before Stage D, so the human always has the last word.

### 4.5 Batch mode

Replaces the human with a fixed two-turn script: contact sheet plus "propose the final spec now, realistic materials only, then emit [SPEC]"; if no valid spec, one force-finalize retry, then the template fallback: `"a photograph of a {category} made of typical realistic materials, product photography, plain neutral gray background, soft even studio lighting"`.
`--material-hint "<text>"` optionally injects a user-level steer for the whole batch (for example "medieval, weathered" for a batch of props).
The spec fallback rate is a tracked batch metric.

### 4.6 Authoritative dataset metadata

Caption and appearance spec are produced from the whole-object renders; the prompt describes the whole object, which is what global texturing needs.

The dataset category of an asset is treated as authoritative ground truth (2026-07-29; motivated by a dishwasher whose Stage V caption re-classified it as a sideboard and produced sideboard textures).
The metadata already on disk per asset is `meta.json` `model_cat` (parsed into `asset.category` by Stage R) and the `semantics.txt` per-link labels (persisted by Stage R as `asset.semantic_labels`).

The mechanism, all in `pbr_texture_pipeline/vlm.py`:

- **Metadata note**: `articulated_context(job)` builds one authoritative paragraph per job from the normalized category (`normalize_category` splits CamelCase and lowercases, so "StorageFurniture" becomes "storage furniture") plus the union of `asset.semantic_labels` and the group labels.
  The note states the category as ground truth and instructs the model to describe, not re-classify.
  It is given to BOTH turns: the caption turn's user message opens with it (the caption system prompt says dataset metadata is authoritative and never to contradict it), and the spec turn's opening user message repeats it.
  `articulated_context` also carries the articulation summary, and is shared by `run_auto` (batch) and the VLM worker's `open` op (interactive), so both paths see identical context.
  An asset without a metadata category gets no note and behaves exactly as before.
- **Authoritative spec-turn phrasing**: the articulation summary states the category as dataset ground truth and instructs the model to set the spec's `category` field to exactly that value; the spec system prompt's category rule says a stated metadata category must not be re-classified.
- **Enforcement**: after spec extraction (and after any force-finalize or template fallback), `apply_category_metadata` force-sets `spec.category` to the normalized metadata category and stamps `spec.category_source = "metadata"`.
  A VLM answer that disagrees (substring match in either direction counts as agreement, so "built-in dishwasher" matches "dishwasher") is kept as `spec.vlm_category` and logged, never kept as the category.
  In batch mode, when the mismatching spec's `ref_prompt` also fails to mention the category, one corrective regenerate turn ("dataset metadata says this object is a {category}, not a {vlm_category}; re-emit the [SPEC] block") is spent before the force-set result is saved.
  The worker's `save` op applies the same force-set (without the retry, since the user is in the loop), so an interactive session cannot persist a spec whose category contradicts metadata.
- **Gate A16** (section 15) checks the invariant over existing job dirs.

## 5. Stage D: reference-image generation

### 5.1 Model and ControlNet choice

Base model: `Qwen/Qwen-Image`, the 20B MMDiT text-to-image foundation model, driven through `QwenImageControlNetPipeline` (diffusers >= 0.35).
Two control signals are used simultaneously via `QwenImageMultiControlNetModel`, which wraps the InstantX `Qwen-Image-ControlNet-Union` adapter (canny, soft-edge, depth, and pose in one checkpoint): depth locks silhouette and coarse 3D structure, while canny preserves fine edge detail (panel seams, handles, surface features) that depth alone loses.
The multi-controlnet wrapper runs the single loaded controlnet once per control image and sums the resulting block samples; there is no additional weight memory, only a modest increase in activation memory during the controlnet forward passes.
Weights are fetched once by `scripts/download_controlnet.py` into the shared HF cache.

### 5.2 Generation settings

1024x1024 (matches the control render), bf16, `enable_model_cpu_offload()` (the 20B DiT plus its 7B Qwen2.5-VL text encoder exceed a single A40, so modules stream on demand and only one is GPU-resident at a time), 30 steps, `true_cfg_scale=4.0` (real classifier-free guidance; the negative prompt only takes effect above 1), depth `controlnet_conditioning_scale=0.9` default (UI slider 0.5-1.0; higher means tighter pose, flatter appearance), canny `canny_scale=0.5` default (UI slider 0.0-1.0; higher means more edge fidelity in the generated reference).

### 5.3 Background handling: three-layer defense

The goal is that backend RMBG preprocessing becomes trivial and deterministic:

1. Prompt and negative boilerplate enforce a plain neutral gray background and soft even lighting; baked shadows are the main enemy of PBR decomposition, so "harsh shadows, strong reflections" are explicitly in the negative prompt. The background is gray rather than white so that objects appear against a clean neutral backdrop; this gives RMBG a clear foreground/background boundary.
2. The control depth map has a black (empty) background, which strongly biases the generator toward clean backdrops.
3. pbr-texture-pipeline runs RMBG-2.0 itself to produce `chosen_rgba.png` and passes RGBA to the backends; `Trellis2TexturingPipeline.preprocess_image` uses the alpha channel directly when present and never re-runs rembg, guaranteeing the same cutout everywhere. The UI shows the cutout for approval.

### 5.4 Candidate grid and reroll

Each roll produces 4 candidates (seeds `base_seed + i`, generated sequentially with batch size 1 to bound VRAM), shown as a 2x2 gallery.
The user clicks to choose; "Reroll" bumps `base_seed += 4`; sliders for conditioning scale and guidance persist per job.
Every candidate gets a JSON sidecar recording `{seed, prompt_hash, cn_scale, canny_scale, guidance, steps, controlnet}` for reproducibility.

### 5.5 Batch auto-selection policy

Generate 4 candidates, filter by silhouette IoU(RMBG cutout mask, `control/mask.png`) >= 0.85, then pick max CLIP(prompt, image).
If all fail the IoU gate, take the max-IoU candidate and flag the job `needs_review` (still textured unless `--strict`).

### 5.6 Open-pose reference

A second Qwen-Image plus depth+canny multi-ControlNet generation from `control/open/{depth,canny}.png`, same appearance prompt and seed, written to `ref/generated_open/` with its own RMBG cutout `chosen_rgba_open.png`.
Same VLM/manual selection flow as the rest-pose reference in the app; batch mode picks the same candidate index as the rest-pose choice.
Skipped entirely if `articulated.global.open_pose_pass: false`.

## 6. Stage P: material plan

A single inexpensive VLM call produces a per-group material plan (`vlm/plan.json`, extracted from a `[PLAN]` block with a catalog fallback on parse failure).
It is kept as material metadata: PBR hints, the app's per-group table, and context for the Stage E/T refine grading (section 7.7).
It does not drive texturing directly; global texturing (section 7) samples one shared decoded field regardless of the plan.

## 7. Stage T: texturing

### 7.1 Why global mode replaced per-part texturing

The original articulated pipeline textured each semantic part group independently: every group got its own reference image (a Qwen-Image material swatch or a crop of the global reference) and its own backend run, and the results were reassembled with forward kinematics.
In practice it produced poor results, with these observed failure modes:

1. Part textures were mutually independent: nothing enforced that a drawer front matched the cabinet body in wood grain, tone, or wear.
2. Part references were close-ups (swatches, crops) that did not show the part in its actual pose, so backends textured parts without knowing their orientation or role in the whole object.
3. All rendering (conditioning, eval, judge) happened at rest pose only, so surfaces revealed by articulation (drawer interiors, inside faces of doors) were never seen by any model, were textured without visual evidence, and were never evaluated.
4. Small parts (handles, knobs, hinges) each consumed a full backend run for a few hundred faces, and seams between adjacent parts were a persistent risk.

This per-part path was removed from the codebase entirely on 2026-07-28: no mode knob, no fallback.
The v1-only config keys it used (`texture_mode`, swatch settings, crop-ref thresholds, `hunyuan_part_ref`, `part_ref_mode`) and the `swatches.py` module were removed with it.
`catalog.py` (the material catalog) survives as the Stage P plan-fallback source.

### 7.2 Core idea

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

Residual limitations, accepted with mitigations: color mixing at contact surfaces (a drawer side and the cabinet wall it touches occupy nearly the same region of the field and receive near-identical colors, which matches real furniture and is usually acceptable; targeted retexturing, section 7.7, covers the worst cases); field resolution bounds detail (the field has a fixed voxel count for the whole object, so a handle spanning a few voxels gets a high-resolution but smooth, detail-free texture; Stage E records how many field voxels each group spans, which predicts this failure directly, and the phase-2 per-part detail pass, section 16, is the planned recovery).

### 7.3 Scope decisions

1. One global TRELLIS.2 run, per-group atlases baked from the shared decoded field. This requires calling into TRELLIS.2 internals from the adapter (splitting the run into a field-decode step and a bake step, then calling the bake once per group); the backend is no longer an opaque subprocess that returns one finished GLB.
2. The open-pose pass ships as a first-class part of the pipeline: a second global pass with joints opened, so the reference image and shape conditioning actually see interiors (section 7.6).

### 7.4 Stage T driver

Stage T is one `trellis2` adapter run per job (`pbr_texture_pipeline/backends/trellis2_adapter.py`):

1. **Load the merged mesh and face ranges**: Stage T consumes `input/mesh_norm.glb` and `input/face_ranges.json` exactly as Stage R wrote them; it never rebuilds the merge (section 3.8 explains why).
2. **Global pass A (rest pose)**: the adapter runs TRELLIS.2 through shape encoding, image conditioning, and field decoding on the merged mesh, using `chosen_rgba.png` and the real `camera.json` (with re-pose applied), then stops before baking. The decoded field stays in memory.
3. **Global pass B (open pose)**: the same, on the open-pose merged mesh with `chosen_rgba_open.png` and `camera_open.json`. Skipped if `articulated.global.open_pose_pass: false`.
4. **Per-group bake**: for each group, inside the adapter process: slice the group's faces from the merged mesh using the recorded face ranges, and UV-unwrap the group submesh (`cumesh.uv_unwrap`, the same routine TRELLIS uses for input without UVs); bake the atlas (size `per_link_texture_size`) from the pass A field, sampling at the group's surface points expressed in pass A's normalized frame; if pass B ran, compute a per-texel rest-pose visibility mask (the same visibility test as the Stage R occlusion statistics, applied to each texel's surface point), re-sample texels hidden at rest from the pass B field in pass B's frame, blended over a band of `blend_band_texels` at the visibility boundary so no hard line crosses the part; fill empty texels by inpainting (`cv2.inpaint`, as TRELLIS does), emit `textured/trellis2/groups/<gid>.glb`, mapped back to the link frame.
5. **Tiny groups**: baked from the field like every other group, with a smaller atlas (`tiny_texture_size`). A full backend run for 20 faces was wasteful under the old per-part path, but a group bake costs almost nothing once the field exists, so that shortcut lost its rationale; the adapter writes its own neutral constant-PBR GLB only as a fallback when UV unwrapping fails or the geometry is degenerate.
6. **Assembly**: `assemble_backend` collects `groups/<gid>.glb` (already bbox-checked per bake), writes the textured URDF with collision symlinks (`articulated.collision_mode: symlink`), and assembles the rest-pose `assembled.glb`.

Adapter contract: the pairs-file schema gains a `global` pair kind carrying `{merged_mesh, merged_mesh_open, image, image_open, camera_json, camera_json_open, face_ranges, out_dir, groups: [{group_id, out_glb, texture_size}]}`.
`registry.py` treats it like any other pair; success is still the `[PBR_RESULT]` marker plus every expected group GLB existing on disk.

Resume granularity: one global field decode is the smallest unit of work; if any group bake is missing, the pass re-runs and re-bakes only the missing groups.
Persisting the decoded field to disk for finer-grained resume is off by default (`persist_field: false`); the field is several gigabytes and re-decoding takes minutes.
The adapter merges `pass_a.json`/`pass_b.json` metadata on partial (resume) runs instead of rewriting them, so a resumed job does not lose earlier groups' bake stats that Stage E diagnostics read.

TRELLIS.2 forces `alphaMode='OPAQUE'` on its output GLBs; all glass parts get a frosted opaque look, which is the accepted compromise (section 1.5).

### 7.5 Orient-Anything-V2 integration

Repo: vendored as the `Orient-Anything-V2` git submodule (NeurIPS 2025; built on VGGT).
API: `VGGT_OriAny_Ref(out_dim=900)` plus `utils/app_utils.inf_single_case(model, pil_ref, pil_tgt=None)`; input is a single image, output is `{ref_az_pred, ref_el_pred, ref_ro_pred, ref_alpha_pred}` (azimuth, elevation, in-plane rotation, symmetry class).
Checkpoint `rotmod_realrotaug_best.pt` (5.05 GB) from Hugging Face repo `Viglong/OriAnyV2_ckpt`.

Usage in Stage R (see also section 3.5): run the model on the 8 contact-sheet panel renders (backgrounds are already clean, so no background removal is needed); the predicted azimuth per panel identifies which panel faces the camera, and because panels are 45 degrees apart, cross-panel agreement is the confidence check; on a confident prediction, use that panel for the re-pose, otherwise fall back to `articulated.front_panel` (2); record the decision and the per-panel predictions in `camera.json`.
The symmetry class (`ref_alpha_pred`) is recorded for possible future multi-reference view-selection use; no current behavior depends on it.

Execution model: a dedicated `orianyv2` conda env (Python 3.11 per its README; the `bpy` dependency is only used by the demo's axis-overlay renderer and is not installed).
Invoked as a batch subprocess (`conda run -n orianyv2 python scripts/orient_infer.py --images ... --out json`), following the backend-adapter pattern: load the model once, process many images, print JSON.
Not a persistent worker; it runs for seconds per job during Stage R, on whichever GPU is free.
One-time setup script `scripts/download_orient_anything.py` places the weights in the shared HF cache.

### 7.6 Open-pose pass design

- **Opened configuration**: every movable joint is set to `articulated.global.open_pose_frac` (default 0.8) of its clamped range, prismatic and revolute alike. Fixed joints and continuous joints without limits stay at rest. One shared configuration for the whole object; no per-joint search.
- **Self-collision is not solved**: PartNet-Mobility assets open cleanly at fractions below the maximum, and the fraction is configurable per run; gate A12 (section 15.2) verifies the opened configuration is valid.
- **Combining the two passes is per-texel, not per-part**: every group's atlas is baked from the rest-pose field, and only texels whose surface points were hidden at rest are overwritten from the open-pose field, blended over `blend_band_texels` at the boundary. A whole-part rule was rejected because many groups mix visible and hidden surfaces in a single part (a drawer is one group containing both its front face and its interior); any per-part rule either recolors the drawer front from the open-pose field (risking a visible mismatch when closed) or leaves the interior with rest-pose guesses. Per-texel masking guarantees that every surface visible at rest keeps exactly the pass A result.
- **Consistency between passes**: same appearance prompt, same seed, same category. The two reference images can still differ in detail; this is acceptable because pass B contributes only texels hidden at rest, so any style difference is confined to interior surfaces and the narrow blend band.

### 7.7 Evaluation and targeted retexturing

Texture globally first, measure, then repair only what is broken:

1. **Select candidates** (heuristics, no model calls): after Stage E, a group is a retexturing candidate if any of the following hold: its `occluded_frac_rest` exceeds `refine.occluded_frac_threshold` and no open-pose pass ran (its hidden surfaces were textured without visual evidence); it spans fewer than `refine.min_field_voxels` field voxels, or its blur estimate is below `refine.blur_flag_ratio` times the object median (the field resolution was too coarse for the part); its bake emitted warnings.
2. **Confirm** (one VLM call per candidate): per-group crops from the state renders for the candidates only are graded by the Stage V VLM (good / blurry / wrong material / missing texture) with reasoning, written to `vlm/refine.json`.
3. **Retexture** (targeted, through the same global path): the confirmed-bad `groups/<gid>.glb` files are moved aside, and the missing-GLB resume rule re-bakes exactly that set from a fresh-seed global decode (the same seed would deterministically reproduce the rejected texels). Reassembly and a fresh eval/judge follow.

Candidate selection and VLM confirmation (steps 1-2) always run, so how often retexturing would trigger is measured continuously.
Actual retexturing execution (step 3) is off by default (`articulated.refine.enabled: false`); the per-group field bake is expected to make it rarely necessary.
It can be forced per run with `--refine` and is exercised by gate A14.
Phase 2 (section 16) upgrades step 3 to the per-part detail pass, which restores full field resolution for the flagged part while keeping consistency through the shared reference image.

## 8. Stage E: evaluation

Turntable previews (or a 4-view PNG strip), a condition-view render for alignment scoring, and `eval/metrics.json` (silhouette IoU for pose fidelity; CLIP(prompt, chosen reference) and CLIP(prompt, condition-view render of the textured output) for semantic fidelity; LPIPS(chosen reference, condition-view render) for transfer fidelity), and a top-level index row.

Additionally:

- **Articulated-state renders**: for each movable joint, render the assembled textured object at 0%, 50%, and 100% of the joint's range (canonical camera plus one three-quarter view), written to `eval/states/trellis2/`, using the same forward-kinematics code the joint viewer uses in the browser.
- **Per-group diagnostics** (`eval/diagnostics.json`): the occlusion statistics carried from Stage R, the number of field voxels each group spans (the direct predictor of field-resolution blur), the per-group mean texel gradient measured from the state renders (a render-based blur estimate), and any bounding-box or bake warnings. This is the input for retexturing candidate selection (section 7.7).

## 9. Stage J: judge

Stage J records the texturing result for each job.
Since there is only one backend (TRELLIS.2), Stage J always records a walkover verdict without a VLM call.
The judge sheet is still rendered for archival and visual review.

Two phases per job:

1. **Sheet prep** (`trellis2` env, `pbr_texture_pipeline/judge.py`): render `previews/trellis2_judgesheet.png` from the textured GLB, a 2x2 rest-pose grid (front/right/back/left, `judge.sheet_yaws_deg`, 512 px tiles composed to a 1024-wide sheet) plus an open-state row (all movable joints at 100% of their clamped range, front and three-quarter views) so the VLM sees interiors and moving-part boundaries, reusing `eval.load_output_colored` and `rendering.render_appearance`.
In batch this runs in the batch process; in the app it is an imaging-worker op (`judge_sheets`), never inside the VLM worker, which has no renderer.
2. **Materialization** (orchestrator, `judge.finalize`): write `judge/verdict.json` (schema: winner, method `walkover`, label_map) and record the winner in `job.json`. `JobDir.final_glb()` resolves to the TRELLIS.2 output.

Degenerate cases: zero GLBs is stage `error`.
Re-running judge re-finalizes idempotently (crash recovery when `verdict.json` exists but judge status is not `done`, with no VLM call).
Judge does not require Stage E to be `done`: it renders its own sheets from texture outputs only, preserving stage independence.

The full VLM comparison machinery (A/B label assignment, multi-sheet comparison, verdict extraction) remains in the code for future use if a second backend is added; with one backend it is never invoked.

## 10. Gradio app and GPU memory strategy

### 10.1 Process architecture

The Gradio app (`app.py`) is a thin orchestrator with zero CUDA in-process.
Three worker types, IPC via JSON-lines over stdin/stdout (`workers/ipc.py`; request `{op, args}` -> response `{ok, paths}`):

1. **Imaging worker** (persistent subprocess, GPU 1 in dual mode): nvdiffrast rendering (Stage R), Qwen-Image + ControlNet (20B, bf16, model CPU offload, ~40 GB peak during a diffuse), RMBG-2.0. One process because all are needed in the same interactive phase and rendering itself is cheap (~1 GB).
2. **VLM worker** (persistent subprocess, GPU 0): Qwen3.6-35B-A3B AWQ under vLLM in the dedicated `vlm` env (~20 GB weights plus a bounded pre-allocated pool), kept separate so either worker can be dropped independently; it stays resident so caption, spec, chat, and judge calls are always live.
3. **Texturing worker** (ephemeral subprocess, GPU 1): the TRELLIS.2 backend adapter, launched per texture request, exits when done (~21-24 GB peak). It shares GPU 1 with the imaging worker, but Stage D and Stage T never overlap for a job, so `manager.py` drops the imaging worker's Qwen-Image model (`unload`) before a texture run and it reloads lazily on the next diffuse.

`manager.py` owns GPU placement (via `CUDA_VISIBLE_DEVICES` per worker environment), VRAM polling via `nvidia-smi` (queue rather than OOM; the node is shared with no SLURM arbitration), and this unload/reload dance.

### 10.2 VRAM budgets

- Dual A40 (current node): GPU 0 holds the VLM (~20 GB AWQ weights plus vLLM's pre-allocated pool) plus the renderer, resident so caption/spec/chat/judge stay live; GPU 1 runs Qwen-Image Stage D (model CPU offload, ~40 GB peak) and, on demand, texturing. The manager drops the imaging worker's Qwen-Image model before each texture run; the ~40-60 s reload cost is paid once per stage switch.
- Single GPU fallback (`gpu_mode: single`): the orchestrator enforces a stage-exclusion rule. Before launching a texturing worker it suspends both persistent workers (via `unload_model()` or termination) and restarts them lazily on the next chat or reroll request. The reload cost is paid once per stage switch.
- `manager._wait_for_vram` polls every 5 s but proceeds anyway after a 1800 s (30-minute) timeout, at OOM risk, and skips the check entirely if `nvidia-smi` is unavailable.

### 10.3 UI layout

A single `gr.Blocks` wizard, one job per session, state is `gr.State(job_id)` only; everything else lives on disk in the job dir, so sessions survive app restarts and batch jobs are browsable identically.

- **Tab 1, Asset**: multi-file URDF upload that validates the reference closure; `gr.Model3D` of the blank mesh; contact sheet gallery; front-view radio (pre-filled from the VLM's `front_view_index`); yaw/pitch nudge sliders; "Render control maps" button with depth/canny previews. Re-posing is automatic whenever a non-front panel is chosen; there is no separate re-pose checkbox.
- **Tab 2, Appearance chat**: `gr.Chatbot` plus message box; live spec panel (editable raw JSON); "Finalize spec" button.
- **Tab 3, Reference image**: control map thumbnail; editable prompt/negative pre-filled from the spec; seed, depth CN scale, canny CN scale, guidance sliders; 2x2 candidate gallery; "Reroll"; chosen image plus cutout preview; "Approve".
- **Tab 4, Texture and review**: Stage P runs automatically before the global bake when the Texture button is pressed; the joint-slider viewer core (`articulated/viewer.py`) is embedded as FastAPI iframe routes, alongside per-group tables, pass A/B reference images, and a `gr.Model3D` viewer; "Judge outputs" runs Stage J and displays the result. There are no in-app download buttons and no "send back to Tab 3" control; rich visual review and downloads live in `scripts/build_viewer.py` and the job directory itself.
- **Tab 5, Jobs browser**: a text-only table over `jobs/*/job.json` for reviewing batch output (winner column from Stage J, approve/flag toggle written back to `job.json`); reference thumbnails and preview strips are not built here, they live in `scripts/build_viewer.py`.

## 11. Batch CLI

```bash
# PartNet-Mobility
conda run -n trellis2 python -m pbr_texture_pipeline.batch \
  --assets 'partnet_mobility/*/mobility.urdf' --jobs-root jobs/ \
  --backends trellis2 \
  --stages render,vlm,diffuse,plan,texture,eval,judge \
  --candidates 4 --seed 42 \
  --material-hint "clean, factory-new" \
  --select iou+clip \
  --gpu-mode dual \
  --limit 100 --resume

# Articraft-10K (extract first, then batch)
bash scripts/extract_articraft.sh --limit 100
conda run -n trellis2 python -m pbr_texture_pipeline.batch \
  --assets 'articraft_extracted/*/model.urdf' --jobs-root jobs_v2/ \
  --stages render,vlm,diffuse,plan,texture,eval,judge --resume
```

Full flag list: `--assets` (required; accepts URDF paths - `mobility.urdf` or `model.urdf`), `--jobs-root` (default `jobs`), `--backends` (default `trellis2`), `--stages` (default `render,vlm,diffuse,plan,texture,eval,judge`; accepts the R/V/D/P/T/E/J aliases), `--candidates`, `--seed`, `--material-hint`, `--select {iou+clip,iou}`, `--gpu-mode {dual,single}`, `--limit`, `--resume`, `--strict`, `--spec-cache-by-category`, `--canny-scale` (override canny ControlNet scale; default from config.yaml), `--group-by {semantic,link}`, `--refine` (force targeted retexturing on for this run).

Execution is stage-major for model-load efficiency, mirroring the `pbr_compare` sweeps:

1. Stage R for all assets (renderer only).
2. Stage V for all (VLM loaded once; two-turn auto script per section 4.5).
3. Stage D for all (Qwen-Image loaded once; auto-select per section 5.5; IoU failures flagged `needs_review` but still textured unless `--strict`).
4. Stage P for all (one VLM worker over all jobs).
5. Stage T: TRELLIS.2 loaded once, iterating over all approved pairs via `--pairs-file`; one global run per job (section 7.4).
6. Stage E: previews, `metrics.json`, articulated-state renders and diagnostics, and a top-level index (HTML). Targeted retexturing (section 7.7) runs as a sub-step inside this stage, gated by `--refine` or `articulated.refine.enabled`; it is not a separate `--stages` token.
7. Stage J: judge sheets for all jobs first (renderer in the batch process), then walkovers recorded for all; the index is rewritten so its winner column is fresh.

`--resume` skips any stage marked done in `job.json`.
On dual GPU, Stage D (GPU 0-ish workload) and Stage T for completed jobs (GPU 1) could overlap in principle; execution stays sequential for simplicity.

## 12. Configuration reference (`config.yaml`)

Every module reads paths and defaults from `config.yaml`; nothing is hardcoded.

- `gpu_mode`: `dual` or `single` (section 10.2).
- `env`: env names and interpreter paths for `trellis2` (primary: render, diffuse, app, batch, and the TRELLIS.2 adapter), `vlm_name` (Stage V/J worker), `orient_name` (Orient-Anything-V2 subprocess); `hf_cache` and `torch_cache` paths (both outside `/home`).
- `backend_envs`: `trellis2` subprocess env name for Stage T.
- `models`: HF repo ids for `vlm`, `qwen_image`, `rmbg`, `trellis2`, `clip`, `controlnet`, `orient`, all resolved inside `hf_cache`.
- `repos`: local paths to the `trellis2` and `orient` repos (adapter and tool cwd requirements).
- `render`: resolution, ssaa, `r`/`fov_deg` camera defaults, canonical yaw/pitch, contact-view pitch, canny thresholds.
- `vlm`: vLLM engine knobs (`max_model_len`, `gpu_memory_utilization`), per-call token budgets (`caption_max_new_tokens`, `spec_max_new_tokens`), and `limit_mm_per_prompt_images` (3, sized for Stage J's three images).
- `diffusion`: candidate count, seed, `cn_scale` (depth), `canny_scale`, `guidance` (`true_cfg_scale`), steps, resolution.
- `select`: `iou_threshold` for batch auto-selection.
- `texture`: `trellis2_resolution`, `trellis2_texture_size`.
- `articulated`: the URDF job configuration, all keys top-level under `articulated:` unless noted:
  - `global`: `open_pose_pass`, `open_pose_frac`, `per_link_texture_size`, `tiny_texture_size`, `blend_band_texels`, `persist_field`, `multi_ref` (phase-2 experiment flag, unused today).
  - `orient`: `enabled`, `env`, `ckpt_file`, `min_agreement_deg`.
  - `refine`: `enabled` (execution; selection and confirmation always run), `blur_flag_ratio`, `occluded_frac_threshold`, `min_field_voxels`.
  - `group_by` (`semantic` default, or `link`), `front_panel` (fallback front panel, 2), `rest_pose` (`zero_clamped`, the only supported policy), `min_group_faces` / `min_group_area_frac` (tiny-group thresholds), `plan_max_new_tokens`, `collision_mode` (`symlink`).
- `judge`: `default_winner`, `max_new_tokens`, `sheet_yaws_deg`, `sheet_tile_res`, `sheet_pitch_deg`.

## 13. Risks, invariants, and gotchas

1. **Canonical-front convention**: `yaw = pi, pitch = 0` in `render_utils` coordinates; load-bearing for every render.
2. **Depth encoding polarity**: analytic depth must be normalized to MiDaS-style inverse depth (near = white, far = black) or ControlNet produces inside-out objects.
3. **Baked lighting in generated references**: even with prompt discipline, the diffusion model bakes shadows and highlights, which the backend may transfer into albedo. Mitigations: negative-prompt terms, lower CN-scale trials.
4. **VLM JSON reliability**: the VLM occasionally emits malformed JSON; mitigated by retry plus template fallback, with the fallback rate tracked as a batch metric.
5. **Env drift**: adapters and workers must read paths and env names from `config.yaml`, never hardcode them, because the env-to-stage mapping has changed multiple times and will likely change again.
6. **Shared-node GPU contention without SLURM**: check free VRAM and queue rather than OOM (section 10.1); the 1800 s wait timeout is a real risk window, not a guarantee.
7. **PartNet-Mobility Z-up world frame**: rendering normalizes with `up="z"` (no axis swap); the exported `mesh_norm.glb` files are converted to Y-up glTF convention so TRELLIS.2's `preprocess_mesh` correctly round-trips to Z-up internal (section 3.2). If a render looks wrong, check that the Z-up/Y-up boundary is crossed exactly once.
8. **`to_link` frame chain**: the global adapter bakes in the TRELLIS field frame and maps each group to its link frame with the pair's `to_link` matrix, composed orchestrator-side; a `to_link` bounds mismatch fails the group loudly. Bbox shrinkage up to 5% of extent is tolerated, because `cumesh.uv_unwrap` welds sliver faces on low-face-count groups (seen on 48-face PartNet parts); gate A9 pins the frame chain it inverts.
9. **`face_ranges.json` is a recording, not a re-derivable fact**: it is only valid against the exact merged-mesh face array Stage R wrote it from; Stage T must load it, never rebuild the merge (section 3.8).
10. **Resumed jobs**: the adapter merges `pass_a.json`/`pass_b.json` metadata on partial runs instead of rewriting them; losing this would silently drop earlier groups' bake stats that Stage E diagnostics depend on.
11. **Color mixing at contact surfaces** is inherent to a shared field (section 7.2); accepted and documented, with targeted retexturing covering the worst cases.
12. **Open pose can reveal geometry PartNet never modeled** (missing interior faces render as holes); this affects the depth maps, not the bake. Gate A12 catches assets where the open-pose pass is not worth running, and the judge sees the open state as a backstop.
13. **Open-pose reference quality**: Qwen-Image with depth+canny ControlNet has likely never seen control maps of, say, a cabinet with every drawer and door open at once, so the pass B reference may come out confused, and it is the sole visual evidence for interior surfaces. Mitigations: the judge sees the open state, targeted retexturing covers bad interiors, and gate A13 includes inspection of the pass B references.
14. **Transparency**: TRELLIS.2 forces `alphaMode='OPAQUE'` on output GLBs; all glass parts get a frosted opaque look, which is the accepted compromise (section 1.4).
15. **Hours-long Stage T runs** on large assets vs. Gradio session drops: all state lives in the job dir and resume is group-granular, so a dropped session loses no completed work; large batches are still better suited to the batch CLI than the app.

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
    articulated/                # URDF asset support
      urdf.py                   # URDF parse, grouping, FK (rest + open pose), textured-URDF rewrite, assembly
      appearance.py              # colored-mesh loading, group visibility
      catalog.py                 # material catalog (Stage P plan fallback)
      orient.py                  # Orient-Anything-V2 subprocess wrapper, cross-panel agreement
      stages.py                  # stage drivers shared by batch.py and app.py
      viewer.py                  # joint-slider viewer core (standalone script + app iframe)
    workers/
      imaging_worker.py          # persistent: rendering + Qwen-Image + RMBG (GPU 1 in dual mode)
      vlm_worker.py               # persistent: Qwen3.6 (GPU 0)
      manager.py                  # GPU placement, VRAM polling, unload/reload orchestration
      ipc.py                       # JSON-lines protocol
    backends/
      registry.py                 # name -> {script, env, cwd}
      _adapter_common.py           # shared adapter helpers
      trellis2_adapter.py           # field decode + per-group bake
      visibility.py                 # occlusion/visibility helpers shared by Stage R and the adapter
    app.py                        # Gradio orchestrator (Tabs 1-5)
    batch.py                      # CLI (section 11)
  scripts/
    verify_articulated.py          # gates A1, A9, A11, A12a/b, A16
    verify_global_bake.py           # gate A10: per-group vs whole-mesh bake
    articulated_viewer.py           # standalone joint-slider viewer server for one job
    drive_pipeline.py                # headless driver for app callbacks (e2e-test skill)
    orient_infer.py                  # Orient-Anything-V2 subprocess entry point
    download_controlnet.py
    download_orient_anything.py
    download_vlm.py
    build_viewer.py                    # browsable HTML gallery of textured results
    prep_partnet.py                     # PartNet-Mobility test asset staging (Z-up OBJs -> Y-up GLB)
    extract_articraft.sh                # Articraft-10K tar.gz extraction
    run_one_by_one.sh                   # sequential single-asset batch runner
  partnet_mobility/              # pristine PartNet-Mobility test assets (never modify)
  articraft_extracted/           # extracted Articraft-10K assets (gitignored)
  jobs/                          # runtime output for PartNet-Mobility (gitignored)
  jobs_v2/                       # runtime output for Articraft-10K (gitignored)
```

## 15. Verification and gates

Gate A1 (FK, grouping, and metadata-optional-fallback correctness) is still live and automated (`scripts/verify_articulated.py` with no `--jobs-root`), because URDF parsing and grouping did not change when the per-part texturing mechanism was replaced.
The remaining original gates (A2-A8: per-part Stage R dry run, denorm/swatch round trip, material-plan-at-scale, the Hunyuan swatch out-of-distribution gate, the per-part end-to-end and app gates) tested the removed per-part swatch/crop-ref mechanism directly and are retired; they do not apply to global texturing and are not run.

Current gates (`scripts/verify_articulated.py --jobs-root jobs/` covers A9/A11/A12a/A12b/A16; `verify_global_bake.py` covers A10; A13/A14 run through the `e2e-test` skill):

- **A9 merge/split round trip** (no GPU): slicing `input/mesh_norm.glb` by `input/face_ranges.json` (the persisted files, not a rebuild) reproduces per-group meshes that match direct group-mesh construction after undoing the Y-up glTF export swap, normalization, re-pose, and forward kinematics, on all 5 PartNet test assets.
- **A10 per-group bake correctness** (GPU): on one cabinet, per-group bakes from a single pass A produce an assembled result visually consistent with one whole-mesh bake from the same field (no seams beyond contact edges), and handles are measurably sharper (per-group texel density reported).
- **A11 orientation agreement**: Orient-Anything-V2 front selection matches `front_panel: 2` on all 5 PartNet assets; on disagreement it falls back and logs the discrepancy.
- **A12 open-pose validity**, two parts: (a) no GPU, the opened configuration produces finite bounds and every joint value stays inside its limits; (b) GPU, condition-camera mask coverage at open pose is at least that of rest pose on drawer/door assets, and interiors are visible in `control/open/depth.png`.
- **A13 end to end**: the `e2e-test` skill passes, with a judge verdict written (walkover to trellis2 by design). The pass B reference images are inspected as part of this gate.
- **A14 targeted retexturing**: with retexturing forced on (`--refine`), at least one group completes the full sequence: selected by heuristics, confirmed by the VLM, re-baked from a fresh-seed global decode, and reassembled.
- **A15 (blind comparison against the removed per-part pipeline)**: dropped by decision; global shipped without this comparison.
- **A16 spec category matches metadata** (no GPU): for every job with a metadata category and a Stage V `vlm/spec.json`, `normalize_category(spec.category)` matches the metadata category and `spec.category_source == "metadata"` (section 4.6). Run against pre-fix job dirs it fails, confirming it detects the original mis-classification bug.

All currently-applicable gates (A1 and A9-A14) have passed on the 5 PartNet-Mobility test assets; A16 has passed on Stage V runs made after the metadata-authority change (12085, 8930), and pre-fix job dirs fail it by design.

## 16. Phases and deferred work

- **Shipped**: global articulated texturing with per-group field bake and per-texel pass masking, the open-pose pass, Orient-Anything-V2 front detection, articulated-state renders in Stage E/J, retexturing diagnostics with execution off by default.
- **Deferred, each to be gated separately if pursued**:
  - **Per-part detail pass**: for a part flagged by the field-resolution diagnostics, re-run the TRELLIS field decode on that part alone, so it gets the entire field to itself, conditioned on a crop of the whole-object reference image rather than a material swatch. Detail returns without losing consistency, because the part's reference comes from the same picture as everyone else's.
  - Multi-reference conditioning for TRELLIS (`get_cond` already accepts a list of images; feed front and back references).
  - Enabling targeted retexturing by default, if the diagnostics show it triggers often enough to matter.
  - Per-category VLM spec caching for homogeneous batches.

## 17. Key reference files

- `TRELLIS.2/trellis2/renderers/mesh_renderer.py` (submodule) - control-map rendering API (mask/depth/normal).
- `TRELLIS.2/trellis2/pipelines/trellis2_texturing.py` (submodule) - mesh normalization and alpha-path image preprocessing that Stage R replicates and the global adapter's field decode/bake split builds on.
- Prior internal R&D (`pbr_compare`, `trellis_pbr`, `Qwen-Image` usage notes; not part of this repo) originated the verified backend APIs, the VLM chat patterns `pbr_texture_pipeline/vlm.py` ports, and the Qwen-Image diffusers usage Stage D ports.

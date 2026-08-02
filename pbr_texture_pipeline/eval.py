"""Stage E: evaluation - previews + alignment/consistency metrics + index (PRD sections 2, 11 gate 6).

Reads a job's textured outputs and produces, per backend:
  previews/<backend>_condview.png    - render from the condition camera (yaw=pi), for alignment
  previews/<backend>_turntable.mp4   - orbit preview (falls back to a 4-view PNG strip)
and eval/metrics.json with:
  silhouette IoU (pose fidelity)            = IoU(output condition-view mask, control/mask.png)
  CLIP(prompt, chosen reference)            (semantic fidelity of the reference itself)
  CLIP(prompt, condition-view render)       (semantic fidelity transferred to the output)
  LPIPS(chosen reference, condition-view)   (appearance transfer fidelity)
  hunyuan albedo shadow-bake spot check     (PRD risk 3)

Runs in the `trellis2` env (needs the TRELLIS.2 renderer for the appearance render). The
render path is validated end-to-end: each backend exports in a different frame, so every
output is re-normalized into the internal Z-up frame (rendering.to_internal_frame) and, if
Stage R re-posed, re-posed by R before rendering, so one camera renders all backends.
"""
from __future__ import annotations

import csv
import html
import math
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from PIL import Image

from pbr_texture_pipeline import rendering as R
from pbr_texture_pipeline.config import load_config
from pbr_texture_pipeline.jobdir import JobDir

_CFG = load_config()

# Turntable defaults (kept small: previews are for review, not archival quality).
TURNTABLE_FRAMES = 24
TURNTABLE_RES = 512
STRIP_YAWS_DEG = (0, 90, 180, 270)   # 4-view PNG strip fallback, relative to canonical front

_lpips_model = None  # lazy; "unavailable" sentinel if weights/pkg missing


# --- output-mesh loading (per-backend frame -> internal Z-up) -----------------
def load_output_colored(glb_path: str, camera: Optional[dict], up: str = "y") -> "R.Mesh":
    """Load a textured output GLB into a vertex-colored TRELLIS.2 Mesh in the internal frame.

    Colours come from `visual.to_color()` (samples the base-color texture per vertex). Vertices
    are re-normalized into the internal Z-up frame; if Stage R re-posed, the same R is applied
    so the appearance render lines up with the condition camera and the control maps.
    `up` names the GLB's source frame: "y" for backend outputs in glTF Y-up (flat mesh jobs),
    "z" for articulated assembled.glb, whose baked FK vertices are in the Z-up URDF world
    frame.
    """
    scene = trimesh.load(glb_path)
    verts = faces = colors = None
    if not (isinstance(scene, trimesh.Scene) and len(scene.geometry) > 1):
        try:
            tm = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
            colors = np.asarray(tm.visual.to_color().vertex_colors)[:, :3].astype(
                np.float32) / 255.0
            verts, faces = tm.vertices, tm.faces
        except Exception:  # noqa: BLE001 - fall through to the robust loader
            verts = None
    if verts is None:
        # Articulated assembled.glb (many geometries) or a visual to_color() can't sample:
        # the articulated appearance loader handles per-geometry colors + node transforms.
        from pbr_texture_pipeline.articulated.appearance import load_colored_arrays
        verts, faces, colors = load_colored_arrays(glb_path)
    verts = R.to_internal_frame(verts, up=up)
    if camera and camera.get("repose_applied") and camera.get("R") is not None:
        verts = verts @ np.asarray(camera["R"], dtype=np.float64).T
    return R.colored_mesh_repr(verts, faces, colors)


# --- previews ----------------------------------------------------------------
def render_condview(job: JobDir, backend: str, camera: Optional[dict]) -> Optional[Path]:
    """Render the textured output from the condition camera -> previews/<backend>_condview.png."""
    glb = job.output_glb(backend)
    if glb is None:
        return None
    mesh = load_output_colored(str(glb), camera, up="z" if job.kind == "urdf" else "y")
    res = int(_CFG.get("render.resolution"))
    ssaa = int(_CFG.get("render.ssaa"))
    out = R.render_appearance(mesh, R.CANONICAL_YAW, R.CANONICAL_PITCH, res, ssaa)
    Image.fromarray(out["rgb"], mode="RGB").save(job.preview_condview(backend))
    return job.preview_condview(backend)


def render_turntable(job: JobDir, backend: str, camera: Optional[dict],
                     n_frames: int = TURNTABLE_FRAMES) -> Optional[Path]:
    """Orbit the textured output; write an mp4 (imageio-ffmpeg) or a 4-view PNG strip fallback."""
    glb = job.output_glb(backend)
    if glb is None:
        return None
    mesh = load_output_colored(str(glb), camera, up="z" if job.kind == "urdf" else "y")
    pitch = math.radians(15.0)

    frames = []
    for k in range(n_frames):
        yaw = R.CANONICAL_YAW + 2 * math.pi * k / n_frames
        out = R.render_appearance(mesh, yaw, pitch, TURNTABLE_RES, ssaa=2)
        frames.append(out["rgb"])

    mp4 = job.preview_turntable(backend)
    try:
        import imageio.v2 as imageio
        imageio.mimsave(str(mp4), frames, fps=12, quality=8, macro_block_size=None)
        return mp4
    except Exception as e:  # noqa: BLE001  - no ffmpeg / codec: fall back to a static strip
        print(f"[eval] turntable mp4 unavailable ({e}); writing 4-view PNG strip instead.")
        strip = _four_view_strip(mesh, pitch)
        strip_path = job.path("previews", f"{backend}_turntable.png")
        Image.fromarray(strip, mode="RGB").save(strip_path)
        return strip_path


def _four_view_strip(mesh: "R.Mesh", pitch: float) -> np.ndarray:
    tiles = []
    for deg in STRIP_YAWS_DEG:
        out = R.render_appearance(mesh, R.CANONICAL_YAW + math.radians(deg), pitch,
                                  TURNTABLE_RES, ssaa=2)
        tiles.append(out["rgb"])
    return np.concatenate(tiles, axis=1)


# --- articulated states (PRD_articulated_v2, Stage E additions) ---------------
def articulated_state_mesh(job: JobDir, backend: str, joint_frac: float,
                           only: Optional[set] = None) -> Optional["R.Mesh"]:
    """Vertex-colored Mesh of a backend's textured groups posed at joint_frac.

    Never transforms assembled.glb (its rest-pose FK is baked into the vertices); instead
    each per-group GLB is loaded and placed with the opened-configuration FK, then brought
    into the internal frame with the usual normalization + camera re-pose. Returns None when
    no group GLBs exist.
    """
    from pbr_texture_pipeline.articulated import urdf as U
    from pbr_texture_pipeline.articulated.appearance import load_colored_arrays

    asset_state = job.state.get("asset")
    if not asset_state:
        return None
    asset = U.parse_asset(asset_state["asset_dir"])
    fk = U.link_world_transforms_at(asset, joint_frac, only)
    camera = job.read_json(job.camera_json()) if job.camera_json().is_file() else {}

    verts, faces, colors, off = [], [], [], 0
    for g in asset_state["groups"]:
        glb = job.textured_group_glb(backend, g["group_id"])
        if not glb.is_file():
            continue
        v, f, c = load_colored_arrays(str(glb))
        T = fk.get(g["link"], np.eye(4))
        verts.append(np.asarray(v, dtype=np.float64) @ T[:3, :3].T + T[:3, 3])
        faces.append(np.asarray(f) + off)
        colors.append(c)
        off += len(v)
    if not verts:
        return None
    v = R.to_internal_frame(np.concatenate(verts, axis=0), up="z")  # FK frame is Z-up
    if camera.get("repose_applied") and camera.get("R") is not None:
        v = v @ np.asarray(camera["R"], dtype=np.float64).T
    return R.colored_mesh_repr(v, np.concatenate(faces, axis=0),
                               np.concatenate(colors, axis=0))


def render_articulated_states(job: JobDir, backend: str,
                              resolution: int = 512) -> dict:
    """eval/states/<backend>/: every movable joint at 0/50/100% of its clamped range,
    rendered from the canonical camera and one three-quarter view."""
    import math as _math

    from pbr_texture_pipeline.articulated import urdf as U

    out_dir = job.path("eval", "states", backend)
    out_dir.mkdir(parents=True, exist_ok=True)
    asset = U.parse_asset(job.state["asset"]["asset_dir"])
    movable = [child for child, j in asset.joints.items()
               if abs(U.opened_pose_q(j, 1.0, asset) - U.rest_pose_q(j)) > 1e-9]
    pitch = _math.radians(15.0)
    views = {"front": R.CANONICAL_YAW, "threequarter": R.CANONICAL_YAW + _math.radians(45)}

    manifest: dict = {"backend": backend, "joints": {}, "renders": []}
    for child in movable:
        jname = asset.joints[child].name or child
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in jname)
        manifest["joints"][jname] = {"child_link": child, "renders": []}
        for pct in (0, 50, 100):
            mesh = articulated_state_mesh(job, backend, pct / 100.0, only={child})
            if mesh is None:
                continue
            for tag, yaw in views.items():
                out = R.render_appearance(mesh, yaw, pitch, resolution, 2)
                path = out_dir / f"{safe}_{pct:03d}_{tag}.png"
                Image.fromarray(out["rgb"], mode="RGB").save(path)
                manifest["joints"][jname]["renders"].append(str(path))
                manifest["renders"].append(str(path))
    job.write_json(out_dir / "manifest.json", manifest)
    return manifest


def render_group_crops(job: JobDir, backend: str, group_ids: list[str],
                       resolution: int = 512) -> dict:
    """Close-up renders for retexture confirmation: each group's textured GLB alone (its own
    normalization makes it fill the frame), front + three-quarter side by side, written to
    eval/refine_crops/<backend>/<gid>.png. Returns {group_id: path}."""
    import math as _math

    from pbr_texture_pipeline.articulated import urdf as U
    from pbr_texture_pipeline.articulated.appearance import load_colored_arrays

    out_dir = job.path("eval", "refine_crops", backend)
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_state = job.state["asset"]
    asset = U.parse_asset(asset_state["asset_dir"])
    fk = U.link_world_transforms(asset)
    camera = job.read_json(job.camera_json()) if job.camera_json().is_file() else {}
    link_of = {g["group_id"]: g["link"] for g in asset_state["groups"]}
    pitch = _math.radians(15.0)

    crops: dict = {}
    for gid in group_ids:
        glb = job.textured_group_glb(backend, gid)
        if not glb.is_file() or gid not in link_of:
            continue
        try:
            v, f, c = load_colored_arrays(str(glb))
            T = fk.get(link_of[gid], np.eye(4))
            v = np.asarray(v, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
            v = R.to_internal_frame(v, up="z")  # FK frame is Z-up
            if camera.get("repose_applied") and camera.get("R") is not None:
                v = v @ np.asarray(camera["R"], dtype=np.float64).T
            mesh = R.colored_mesh_repr(v, f, c)
            tiles = [R.render_appearance(mesh, R.CANONICAL_YAW + _math.radians(d), pitch,
                                         resolution, 2)["rgb"] for d in (0.0, 45.0)]
            path = out_dir / f"{gid}.png"
            Image.fromarray(np.concatenate(tiles, axis=1), mode="RGB").save(path)
            crops[gid] = str(path)
        except Exception as e:  # noqa: BLE001
            print(f"[eval] refine crop failed for {gid}: {e}")
    return crops


def _texture_blur_estimate(glb_path: str) -> Optional[float]:
    """Mean gradient magnitude of a group GLB's base-color texture (higher = sharper).

    Texture-space stand-in for the PRD's render-based estimate: it measures the baked
    detail directly and is independent of screen coverage, which suits the relative
    per-group comparison the retexture heuristics make.
    """
    try:
        import cv2

        scene = trimesh.load(glb_path)
        geoms = list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]
        grads = []
        for g in geoms:
            mat = getattr(g.visual, "material", None)
            tex = getattr(mat, "baseColorTexture", None)
            if tex is None:
                continue
            arr = np.asarray(tex.convert("L"), dtype=np.float32)
            gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1)
            grads.append(float(np.mean(np.hypot(gx, gy))))
        return float(np.mean(grads)) if grads else None
    except Exception:  # noqa: BLE001
        return None


def write_diagnostics(job: JobDir, backends: list[str]) -> Optional[Path]:
    """eval/diagnostics.json: per-group occlusion (Stage R), field-voxel span and pass-B
    fraction (global adapter metadata), texture blur estimate, and bake warnings. This is
    the input for retexturing candidate selection (PRD_articulated_v2 section 7)."""
    asset_state = job.state.get("asset")
    if not asset_state:
        return None
    pass_a = {}
    pa_path = job.path("textured", "trellis2", "global", "pass_a.json")
    if pa_path.is_file():
        pass_a = job.read_json(pa_path).get("groups", {})

    groups: dict = {}
    blurs = []
    for g in asset_state["groups"]:
        gid = g["group_id"]
        meta = pass_a.get(gid, {})
        glb = job.textured_group_glb("trellis2", gid)
        blur = _texture_blur_estimate(str(glb)) if glb.is_file() else None
        if blur is not None and not meta.get("constant_pbr"):
            blurs.append(blur)
        groups[gid] = {
            "occluded_frac_rest": g.get("occluded_frac_rest"),
            "occluded_frac_open": g.get("occluded_frac_open"),
            "tiny": bool(g.get("tiny")),
            "n_faces": g.get("n_faces"),
            "field_voxels": meta.get("field_voxels"),
            "texels": meta.get("texels"),
            "pass_b_frac": meta.get("pass_b_frac"),
            "constant_pbr": bool(meta.get("constant_pbr")),
            "bake_error": meta.get("error"),
            "blur_estimate": blur,
        }
    diag = {
        "job_id": job.job_id,
        "backends": backends,
        "open_pose_pass_ran": pa_path.parent.joinpath("pass_b.json").is_file(),
        "object_median_blur": float(np.median(blurs)) if blurs else None,
        "groups": groups,
    }
    out = job.path("eval", "diagnostics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    job.write_json(out, diag)
    return out


# --- metric helpers ----------------------------------------------------------
def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        a_img = Image.fromarray((a * 255).astype(np.uint8)).resize((b.shape[1], b.shape[0]),
                                                                   Image.NEAREST)
        a = np.asarray(a_img) > 127
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _clip(prompt: str, image_path: Path) -> Optional[float]:
    """Reuse the Stage-D CLIP scorer (lazy-loads + caches CLIP; returns None if unavailable)."""
    from pbr_texture_pipeline.diffusion import clip_score
    return clip_score(prompt, image_path)


def _lpips(ref_rgb: np.ndarray, cond_rgb: np.ndarray) -> Optional[float]:
    """LPIPS(reference, condition-view). Both composited to the same size on white; None if unavailable."""
    global _lpips_model
    if _lpips_model == "unavailable":
        return None
    try:
        import torch
        if _lpips_model is None:
            import lpips
            _lpips_model = lpips.LPIPS(net="alex").cuda().eval()
    except Exception as e:  # noqa: BLE001
        print(f"[eval] LPIPS unavailable ({e}); skipping transfer-fidelity metric.")
        _lpips_model = "unavailable"
        return None

    import torch

    def _prep(arr: np.ndarray) -> "torch.Tensor":
        img = Image.fromarray(arr).convert("RGB").resize((256, 256), Image.BILINEAR)
        t = torch.tensor(np.asarray(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        return (t * 2 - 1).unsqueeze(0).cuda()

    with torch.no_grad():
        d = _lpips_model(_prep(ref_rgb), _prep(cond_rgb))
    return float(d.reshape(-1)[0])


def _composite_on_white(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Put a black-bg render on white so LPIPS compares object appearance, not background."""
    m = (mask > 0.5)[..., None]
    white = np.full_like(rgb, 255)
    return np.where(m, rgb, white)


def hunyuan_albedo_shadow(job: JobDir) -> Optional[dict]:
    """Spot-check Hunyuan's exported albedo for baked shadows (PRD risk 3).

    Heuristic proxy (not a hard gate): over textured (non-near-white) texels, report the
    fraction that are markedly dark. High dark_fraction on an object the spec described as
    light-coloured suggests shadows/AO were baked into albedo.
    """
    d = job.path("textured", "hunyuan")
    albedo = None
    for name in ("textured_mesh.jpg", "textured_mesh_albedo.jpg"):
        if (d / name).is_file():
            albedo = d / name
            break
    if albedo is None:
        # rglob: articulated jobs write per-group albedos under textured/hunyuan/groups/.
        cands = [p for p in sorted(d.rglob("*albedo*.jpg"))] + \
                [p for p in sorted(d.rglob("*.jpg"))
                 if "metallic" not in p.name and "roughness" not in p.name]
        albedo = cands[0] if cands else None
    if albedo is None:
        return None
    lum = np.asarray(Image.open(albedo).convert("L"), dtype=np.float32) / 255.0
    textured = lum < 0.97           # drop the near-white UV-atlas background
    if textured.sum() == 0:
        return {"albedo": str(albedo), "dark_fraction": 0.0, "mean_luma": 1.0}
    vals = lum[textured]
    return {
        "albedo": str(albedo),
        "mean_luma": float(vals.mean()),
        "dark_fraction": float((vals < 0.25).mean()),   # markedly dark texels
        "note": "heuristic shadow-bake proxy; high dark_fraction on a light object = suspect",
    }


# --- per-job driver ----------------------------------------------------------
def run_eval(job: JobDir, backends: list[str]) -> dict:
    """Render previews + compute metrics.json for a job's textured backends. Sets stage eval."""
    camera = None
    if job.camera_json().is_file():
        camera = job.read_json(job.camera_json())
    spec = job.read_json(job.spec()) if job.spec().is_file() else {}
    prompt = spec.get("ref_prompt", "")

    ctrl_mask = None
    if job.control("mask").is_file():
        ctrl_mask = np.asarray(Image.open(job.control("mask")).convert("L")) > 127

    ref_rgb = None
    if job.chosen().is_file():
        ref_rgb = np.asarray(Image.open(job.chosen()).convert("RGB"))

    metrics: dict = {
        "job_id": job.job_id,
        "prompt": prompt,
        "clip_prompt_vs_ref": (_clip(prompt, job.chosen()) if (prompt and job.chosen().is_file())
                               else None),
        "backends": {},
    }

    for backend in backends:
        glb = job.output_glb(backend)
        if glb is None:
            continue
        entry: dict = {"glb": str(glb)}

        # Appearance render from the condition camera (drives IoU / CLIP / LPIPS).
        mesh = load_output_colored(str(glb), camera, up="z" if job.kind == "urdf" else "y")
        res = int(_CFG.get("render.resolution"))
        cond = R.render_appearance(mesh, R.CANONICAL_YAW, R.CANONICAL_PITCH, res,
                                   int(_CFG.get("render.ssaa")))
        Image.fromarray(cond["rgb"], mode="RGB").save(job.preview_condview(backend))
        entry["condview"] = str(job.preview_condview(backend))

        out_mask = cond["mask"] > 0.5
        entry["silhouette_iou"] = (_mask_iou(out_mask, ctrl_mask)
                                   if ctrl_mask is not None else None)

        entry["clip_prompt_vs_condview"] = (_clip(prompt, Path(entry["condview"]))
                                            if prompt else None)

        if ref_rgb is not None:
            entry["lpips_ref_vs_condview"] = _lpips(
                ref_rgb, _composite_on_white(cond["rgb"], cond["mask"]))
        else:
            entry["lpips_ref_vs_condview"] = None

        # Turntable preview (separate load path handles the mp4/strip fallback).
        tt = render_turntable(job, backend, camera)
        entry["turntable"] = str(tt) if tt else None

        metrics["backends"][backend] = entry

    if "hunyuan" in backends:
        hs = hunyuan_albedo_shadow(job)
        if hs is not None:
            metrics["hunyuan_albedo_shadow"] = hs

    # Articulated additions (PRD_articulated_v2): joint-state renders per textured backend
    # and the per-group diagnostics that feed retexture candidate selection.
    if job.kind == "urdf":
        for backend in backends:
            if backend in metrics["backends"]:
                try:
                    manifest = render_articulated_states(job, backend)
                    metrics["backends"][backend]["state_renders"] = len(manifest["renders"])
                except Exception as e:  # noqa: BLE001
                    print(f"[eval] state renders failed for {backend}: {e}")
        diag = write_diagnostics(job, backends)
        metrics["diagnostics"] = str(diag) if diag else None

    job.write_json(job.metrics(), metrics)
    return metrics


# --- top-level index over all jobs (PRD section 8 step 5) --------------------
_INDEX_FIELDS = ["job_id", "category", "prompt", "backend", "winner", "silhouette_iou",
                 "clip_prompt_vs_condview", "lpips_ref_vs_condview", "needs_review"]


def _index_rows(jobs_root: str) -> list[dict]:
    rows = []
    for job in JobDir.load_all(jobs_root):
        spec = job.read_json(job.spec()) if job.spec().is_file() else {}
        metrics = job.read_json(job.metrics()) if job.metrics().is_file() else {}
        needs_review = any(job.status(s) == "needs_review"
                           for s in ("diffuse", "texture", "judge"))
        winner = job.winner() or ""
        bmap = metrics.get("backends", {})
        if not bmap:
            rows.append({"job_id": job.job_id, "category": spec.get("category", ""),
                         "prompt": spec.get("ref_prompt", ""), "backend": "",
                         "winner": winner, "silhouette_iou": "",
                         "clip_prompt_vs_condview": "",
                         "lpips_ref_vs_condview": "", "needs_review": needs_review})
            continue
        for backend, e in bmap.items():
            rows.append({
                "job_id": job.job_id, "category": spec.get("category", ""),
                "prompt": spec.get("ref_prompt", ""), "backend": backend,
                "winner": winner,
                "silhouette_iou": e.get("silhouette_iou", ""),
                "clip_prompt_vs_condview": e.get("clip_prompt_vs_condview", ""),
                "lpips_ref_vs_condview": e.get("lpips_ref_vs_condview", ""),
                "needs_review": needs_review,
            })
    return rows


def write_index(jobs_root: str) -> dict:
    """Assemble a top-level HTML + CSV index across every job (PRD section 8)."""
    rows = _index_rows(jobs_root)
    root = Path(jobs_root)
    csv_path = root / "index.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)

    html_path = root / "index.html"
    with open(html_path, "w") as f:
        f.write(_render_html(rows))
    return {"csv": str(csv_path), "html": str(html_path), "n_rows": len(rows)}


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return html.escape(str(v))


def _render_html(rows: list[dict]) -> str:
    head = ("<!doctype html><meta charset=utf-8><title>pbr-texture-pipeline jobs</title>"
            "<style>body{font-family:system-ui,sans-serif;margin:2rem}"
            "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;"
            "padding:.35rem .6rem;font-size:14px;text-align:left}"
            "th{background:#f3f3f3}tr:hover{background:#fafafa}"
            ".rev{background:#fff3cd}</style>"
            f"<h1>pbr-texture-pipeline jobs ({len(rows)} rows)</h1><table><tr>"
            + "".join(f"<th>{html.escape(c)}</th>" for c in _INDEX_FIELDS) + "</tr>")
    body = []
    for r in rows:
        cls = " class=rev" if r.get("needs_review") else ""
        body.append("<tr%s>%s</tr>" % (
            cls, "".join(f"<td>{_fmt(r.get(c, ''))}</td>" for c in _INDEX_FIELDS)))
    return head + "".join(body) + "</table>"

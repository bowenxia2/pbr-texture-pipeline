"""Per-stage drivers for articulated URDF jobs.

batch.py and the app's workers call these so interactive and batch stay one code path
(PRD.md, articulated extension). Heavy imports (torch, renderer, trimesh) are deferred
inside functions; importing this module never touches CUDA.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pbr_texture_pipeline.config import load_config

_CFG = load_config()


def effective_category(job) -> str:
    """meta.json category when present, else Stage V's spec.category, else 'object'."""
    cat = job.state.get("asset", {}).get("category")
    if cat:
        return cat
    if job.spec().is_file():
        cat = job.read_json(job.spec()).get("category")
        if cat:
            return cat
    return "object"


# --- Stage R (WS2) ------------------------------------------------------------
def merge_groups(group_world: dict) -> tuple:
    """Concatenate per-group (vertices, faces) arrays into one Trimesh in the dict's
    insertion order, recording the face range each group occupies in the merged face array.

    Returns (mesh, face_ranges) with face_ranges[gid] = [start, stop) into mesh.faces.
    Global texture mode (PRD.md section 7) slices the merged mesh back into groups by
    these ranges, so they are only valid against the exact merge built here; Stage R
    persists them to input/face_ranges.json and Stage T never rebuilds the merge.
    """
    import trimesh

    all_v, all_f = [], []
    face_ranges: dict[str, list[int]] = {}
    v_off = f_off = 0
    for gid, (v, f) in group_world.items():
        f = np.asarray(f)
        all_v.append(np.asarray(v))
        all_f.append(f + v_off)
        face_ranges[gid] = [f_off, f_off + len(f)]
        v_off += len(v)
        f_off += len(f)
    mesh = trimesh.Trimesh(vertices=np.concatenate(all_v, axis=0),
                           faces=np.concatenate(all_f, axis=0), process=False)
    return mesh, face_ranges


def render_open_controls(job, mesh_repr, R_mat) -> dict:
    """Control maps + camera_open.json for the open-pose merged mesh (global texture mode).

    Composed from the rendering primitives at the canonical condition camera; writes
    control/open/{depth,normal,canny,mask}.png and control/camera_open.json with the same
    schema as camera.json. Deliberately duplicates the file-writing glue of
    rendering.render_control_maps so that function and the JobDir path helpers stay
    untouched for the main render path.
    """
    import cv2
    from PIL import Image

    from pbr_texture_pipeline import rendering as R

    resolution = int(_CFG.get("render.resolution"))
    ssaa = int(_CFG.get("render.ssaa"))
    out_dir = job.path("control", "open")
    out_dir.mkdir(parents=True, exist_ok=True)

    r = R.render_view(mesh_repr, R.CANONICAL_YAW, R.CANONICAL_PITCH, resolution, ssaa,
                      return_types=("mask", "depth", "normal"))
    mask, depth, normal = r["mask"], r["depth"], r["normal"]

    Image.fromarray(R.depth_to_controlnet(depth, mask), mode="L").save(out_dir / "depth.png")
    normal_img = (np.clip(normal.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(normal_img, mode="RGB").save(out_dir / "normal.png")
    lo, hi = _CFG.get("render.canny_thresholds")
    edges = cv2.Canny(normal_img, int(lo), int(hi))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    Image.fromarray(edges, mode="L").save(out_dir / "canny.png")
    mask_u8 = ((mask > 0.5) * 255).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(out_dir / "mask.png")

    coverage = float((mask > 0.5).mean())
    cam = {
        "yaw": R.CANONICAL_YAW, "pitch": R.CANONICAL_PITCH, "r": R.R, "fov_deg": R.FOV_DEG,
        "resolution": resolution, "repose_applied": R_mat is not None,
        "R": R_mat.tolist() if R_mat is not None else None,
        "mask_coverage": coverage,
    }
    job.write_json(job.path("control", "camera_open.json"), cam)
    return {"mask_coverage_open": coverage}


def render(job) -> dict:
    """Articulated Stage R: group meshes, rest-pose clay assembly + control maps,
    per-group occlusion stats. Requires the trellis2 env (renderer + CUDA)."""
    import trimesh

    from pbr_texture_pipeline import rendering as R
    from pbr_texture_pipeline.articulated import urdf as U

    urdf_path = Path(job.state["mesh_source"])
    asset_dir = urdf_path.parent

    # Preflight: the URDF's transitive reference closure must exist on disk (decision 6).
    missing = [r for r in U.required_files(urdf_path) if not (asset_dir / r).is_file()]
    if missing:
        raise FileNotFoundError(
            f"asset {asset_dir} is missing {len(missing)} referenced file(s): "
            + ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""))

    asset = U.parse_asset(asset_dir)
    group_by = (job.state.get("params", {}).get("group_by")
                or str(_CFG.get("articulated.group_by", "semantic")))
    groups = U.build_groups(asset, group_by)
    fk = U.link_world_transforms(asset)

    # 1. Per-group merged meshes (link frame) + norm sidecars.
    merged: dict[str, "trimesh.Trimesh"] = {}
    for g in groups:
        m = U.merge_group_mesh(asset, g)
        job.group_dir(g.group_id).mkdir(parents=True, exist_ok=True)
        m.export(job.group_mesh(g.group_id))
        merged[g.group_id] = m

    total_area = sum(float(m.area) for m in merged.values()) or 1.0
    min_faces = int(_CFG.get("articulated.min_group_faces", 20))
    min_area_frac = float(_CFG.get("articulated.min_group_area_frac", 0.0005))

    group_world: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    group_meta: list[dict] = []
    for g in groups:
        m = merged[g.group_id]
        center, scale = U.norm_params(m)
        area_frac = float(m.area) / total_area
        tiny = (len(m.faces) < min_faces) or (area_frac < min_area_frac)
        job.write_json(job.group_norm(g.group_id), {
            "center": [float(c) for c in center], "scale": float(scale),
            "faces": int(len(m.faces)), "area": float(m.area), "area_frac": area_frac,
            "bounds_link": [[float(x) for x in b] for b in m.bounds],
        })
        T = fk.get(g.link, np.eye(4))
        v = np.asarray(m.vertices, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
        group_world[g.group_id] = (v, np.asarray(m.faces))
        group_meta.append({
            "group_id": g.group_id, "link": g.link, "label": g.label,
            "semantic_id": g.semantic_id,
            "objs": [vis.obj or f"<{vis.primitive['type']}>" for vis in g.visuals],
            "n_faces": int(len(m.faces)), "area_frac": area_frac, "motion": g.motion,
            "tiny": bool(tiny),
        })

    # 2. Rest-pose FK clay assembly -> mesh_norm + contact sheet + control maps (unchanged
    #    downstream: Stage D runs on this assembled depth). The URDF world frame is Z-up
    #    (verified on the test assets: cart wheels / table feet at min z), so normalize with
    #    up="z" (center+scale only, no axis swap for rendering). The exported mesh_norm.glb
    #    is then swapped to Y-up glTF convention (export_yup) so TRELLIS.2's preprocess_mesh
    #    (which assumes Y-up) correctly round-trips to Z-up internal. Azimuth is handled by
    #    the standard repose machinery: the canonical camera sees the front, and camera.json
    #    records R so Stage E/J un-rotate.
    assembled, face_ranges = merge_groups(group_world)
    all_v = np.asarray(assembled.vertices)
    norm0 = R.preprocess_mesh(assembled, up="z")

    # Front detection (PRD.md section 7.5): render the contact sheet from the
    # un-reposed mesh first, let Orient-Anything-V2 pick the front panel from those panel
    # files, then re-pose and re-render so Stage V keeps seeing front-aligned panels.
    # Config front_panel stays the fallback (detection disabled, unconfident, or failed).
    front_panel = int(_CFG.get("articulated.front_panel", 2))
    orient_decision = None
    if bool(_CFG.get("articulated.orient.enabled", False)):
        from pbr_texture_pipeline.articulated import orient as O

        R.render_contact_sheet(job, R.to_mesh_repr(norm0))
        orient_decision = O.detect_front_panel(job)
        front_panel = int(orient_decision["front_panel"])
    R_mat = R.repose_matrix(front_panel) if front_panel else None
    norm = R.apply_repose(norm0, R_mat) if R_mat is not None else norm0
    R.export_yup(norm, job.mesh_norm())
    mesh_repr = R.to_mesh_repr(norm)
    vis_res = int(_CFG.get("render.resolution"))
    cs = R.render_contact_sheet(job, mesh_repr, also_depth=True,
                                depth_resolution=vis_res)
    info = R.render_control_maps(job, mesh_repr, repose_applied=R_mat is not None,
                                 R_mat=R_mat)
    if orient_decision is not None:
        cam = job.read_json(job.camera_json())
        cam["orient"] = orient_decision
        job.write_json(job.camera_json(), cam)
        info["orient_method"] = orient_decision.get("method")
        info["orient_front_panel"] = front_panel

    # 2b. Open-pose merged mesh (PRD.md section 7.6): the same groups merged in
    #     the same order with movable joints opened, so the rest-pose face ranges apply to
    #     both meshes. Same normalization + re-pose machinery; its own norm params (the open
    #     bounds differ from rest).
    open_frac = float(_CFG.get("articulated.global.open_pose_frac", 0.8))
    fk_open = U.link_world_transforms_at(asset, open_frac)
    group_world_open = {}
    for g in groups:
        m = merged[g.group_id]
        T = fk_open.get(g.link, np.eye(4))
        v = np.asarray(m.vertices, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
        group_world_open[g.group_id] = (v, np.asarray(m.faces))
    assembled_open, face_ranges_open = merge_groups(group_world_open)
    assert face_ranges_open == face_ranges, "open-pose merge diverged from rest-pose merge"
    norm_open = R.preprocess_mesh(assembled_open, up="z")
    if R_mat is not None:
        norm_open = R.apply_repose(norm_open, R_mat)
    R.export_yup(norm_open, job.path("input", "mesh_norm_open.glb"))

    center_rest, scale_rest = U.norm_params(assembled)
    center_open, scale_open = U.norm_params(assembled_open)
    job.write_json(job.path("input", "face_ranges.json"), {
        "group_order": list(group_world.keys()),
        "face_ranges": face_ranges,
        "norm": {"center": [float(c) for c in center_rest], "scale": float(scale_rest)},
        "norm_open": {"center": [float(c) for c in center_open], "scale": float(scale_open)},
        "open_pose_frac": open_frac,
    })

    # 2c. Open-pose control maps + rest-pose visibility npz + per-group occlusion statistics
    #     (PRD.md section 3.8). The rest-pose depth buffers were captured during the contact
    #     sheet render above; save them as the visibility npz that the adapter reads.
    from pbr_texture_pipeline.backends import visibility as VIS

    mesh_repr_open = R.to_mesh_repr(norm_open)
    info.update(render_open_controls(job, mesh_repr_open, R_mat))
    VIS.save_views(job.path("control", "visibility_rest.npz"),
                   cs["depths"], cs["extrinsics"], cs["intrinsics"])

    def _internal(v: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
        w = (np.asarray(v, dtype=np.float64) - center) * scale
        return w if R_mat is None else w @ R_mat.T

    views_rest = VIS.load_views(job.path("control", "visibility_rest.npz"))
    n_samples = 1024
    for meta in group_meta:
        gid = meta["group_id"]
        v, f = group_world[gid]
        surf = trimesh.Trimesh(vertices=v, faces=f, process=False)
        pts, _fid = trimesh.sample.sample_surface(surf, n_samples)
        vis = VIS.visible_from_any(_internal(pts, center_rest, scale_rest), views_rest)
        meta["occluded_frac_rest"] = float(1.0 - vis.mean())

    job.set_asset({
        "asset_dir": str(asset_dir), "urdf": str(urdf_path),
        "category": asset.category, "group_by": group_by,
        "semantic_labels": sorted({label for (_motion, label) in asset.semantics.values()}),
        "bbox_world": {"min": [float(x) for x in all_v.min(axis=0)],
                       "max": [float(x) for x in all_v.max(axis=0)]},
        "groups": group_meta,
    })
    info["n_groups"] = len(group_meta)
    info["n_tiny"] = sum(1 for m in group_meta if m["tiny"])
    return info


# --- Stage T pair construction ------------------------------------------------
def global_texture_pair(job, seed: int):
    """One trellis2 pair for a job, or None when every group already has
    textured/trellis2/groups/<gid>.glb (the resume rule).

    Everything the adapter must not compute itself arrives here as plain data: per group a
    to_link 4x4 mapping the adapter's output frame (the pipeline re-normalization of
    mesh_norm.glb, with TRELLIS's axis swap already undone by its bake) back to the link
    frame. The chain inverts: adapter renorm -> Y-up-to-Z-up (undo the glTF export swap) ->
    re-pose -> Stage R normalization -> rest-pose FK. Gate A9 pins the frame chain.
    """
    import trimesh

    from pbr_texture_pipeline.articulated import urdf as U

    asset_state = job.state["asset"]
    franges_path = job.path("input", "face_ranges.json")
    if not franges_path.is_file():
        raise FileNotFoundError(f"{franges_path} missing; re-run Stage R for {job.job_id}")
    franges = job.read_json(franges_path)

    cam = job.read_json(job.camera_json())
    Rz = (np.asarray(cam["R"], dtype=np.float64)
          if cam.get("repose_applied") and cam.get("R") is not None else np.eye(3))
    c1 = np.asarray(franges["norm"]["center"], dtype=np.float64)
    s1 = float(franges["norm"]["scale"])
    merged = trimesh.load(str(job.mesh_norm()), force="mesh", process=False)
    c2, s2 = U.norm_params(merged)

    def _mat4(lin: np.ndarray, t: np.ndarray) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = lin
        M[:3, 3] = t
        return M

    # Y-up (glTF file) -> Z-up (internal): same swap as preprocess_mesh(up="y")
    _YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

    base = (_mat4(np.eye(3) / s1, c1)          # undo Stage R normalization (Z-up)
            @ _mat4(Rz.T, np.zeros(3))         # undo re-pose
            @ _mat4(_YUP_TO_ZUP, np.zeros(3))  # undo glTF Y-up export swap
            @ _mat4(np.eye(3) / s2, c2))       # undo adapter normalization (Y-up file)

    asset = U.parse_asset(asset_state["asset_dir"])
    fk = U.link_world_transforms(asset)
    tex_size = int(_CFG.get("articulated.global.per_link_texture_size", 1024))
    tiny_size = int(_CFG.get("articulated.global.tiny_texture_size", 256))

    groups = []
    for g in asset_state["groups"]:
        gid = g["group_id"]
        out_glb = job.textured_group_glb("trellis2", gid)
        if out_glb.is_file():
            continue
        to_link = np.linalg.inv(fk.get(g["link"], np.eye(4))) @ base
        norm = job.read_json(job.group_norm(gid))
        entry = {
            "group_id": gid, "out_glb": str(out_glb),
            "texture_size": tiny_size if g.get("tiny") else tex_size,
            "to_link": to_link.tolist(),
            "bounds_link": norm.get("bounds_link"),
        }
        groups.append(entry)
    if not groups:
        return None

    pair = {
        "kind": "global",
        "merged_mesh": str(job.mesh_norm()),
        "image": str(job.chosen_rgba()),
        "face_ranges": str(franges_path),
        "seed": int(seed),
        "out_dir": str(job.path("textured", "trellis2", "global")),
        "blend_band_texels": int(_CFG.get("articulated.global.blend_band_texels", 8)),
        "groups": groups,
    }
    if bool(_CFG.get("articulated.global.open_pose_pass", True)):
        open_img = job.path("ref", "chosen_rgba_open.png")
        open_mesh = job.path("input", "mesh_norm_open.glb")
        vis_rest = job.path("control", "visibility_rest.npz")
        if open_img.is_file() and open_mesh.is_file() and vis_rest.is_file():
            pair.update({"merged_mesh_open": str(open_mesh), "image_open": str(open_img),
                         "visibility_rest": str(vis_rest)})
        else:
            print(f"[global] {job.job_id}: open-pose pass artifacts missing; "
                  "running rest-pose only")
    return pair


# --- targeted retexturing (PRD.md section 7.7) ---------------------------------
def select_retexture_candidates(job) -> list[dict]:
    """Apply the retexture heuristics to eval/diagnostics.json. Always runs (the selection
    is a tracked metric even while execution is disabled); writes the candidate list back
    into the diagnostics file and returns it."""
    diag_path = job.path("eval", "diagnostics.json")
    if not diag_path.is_file():
        return []
    diag = job.read_json(diag_path)
    occ_thr = float(_CFG.get("articulated.refine.occluded_frac_threshold", 0.6))
    min_vox = int(_CFG.get("articulated.refine.min_field_voxels", 8))
    blur_ratio = float(_CFG.get("articulated.refine.blur_flag_ratio", 0.35))
    open_ran = bool(diag.get("open_pose_pass_ran"))
    median_blur = diag.get("object_median_blur")

    candidates = []
    for gid, g in diag.get("groups", {}).items():
        reasons = []
        occ = g.get("occluded_frac_rest")
        if not open_ran and occ is not None and occ > occ_thr:
            reasons.append(f"occluded_frac_rest {occ:.2f} > {occ_thr} without an open-pose pass")
        fv = g.get("field_voxels")
        if fv is not None and fv < min_vox:
            reasons.append(f"field_voxels {fv} < {min_vox}")
        blur = g.get("blur_estimate")
        if blur is not None and median_blur and blur < blur_ratio * median_blur:
            reasons.append(f"blur estimate {blur:.1f} < {blur_ratio} x median {median_blur:.1f}")
        if g.get("bake_error") or g.get("constant_pbr"):
            reasons.append("bake warning")
        if reasons:
            candidates.append({"group_id": gid, "reasons": reasons})
    diag["retexture_candidates"] = candidates
    job.write_json(diag_path, diag)
    return candidates


def retexture_groups(job, backend: str, group_ids: list[str]) -> list[str]:
    """Targeted retexture prep: move the named groups' textured GLBs aside so the
    missing-GLB resume rule (global_texture_pair lists only absent groups) re-emits exactly
    this set on the next Stage T run. Returns the groups actually moved."""
    moved = []
    for gid in group_ids:
        glb = job.textured_group_glb(backend, gid)
        if glb.is_file():
            aside = glb.with_suffix(".glb.retex")
            aside.unlink(missing_ok=True)
            glb.rename(aside)
            moved.append(gid)
    return moved


# --- assembly (end of Stage T) ------------------------------------------------
def assemble_backend(job, backend: str) -> dict:
    """Collect groups/<gid>.glb, write the textured URDF (+ collision symlinks) and the
    rest-pose assembled.glb. Returns per-group status + counts.

    The adapter writes (and bbox-checks) groups/<gid>.glb directly, tiny groups included,
    so collection here is only an existence check."""
    from pbr_texture_pipeline.articulated import urdf as U

    asset_state = job.state["asset"]
    status: dict[str, str] = {}
    for g in asset_state["groups"]:
        gid = g["group_id"]
        status[gid] = "ok" if job.textured_group_glb(backend, gid).is_file() else "error"

    # 2. Textured URDF: a link is rewritten only when ALL its groups succeeded (mixing
    #    textured groups with original visuals would duplicate geometry).
    by_link: dict[str, list[dict]] = {}
    for g in asset_state["groups"]:
        by_link.setdefault(g["link"], []).append(g)
    group_glbs: dict[str, list[tuple[str, str]]] = {}
    for link, gs in by_link.items():
        if all(status.get(g["group_id"]) == "ok" for g in gs):
            group_glbs[link] = [(g["group_id"], f"groups/{g['group_id']}.glb") for g in gs]

    tex_dir = job.textured_dir(backend)
    U.write_textured_urdf(asset_state["urdf"], str(job.textured_urdf(backend)), group_glbs)

    # 3. Collision meshes: symlink the referenced top-level asset dirs next to the URDF so
    #    <collision> relpaths resolve (collision_mode: symlink).
    if str(_CFG.get("articulated.collision_mode", "symlink")) == "symlink":
        asset_dir = Path(asset_state["asset_dir"])
        tops = set()
        for rel in U.required_files(asset_state["urdf"]):
            tops.add(rel.split("/", 1)[0])
        for top in sorted(tops):
            src_dir = asset_dir / top
            link_path = tex_dir / top
            if src_dir.exists() and not link_path.exists():
                link_path.symlink_to(src_dir)

    # 4. Rest-pose assembled scene.
    n_ok = sum(1 for s in status.values() if s == "ok")
    assembled = None
    if n_ok:
        assembled = U.assemble_scene(job, backend)

    return {"groups": status, "n_ok": n_ok, "n_total": len(status),
            "textured_urdf": str(job.textured_urdf(backend)),
            "assembled_glb": str(assembled) if assembled else None}

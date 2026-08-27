"""Per-stage drivers for articulated URDF jobs.

batch.py and the app's workers call these so interactive and batch stay one code path.
Heavy imports (torch, renderer, trimesh) are deferred inside functions; importing this
module never touches CUDA.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pbr_texture_pipeline.config import load_config

_CFG = load_config()


def effective_category(job) -> str:
    """meta.json category when present, else 'object'."""
    cat = job.state.get("asset", {}).get("category")
    if cat:
        return cat
    return "object"


# --- Stage R ------------------------------------------------------------------
def merge_groups(group_world: dict) -> tuple:
    """Concatenate per-group (vertices, faces) arrays into one Trimesh in the dict's
    insertion order, recording the face range each group occupies in the merged face array.

    Returns (mesh, face_ranges) with face_ranges[gid] = [start, stop) into mesh.faces.
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


def render(job) -> dict:
    """Articulated Stage R: group meshes, rest-pose assembly + front-view rendering.

    1. Parse URDF, build groups, compute rest-pose FK.
    2. Merge groups into a single rest-pose mesh for TRELLIS.2 (mesh_norm.glb + face_ranges).
    3. Detect front via Orient-Anything-V2 (optional).
    4. Render front textured RGBA view + depth map via pyrender.
    """
    import trimesh

    from pbr_texture_pipeline import rendering as R
    from pbr_texture_pipeline.articulated import urdf as U

    urdf_path = Path(job.state["mesh_source"])
    asset_dir = urdf_path.parent

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
    merged_textured: dict[str, "trimesh.Trimesh"] = {}
    for g in groups:
        m = U.merge_group_mesh(asset, g)
        job.group_dir(g.group_id).mkdir(parents=True, exist_ok=True)
        m.export(job.group_mesh(g.group_id))
        merged[g.group_id] = m
        merged_textured[g.group_id] = U.merge_group_mesh(asset, g, with_materials=True)

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

    # 2. Rest-pose FK clay assembly -> mesh_norm.glb + face_ranges (still needed by Stage T).
    assembled, face_ranges = merge_groups(group_world)
    all_v = np.asarray(assembled.vertices)
    norm0 = R.preprocess_mesh(assembled, up="z")

    # Front detection: render a quick contact sheet for Orient-V2, then discard.
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

    center_rest, scale_rest = U.norm_params(assembled)
    job.write_json(job.path("input", "face_ranges.json"), {
        "group_order": list(group_world.keys()),
        "face_ranges": face_ranges,
        "norm": {"center": [float(c) for c in center_rest], "scale": float(scale_rest)},
    })

    cam = {
        "repose_applied": R_mat is not None,
        "R": R_mat.tolist() if R_mat is not None else None,
    }
    if orient_decision is not None:
        cam["orient"] = orient_decision
    job.write_json(job.camera_json(), cam)

    # 3. Render front textured view + depth map via pyrender.
    num_views = int(_CFG.get("render.num_views", 1))
    resolution = int(_CFG.get("render.resolution", 1024))
    group_meshes_textured = []
    for g in groups:
        T = fk.get(g.link, np.eye(4))
        group_meshes_textured.append((g.group_id, merged_textured[g.group_id], T))
    render_out = R.render_textured_views(job, group_meshes_textured, R_mat,
                                         num_views=num_views, resolution=resolution)

    # Remove stale contact-sheet views (8 panels at 45-deg) beyond the textured set.
    for k in range(num_views, R.CONTACT_N):
        stale = job.render_view(k)
        if stale.exists():
            stale.unlink()

    info = {}
    if orient_decision is not None:
        info["orient_method"] = orient_decision.get("method")
        info["orient_front_panel"] = front_panel

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


# --- Stage V ------------------------------------------------------------------

def vlm(job) -> dict:
    """Stage V: VLM material analysis + texture quality classification."""
    front_white = job.render_front_white()
    if not front_white.is_file():
        raise FileNotFoundError(
            f"front_white.png missing; re-run Stage R for {job.job_id}")
    out_path = job.vlm_materials()
    class_path = job.vlm_classification()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    from pbr_texture_pipeline.backends import registry

    category = effective_category(job)
    items = [{
        "image": str(front_white),
        "output": str(out_path),
        "classification_output": str(class_path),
        "category": category,
    }]
    items_path = job.root / "_items_vlm.json"
    with open(items_path, "w") as f:
        json.dump(items, f)
    try:
        registry.vlm_infer_batch(str(items_path))
    finally:
        items_path.unlink(missing_ok=True)

    if not out_path.is_file():
        raise RuntimeError(f"VLM did not produce {out_path}")
    materials = out_path.read_text().strip()
    classification = "edit"
    if class_path.is_file():
        classification = class_path.read_text().strip().lower()
        if classification not in ("edit", "generate"):
            classification = "edit"
    return {"materials": materials, "classification": classification}


# --- Stage E ------------------------------------------------------------------
def imageedit(job) -> dict:
    """Stage E: enhance or generate front-panel image based on VLM classification."""
    front_white = job.render_front_white()
    canny = job.render_canny(0)
    materials_path = job.vlm_materials()

    for p, label in [(front_white, "front_white.png"),
                     (canny, "canny_0.png"),
                     (materials_path, "materials.txt")]:
        if not p.is_file():
            raise FileNotFoundError(
                f"{label} missing; re-run earlier stages for {job.job_id}")

    materials = materials_path.read_text().strip()
    classification_path = job.vlm_classification()
    classification = "edit"
    if classification_path.is_file():
        classification = classification_path.read_text().strip().lower()
        if classification not in ("edit", "generate"):
            classification = "edit"

    out_path = job.enhanced_view(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    from pbr_texture_pipeline.backends import registry

    if classification == "generate":
        category = effective_category(job)
        items = [{
            "canny": str(canny),
            "materials": materials,
            "output": str(out_path),
            "category": category,
        }]
        items_path = job.root / "_items_imagegen.json"
        with open(items_path, "w") as f:
            json.dump(items, f)
        try:
            registry.imagegen_infer_batch(str(items_path))
        finally:
            items_path.unlink(missing_ok=True)
    else:
        items = [{
            "source": str(front_white),
            "canny": str(canny),
            "materials": materials,
            "output": str(out_path),
        }]
        items_path = job.root / "_items_imageedit.json"
        with open(items_path, "w") as f:
            json.dump(items, f)
        try:
            registry.imageedit_infer_batch(str(items_path))
        finally:
            items_path.unlink(missing_ok=True)

    if not out_path.is_file():
        raise RuntimeError(f"Stage E did not produce {out_path}")
    return {"enhanced": str(out_path), "path": classification}


# --- Stage T pair construction ------------------------------------------------
def global_texture_pair(job, seed: int):
    """One trellis2 pair for a job, or None when every group already has
    textured/trellis2/groups/<gid>.glb (the resume rule).

    Everything the adapter must not compute itself arrives here as plain data: per group a
    to_link 4x4 mapping the adapter's output frame back to the link frame.
    The chain inverts: adapter renorm -> Y-up-to-Z-up (undo the glTF export swap) ->
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

    enhanced = job.enhanced_view(0)
    image_paths = [str(enhanced)] if enhanced.is_file() else [str(job.render_view(0))]
    pair = {
        "kind": "global",
        "merged_mesh": str(job.mesh_norm()),
        "image": image_paths,
        "face_ranges": str(franges_path),
        "seed": int(seed),
        "out_dir": str(job.path("textured", "trellis2", "global")),
        "groups": groups,
    }
    return pair


# --- assembly (end of Stage T) ------------------------------------------------
def assemble_backend(job, backend: str) -> dict:
    """Collect groups/<gid>.glb, write the textured URDF (+ collision symlinks), the
    rest-pose assembled.glb, and the original URDF + assembled_original.glb for comparison.
    Returns per-group status + counts."""
    import shutil

    from pbr_texture_pipeline.articulated import urdf as U

    asset_state = job.state["asset"]
    status: dict[str, str] = {}
    for g in asset_state["groups"]:
        gid = g["group_id"]
        status[gid] = "ok" if job.textured_group_glb(backend, gid).is_file() else "error"

    by_link: dict[str, list[dict]] = {}
    for g in asset_state["groups"]:
        by_link.setdefault(g["link"], []).append(g)
    group_glbs: dict[str, list[tuple[str, str]]] = {}
    for link, gs in by_link.items():
        if all(status.get(g["group_id"]) == "ok" for g in gs):
            group_glbs[link] = [(g["group_id"], f"groups/{g['group_id']}.glb") for g in gs]

    tex_dir = job.textured_dir(backend)
    U.write_textured_urdf(asset_state["urdf"], str(job.textured_urdf(backend)), group_glbs)

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

    shutil.copy2(asset_state["urdf"], str(job.original_urdf(backend)))

    n_ok = sum(1 for s in status.values() if s == "ok")
    assembled = None
    if n_ok:
        assembled = U.assemble_scene(job, backend)

    original_glb = U.assemble_original_scene(job, backend)

    return {"groups": status, "n_ok": n_ok, "n_total": len(status),
            "textured_urdf": str(job.textured_urdf(backend)),
            "original_urdf": str(job.original_urdf(backend)),
            "assembled_glb": str(assembled) if assembled else None,
            "assembled_original_glb": str(original_glb)}

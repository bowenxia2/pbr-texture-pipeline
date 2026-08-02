"""TRELLIS.2 global texturing adapter for articulated jobs (PRD_articulated_v2).

One pair per job: decode the PBR voxel field ONCE for the whole merged rest-pose mesh
(splitting Trellis2TexturingPipeline.run before its bake step), then bake every part group
its own UV atlas from that shared field by slicing the merged mesh with the face ranges
Stage R persisted. With the optional open-pose fields present, a second field is decoded
from the open-pose merged mesh and texels whose surface points are hidden at rest are
re-sampled from it, blended over a band at the visibility boundary.

Runs in the `trellis2` env with cwd = TRELLIS.2/ (so texturing_pipeline.json resolves).
No pbr-texture-pipeline imports: `_adapter_common` and `visibility` are imported from the script's own
directory, and everything needing FK/repose/camera math (the per-group `to_link` matrices)
arrives precomputed in the pairs file.

Pair schema (one entry per job):
  {"kind": "global", "merged_mesh", "image", "face_ranges", "seed", "out_dir",
   "blend_band_texels", "groups": [{"group_id", "out_glb", "texture_size", "to_link" (4x4),
                                    "bounds_link" ([[min],[max]], optional)}],
   optional open-pose pass: "merged_mesh_open", "image_open", "visibility_rest" (npz)}

Emits one [PBR_RESULT] line per baked group and a final summary line per pair.
Success for a group = its result line ok:true plus out_glb on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Launched with cwd = TRELLIS.2/; `python /abs/script.py` only adds the script dir to
# sys.path, so add cwd for `import trellis2`.
sys.path.insert(0, os.getcwd())

import numpy as np
import trimesh
from PIL import Image

from _adapter_common import RESULT_MARKER, emit_result, norm_params  # noqa: F401
import visibility as VIS

_BBOX_TOL = 1e-3  # same relative tolerance as the orchestrator's per-part bbox check
_BBOX_SHRINK_TOL = 5e-2  # max extent fraction uv_unwrap may lose by welding sliver faces


def _load_mesh(path: str) -> trimesh.Trimesh:
    """Load preserving vertex/face order: face ranges are only valid against the exact
    arrays Stage R exported (gate A9 verifies this load path)."""
    return trimesh.load(path, force="mesh", process=False)


def _unswap(v: np.ndarray) -> np.ndarray:
    """Field/internal frame -> the merged-mesh file frame (inverse of preprocess_mesh's
    axis swap): (x, y, z) -> (x, z, -y)."""
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)


def _decode_field(pipe, mesh_path: str, image_path: str, seed: int, resolution: int):
    """The front half of Trellis2TexturingPipeline.run: normalize + condition + sample +
    decode, stopping before the bake. Returns (pbr_voxel, preprocessed_mesh, center, scale)
    where center/scale are the pipeline's re-normalization of the loaded file (needed to map
    field-frame points back into the file frame)."""
    import torch

    mesh = _load_mesh(mesh_path)
    center, scale = norm_params(mesh.vertices)
    image = pipe.preprocess_image(Image.open(image_path))
    mesh_p = pipe.preprocess_mesh(mesh)
    torch.manual_seed(seed)
    cond = pipe.get_cond([image], 512 if resolution == 512 else 1024)
    shape_slat = pipe.encode_shape_slat(mesh_p, resolution)
    tex_model = (pipe.models["tex_slat_flow_model_512"] if resolution == 512
                 else pipe.models["tex_slat_flow_model_1024"])
    tex_slat = pipe.sample_tex_slat(cond, tex_model, shape_slat, {})
    pbr_voxel = pipe.decode_tex_slat(tex_slat)
    return pbr_voxel, mesh_p, center, scale


def _sample_field(pipe, pbr_voxel, pos, resolution: int):
    """Trilinear field sample at field-frame positions pos [N,3] -> attrs [N,C] (torch)."""
    import flex_gemm
    import torch

    return flex_gemm.ops.grid_sample.grid_sample_3d(
        pbr_voxel.feats,
        pbr_voxel.coords,
        shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
        grid=((pos + 0.5) * resolution).reshape(1, -1, 3),
        mode="trilinear",
    )


def _bake_group(pipe, ctx, group: dict, faces: np.ndarray, mesh_a, field_a,
                resolution: int, open_ctx: dict | None) -> tuple[trimesh.Trimesh, dict]:
    """Bake one group's atlas from the shared field(s); returns (mesh in link frame, stats).

    Mirrors Trellis2TexturingPipeline.postprocess_mesh (uv unwrap, rasterize in UV space,
    field sample, inpaint, PBR material, axis un-swap) with three extensions: the geometry
    is a face-range slice of the merged mesh, hidden-at-rest texels are optionally
    re-sampled from the open-pose field, and the result is mapped to the link frame by the
    orchestrator-provided to_link matrix.
    """
    import cumesh
    import cv2
    import nvdiffrast.torch as dr
    import torch

    texture_size = int(group["texture_size"])
    start, stop = group["face_range"]
    f = faces[start:stop]
    used = np.unique(f)
    remap = -np.ones(len(mesh_a.vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    sub = trimesh.Trimesh(vertices=np.asarray(mesh_a.vertices)[used], faces=remap[f],
                          process=False)
    normals = np.asarray(sub.vertex_normals)

    vertices_torch = torch.from_numpy(np.asarray(sub.vertices)).float().cuda()
    faces_torch = torch.from_numpy(np.asarray(sub.faces)).int().cuda()
    _cumesh = cumesh.CuMesh()
    _cumesh.init(vertices_torch, faces_torch)
    vertices_torch, faces_torch, uvs_torch, vmap = _cumesh.uv_unwrap(return_vmaps=True)
    vertices_torch = vertices_torch.cuda()
    faces_torch = faces_torch.cuda()
    uvs_torch = uvs_torch.cuda()
    vmap_np = vmap.cpu().numpy()
    vertices = vertices_torch.cpu().numpy()
    faces_out = faces_torch.cpu().numpy()
    uvs = uvs_torch.cpu().numpy()
    normals = normals[vmap_np]

    uv_clip = torch.cat([uvs_torch * 2 - 1, torch.zeros_like(uvs_torch[:, :1]),
                         torch.ones_like(uvs_torch[:, :1])], dim=-1).unsqueeze(0)
    rast, _ = dr.rasterize(ctx, uv_clip, faces_torch, resolution=[texture_size, texture_size])
    mask_t = rast[0, ..., 3] > 0
    pos = dr.interpolate(vertices_torch.unsqueeze(0), rast, faces_torch)[0][0]

    n_ch = field_a.shape[1]
    attrs = torch.zeros(texture_size, texture_size, n_ch, device=pos.device)
    attrs_a = _sample_field(pipe, field_a, pos[mask_t], resolution)
    stats = {
        "texture_size": texture_size, "n_faces": int(stop - start),
        "texels": int(mask_t.sum().item()),
        "field_voxels": int(torch.unique(
            ((pos[mask_t] + 0.5) * resolution).long(), dim=0).shape[0]),
        "pass_b_frac": 0.0,
    }

    if open_ctx is not None:
        # Per-texel rest-pose visibility: map texel positions from field frame to the
        # merged-mesh file frame the npz cameras live in, then test against the depth
        # buffers with the same routine Stage R used for the occlusion statistics.
        pos_np = pos[mask_t].cpu().numpy().astype(np.float64)
        pts_file = _unswap(pos_np) / open_ctx["scale_rest"] + open_ctx["center_rest"]
        visible = VIS.visible_from_any(pts_file, open_ctx["views_rest"])

        # Same face slice of the open-pose mesh: identical topology, so the unwrapped
        # vertex map carries over and the same rasterization interpolates open positions.
        v_open_sub = np.asarray(open_ctx["mesh_open"].vertices)[used][vmap_np]
        v_open_t = torch.from_numpy(v_open_sub).float().cuda()
        pos_open = dr.interpolate(v_open_t.unsqueeze(0), rast, faces_torch)[0][0]
        attrs_b = _sample_field(pipe, open_ctx["field_open"], pos_open[mask_t], resolution)

        mask_np = mask_t.cpu().numpy()
        hidden = np.zeros(mask_np.shape, dtype=np.uint8)
        hidden[mask_np] = (~visible).astype(np.uint8)
        band = max(int(open_ctx["blend_band_texels"]), 1)
        dist = cv2.distanceTransform(hidden, cv2.DIST_L2, 3)
        w_full = np.clip(dist / band, 0.0, 1.0) * (hidden > 0)
        w = torch.from_numpy(w_full[mask_np].astype(np.float32)).to(pos.device)[:, None]
        attrs_a = attrs_a * (1 - w) + attrs_b * w
        stats["pass_b_frac"] = float((~visible).mean()) if len(visible) else 0.0

    attrs[mask_t] = attrs_a

    # Texture construction, verbatim from postprocess_mesh.
    mask_np = mask_t.cpu().numpy()
    layout = pipe.pbr_attr_layout
    base_color = np.clip(attrs[..., layout["base_color"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., layout["metallic"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., layout["roughness"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., layout["alpha"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    inpaint_mask = (~mask_np).astype(np.uint8)
    base_color = cv2.inpaint(base_color, inpaint_mask, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, inpaint_mask, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, inpaint_mask, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, inpaint_mask, 1, cv2.INPAINT_TELEA)[..., None]
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(
            np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
        doubleSided=True,
    )

    # Field frame -> file frame (un-swap), then -> link frame (to_link).
    T = np.asarray(group["to_link"], dtype=np.float64)
    v_out = _unswap(vertices.astype(np.float64))
    v_link = v_out @ T[:3, :3].T + T[:3, 3]
    n_out = _unswap(normals.astype(np.float64)) @ T[:3, :3].T
    n_out /= np.maximum(np.linalg.norm(n_out, axis=1, keepdims=True), 1e-12)
    uvs[:, 1] = 1 - uvs[:, 1]
    out = trimesh.Trimesh(vertices=v_link, faces=faces_out, vertex_normals=n_out,
                          process=False,
                          visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
    return out, stats


def _constant_pbr(group: dict, faces: np.ndarray, mesh_a) -> trimesh.Trimesh:
    """Neutral constant-PBR fallback when unwrap/bake fails (rare; no plan lookup here)."""
    start, stop = group["face_range"]
    f = faces[start:stop]
    used = np.unique(f)
    remap = -np.ones(len(mesh_a.vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    T = np.asarray(group["to_link"], dtype=np.float64)
    v_out = _unswap(np.asarray(mesh_a.vertices)[used].astype(np.float64))
    v_link = v_out @ T[:3, :3].T + T[:3, 3]
    mesh = trimesh.Trimesh(vertices=v_link, faces=remap[f], process=False)
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[128, 128, 128, 255], metallicFactor=0.0, roughnessFactor=0.9)
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return mesh


def _slice_link_bounds(group: dict, faces: np.ndarray, mesh_a) -> np.ndarray:
    """Link-frame bounds of the exact face-range slice mapped by to_link (no unwrap)."""
    start, stop = group["face_range"]
    used = np.unique(faces[start:stop])
    T = np.asarray(group["to_link"], dtype=np.float64)
    v = _unswap(np.asarray(mesh_a.vertices)[used].astype(np.float64))
    v = v @ T[:3, :3].T + T[:3, 3]
    return np.stack([v.min(axis=0), v.max(axis=0)])


def _check_bounds(out: trimesh.Trimesh, group: dict,
                  slice_bounds: np.ndarray) -> tuple[str | None, str | None]:
    """Own bbox sanity check vs the input group's link-frame bounds (from the pair), so the
    orchestrator's assemble step needs no changes. Two causes are separated:

    to_link correctness: the exact input slice mapped by to_link must reproduce the
    orchestrator's link-frame bounds. A mismatch means a bad transform, which the
    constant-PBR fallback could not place correctly either, so it fails the group loudly.

    Bake fidelity: cumesh uv_unwrap welds degenerate sliver faces, so the baked mesh may
    legitimately shrink a little inside the slice bounds (seen on 48-face PartNet groups).
    Shrinkage beyond _BBOX_SHRINK_TOL of the extent means the bake mangled the geometry
    and the constant-PBR fallback (exact slice geometry) should take over.

    Returns (fatal_error, warning); at most one is set.
    """
    bounds = group.get("bounds_link")
    if bounds is None:
        return None, None
    ref = np.asarray(bounds, dtype=np.float64)
    extent = float((ref[1] - ref[0]).max()) or 1.0
    err_t = float(np.abs(slice_bounds - ref).max())
    if err_t > _BBOX_TOL * extent:
        return (f"to_link bounds mismatch: bbox err {err_t:.5f} "
                f"> {_BBOX_TOL * extent:.5f}"), None
    err_b = float(np.abs(np.asarray(out.bounds) - ref).max())
    if err_b > _BBOX_SHRINK_TOL * extent:
        return (f"bake geometry loss: bbox err {err_b:.5f} "
                f"> {_BBOX_SHRINK_TOL * extent:.5f}"), None
    if err_b > _BBOX_TOL * extent:
        return None, f"bbox shrank {err_b:.5f} (uv_unwrap welded sliver faces)"
    return None, None


def _run_pair(pipe, ctx, pair: dict, resolution: int) -> None:
    import torch

    out_dir = pair["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    seed = int(pair.get("seed", 42))

    with open(pair["face_ranges"]) as fp:
        franges = json.load(fp)

    todo = [dict(g, face_range=franges["face_ranges"][g["group_id"]])
            for g in pair["groups"] if not os.path.isfile(g["out_glb"])]
    for g in pair["groups"]:
        if os.path.isfile(g["out_glb"]):
            emit_result({"backend": "trellis2_global", "group_id": g["group_id"],
                         "glb_path": g["out_glb"], "ok": True, "skipped": True})
    if not todo:
        emit_result({"backend": "trellis2_global", "kind": "global_summary",
                     "glb_path": pair["groups"][-1]["out_glb"] if pair["groups"] else None,
                     "ok": True, "n_baked": 0, "n_failed": 0})
        return

    with torch.no_grad():
        field_a, mesh_a, center_rest, scale_rest = _decode_field(
            pipe, pair["merged_mesh"], pair["image"], seed, resolution)
        faces = np.asarray(mesh_a.faces)

        open_ctx = None
        if pair.get("merged_mesh_open") and pair.get("image_open"):
            field_b, mesh_b_p, _c, _s = _decode_field(
                pipe, pair["merged_mesh_open"], pair["image_open"], seed, resolution)
            open_ctx = {
                "field_open": field_b,
                # Positions are interpolated from the open mesh in ITS field frame, so keep
                # the preprocessed open mesh (same face/vertex order as the file).
                "mesh_open": mesh_b_p,
                "views_rest": VIS.load_views(pair["visibility_rest"]),
                "center_rest": np.asarray(center_rest, dtype=np.float64),
                "scale_rest": float(scale_rest),
                "blend_band_texels": int(pair.get("blend_band_texels", 8)),
            }

        pass_meta: dict[str, dict] = {}
        n_ok = n_fail = 0
        for group in todo:
            gid = group["group_id"]
            glb_path = group["out_glb"]
            os.makedirs(os.path.dirname(glb_path), exist_ok=True)
            try:
                out, stats = _bake_group(pipe, ctx, group, faces, mesh_a, field_a,
                                         resolution, open_ctx)
                fatal, warn = _check_bounds(out, group,
                                            _slice_link_bounds(group, faces, mesh_a))
                if fatal is not None:
                    raise RuntimeError(fatal)
                if warn is not None:
                    stats["bbox_warn"] = warn
                out.export(glb_path, extension_webp=True)
                pass_meta[gid] = stats
                n_ok += 1
                emit_result({"backend": "trellis2_global", "group_id": gid,
                             "glb_path": glb_path, "ok": True, **stats})
            except Exception as e:  # noqa: BLE001
                # to_link mismatches mean a bad transform and must fail loudly; bake/unwrap
                # failures (including bake geometry loss) degrade to a constant-PBR GLB.
                if "to_link bounds mismatch" in str(e):
                    n_fail += 1
                    emit_result({"backend": "trellis2_global", "group_id": gid,
                                 "glb_path": None, "ok": False, "error": str(e)})
                    continue
                try:
                    _constant_pbr(group, faces, mesh_a).export(glb_path)
                    pass_meta[gid] = {"constant_pbr": True, "error": str(e)}
                    n_ok += 1
                    emit_result({"backend": "trellis2_global", "group_id": gid,
                                 "glb_path": glb_path, "ok": True, "constant_pbr": True,
                                 "error": str(e)})
                except Exception as e2:  # noqa: BLE001
                    n_fail += 1
                    emit_result({"backend": "trellis2_global", "group_id": gid,
                                 "glb_path": None, "ok": False,
                                 "error": f"{e}; fallback failed: {e2}"})

    # Group-granular resume: a partial run bakes only the missing groups, so the metadata
    # of previously baked groups must be merged, not clobbered.
    def _merged(path: str, key: str, fresh: dict) -> dict:
        prev = {}
        if os.path.isfile(path):
            try:
                with open(path) as fp:
                    prev = json.load(fp).get(key, {})
            except Exception:  # noqa: BLE001
                prev = {}
        prev.update(fresh)
        return prev

    pass_a_path = os.path.join(out_dir, "pass_a.json")
    with open(pass_a_path, "w") as fp:
        json.dump({"resolution": resolution, "seed": seed,
                   "field_voxels_total": int(field_a.coords.shape[0]),
                   "groups": _merged(pass_a_path, "groups", pass_meta)}, fp, indent=2)
    if open_ctx is not None:
        pass_b_path = os.path.join(out_dir, "pass_b.json")
        fresh_b = {g: m.get("pass_b_frac", 0.0) for g, m in pass_meta.items()}
        with open(pass_b_path, "w") as fp:
            json.dump({"seed": seed,
                       "field_voxels_total": int(open_ctx["field_open"].coords.shape[0]),
                       "pass_b_frac": _merged(pass_b_path, "pass_b_frac", fresh_b)},
                      fp, indent=2)

    last_ok = next((g["out_glb"] for g in reversed(pair["groups"])
                    if os.path.isfile(g["out_glb"])), None)
    emit_result({"backend": "trellis2_global", "kind": "global_summary",
                 "glb_path": last_ok, "ok": n_fail == 0,
                 "n_baked": n_ok, "n_failed": n_fail})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file", required=True,
                    help="JSON list of global pairs (one per job)")
    ap.add_argument("--resolution", type=int, default=1024, choices=[512, 1024])
    ap.add_argument("--texture-size", type=int, default=2048,
                    help="unused (per-group texture_size comes from the pair); kept for CLI parity")
    args = ap.parse_args()

    import nvdiffrast.torch as dr
    from trellis2.pipelines import Trellis2TexturingPipeline

    pipe = Trellis2TexturingPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
    pipe.cuda()
    ctx = dr.RasterizeCudaContext()

    with open(args.pairs_file) as fp:
        pairs = json.load(fp)
    for pair in (pairs if isinstance(pairs, list) else [pairs]):
        try:
            _run_pair(pipe, ctx, pair, args.resolution)
        except Exception as e:  # noqa: BLE001
            emit_result({"backend": "trellis2_global", "kind": "global_summary",
                         "glb_path": None, "ok": False, "error": str(e)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

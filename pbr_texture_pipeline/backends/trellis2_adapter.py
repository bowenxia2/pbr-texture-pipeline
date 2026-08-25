"""TRELLIS.2 global texturing adapter for articulated jobs.

One pair per job: decode the PBR voxel field ONCE for the whole merged rest-pose mesh
(splitting Trellis2TexturingPipeline.run before its bake step), then bake every part group
its own UV atlas from that shared field by slicing the merged mesh with the face ranges
Stage R persisted. Multi-image conditioning: each view's tokens are concatenated by the
forked TRELLIS.2 so the model cross-attends to all of them.

Runs in the `trellis2` env with cwd = TRELLIS.2/ (so texturing_pipeline.json resolves).
No pbr-texture-pipeline imports: `_adapter_common` is imported from the script's own
directory, and everything needing FK/repose/camera math (the per-group `to_link` matrices)
arrives precomputed in the pairs file.

Pair schema (one entry per job):
  {"kind": "global", "merged_mesh", "image": [list of RGBA paths],
   "face_ranges", "seed", "out_dir",
   "groups": [{"group_id", "out_glb", "texture_size", "to_link" (4x4),
               "bounds_link" ([[min],[max]], optional)}]}

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

_BBOX_TOL = 1e-3  # same relative tolerance as the orchestrator's per-part bbox check
_BBOX_SHRINK_TOL = 5e-2  # max extent fraction uv_unwrap may lose by welding sliver faces


def _load_mesh(path: str) -> trimesh.Trimesh:
    """Load preserving vertex/face order: face ranges are only valid against the exact
    arrays Stage R exported."""
    return trimesh.load(path, force="mesh", process=False)


def _unswap(v: np.ndarray) -> np.ndarray:
    """Field/internal Z-up frame -> Y-up file frame (inverse of preprocess_mesh's
    Y-to-Z axis swap): (x, y, z) -> (x, z, -y)."""
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)


def _decode_field(pipe, mesh_path: str, image_paths: list[str], seed: int, resolution: int):
    """The front half of Trellis2TexturingPipeline.run: normalize + condition + sample +
    decode, stopping before the bake. Returns (pbr_voxel, preprocessed_mesh, center, scale)
    where center/scale are the pipeline's re-normalization of the loaded file (needed to map
    field-frame points back into the file frame).

    Accepts a list of image paths; each is preprocessed and their tokens are concatenated
    by get_cond so the model cross-attends to all views.
    """
    import torch

    mesh = _load_mesh(mesh_path)
    center, scale = norm_params(mesh.vertices)
    images = [pipe.preprocess_image(Image.open(p)) for p in image_paths]
    mesh_p = pipe.preprocess_mesh(mesh)
    torch.manual_seed(seed)
    cond = pipe.get_cond(images, 512 if resolution == 512 else 1024)
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


def _split_sharp(mesh, angle_deg: float = 60.0):
    """Split vertices at edges where the face-face angle exceeds *angle_deg*.

    After splitting, ``trimesh.Trimesh.vertex_normals`` averages only within
    each smooth region, giving correct hard edges at creases while keeping
    shared vertices on smooth surfaces (important for xatlas chart quality).
    """
    from collections import defaultdict

    verts = np.asarray(mesh.vertices)
    faces_arr = np.asarray(mesh.faces)
    fn = np.asarray(mesh.face_normals)
    cos_thresh = np.cos(np.radians(angle_deg))

    v2f = defaultdict(list)
    for fi, face in enumerate(faces_arr):
        for vi in face:
            v2f[int(vi)].append(fi)

    new_verts = list(verts)
    new_faces = faces_arr.copy()

    for vi in range(len(verts)):
        adj = v2f[vi]
        if len(adj) <= 1:
            continue

        face_others = {
            fi: {int(v) for v in new_faces[fi] if v != vi} for fi in adj
        }

        visited = set()
        groups = []
        for fi_start in adj:
            if fi_start in visited:
                continue
            group = []
            stack = [fi_start]
            while stack:
                fi = stack.pop()
                if fi in visited:
                    continue
                visited.add(fi)
                group.append(fi)
                for fj in adj:
                    if fj in visited:
                        continue
                    if face_others[fi] & face_others[fj]:
                        if np.dot(fn[fi], fn[fj]) >= cos_thresh:
                            stack.append(fj)
            groups.append(group)

        if len(groups) <= 1:
            continue

        for group in groups[1:]:
            new_vi = len(new_verts)
            new_verts.append(verts[vi].copy())
            for fi in group:
                for j in range(3):
                    if new_faces[fi, j] == vi:
                        new_faces[fi, j] = new_vi
                        break

    return trimesh.Trimesh(
        vertices=np.asarray(new_verts), faces=new_faces, process=False
    )


def _bake_group(pipe, ctx, group: dict, faces: np.ndarray, mesh_a, field_a,
                resolution: int) -> tuple[trimesh.Trimesh, dict]:
    """Bake one group's atlas from the shared field; returns (mesh in link frame, stats).

    Mirrors Trellis2TexturingPipeline.postprocess_mesh (uv unwrap, rasterize in UV space,
    field sample, inpaint, PBR material, axis un-swap) with two extensions: the geometry
    is a face-range slice of the merged mesh, and the result is mapped to the link frame by
    the orchestrator-provided to_link matrix.
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
    sub.merge_vertices(merge_norm=True)
    sub = _split_sharp(sub)

    vertices_torch = torch.from_numpy(np.asarray(sub.vertices)).float().cuda()
    faces_torch = torch.from_numpy(np.asarray(sub.faces)).int().cuda()
    _cumesh = cumesh.CuMesh()
    _cumesh.init(vertices_torch, faces_torch)
    vertices_torch, faces_torch, uvs_torch, vmap = _cumesh.uv_unwrap(return_vmaps=True)
    vertices_torch = vertices_torch.cuda()
    faces_torch = faces_torch.cuda()
    uvs_torch = uvs_torch.cuda()
    vertices = vertices_torch.cpu().numpy()
    faces_out = faces_torch.cpu().numpy()
    uvs = uvs_torch.cpu().numpy()
    normals = np.asarray(trimesh.Trimesh(
        vertices=vertices, faces=faces_out, process=False).vertex_normals)

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
    }

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
    sub = trimesh.Trimesh(vertices=np.asarray(mesh_a.vertices)[used], faces=remap[f],
                          process=False)
    sub.merge_vertices(merge_norm=True)
    T = np.asarray(group["to_link"], dtype=np.float64)
    v_out = _unswap(np.asarray(sub.vertices, dtype=np.float64))
    v_link = v_out @ T[:3, :3].T + T[:3, 3]
    mat = trimesh.visual.material.PBRMaterial(
        baseColorFactor=np.array([180, 180, 180, 255], dtype=np.uint8),
        metallicFactor=0.0, roughnessFactor=0.8, alphaMode="OPAQUE")
    return trimesh.Trimesh(vertices=v_link, faces=sub.faces, process=False,
                           visual=trimesh.visual.TextureVisuals(material=mat))


def _check_bounds(mesh_out, group, bounds_from_merged):
    """Compare baked group bounds to orchestrator expectation; return (fatal, warn) msgs."""
    bl = group.get("bounds_link")
    if bl is None:
        return None, None
    expected_min, expected_max = np.array(bl[0]), np.array(bl[1])
    got_min, got_max = mesh_out.bounds
    extent = np.maximum(expected_max - expected_min, 1e-8)
    err = np.max(np.abs(np.concatenate([(got_min - expected_min) / extent,
                                        (got_max - expected_max) / extent])))
    if err > _BBOX_TOL:
        err_b = np.max(np.clip(expected_min - got_min, 0, None) +
                       np.clip(got_max - expected_max, 0, None)) / np.max(extent)
        if err_b > _BBOX_SHRINK_TOL:
            return f"to_link bounds mismatch: rel err {err:.5f}", None
        return None, f"bbox shrank {err_b:.5f} (uv_unwrap welded sliver faces)"
    return None, None


def _slice_link_bounds(group, faces, mesh_a):
    """Compute link-frame bounds from the merged-mesh slice (for comparison)."""
    start, stop = group["face_range"]
    f = faces[start:stop]
    used = np.unique(f)
    sub_v = np.asarray(mesh_a.vertices)[used]
    T = np.asarray(group["to_link"], dtype=np.float64)
    v_out = _unswap(sub_v.astype(np.float64))
    v_link = v_out @ T[:3, :3].T + T[:3, 3]
    return v_link.min(axis=0), v_link.max(axis=0)


def _run_pair(pipe, ctx, pair: dict, resolution: int) -> None:
    import torch

    out_dir = pair["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    seed = int(pair.get("seed", 42))

    with open(pair["face_ranges"]) as fp:
        franges = json.load(fp)

    def _glb_valid(path: str) -> bool:
        return os.path.isfile(path) and os.path.getsize(path) > 0

    todo = [dict(g, face_range=franges["face_ranges"][g["group_id"]])
            for g in pair["groups"] if not _glb_valid(g["out_glb"])]
    for g in pair["groups"]:
        if _glb_valid(g["out_glb"]):
            emit_result({"backend": "trellis2", "group_id": g["group_id"],
                         "glb_path": g["out_glb"], "ok": True, "skipped": True})
    if not todo:
        emit_result({"backend": "trellis2", "kind": "global_summary",
                     "glb_path": pair["groups"][-1]["out_glb"] if pair["groups"] else None,
                     "ok": True, "n_baked": 0, "n_failed": 0})
        return

    with torch.no_grad():
        field_a, mesh_a, center_rest, scale_rest = _decode_field(
            pipe, pair["merged_mesh"], pair["image"], seed, resolution)
        faces = np.asarray(mesh_a.faces)

        pass_meta: dict[str, dict] = {}
        n_ok = n_fail = 0
        for group in todo:
            gid = group["group_id"]
            glb_path = group["out_glb"]
            os.makedirs(os.path.dirname(glb_path), exist_ok=True)
            try:
                out, stats = _bake_group(pipe, ctx, group, faces, mesh_a, field_a,
                                         resolution)
                fatal, warn = _check_bounds(out, group,
                                            _slice_link_bounds(group, faces, mesh_a))
                if fatal is not None:
                    raise RuntimeError(fatal)
                if warn is not None:
                    stats["bbox_warn"] = warn
                out.export(glb_path, extension_webp=True)
                pass_meta[gid] = stats
                n_ok += 1
                emit_result({"backend": "trellis2", "group_id": gid,
                             "glb_path": glb_path, "ok": True, **stats})
            except Exception as e:  # noqa: BLE001
                if "to_link bounds mismatch" in str(e):
                    n_fail += 1
                    emit_result({"backend": "trellis2", "group_id": gid,
                                 "glb_path": None, "ok": False, "error": str(e)})
                    continue
                try:
                    _constant_pbr(group, faces, mesh_a).export(glb_path)
                    pass_meta[gid] = {"constant_pbr": True, "error": str(e)}
                    n_ok += 1
                    emit_result({"backend": "trellis2", "group_id": gid,
                                 "glb_path": glb_path, "ok": True, "constant_pbr": True,
                                 "error": str(e)})
                except Exception as e2:  # noqa: BLE001
                    if os.path.isfile(glb_path) and os.path.getsize(glb_path) == 0:
                        os.unlink(glb_path)
                    n_fail += 1
                    emit_result({"backend": "trellis2", "group_id": gid,
                                 "glb_path": None, "ok": False,
                                 "error": f"{e}; fallback failed: {e2}"})

    # Group-granular resume: merge metadata with previous runs.
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

    last_ok = next((g["out_glb"] for g in reversed(pair["groups"])
                    if os.path.isfile(g["out_glb"])), None)
    emit_result({"backend": "trellis2", "kind": "global_summary",
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
            emit_result({"backend": "trellis2", "kind": "global_summary",
                         "glb_path": None, "ok": False, "error": str(e)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

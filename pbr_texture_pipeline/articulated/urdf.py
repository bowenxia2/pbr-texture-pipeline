"""URDF parsing, semantic grouping, rest-pose FK, and textured-URDF rewriting.

Port of trellis_pbr/urdf_utils.py generalized from per-link to per-(link, visual-name) groups
(PRD.md, articulated extension). PartNet-Mobility assets are static OBJ fragments assembled by a
URDF: each <link> holds <visual> elements referencing OBJs with per-visual <origin> transforms
in the link frame; joints connect links with fixed/revolute/prismatic types.

Metadata policy (decision 6): only mobility.urdf plus the files it transitively references are
required. semantics.txt / result.json / meta.json / bounding_box.json are used when present:
  - group labels come from the URDF <visual name="label-id"> attribute (verified identical to
    result.json leaves), so grouping never needs result.json;
  - visuals missing `name` degrade the asset to group_by="link" with label = link name;
  - motion type falls back from semantics.txt to the URDF joint type;
  - category falls back from meta.json to Stage V's spec.category (callers handle that);
  - bounding_box.json only feeds the A1 bbox cross-check (skipped when absent).
"""
from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# --- low-level parsing --------------------------------------------------------
def parse_origin(elem) -> np.ndarray:
    """4x4 transform for a URDF <origin> element (xyz + rpy, fixed-axis XYZ)."""
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if elem is not None:
        if elem.get("xyz"):
            xyz = [float(v) for v in elem.get("xyz").split()]
        if elem.get("rpy"):
            rpy = [float(v) for v in elem.get("rpy").split()]
    T = _euler_sxyz(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = xyz
    return T


def _euler_sxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis (extrinsic) XYZ euler -> 4x4 (== tf.euler_matrix(..., axes='sxyz'))."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    return T


_NAME_ID_RE = re.compile(r"^(.*?)-(\d+)$")


def split_visual_name(name: str) -> tuple[str, Optional[int]]:
    """URDF visual name "glass-12" -> ("glass", 12); no trailing id -> (name, None)."""
    m = _NAME_ID_RE.match(name)
    if m:
        return m.group(1), int(m.group(2))
    return name, None


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _parse_primitive(geom_elem) -> Optional[dict]:
    """Parse a URDF <geometry> element for box/cylinder/sphere primitives.
    Returns None if the geometry is a <mesh> (handled separately)."""
    box = geom_elem.find("box")
    if box is not None:
        return {"type": "box", "size": [float(v) for v in box.get("size").split()]}
    cyl = geom_elem.find("cylinder")
    if cyl is not None:
        return {"type": "cylinder", "radius": float(cyl.get("radius")),
                "length": float(cyl.get("length"))}
    sph = geom_elem.find("sphere")
    if sph is not None:
        return {"type": "sphere", "radius": float(sph.get("radius"))}
    return None


def primitive_to_trimesh(prim: dict):
    """Convert a primitive geometry dict to a trimesh object."""
    import trimesh
    if prim["type"] == "box":
        return trimesh.creation.box(extents=prim["size"])
    elif prim["type"] == "cylinder":
        return trimesh.creation.cylinder(radius=prim["radius"], height=prim["length"])
    elif prim["type"] == "sphere":
        return trimesh.creation.icosphere(radius=prim["radius"])
    raise ValueError(f"unknown primitive type {prim['type']!r}")


# --- data model ---------------------------------------------------------------
@dataclass
class Visual:
    obj: Optional[str]            # OBJ relpath from asset_dir (None for primitive geometry)
    origin: np.ndarray            # 4x4 link-frame transform
    name: Optional[str]           # <visual name="..."> attribute (None when absent)
    primitive: Optional[dict] = None  # e.g. {"type": "box", "size": [x,y,z]}


@dataclass
class Joint:
    name: str
    type: str                     # fixed | revolute | continuous | prismatic | ...
    parent: str
    child: str
    origin: np.ndarray            # 4x4
    axis: np.ndarray              # 3, child-frame
    lower: Optional[float]
    upper: Optional[float]


@dataclass
class AssetInfo:
    asset_dir: Path
    urdf_path: Path
    links: dict[str, list[Visual]]        # every link, including visual-less ones (base)
    joints: dict[str, Joint]              # child link name -> joint
    root_link: str
    category: Optional[str]               # meta.json model_cat, or None
    semantics: dict[str, tuple[str, str]]  # link -> (motion, label) from semantics.txt
    bbox: Optional[dict]                  # bounding_box.json {min, max}, or None
    result_tree: Optional[list] = None    # result.json (human-readable hierarchy), or None
    all_visuals_named: bool = True


@dataclass
class Group:
    group_id: str
    link: str
    label: str                    # semantic label ("glass"); link name when degraded
    semantic_id: Optional[int]    # trailing -<id> of the visual name, or None
    visuals: list[Visual] = field(default_factory=list)
    motion: str = "static"        # semantics.txt motion, or URDF-joint-type fallback


def find_urdf(asset_dir: str | os.PathLike[str]) -> Path:
    """Find the URDF file in an asset directory: mobility.urdf or model.urdf."""
    asset_dir = Path(asset_dir)
    for name in ("mobility.urdf", "model.urdf"):
        p = asset_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"no mobility.urdf or model.urdf in {asset_dir}")


def parse_asset(asset_dir: str | os.PathLike[str]) -> AssetInfo:
    """Parse an asset directory. A URDF file (mobility.urdf or model.urdf) is required;
    all other metadata optional."""
    asset_dir = Path(asset_dir)
    urdf_path = find_urdf(asset_dir)
    root = ET.parse(urdf_path).getroot()

    links: dict[str, list[Visual]] = {}
    all_named = True
    for link in root.findall("link"):
        name = link.get("name")
        visuals = []
        for vis in link.findall("visual"):
            geom = vis.find("geometry")
            if geom is None:
                continue
            mesh_elem = geom.find("mesh")
            prim = None
            obj_path = None
            if mesh_elem is not None:
                obj_path = mesh_elem.get("filename")
            else:
                prim = _parse_primitive(geom)
                if prim is None:
                    continue
            vname = vis.get("name")
            if not vname:
                all_named = False
            visuals.append(Visual(obj=obj_path,
                                  origin=parse_origin(vis.find("origin")),
                                  name=vname, primitive=prim))
        links[name] = visuals

    joints: dict[str, Joint] = {}
    for j in root.findall("joint"):
        parent = j.find("parent")
        child = j.find("child")
        if parent is None or child is None:
            continue
        axis_elem = j.find("axis")
        axis = np.array([float(v) for v in axis_elem.get("xyz").split()]) \
            if axis_elem is not None and axis_elem.get("xyz") else np.array([1.0, 0.0, 0.0])
        limit = j.find("limit")
        lower = float(limit.get("lower")) if limit is not None and limit.get("lower") else None
        upper = float(limit.get("upper")) if limit is not None and limit.get("upper") else None
        joints[child.get("link")] = Joint(
            name=j.get("name"), type=j.get("type", "fixed"),
            parent=parent.get("link"), child=child.get("link"),
            origin=parse_origin(j.find("origin")), axis=axis, lower=lower, upper=upper)

    children = set(joints.keys())
    root_link = next((n for n in links if n not in children), next(iter(links)))

    category = None
    meta_path = asset_dir / "meta.json"
    if meta_path.is_file():
        try:
            category = json.loads(meta_path.read_text()).get("model_cat")
        except (OSError, json.JSONDecodeError):
            category = None
    if not category:
        robot_name = root.get("name")
        if robot_name:
            category = robot_name.replace("_", " ").strip() or None

    semantics: dict[str, tuple[str, str]] = {}
    sem_path = asset_dir / "semantics.txt"
    if sem_path.is_file():
        for line in sem_path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                semantics[parts[0]] = (parts[1], parts[2])

    bbox = None
    bbox_path = asset_dir / "bounding_box.json"
    if bbox_path.is_file():
        try:
            bbox = json.loads(bbox_path.read_text())
        except (OSError, json.JSONDecodeError):
            bbox = None

    result_tree = None
    result_path = asset_dir / "result.json"
    if result_path.is_file():
        try:
            result_tree = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            result_tree = None

    return AssetInfo(asset_dir=asset_dir, urdf_path=urdf_path, links=links, joints=joints,
                     root_link=root_link, category=category, semantics=semantics, bbox=bbox,
                     result_tree=result_tree, all_visuals_named=all_named)


# --- required-files closure (app upload validation + batch preflight) ---------
def _obj_mtllibs(obj_path: Path) -> list[str]:
    refs = []
    try:
        with open(obj_path, errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s.startswith("mtllib "):
                    refs.append(s[len("mtllib "):].strip())
                elif s.startswith(("v ", "f ")):
                    break  # header is over; mtllib always precedes geometry in these assets
    except OSError:
        pass
    return refs


def _mtl_maps(mtl_path: Path) -> list[str]:
    refs = []
    try:
        with open(mtl_path, errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s.startswith("map_"):
                    parts = s.split(None, 1)
                    if len(parts) == 2:
                        # Drop any map options (-o, -s, ...): the filename is the last token.
                        refs.append(parts[1].split()[-1])
    except OSError:
        pass
    return refs


def required_files(urdf_path: str | os.PathLike[str]) -> list[str]:
    """Transitive reference closure of a URDF, as sorted relpaths from its directory.

    URDF <visual>/<collision> mesh files, mtllib refs inside those OBJs, and map_* image refs
    inside those MTLs. Primitive geometries (box/cylinder/sphere) have no file refs and are
    skipped. Relpaths are normalized (`textured_objs/../images/x.jpg` -> `images/x.jpg`) so
    they can be checked against files on disk or an upload set.
    """
    urdf_path = Path(urdf_path)
    asset_dir = urdf_path.parent
    root = ET.parse(urdf_path).getroot()

    out: set[str] = set()
    mesh_rels: set[str] = set()
    for mesh in root.findall(".//visual/geometry/mesh") + root.findall(".//collision/geometry/mesh"):
        fn = mesh.get("filename")
        if fn:
            mesh_rels.add(os.path.normpath(fn).replace(os.sep, "/"))
    out |= mesh_rels

    for mesh_rel in sorted(mesh_rels):
        if not mesh_rel.lower().endswith(".obj"):
            continue
        obj_path = asset_dir / mesh_rel
        obj_dir = os.path.dirname(mesh_rel)
        for mtl in _obj_mtllibs(obj_path):
            mtl_rel = os.path.normpath(os.path.join(obj_dir, mtl)).replace(os.sep, "/")
            out.add(mtl_rel)
            mtl_dir = os.path.dirname(mtl_rel)
            for img in _mtl_maps(asset_dir / mtl_rel):
                out.add(os.path.normpath(os.path.join(mtl_dir, img)).replace(os.sep, "/"))
    return sorted(out)


# --- rest pose + forward kinematics -------------------------------------------
def rest_pose_q(joint: Joint) -> float:
    """Rest pose = q=0 clamped into the joint limits (all 5 verified assets contain 0)."""
    if joint.lower is None or joint.upper is None:
        return 0.0
    return min(max(0.0, joint.lower), joint.upper)


def motion_transform(joint: Joint, q: float) -> np.ndarray:
    """Joint motion transform at value q (identity for fixed joints / q == 0)."""
    T = np.eye(4)
    if q == 0.0:
        return T
    axis = joint.axis / (np.linalg.norm(joint.axis) or 1.0)
    if joint.type in ("revolute", "continuous"):
        c, s = np.cos(q), np.sin(q)
        x, y, z = axis
        C = 1 - c
        T[:3, :3] = np.array([
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ])
    elif joint.type == "prismatic":
        T[:3, 3] = axis * q
    return T


def opened_pose_q(joint: Joint, frac: float, asset: Optional[AssetInfo] = None) -> float:
    """Opened joint value: the clamped rest value lerped by `frac` toward the OPEN limit
    (PRD.md section 7.6). Fixed joints and joints without limits (continuous)
    stay at rest; the value stays inside the limits for any frac in [0, 1].

    With `asset` given, the open limit is the geometric choice from open_target_q (the
    direction that moves the child geometry away from the assembly). Without it, the limit
    farther from rest is used - correct when rest sits at the closed end of the range, which
    holds for 4 of the 5 test assets but NOT for dishwasher 12085 (rest mid-range), so every
    caller that has the asset should pass it.
    """
    rest = rest_pose_q(joint)
    if joint.type == "fixed" or joint.lower is None or joint.upper is None:
        return rest
    if asset is not None:
        target = open_target_q(asset, joint)
    else:
        target = joint.upper if (joint.upper - rest) >= (rest - joint.lower) else joint.lower
    return rest + frac * (target - rest)


_OPEN_TARGET_CACHE: dict[tuple[str, str], float] = {}
_LINK_CORNER_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _link_bounds_corners(asset: AssetInfo) -> dict[str, np.ndarray]:
    """{link: [8,3] AABB corners of its visuals' geometry in the link frame}, cached per
    asset (one OBJ load pass; only open_target_q needs it). The AABB of affinely transformed
    corners bounds the AABB of the transformed vertices, which is all the direction
    comparison needs."""
    import trimesh

    key = str(asset.urdf_path)
    cached = _LINK_CORNER_CACHE.get(key)
    if cached is not None:
        return cached
    out: dict[str, np.ndarray] = {}
    for link, visuals in asset.links.items():
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for vis in visuals:
            try:
                if vis.primitive is not None:
                    m = primitive_to_trimesh(vis.primitive)
                else:
                    m = trimesh.load(str(asset.asset_dir / vis.obj), force="mesh",
                                     process=False, skip_materials=True)
            except Exception:  # noqa: BLE001
                continue
            v = np.asarray(m.vertices, dtype=np.float64)
            if len(v) == 0:
                continue
            v = v @ np.asarray(vis.origin[:3, :3]).T + np.asarray(vis.origin[:3, 3])
            lo = np.minimum(lo, v.min(axis=0))
            hi = np.maximum(hi, v.max(axis=0))
        if np.isfinite(lo).all():
            out[link] = np.array([[x, y, z] for x in (lo[0], hi[0])
                                  for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    _LINK_CORNER_CACHE[key] = out
    return out


def open_target_q(asset: AssetInfo, joint: Joint) -> float:
    """The limit value that OPENS this joint: of the two limits, the one that grows the
    assembled bounding box more (a shelf slides out of the body, a door swings away from
    the frame; moving INTO the body cannot grow the assembly's bounds).

    PartNet-Mobility records no open direction, and rest = clamped q=0 can sit mid-range:
    dishwasher 12085's shelf has limits [-0.58, 0.14] around rest 0, where the limit
    farther from rest slides the shelf INSIDE the body (assembled extent 1.49 -> 0.91)
    while +0.14 pulls it out (-> 1.62). Pure FK + cached per-link bounds, no rendering and
    no joint-state search; cached per (asset, joint). Near-equal volumes (a freestanding
    door swinging either way) fall back to the limit farther from rest."""
    rest = rest_pose_q(joint)
    key = (str(asset.urdf_path), joint.name or joint.child)
    if key in _OPEN_TARGET_CACHE:
        return _OPEN_TARGET_CACHE[key]

    corners = _link_bounds_corners(asset)

    def volume(q: float) -> float:
        memo: dict[str, np.ndarray] = {}

        def world(link: str) -> np.ndarray:
            if link in memo:
                return memo[link]
            j = asset.joints.get(link)
            if j is None:
                memo[link] = np.eye(4)
            else:
                qj = q if j is joint else rest_pose_q(j)
                memo[link] = world(j.parent) @ j.origin @ motion_transform(j, qj)
            return memo[link]

        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for link in asset.links:
            if link not in corners:
                continue
            T = world(link)
            pts = corners[link] @ T[:3, :3].T + T[:3, 3]
            lo = np.minimum(lo, pts.min(axis=0))
            hi = np.maximum(hi, pts.max(axis=0))
        return float(np.prod(hi - lo)) if np.isfinite(lo).all() else 0.0

    farther = joint.upper if (joint.upper - rest) >= (rest - joint.lower) else joint.lower
    target = farther
    v_lo, v_hi = volume(joint.lower), volume(joint.upper)
    if v_lo > 0 and v_hi > 0 and abs(v_lo - v_hi) > 1e-6 * max(v_lo, v_hi):
        best = joint.lower if v_lo > v_hi else joint.upper
        # A no-motion pick (rest already at that limit) would freeze the joint; keep the
        # farther limit instead.
        if abs(best - rest) > 1e-9:
            target = best
    _OPEN_TARGET_CACHE[key] = target
    return target


def link_world_transforms_at(asset: AssetInfo, joint_frac: float,
                             only: Optional[set[str]] = None) -> dict[str, np.ndarray]:
    """World transform per link with movable joints opened to `joint_frac` of their range
    (opened_pose_q). `only`, when given, restricts opening to the named child links (Stage E
    single-joint state renders); every other joint stays at rest. joint_frac == 0 reproduces
    link_world_transforms."""
    memo: dict[str, np.ndarray] = {}

    def world(link: str) -> np.ndarray:
        if link in memo:
            return memo[link]
        j = asset.joints.get(link)
        if j is None:
            memo[link] = np.eye(4)
        else:
            opened = only is None or link in only
            q = opened_pose_q(j, joint_frac, asset) if opened else rest_pose_q(j)
            memo[link] = world(j.parent) @ j.origin @ motion_transform(j, q)
        return memo[link]

    for name in asset.links:
        world(name)
    return memo


def link_world_transforms(asset: AssetInfo) -> dict[str, np.ndarray]:
    """World transform per link at rest pose: product of joint <origin> transforms from the
    base, with a motion transform only when the clamped rest q != 0. Verified numerically on
    8930 (a tree, not a chain): assembled vertices == R_base @ raw_concat exactly."""
    memo: dict[str, np.ndarray] = {}

    def world(link: str) -> np.ndarray:
        if link in memo:
            return memo[link]
        j = asset.joints.get(link)
        if j is None:
            memo[link] = np.eye(4)
        else:
            memo[link] = world(j.parent) @ j.origin @ motion_transform(j, rest_pose_q(j))
        return memo[link]

    for name in asset.links:
        world(name)
    return memo


# --- grouping -----------------------------------------------------------------
def link_motion(asset: AssetInfo, link: str) -> str:
    """semantics.txt motion for a link, else derived from its URDF joint type."""
    if link in asset.semantics:
        return asset.semantics[link][0]
    j = asset.joints.get(link)
    if j is None:
        return "static"
    return {"revolute": "hinge", "continuous": "hinge", "prismatic": "slider",
            "fixed": "static", "floating": "free", "planar": "free"}.get(j.type, "static")


def build_groups(asset: AssetInfo, group_by: str = "semantic") -> list[Group]:
    """Partition every visual into texture groups.

    group_by="semantic": one group per (link, visual name); group_id = "<link>__<name>"
    sanitized; label = the name with its trailing -<id> stripped. Assets with any unnamed
    visual degrade to link grouping (metadata fallback, decision 6).
    group_by="link": one group per link; label = semantics.txt label, else link name.
    """
    if group_by not in ("semantic", "link"):
        raise ValueError(f"unknown group_by {group_by!r}")
    if group_by == "semantic" and not asset.all_visuals_named:
        group_by = "link"

    groups: dict[str, Group] = {}
    for link, visuals in asset.links.items():
        for vis in visuals:
            if group_by == "semantic":
                label, sem_id = split_visual_name(vis.name)
                gid = _sanitize(f"{link}__{vis.name}")
            else:
                label = asset.semantics[link][1] if link in asset.semantics else link
                sem_id = None
                gid = _sanitize(link)
            g = groups.get(gid)
            if g is None:
                g = Group(group_id=gid, link=link, label=label, semantic_id=sem_id,
                          motion=link_motion(asset, link))
                groups[gid] = g
            g.visuals.append(vis)
    return list(groups.values())


# --- mesh merging + normalization ---------------------------------------------
def merge_group_mesh(asset: AssetInfo, group: Group, with_materials: bool = False):
    """Load a group's visuals (OBJ meshes or primitives), bake per-visual origins,
    concatenate -> one Trimesh in the link frame.

    When *with_materials* is False (default), materials are stripped - the geometry-only
    mesh is what TRELLIS re-UVs and face_ranges/norm_params operate on.
    When True, OBJ materials and textures are preserved for pyrender textured rendering.
    """
    import trimesh

    parts = []
    for vis in group.visuals:
        if vis.primitive is not None:
            m = primitive_to_trimesh(vis.primitive)
        else:
            obj_path = str(asset.asset_dir / vis.obj)
            kwargs = dict(force="mesh", process=False,
                          skip_materials=not with_materials)
            if with_materials:
                kwargs["resolver"] = trimesh.visual.resolvers.FilePathResolver(
                    obj_path, allow_anywhere=True)
            m = trimesh.load(obj_path, **kwargs)
        if m.vertices.shape[0] == 0:
            continue
        m = m.copy()
        m.apply_transform(vis.origin)
        parts.append(m)
    if not parts:
        raise ValueError(f"group {group.group_id} has no loadable geometry")
    merged = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    if not with_materials:
        merged.visual = trimesh.visual.ColorVisuals(mesh=merged)
    return merged


def norm_params(mesh) -> tuple[np.ndarray, float]:
    """Replicate TRELLIS Trellis2TexturingPipeline.preprocess_mesh normalization so it can be
    inverted: v' = (v - center) * scale (TRELLIS swaps Y/Z then undoes it on output, so the
    net mapping back to the link frame is center+scale only)."""
    vmin = mesh.vertices.min(axis=0)
    vmax = mesh.vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    scale = 0.99999 / (vmax - vmin).max()
    return center, float(scale)


def denormalize(textured, center: np.ndarray, scale: float):
    """Map a TRELLIS-output (normalized) mesh back into the original link frame in place."""
    textured.vertices = textured.vertices / scale + center
    return textured


# --- axis convention ----------------------------------------------------------
# URDF world frame is Z-up; glTF convention is Y-up.
# (x, y, z) -> (x, z, -y) matches rendering.export_yup.
_Z_TO_Y = np.array([
    [1,  0,  0, 0],
    [0,  0,  1, 0],
    [0, -1,  0, 0],
    [0,  0,  0, 1],
], dtype=np.float64)


# --- textured URDF + assembled scene ------------------------------------------
def write_textured_urdf(src_urdf: str | os.PathLike[str], dst_urdf: str | os.PathLike[str],
                        group_glbs: dict[str, list[tuple[str, str]]]) -> None:
    """Rewrite a URDF with textured per-group GLB visuals.

    `group_glbs` maps link name -> [(group_id, glb_relpath)] (relpaths relative to dst_urdf's
    directory). Each listed link's <visual> elements are replaced by one <visual> per group
    (identity origin: group geometry is already in the link frame). <collision> elements,
    joints, and links absent from the map (failed links) are left verbatim.
    """
    tree = ET.parse(src_urdf)
    root = tree.getroot()
    for link in root.findall("link"):
        name = link.get("name")
        entries = group_glbs.get(name)
        if not entries:
            continue
        for vis in link.findall("visual"):
            link.remove(vis)
        for gid, relpath in entries:
            vis = ET.SubElement(link, "visual")
            vis.set("name", gid)
            origin = ET.SubElement(vis, "origin")
            origin.set("xyz", "0 0 0")
            origin.set("rpy", "0 0 0")
            geom = ET.SubElement(vis, "geometry")
            mesh = ET.SubElement(geom, "mesh")
            mesh.set("filename", relpath)
    ET.indent(tree, space="\t")
    tree.write(dst_urdf, xml_declaration=True, encoding="utf-8")


def assemble_scene(job, backend: str):
    """Rest-pose FK assembly of a backend's textured group GLBs -> textured/<backend>/assembled.glb.

    Groups whose GLB is missing (failed) are skipped; the caller decides how to grade that.
    Returns the output path.
    """
    import trimesh

    asset_state = job.state["asset"]
    asset = parse_asset(asset_state["asset_dir"])
    fk = link_world_transforms(asset)
    scene = trimesh.Scene()
    for g in asset_state["groups"]:
        glb = job.textured_group_glb(backend, g["group_id"])
        if not glb.is_file() or glb.stat().st_size == 0:
            continue
        loaded = trimesh.load(str(glb))
        sub = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
        T = _Z_TO_Y @ fk.get(g["link"], np.eye(4))
        for node_name in list(sub.graph.nodes_geometry):
            transform, geom_name = sub.graph[node_name]
            scene.add_geometry(sub.geometry[geom_name], transform=T @ transform,
                               node_name=f"{g['group_id']}__{node_name}")
    out = job.assembled_glb(backend)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out))
    return out


def assemble_original_scene(job, backend: str):
    """Rest-pose FK assembly of the original textured meshes -> assembled_original.glb.

    Mirrors assemble_scene but loads per-group meshes from the source asset with
    materials preserved, giving a direct visual comparison against the textured output.
    """
    import trimesh

    asset_state = job.state["asset"]
    asset = parse_asset(asset_state["asset_dir"])
    groups = build_groups(asset, asset_state.get("group_by", "semantic"))
    fk = link_world_transforms(asset)
    scene = trimesh.Scene()
    for g in groups:
        try:
            m = merge_group_mesh(asset, g, with_materials=True)
        except Exception:
            continue
        T = _Z_TO_Y @ fk.get(g.link, np.eye(4))
        scene.add_geometry(m, transform=T, node_name=g.group_id)
    out = job.original_glb(backend)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out))
    return out

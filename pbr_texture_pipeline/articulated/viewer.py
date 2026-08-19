"""Joint-slider viewer core (port of trellis_pbr/viewer.py; see PRD.md, articulated extension).

Side-by-side Three.js viewer: original OBJ+MTL asset vs a backend's textured per-group GLBs,
with synced joint sliders. Refactored into an HTML template + path-resolution helpers so the
same core serves two frontends:
  - scripts/articulated_viewer.py: thin stdlib http.server wrapper (standalone).
  - pbr_texture_pipeline.app: FastAPI routes mounted on Gradio's app (/viewer/{job_id}/{backend}).

The page requests `asset/<relpath>` (original asset files) and `job/<relpath>` (job-dir files)
as RELATIVE urls, so it works under any mount prefix. CUDA-free, stdlib+ET only.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

MIME = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json",
    ".obj": "text/plain",
    ".mtl": "text/plain",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ktx2": "image/ktx2",
}


# --- URDF parsing (viewer-flavored: raw xyz/rpy lists for Three.js) -----------
def _parse_origin(el):
    if el is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    return xyz, rpy


def _mtl_from_obj(obj_path: Path) -> Optional[str]:
    try:
        with open(obj_path, errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("mtllib "):
                    return s[7:].strip()
                if s.startswith(("v ", "f ")):
                    break
    except OSError:
        pass
    return None


def parse_urdf(path: Path) -> dict:
    """Parse a URDF file into {links, joints} with raw xyz/rpy (Three.js consumes these)."""
    root = ET.parse(path).getroot()
    links: dict[str, dict] = {}
    for link_el in root.findall("link"):
        visuals = []
        for vis_el in link_el.findall("visual"):
            xyz, rpy = _parse_origin(vis_el.find("origin"))
            geom = vis_el.find("geometry")
            if geom is None:
                continue
            mesh_el = geom.find("mesh")
            if mesh_el is not None:
                visuals.append({"mesh": mesh_el.get("filename", ""), "xyz": xyz, "rpy": rpy})
                continue
            box_el = geom.find("box")
            if box_el is not None:
                visuals.append({"primitive": "box",
                                "size": [float(v) for v in box_el.get("size").split()],
                                "xyz": xyz, "rpy": rpy})
                continue
            cyl_el = geom.find("cylinder")
            if cyl_el is not None:
                visuals.append({"primitive": "cylinder",
                                "radius": float(cyl_el.get("radius")),
                                "length": float(cyl_el.get("length")),
                                "xyz": xyz, "rpy": rpy})
                continue
            sph_el = geom.find("sphere")
            if sph_el is not None:
                visuals.append({"primitive": "sphere",
                                "radius": float(sph_el.get("radius")),
                                "xyz": xyz, "rpy": rpy})
        links[link_el.get("name")] = {"visuals": visuals}

    joints: dict[str, dict] = {}
    for j_el in root.findall("joint"):
        parent_el = j_el.find("parent")
        child_el = j_el.find("child")
        xyz, rpy = _parse_origin(j_el.find("origin"))
        ax_el = j_el.find("axis")
        axis = ([float(v) for v in ax_el.get("xyz", "1 0 0").split()]
                if ax_el is not None else [1.0, 0.0, 0.0])
        lim_el = j_el.find("limit")
        limit = ({"lower": float(lim_el.get("lower", "-3.14159")),
                  "upper": float(lim_el.get("upper", "3.14159"))}
                 if lim_el is not None else None)
        joints[j_el.get("name")] = {
            "type": j_el.get("type", "fixed"),
            "parent": parent_el.get("link") if parent_el is not None else None,
            "child": child_el.get("link") if child_el is not None else None,
            "origin_xyz": xyz, "origin_rpy": rpy, "axis": axis, "limit": limit,
        }
    return {"links": links, "joints": joints}


def find_root_link(links: dict, joints: dict) -> str:
    children = {j["child"] for j in joints.values() if j["child"]}
    for name in links:
        if name not in children:
            return name
    return next(iter(links))


# --- scene data ---------------------------------------------------------------
def build_scene_data(job, backend: str) -> dict:
    """Assemble the JSON the viewer page consumes for one (job, backend)."""
    asset_state = job.state.get("asset") or {}
    asset_dir = Path(asset_state["asset_dir"])
    orig = parse_urdf(Path(asset_state["urdf"]))
    textured_urdf = job.textured_urdf(backend)
    textured = parse_urdf(textured_urdf) if textured_urdf.is_file() else {"links": {}}

    links: dict[str, dict] = {}
    for link_name, orig_link in orig["links"].items():
        rich_visuals = []
        for vis in orig_link["visuals"]:
            rich_visuals.append({**vis, "mtl": _mtl_from_obj(asset_dir / vis["mesh"])})
        result_link = textured["links"].get(link_name, {"visuals": []})
        # N textured GLBs per link (one per group), relpaths relative to textured/<backend>/.
        result_glbs = [f"textured/{backend}/{v['mesh']}"
                       for v in result_link["visuals"] if v["mesh"].endswith(".glb")]
        links[link_name] = {"original_visuals": rich_visuals, "result_glbs": result_glbs}

    bbox = asset_state.get("bbox_world") or {"min": [-0.5] * 3, "max": [0.5] * 3}
    center = [(bbox["min"][i] + bbox["max"][i]) / 2 for i in range(3)]
    size = float(max(bbox["max"][i] - bbox["min"][i] for i in range(3))) or 1.0

    category = asset_state.get("category")
    if not category and job.spec().is_file():
        category = job.read_json(job.spec()).get("category")

    return {
        "job_id": job.job_id,
        "backend": backend,
        "category": category or "object",
        "links": links,
        "joints": orig["joints"],
        "root_link": find_root_link(orig["links"], orig["joints"]),
        "bbox_center": center,
        "bbox_size": size,
    }


# --- path resolution (shared by both frontends) -------------------------------
def resolve_file(job, kind: str, relpath: str) -> Optional[Path]:
    """Map a viewer request to a file: kind "asset" -> the asset dir, "job" -> the job dir.
    Returns None for traversal attempts or missing files."""
    if kind == "asset":
        root = Path(job.state["asset"]["asset_dir"]).resolve()
    elif kind == "job":
        root = job.root
    else:
        return None
    p = (root / relpath).resolve()
    # Job-dir symlinks (textured/<backend>/textured_objs -> asset dir) resolve outside the
    # job root by design; allow them only into the asset dir.
    asset_root = Path(job.state["asset"]["asset_dir"]).resolve()
    if not (str(p).startswith(str(root) + "/") or str(p).startswith(str(asset_root) + "/")):
        return None
    return p if p.is_file() else None


def mime_for(path: Path) -> str:
    return MIME.get(path.suffix.lower(), "application/octet-stream")


# --- HTML ---------------------------------------------------------------------
HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>pbr-texture-pipeline joint viewer &#8212; __TITLE__</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#13131f;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;
     display:flex;flex-direction:column;height:100vh;overflow:hidden}
header{display:flex;align-items:center;gap:10px;padding:5px 14px;
       background:#191929;border-bottom:1px solid #0f2040;flex-shrink:0;min-height:36px}
h1{font-size:.85rem;font-weight:700;color:#e94560;white-space:nowrap}
.badge{background:#0f2040;color:#4fc3f7;padding:2px 8px;border-radius:4px;
       font-size:.7rem;white-space:nowrap}
.hint{font-size:.65rem;color:#333355;white-space:nowrap;margin-left:auto}
.viewers{display:flex;flex:1;min-height:0;gap:2px;background:#090912}
.pane{flex:1;display:flex;flex-direction:column;min-width:0}
.pane-label{padding:3px 10px;font-size:.68rem;font-weight:700;text-transform:uppercase;
            letter-spacing:.1em;background:#191929;flex-shrink:0}
.pane-label.orig{color:#81c784}
.pane-label.res {color:#4fc3f7}
.wrap{flex:1;position:relative;overflow:hidden}
canvas{display:block;width:100%;height:100%}
.overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
         background:rgba(13,13,25,.8);font-size:.8rem;color:#505070;
         pointer-events:none;transition:opacity .5s}
.controls{background:#191929;border-top:1px solid #0f2040;padding:7px 14px;
          flex-shrink:0;max-height:140px;overflow-y:auto}
.ctrl-hdr{font-size:.62rem;text-transform:uppercase;letter-spacing:.12em;color:#404060;margin-bottom:5px}
.jrow{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.jname{font-size:.7rem;min-width:100px;color:#c0c0e0;white-space:nowrap}
.jtype{font-size:.6rem;color:#404060;min-width:68px;white-space:nowrap}
input[type=range]{flex:1;accent-color:#e94560;cursor:pointer}
.jval{font-size:.7rem;min-width:70px;text-align:right;color:#e94560;
      font-family:monospace;white-space:nowrap}
.nojoint{font-size:.76rem;color:#303050}
</style>
</head>
<body>

<header>
  <h1>pbr-texture-pipeline joint viewer</h1>
  <span class="badge">__TITLE__</span>
  <span class="badge">__CATEGORY__</span>
  <span class="badge">__BACKEND__</span>
  <span class="hint">orbit: left-drag &nbsp; pan: right-drag &nbsp; zoom: scroll</span>
</header>

<div class="viewers">
  <div class="pane">
    <div class="pane-label orig">Original &#8212; untextured OBJ</div>
    <div class="wrap" id="ow">
      <canvas id="oc"></canvas>
      <div class="overlay" id="ol">Loading&#8230;</div>
    </div>
  </div>
  <div class="pane">
    <div class="pane-label res">Textured Result &#8212; __BACKEND__ per-group GLB</div>
    <div class="wrap" id="rw">
      <canvas id="rc"></canvas>
      <div class="overlay" id="rl">Loading&#8230;</div>
    </div>
  </div>
</div>

<div class="controls">
  <div class="ctrl-hdr">Joint Controls</div>
  <div id="jc"></div>
</div>

<script type="importmap">
{"imports":{
  "three":"https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.module.js",
  "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/"
}}
</script>

<script type="module">
import * as THREE from 'three';
import {OBJLoader}     from 'three/addons/loaders/OBJLoader.js';
import {MTLLoader}     from 'three/addons/loaders/MTLLoader.js';
import {GLTFLoader}    from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const SD = __SCENE_DATA__;
const ASSET = 'asset/';   // relative to this page's URL directory
const JOB = 'job/';

function mkRenderer(id) {
  const r = new THREE.WebGLRenderer({canvas: document.getElementById(id), antialias: true});
  r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  r.outputColorSpace = THREE.SRGBColorSpace;
  r.toneMapping = THREE.ACESFilmicToneMapping;
  r.toneMappingExposure = 1.0;
  return r;
}
const R = {o: mkRenderer('oc'), r: mkRenderer('rc')};
const S = {o: new THREE.Scene(), r: new THREE.Scene()};
S.o.background = new THREE.Color(0x1a1a2e);
S.r.background = new THREE.Color(0x1a1a2e);

const CAM = {};
for (const s of ['o','r']) {
  const cam = new THREE.PerspectiveCamera(45, 1, 5e-4, 500);
  const [cx,cy,cz] = SD.bbox_center, d = SD.bbox_size * 1.8;
  cam.position.set(cx + d*.55, cy + d*.35, cz + d);
  cam.lookAt(cx, cy, cz);
  CAM[s] = cam;
}

const CTRL = {};
for (const s of ['o','r']) {
  const c = new OrbitControls(CAM[s], R[s].domElement);
  c.target.set(...SD.bbox_center);
  c.enableDamping = true;
  c.dampingFactor = .07;
  c.update();
  CTRL[s] = c;
}

function addLights(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, .5));
  const h = new THREE.HemisphereLight(0xffffff, 0x334466, 1.0);
  h.position.set(0, 20, 0);
  scene.add(h);
  const d1 = new THREE.DirectionalLight(0xffffff, 2.2);
  d1.position.set(3, 6, 4);
  scene.add(d1);
  const d2 = new THREE.DirectionalLight(0x8899ff, .5);
  d2.position.set(-4, -2, -3);
  scene.add(d2);
}
addLights(S.o);
addLights(S.r);

const ART = {o:{}, r:{}};

function buildTree(side) {
  const scene = S[side];
  const lg = {};
  for (const name of Object.keys(SD.links)) {
    lg[name] = new THREE.Group();
    lg[name].name = name;
  }
  for (const [jname, j] of Object.entries(SD.joints)) {
    if (!j.parent || !j.child) continue;
    const pivot = new THREE.Group();
    pivot.position.set(...j.origin_xyz);
    pivot.setRotationFromEuler(
      new THREE.Euler(j.origin_rpy[0], j.origin_rpy[1], j.origin_rpy[2], 'XYZ')
    );
    const art = new THREE.Group();
    pivot.add(art);
    if (lg[j.child])  art.add(lg[j.child]);
    if (lg[j.parent]) lg[j.parent].add(pivot);
    ART[side][jname] = {group: art, joint: j};
  }
  if (lg[SD.root_link]) scene.add(lg[SD.root_link]);
  return lg;
}

const OL = buildTree('o');
const RL = buildTree('r');

let opend = 0, rpend = 0;

function checkDone() {
  if (opend <= 0) { const e = document.getElementById('ol'); if(e) e.style.opacity='0'; }
  if (rpend <= 0) { const e = document.getElementById('rl'); if(e) e.style.opacity='0'; }
}

for (const ld of Object.values(SD.links)) {
  opend += ld.original_visuals.length;
  rpend += ld.result_glbs.length;
}
if (opend <= 0) checkDone();
if (rpend <= 0) checkDone();

const mtlCache = {};

function loadOriginal() {
  for (const [lname, ld] of Object.entries(SD.links)) {
    const grp = OL[lname];
    if (!grp) continue;
    for (const vis of ld.original_visuals) {
      if (vis.primitive) {
        const mat = new THREE.MeshStandardMaterial({color: 0x888888, roughness: 0.7});
        let geom;
        if (vis.primitive === 'box') {
          geom = new THREE.BoxGeometry(vis.size[0], vis.size[1], vis.size[2]);
        } else if (vis.primitive === 'cylinder') {
          geom = new THREE.CylinderGeometry(vis.radius, vis.radius, vis.length, 32);
          geom.rotateX(Math.PI / 2);
        } else if (vis.primitive === 'sphere') {
          geom = new THREE.SphereGeometry(vis.radius, 32, 16);
        }
        if (geom) {
          const mesh = new THREE.Mesh(geom, mat);
          mesh.position.set(...vis.xyz);
          mesh.setRotationFromEuler(
            new THREE.Euler(vis.rpy[0], vis.rpy[1], vis.rpy[2], 'XYZ')
          );
          grp.add(mesh);
        }
        opend--; checkDone();
        continue;
      }
      const slash = vis.mesh.lastIndexOf('/');
      const dir   = slash >= 0 ? vis.mesh.substring(0, slash + 1) : '';
      const objUrl = ASSET + vis.mesh;

      const addObj = (mats) => {
        const ol = new OBJLoader();
        if (mats) ol.setMaterials(mats);
        ol.load(objUrl,
          (obj) => {
            obj.position.set(...vis.xyz);
            obj.setRotationFromEuler(
              new THREE.Euler(vis.rpy[0], vis.rpy[1], vis.rpy[2], 'XYZ')
            );
            grp.add(obj);
            opend--; checkDone();
          },
          undefined,
          () => { opend--; checkDone(); }
        );
      };

      if (vis.mtl) {
        const key = ASSET + dir + vis.mtl;
        if (!mtlCache[key]) {
          const ml = new MTLLoader();
          ml.setPath(ASSET + dir);
          ml.setResourcePath(ASSET + dir);
          mtlCache[key] = new Promise(res => {
            ml.load(vis.mtl,
              (m) => { m.preload(); res(m); },
              undefined,
              () => res(null)
            );
          });
        }
        mtlCache[key].then(addObj);
      } else {
        addObj(null);
      }
    }
  }
}

function loadResult() {
  const gl = new GLTFLoader();
  for (const [lname, ld] of Object.entries(SD.links)) {
    const grp = RL[lname];
    if (!grp) continue;
    for (const glb of ld.result_glbs) {
      gl.load(JOB + glb,
        (gltf) => { grp.add(gltf.scene); rpend--; checkDone(); },
        undefined,
        () => { rpend--; checkDone(); }
      );
    }
  }
}

loadOriginal();
loadResult();

function applyJoint(jname, val) {
  for (const s of ['o','r']) {
    const e = ART[s][jname];
    if (!e) continue;
    const ax = new THREE.Vector3(...e.joint.axis).normalize();
    if (e.joint.type === 'revolute' || e.joint.type === 'continuous') {
      e.group.setRotationFromAxisAngle(ax, val);
    } else if (e.joint.type === 'prismatic') {
      e.group.position.copy(ax.multiplyScalar(val));
    }
  }
}

const jcEl   = document.getElementById('jc');
const movable = Object.entries(SD.joints).filter(([,j]) => j.type !== 'fixed');

if (!movable.length) {
  jcEl.innerHTML = '<div class="nojoint">No movable joints in this asset.</div>';
} else {
  for (const [jname, j] of movable) {
    const prismatic = j.type === 'prismatic';
    let lo = -Math.PI, hi = Math.PI;
    if (j.limit) { lo = j.limit.lower; hi = j.limit.upper; }
    const step = Math.abs(hi - lo) / 400 || 0.001;
    const unit = prismatic ? 'm' : 'rad';

    const row = document.createElement('div');
    row.className = 'jrow';
    row.innerHTML =
      '<span class="jname">' + jname + '</span>' +
      '<span class="jtype">[' + j.type + ']</span>' +
      '<input type="range" id="s_' + jname + '" min="' + lo + '" max="' + hi +
      '" step="' + step + '" value="0">' +
      '<span class="jval" id="v_' + jname + '">0.000 ' + unit + '</span>';
    jcEl.appendChild(row);

    row.querySelector('input').addEventListener('input', function() {
      const v = parseFloat(this.value);
      document.getElementById('v_' + jname).textContent = v.toFixed(3) + ' ' + unit;
      applyJoint(jname, v);
    });
  }
}

function resize() {
  for (const [s, wid] of [['o','ow'],['r','rw']]) {
    const wrap = document.getElementById(wid);
    const w = wrap.clientWidth, h = wrap.clientHeight;
    if (w > 0 && h > 0) {
      R[s].setSize(w, h, false);
      CAM[s].aspect = w / h;
      CAM[s].updateProjectionMatrix();
    }
  }
}
new ResizeObserver(resize).observe(document.body);
setTimeout(resize, 0);

(function loop() {
  requestAnimationFrame(loop);
  CTRL.o.update(); CTRL.r.update();
  R.o.render(S.o, CAM.o);
  R.r.render(S.r, CAM.r);
})();
</script>
</body>
</html>
"""


def make_html(scene: dict) -> str:
    safe_json = json.dumps(scene).replace("</", "<\\/")
    html = HTML_TEMPLATE
    html = html.replace("__TITLE__", scene["job_id"])
    html = html.replace("__CATEGORY__", scene["category"])
    html = html.replace("__BACKEND__", scene["backend"])
    html = html.replace("__SCENE_DATA__", safe_json)
    return html

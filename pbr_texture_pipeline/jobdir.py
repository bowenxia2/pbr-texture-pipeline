"""Per-job directory and job.json state machine (PRD sections 2, 8).

Every stage reads and writes only its job directory. `job.json` records per-stage
status/params/timestamps/seeds, which makes stages resumable and lets interactive
(human approves) and batch (policy approves) modes share one code path.

Layout (PRD section 2):
    jobs/<job_id>/
      job.json
      input/   original.<ext>  mesh_norm.glb  asset/ (rebuilt URDF tree)
               face_ranges.json  mesh_norm_open.glb
      groups/<gid>/  mesh.glb  norm.json
      views/   view_{00..07}.png  contact_sheet.png
      control/ camera.json  camera_open.json  depth.png normal.png canny.png mask.png
               open/  visibility_rest.npz
      vlm/     transcript.json  spec.json  caption.json  plan.json
      ref/     candidate_{NN}.png(+.json)  chosen.png  chosen_rgba.png
               chosen_rgba_open.png
      textured/<backend>/{groups/<gid>.glb, global/pass_a.json, global/pass_b.json,
               mobility_textured.urdf, assembled.glb}
      previews/<backend>_turntable.mp4  <backend>_condview.png  <backend>_judgesheet.png
      eval/    metrics.json  states/  diagnostics.json
      judge/   verdict.json
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

# Pipeline stages (PRD section 2). Stage P (plan) sits between diffuse and texture:
# the per-group material plan must see the chosen global reference.
STAGES = ("render", "vlm", "diffuse", "plan", "texture", "eval", "judge")
STAGE_ALIASES = {"R": "render", "V": "vlm", "D": "diffuse", "P": "plan", "T": "texture",
                 "E": "eval", "J": "judge"}

# Per-stage status values (PRD section 2).
PENDING = "pending"
RUNNING = "running"
DONE = "done"
ERROR = "error"
NEEDS_REVIEW = "needs_review"
STATUSES = (PENDING, RUNNING, DONE, ERROR, NEEDS_REVIEW)

SCHEMA_VERSION = 3  # v3: kind removed; all jobs are articulated (URDF)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def make_job_id(urdf_path: str | os.PathLike[str], category: str = "") -> str:
    """job_id = <asset_id>_<cat_slug>_urdf_<timestamp>.

    Every PartNet-Mobility asset's file is `mobility.urdf`, so a stem-based id would collide;
    the asset directory name (the numeric PartNet id) disambiguates instead.
    """
    asset_id = re.sub(r"[^A-Za-z0-9._-]", "_", Path(urdf_path).resolve().parent.name) or "asset"
    cat = re.sub(r"[^A-Za-z0-9]+", "_", (category or "").lower()).strip("_")
    mid = f"{asset_id}_{cat}" if cat else asset_id
    return f"{mid}_urdf_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


class JobDir:
    """Handle to one job directory + its job.json state machine."""

    # --- construction --------------------------------------------------------
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).resolve()
        self._state: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        jobs_root: str | os.PathLike[str],
        mesh_path: str | os.PathLike[str],
        params: dict[str, Any] | None = None,
        job_id: Optional[str] = None,
    ) -> "JobDir":
        """Create a fresh job dir + skeleton subtree + initial job.json."""
        job_id = job_id or make_job_id(mesh_path)
        job = cls(Path(jobs_root) / job_id)
        job.root.mkdir(parents=True, exist_ok=False)
        for sub in ("input", "views", "control", "vlm", "ref", "textured", "previews", "eval",
                    "judge"):
            (job.root / sub).mkdir(exist_ok=True)
        now = _now()
        job._state = {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "mesh_source": str(Path(mesh_path).resolve()),
            "created": now,
            "updated": now,
            "params": params or {},
            "stages": {
                s: {"status": PENDING, "params": {}, "seed": None,
                    "started": None, "finished": None, "error": None}
                for s in STAGES
            },
        }
        job._write()
        return job

    @classmethod
    def load(cls, root: str | os.PathLike[str]) -> "JobDir":
        """Load an existing job dir (for --resume / jobs browser)."""
        job = cls(root)
        with open(job.path("job.json")) as f:
            job._state = json.load(f)
        # Backfill stages added after the job was created (e.g. "judge" 2026-07-17, "plan"
        # 2026-07-21) so status()/stage() never KeyError on old job dirs. In-memory only:
        # nothing is written to disk until a real transition touches the job.
        stages = job._state.setdefault("stages", {})
        for s in STAGES:
            stages.setdefault(s, {"status": PENDING, "params": {}, "seed": None,
                                  "started": None, "finished": None, "error": None})
        return job

    @classmethod
    def load_all(cls, jobs_root: str | os.PathLike[str]) -> list["JobDir"]:
        """Load every job under jobs_root that has a job.json (Tab 5 / batch index)."""
        out: list[JobDir] = []
        root = Path(jobs_root)
        if not root.is_dir():
            return out
        for d in sorted(root.iterdir()):
            if (d / "job.json").is_file():
                try:
                    out.append(cls.load(d))
                except (OSError, json.JSONDecodeError):
                    continue
        return out

    # --- state access --------------------------------------------------------
    @property
    def state(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("job state not loaded")
        return self._state

    @property
    def job_id(self) -> str:
        return self.state["job_id"]

    def _stage_key(self, stage: str) -> str:
        stage = STAGE_ALIASES.get(stage, stage)
        if stage not in STAGES:
            raise KeyError(f"unknown stage {stage!r}; expected one of {STAGES}")
        return stage

    def stage(self, stage: str) -> dict[str, Any]:
        return self.state["stages"][self._stage_key(stage)]

    def status(self, stage: str) -> str:
        return self.stage(stage)["status"]

    def is_done(self, stage: str) -> bool:
        """--resume uses this to skip completed stages (PRD section 8)."""
        return self.status(stage) == DONE

    # --- state transitions (shared by interactive + batch) -------------------
    def start(self, stage: str, params: dict[str, Any] | None = None,
              seed: int | None = None) -> None:
        st = self.stage(stage)
        st["status"] = RUNNING
        st["started"] = _now()
        st["error"] = None
        if params is not None:
            st["params"].update(params)
        if seed is not None:
            st["seed"] = seed
        self._touch_and_write()

    def finish(self, stage: str, status: str = DONE,
               params: dict[str, Any] | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"invalid status {status!r}")
        st = self.stage(stage)
        st["status"] = status
        st["finished"] = _now()
        if params is not None:
            st["params"].update(params)
        self._touch_and_write()

    def fail(self, stage: str, error: str) -> None:
        st = self.stage(stage)
        st["status"] = ERROR
        st["finished"] = _now()
        st["error"] = str(error)[:4000]
        self._touch_and_write()

    def reset(self, stage: str) -> None:
        """Return a stage to pending so it re-runs (targeted retexturing invalidates the
        already-recorded eval/judge results for the affected job)."""
        st = self.stage(stage)
        st["status"] = PENDING
        st["error"] = None
        self._touch_and_write()

    def set_param(self, key: str, value: Any) -> None:
        self.state["params"][key] = value
        self._touch_and_write()

    def set_asset(self, asset: dict[str, Any]) -> None:
        """Write the 'asset' section (Stage R fills it after parsing the URDF)."""
        self.state["asset"] = asset
        self._touch_and_write()

    # --- path helpers (every artifact in the PRD section 2 tree) -------------
    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    # input/
    def original(self, ext: str) -> Path:
        return self.path("input", f"original{ext if ext.startswith('.') else '.' + ext}")

    def mesh_norm(self) -> Path:
        return self.path("input", "mesh_norm.glb")

    def mesh_reposed(self) -> Path:
        """Normalized mesh rotated so the chosen front panel faces the canonical camera.
        Written only when a non-zero front is selected in Tab 1; consumed by Stage T so the
        backend textures the same orientation the control maps / reference were rendered from."""
        return self.path("input", "mesh_reposed.glb")

    # views/
    def view(self, i: int) -> Path:
        return self.path("views", f"view_{i:02d}.png")

    def contact_sheet(self) -> Path:
        return self.path("views", "contact_sheet.png")

    # control/
    def camera_json(self) -> Path:
        return self.path("control", "camera.json")

    def control(self, name: str) -> Path:
        """name in {depth, normal, canny, mask} -> control/<name>.png."""
        return self.path("control", f"{name}.png")

    # vlm/
    def transcript(self) -> Path:
        return self.path("vlm", "transcript.json")

    def spec(self) -> Path:
        return self.path("vlm", "spec.json")

    def caption(self) -> Path:
        return self.path("vlm", "caption.json")

    # ref/
    def candidate(self, i: int) -> Path:
        return self.path("ref", f"candidate_{i:02d}.png")

    def candidate_sidecar(self, i: int) -> Path:
        return self.path("ref", f"candidate_{i:02d}.json")

    def chosen(self) -> Path:
        return self.path("ref", "chosen.png")

    def chosen_rgba(self) -> Path:
        return self.path("ref", "chosen_rgba.png")

    # textured/<backend>/
    def textured_dir(self, backend: str) -> Path:
        d = self.path("textured", backend)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def output_glb(self, backend: str) -> Optional[Path]:
        """The textured GLB a backend wrote, or None if absent.

        The assembled rest-pose GLB is the primary output; fall back to any *.glb in the backend
        dir (non-recursive, so per-group GLBs under groups/ never pollute the fallback).
        """
        d = self.path("textured", backend)
        if not d.is_dir():
            return None
        if (d / "assembled.glb").is_file():
            return d / "assembled.glb"
        globbed = sorted(d.glob("*.glb"))
        return globbed[0] if globbed else None

    # --- articulated path helpers (PRD.md) ----------------------------------------
    def asset_dir(self) -> Path:
        """The rebuilt asset tree for app uploads (input/asset/); batch jobs reference the
        source asset dir directly via state["asset"]["asset_dir"]."""
        return self.path("input", "asset")

    def group_dir(self, group_id: str) -> Path:
        return self.path("groups", group_id)

    def group_mesh(self, group_id: str) -> Path:
        """Merged per-group mesh in the link frame (Stage R writes it)."""
        return self.path("groups", group_id, "mesh.glb")

    def group_norm(self, group_id: str) -> Path:
        """center/scale/faces/area sidecar for verification (adapters recompute)."""
        return self.path("groups", group_id, "norm.json")

    def group_coverage(self) -> Path:
        """Per-group visibility at the condition camera (feeds the crop-ref policy)."""
        return self.path("control", "groups", "coverage.json")

    def group_mask(self, group_id: str) -> Path:
        return self.path("control", "groups", f"{group_id}_mask.png")

    def plan(self) -> Path:
        """Stage P material plan (label-level materials + per-group resolution)."""
        return self.path("vlm", "plan.json")

    def textured_group_glb(self, backend: str, group_id: str) -> Path:
        return self.path("textured", backend, "groups", f"{group_id}.glb")

    def textured_urdf(self, backend: str) -> Path:
        return self.path("textured", backend, "mobility_textured.urdf")

    def assembled_glb(self, backend: str) -> Path:
        """Rest-pose FK assembly of all textured groups (what Stage E/J consume)."""
        return self.path("textured", backend, "assembled.glb")

    # previews/
    def preview_turntable(self, backend: str) -> Path:
        return self.path("previews", f"{backend}_turntable.mp4")

    def preview_condview(self, backend: str) -> Path:
        return self.path("previews", f"{backend}_condview.png")

    # eval/
    def metrics(self) -> Path:
        return self.path("eval", "metrics.json")

    # judge/
    def judge_sheet(self, backend: str) -> Path:
        """2x2 multi-view contact sheet of a textured output (front/right/back/left)."""
        return self.path("previews", f"{backend}_judgesheet.png")

    def judge_verdict(self) -> Path:
        return self.path("judge", "verdict.json")

    def winner(self) -> Optional[str]:
        """Winning backend per Stage J, or None if the judge has not decided yet.

        job.json is authoritative; verdict.json covers a crash between the verdict write and
        the job.json update (finalize writes the verdict first).
        """
        w = self.stage("judge").get("params", {}).get("winner")
        if w:
            return w
        if self.judge_verdict().is_file():
            try:
                return self.read_json(self.judge_verdict()).get("winner")
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def final_glb(self) -> Optional[Path]:
        """The recommended mesh: the winner's GLB (both backends' outputs remain on disk after
        Stage J; this just resolves which one the judge preferred). Before judging, resolves
        only if exactly one backend produced a GLB."""
        w = self.winner()
        if w:
            return self.output_glb(w)
        textured = self.path("textured")
        if not textured.is_dir():
            return None
        glbs = [g for g in (self.output_glb(d.name) for d in sorted(textured.iterdir())
                            if d.is_dir()) if g is not None]
        return glbs[0] if len(glbs) == 1 else None

    # --- convenience json io within the job dir ------------------------------
    def write_json(self, rel_path: str | os.PathLike[str], obj: Any) -> Path:
        p = self.path(str(rel_path)) if not os.path.isabs(str(rel_path)) else Path(rel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(p, obj)
        return p

    def read_json(self, rel_path: str | os.PathLike[str]) -> Any:
        p = self.path(str(rel_path)) if not os.path.isabs(str(rel_path)) else Path(rel_path)
        with open(p) as f:
            return json.load(f)

    # --- internals -----------------------------------------------------------
    def _touch_and_write(self) -> None:
        self.state["updated"] = _now()
        self._write()

    def _write(self) -> None:
        _atomic_write_json(self.path("job.json"), self.state)


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Temp-file + os.replace so a crashed stage never leaves a half-written file (PRD 1.1.5)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def resolve_stages(stages: Iterable[str] | None) -> list[str]:
    """Normalize a user stage list (aliases or names) to canonical order."""
    if stages is None:
        return list(STAGES)
    wanted = {STAGE_ALIASES.get(s, s) for s in stages}
    return [s for s in STAGES if s in wanted]

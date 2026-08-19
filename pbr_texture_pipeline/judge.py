"""Stage J: VLM picks the best textured output per job, both outputs are kept.

Three phases, split across envs like the rest of the pipeline:
  1. Sheet prep (this module, trellis2 env): render previews/<backend>_judgesheet.png, a 2x2
     multi-view contact sheet (front/right/back/left) of each textured GLB, reusing
     eval.load_output_colored + rendering.render_appearance.
  2. Verdict (vlm env worker): one deterministic VLM call over (sheet A, sheet B, reference);
     lives in pbr_texture_pipeline.vlm.run_judge, mirroring the existing eval/vlm split.
  3. Materialization (this module, `finalize`): write judge/verdict.json and record the winner
     in job.json. Both textured/<backend>/ dirs stay on disk; the verdict is a recommendation
     with reasoning, not an elimination.

Degenerate cases: one GLB -> walkover verdict without a VLM call; zero GLBs -> stage error
(handled by the orchestrators). Re-running after a verdict already exists re-finalizes
idempotently with the same winner, so the stage is idempotent.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Optional

from pbr_texture_pipeline.config import load_config
from pbr_texture_pipeline.jobdir import DONE, NEEDS_REVIEW, JobDir

_CFG = load_config()

VERDICT_SCHEMA_VERSION = 1


# --- phase 1: judge sheets (needs the TRELLIS.2 renderer; trellis2 env only) --
def render_judge_sheet(job: JobDir, backend: str, camera: Optional[dict]) -> Optional[Path]:
    """Render previews/<backend>_judgesheet.png (2x2: front, right / back, left). Idempotent.

    The GLB check comes FIRST: a backend that never produced a GLB (texture failure) must not
    count as available just because a stale sheet survives from an earlier run.
    """
    glb = job.output_glb(backend)
    if glb is None:
        return None
    out = job.judge_sheet(backend)
    if out.is_file():
        return out
    # Lazy CUDA-heavy imports (repo convention: importing pbr_texture_pipeline.* never touches torch).
    import numpy as np
    from PIL import Image

    from pbr_texture_pipeline import eval as E
    from pbr_texture_pipeline import rendering as R

    mesh = E.load_output_colored(str(glb), camera, up="z")
    yaws_deg = [float(d) for d in _CFG.get("judge.sheet_yaws_deg", [0, 90, 180, 270])]
    res = int(_CFG.get("judge.sheet_tile_res", 512))
    pitch = math.radians(float(_CFG.get("judge.sheet_pitch_deg", 15)))
    tiles = [R.render_appearance(mesh, R.CANONICAL_YAW + math.radians(d), pitch, res, ssaa=2)["rgb"]
             for d in yaws_deg]
    grid = np.concatenate([np.concatenate(tiles[:2], axis=1),
                           np.concatenate(tiles[2:], axis=1)], axis=0)
    # Extra row: the object with every movable joint fully open (front + three-quarter),
    # so the VLM sees interiors and moving-part boundaries when picking a winner.
    try:
        open_mesh = E.articulated_state_mesh(job, backend, 1.0)
    except Exception as e:  # noqa: BLE001
        print(f"[judge] open-state row unavailable for {backend}: {e}")
        open_mesh = None
    if open_mesh is not None:
        row = [R.render_appearance(open_mesh, R.CANONICAL_YAW + math.radians(d), pitch,
                                   res, ssaa=2)["rgb"] for d in (0.0, 45.0)]
        grid = np.concatenate([grid, np.concatenate(row, axis=1)], axis=0)
    Image.fromarray(grid, mode="RGB").save(out)
    return out


def prepare(job: JobDir, backends: list[str]) -> dict:
    """Render judge sheets for every backend that produced a GLB."""
    camera = job.read_json(job.camera_json()) if job.camera_json().is_file() else None
    available, sheets = [], {}
    for b in backends:
        sheet = render_judge_sheet(job, b, camera)
        if sheet is not None:
            available.append(b)
            sheets[b] = str(sheet)
    return {"available": available, "sheets": sheets}


# --- label assignment ---------------------------------------------------------
def label_assignment(job_id: str, available: list[str]) -> dict[str, str]:
    """Deterministic anti-position-bias A/B assignment: stable per job, ~50/50 per jobs root."""
    pair = sorted(available)
    if int(hashlib.md5(job_id.encode()).hexdigest(), 16) % 2:
        pair = pair[::-1]
    return {"A": pair[0], "B": pair[1]}


# --- verdict builders ---------------------------------------------------------
def _verdict_base(winner: str, method: str, reason: str) -> dict:
    return {
        "schema_version": VERDICT_SCHEMA_VERSION,
        "winner": winner,
        "method": method,
        "label_map": None,
        "confidence": "low",
        "criteria": {},
        "reasoning": "",
        "model": None,
        "raw_response": "",
        "reason": reason,
    }


def walkover_verdict(winner: str, reason: str) -> dict:
    """Exactly one textured output exists: no VLM call needed."""
    v = _verdict_base(winner, "walkover", reason)
    v["confidence"] = "high"
    return v


def fallback_verdict(available: list[str], raw_response: str = "",
                     model: Optional[str] = None) -> dict:
    """config judge.default_winner after the VLM failed twice / refused -> needs_review."""
    default = str(_CFG.get("judge.default_winner", "trellis2"))
    winner = default if default in available else available[0]
    v = _verdict_base(winner, "fallback_default",
                      "VLM returned no valid A/B verdict after retry")
    v["raw_response"] = raw_response
    v["model"] = model
    return v


def vlm_verdict(parsed: dict, label_map: dict[str, str], raw_response: str,
                model: Optional[str]) -> dict:
    """Map an extracted A/B verdict (see vlm.extract_verdict) to backend names."""
    v = _verdict_base(label_map[parsed["winner"]], "vlm", "")
    v.update({
        "label_map": label_map,
        "confidence": parsed.get("confidence", "medium"),
        "criteria": parsed.get("criteria", {}),
        "reasoning": parsed.get("reasoning", ""),
        "model": model,
        "raw_response": raw_response,
    })
    return v


# --- phase 3: materialization -------------------------------------------------
def finalize(job: JobDir, verdict: dict) -> dict:
    """Persist the verdict and record the winner. Both textured outputs are kept on disk;
    this only writes judge/verdict.json and updates job.json (crash-recoverable, idempotent).
    """
    winner = verdict["winner"]
    job.write_json(job.judge_verdict(), verdict)
    status = NEEDS_REVIEW if verdict.get("method") == "fallback_default" else DONE
    job.finish("judge", status, params={"winner": winner,
                                        "method": verdict.get("method"),
                                        "confidence": verdict.get("confidence")})
    return {"winner": winner, "status": status, "final_glb": str(job.final_glb() or "")}

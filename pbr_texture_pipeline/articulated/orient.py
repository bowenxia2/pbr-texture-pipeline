"""Orient-Anything-V2 front detection for articulated Stage R (PRD.md section 7.5).

Runs scripts/orient_infer.py as a short-lived subprocess in the orianyv2 conda env over the
8 pre-repose contact-sheet panels, then picks the front panel from the per-panel azimuth
predictions. Because the panels are 45 degrees apart, every panel independently implies
where the front is; the spread of those implied positions is the confidence check. On low
agreement (or any failure) the decision falls back to config articulated.front_panel and
records why.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from pbr_texture_pipeline.config import load_config

_CFG = load_config()
RESULT_MARKER = "[PBR_RESULT]"
PANEL_STEP_DEG = 45.0
N_PANELS = 8


def ckpt_path() -> Optional[Path]:
    """Cached checkpoint file (articulated.orient.ckpt_file inside models.orient), or None."""
    repo_id = _CFG.model("orient")
    filename = str(_CFG.get("articulated.orient.ckpt_file"))
    snap_root = (Path(_CFG.hf_cache) / "hub"
                 / ("models--" + repo_id.replace("/", "--")) / "snapshots")
    if snap_root.is_dir():
        for snap in sorted(snap_root.iterdir()):
            p = snap / filename
            if p.is_file():
                return p
    return None


def infer_panels(image_paths: list) -> Optional[list[dict]]:
    """Run orient_infer.py over the panel images; per-image dicts in order, or None."""
    from pbr_texture_pipeline.backends.registry import _conda_bin

    ckpt = ckpt_path()
    if ckpt is None:
        print("[orient] checkpoint not cached; run scripts/download_orient_anything.py")
        return None
    script = Path(__file__).resolve().parents[2] / "scripts" / "orient_infer.py"
    env_name = str(_CFG.get("articulated.orient.env"))
    cmd = [_conda_bin(), "run", "-n", env_name, "--no-capture-output", "python", str(script),
           "--repo", str(_CFG.repo("orient")), "--ckpt", str(ckpt),
           "--images", *[str(p) for p in image_paths]]
    env = os.environ.copy()
    env.update(_CFG.hf_env())
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    except subprocess.TimeoutExpired:
        print("[orient] inference subprocess timed out")
        return None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            try:
                payload = json.loads(line[len(RESULT_MARKER):].strip())
            except json.JSONDecodeError:
                break
            if payload.get("ok"):
                return payload["results"]
    tail = (proc.stderr or proc.stdout or "")[-500:]
    print(f"[orient] inference failed (rc {proc.returncode}): {tail}")
    return None


def choose_front(az_by_panel: list, min_agreement_deg: float) -> dict:
    """Front-panel choice from per-panel azimuth predictions (degrees; None = panel failed).

    Panel k views the object from canonical-front + k*45deg, so a predicted azimuth az_k
    implies the front sits at k*45 +/- az_k degrees around the ring (the sign depends on the
    model's azimuth handedness, which is resolved here by trying both and keeping the one
    whose implied positions cluster tighter). Agreement is the mean absolute circular
    deviation of the implied positions from their circular mean; the choice is confident
    when it is within min_agreement_deg.
    """
    best = None
    for sign in (1, -1):
        phis = [(k * PANEL_STEP_DEG + sign * az) % 360.0
                for k, az in enumerate(az_by_panel) if az is not None]
        if len(phis) < 2:
            continue
        ang = np.radians(phis)
        mean = math.degrees(math.atan2(np.sin(ang).sum(), np.cos(ang).sum())) % 360.0
        dev = [abs((p - mean + 180.0) % 360.0 - 180.0) for p in phis]
        err = float(np.mean(dev))
        if best is None or err < best["agreement_err_deg"]:
            best = {"sign": sign, "front_deg": float(mean), "agreement_err_deg": err,
                    "implied_front_deg": [float(p) for p in phis]}
    if best is None:
        return {"ok": False, "front_panel": None, "reason": "fewer than 2 panel predictions"}
    best["front_panel"] = int(round(best["front_deg"] / PANEL_STEP_DEG)) % N_PANELS
    best["ok"] = best["agreement_err_deg"] <= float(min_agreement_deg)
    if not best["ok"]:
        best["reason"] = (f"agreement {best['agreement_err_deg']:.1f} deg > "
                          f"{float(min_agreement_deg):.1f} deg")
    return best


def detect_front_panel(job) -> dict:
    """Full Stage R flow: views/view_<k>.png (pre-repose) -> decision dict for camera.json.

    decision["front_panel"] is always usable: the model's confident choice, else the config
    fallback. The raw per-panel predictions and the reason for any fallback are recorded so
    gate A11 and later debugging can see what the model said.
    """
    fallback = int(_CFG.get("articulated.front_panel", 2))
    min_agree = float(_CFG.get("articulated.orient.min_agreement_deg", 30))
    decision: dict = {"fallback_panel": fallback}

    images = [job.view(k) for k in range(N_PANELS)]
    missing = [str(p) for p in images if not p.is_file()]
    results = infer_panels(images) if not missing else None
    if results is None:
        decision.update({"ok": False, "method": "fallback", "front_panel": fallback,
                         "reason": "panel files missing" if missing else "inference failed"})
        return decision

    decision["panels"] = results
    az = [r.get("az") if r.get("ok") else None for r in results]
    choice = choose_front(az, min_agree)
    decision.update(choice)
    if choice["ok"]:
        decision["method"] = "orient_anything_v2"
    else:
        decision["method"] = "fallback"
        decision["predicted_panel"] = choice.get("front_panel")
        decision["front_panel"] = fallback
        print(f"[orient] falling back to panel {fallback}: {choice.get('reason')}")
    return decision

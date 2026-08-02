"""One-time Orient-Anything-V2 checkpoint download (PRD_articulated_v2 section 6).

Fetches the single checkpoint file (config articulated.orient.ckpt_file, ~5 GB) from the
models.orient Hugging Face repo into the shared HF cache at config.env.hf_cache (never
/home). Structure mirrors scripts/download_controlnet.py.

Usage:
  python -m scripts.download_orient_anything            # download if absent
  python -m scripts.download_orient_anything --check    # report presence only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as a script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pbr_texture_pipeline.config import load_config  # noqa: E402


def _pin_cache(cfg) -> None:
    """Force all HF downloads into the shared gscratch cache before importing hub."""
    for k, v in cfg.hf_env().items():
        os.environ.setdefault(k, v)


def _cached_path(repo_id: str, filename: str, cache_dir: str) -> Path | None:
    """Path of the cached file if a snapshot containing it exists, else None."""
    snap_root = Path(cache_dir) / "hub" / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not snap_root.is_dir():
        return None
    for snap in snap_root.iterdir():
        p = snap / filename
        if p.is_file():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report presence only; download nothing")
    args = ap.parse_args()

    cfg = load_config()
    _pin_cache(cfg)
    repo_id = cfg.model("orient")
    filename = str(cfg.get("articulated.orient.ckpt_file"))

    print(f"HF cache: {cfg.hf_cache}/hub")
    cached = _cached_path(repo_id, filename, cfg.hf_cache)
    if cached is not None:
        print(f"[cached] {repo_id}/{filename}\n         {cached}")
        return 0
    if args.check:
        print(f"[MISSING] {repo_id}/{filename}")
        return 1

    from huggingface_hub import hf_hub_download

    print(f"[download] {repo_id}/{filename} -> {cfg.hf_cache}/hub")
    try:
        path = hf_hub_download(repo_id, filename)
    except Exception as e:  # noqa: BLE001
        print(f"[FAILED] {repo_id}/{filename}: {e}")
        return 1
    print(f"[done] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

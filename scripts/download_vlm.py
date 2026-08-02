"""One-time Stage V weight download for pbr_texture_pipeline.

Fetches the configured VLM checkpoint (config models.vlm; Qwen3.6-35B-A3B AWQ 4-bit, ~24 GB)
into the shared HF cache at config.env.hf_cache (never /home).

Usage:
  python -m scripts.download_vlm              # download if missing
  python -m scripts.download_vlm --check      # report presence only, download nothing
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


def _is_cached(repo_id: str, cache_dir: str) -> bool:
    """True if a models--<org>--<name> dir with a snapshot exists in the hub cache."""
    hub = Path(cache_dir) / "hub"
    folder = "models--" + repo_id.replace("/", "--")
    snap = hub / folder / "snapshots"
    return snap.is_dir() and any(snap.iterdir())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report presence only; download nothing")
    args = ap.parse_args()

    cfg = load_config()
    _pin_cache(cfg)
    repo = cfg.model("vlm")
    print(f"HF cache: {cfg.hf_cache}/hub")

    if _is_cached(repo, cfg.hf_cache):
        print(f"[cached] {repo}")
        return 0
    if args.check:
        print(f"[MISSING] {repo}")
        return 1

    from huggingface_hub import snapshot_download
    print(f"[download] {repo} -> {cfg.hf_cache}/hub")
    try:
        snapshot_download(repo)
        print(f"[done] {repo}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[FAILED] {repo}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

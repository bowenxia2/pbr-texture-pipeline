"""One-time Stage D weight download for pbr-texture-pipeline (PRD section 5, risk 5).

Targets the shared HF cache at config.env.hf_cache (never /home). Fetches the InstantX
Qwen-Image ControlNet-Union adapter and, if missing, the Qwen-Image base model. Every other
pbr-texture-pipeline weight is already cached.

Usage:
  python -m scripts.download_controlnet              # ControlNet (+ Qwen-Image base if absent)
  python -m scripts.download_controlnet --check      # report presence only, download nothing
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


def download(repo_id: str, cache_dir: str, check_only: bool) -> bool:
    from huggingface_hub import snapshot_download

    if _is_cached(repo_id, cache_dir):
        print(f"[cached] {repo_id}")
        return True
    if check_only:
        print(f"[MISSING] {repo_id}")
        return False
    print(f"[download] {repo_id} -> {cache_dir}/hub")
    try:
        # ControlNet repos ship both .safetensors and .bin; grab config + safetensors only.
        snapshot_download(
            repo_id,
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
        )
        print(f"[done] {repo_id}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FAILED] {repo_id}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report presence only; download nothing")
    args = ap.parse_args()

    cfg = load_config()
    _pin_cache(cfg)
    cache_dir = cfg.hf_cache

    # ControlNet adapter, plus the Qwen-Image base (large) if it is not already cached.
    targets = [cfg.model("controlnet")]
    if not _is_cached(cfg.model("qwen_image"), cache_dir):
        targets.append(cfg.model("qwen_image"))

    print(f"HF cache: {cache_dir}/hub")
    ok = True
    for repo in targets:
        ok &= download(repo, cache_dir, args.check)

    # Also report the other Stage-D weights so we log what is missing (subtask 0.2.3).
    print("\n-- other Stage D weights --")
    for repo in (cfg.model("qwen_image"), cfg.model("rmbg")):
        _is_cached(repo, cache_dir) and print(f"[cached] {repo}") or (
            not _is_cached(repo, cache_dir) and print(f"[MISSING] {repo}")
        )

    if args.check:
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

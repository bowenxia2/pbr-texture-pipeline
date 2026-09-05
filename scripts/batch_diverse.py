#!/usr/bin/env python3
"""Run the batch pipeline on Articraft-10K assets in batches of N,
round-robining across object types for diversity.

Usage:
    conda run -n trellis2 python scripts/batch_diverse.py [--batch-size 5] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ASSETS_DIR = Path("articraft_extracted")
JOBS_ROOT = "jobs_v2"
DEFAULT_BATCH_SIZE = 5


def get_robot_name(asset_dir: Path) -> str:
    urdf = asset_dir / "model.urdf"
    if urdf.is_file():
        try:
            root = ET.parse(str(urdf)).getroot()
            name = root.get("name", "")
            if name:
                return name.strip().lower()
        except Exception:
            pass
    return asset_dir.name


def interleave(by_type: dict[str, list[Path]]) -> list[Path]:
    """Round-robin across types, longest-first so every batch gets diversity."""
    type_keys = sorted(by_type, key=lambda t: -len(by_type[t]))
    iters = {t: iter(items) for t, items in by_type.items()}
    result = []
    while iters:
        exhausted = []
        for t in type_keys:
            if t not in iters:
                continue
            try:
                result.append(next(iters[t]))
            except StopIteration:
                exhausted.append(t)
        for t in exhausted:
            del iters[t]
            type_keys.remove(t)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the interleaved order and exit")
    ap.add_argument("--start-batch", type=int, default=1,
                    help="1-indexed batch number to start from (for resuming)")
    args = ap.parse_args()

    assets = sorted(
        d for d in ASSETS_DIR.iterdir()
        if d.is_dir() and (d / "model.urdf").is_file()
    )
    print(f"Found {len(assets)} assets")

    by_type: dict[str, list[Path]] = defaultdict(list)
    for a in assets:
        by_type[get_robot_name(a)].append(a)

    print(f"Found {len(by_type)} distinct types:")
    for t, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {len(items):3d}  {t}")

    ordered = interleave(by_type)
    total_batches = (len(ordered) + args.batch_size - 1) // args.batch_size

    if args.dry_run:
        for i, a in enumerate(ordered):
            batch_num = i // args.batch_size + 1
            print(f"  batch {batch_num:2d} | {get_robot_name(a):40s} | {a.name}")
        print(f"\n{total_batches} batches of {args.batch_size}")
        return 0

    for batch_idx in range(total_batches):
        batch_num = batch_idx + 1
        if batch_num < args.start_batch:
            continue
        batch = ordered[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        print(f"\n{'=' * 70}")
        print(f"BATCH {batch_num}/{total_batches} ({len(batch)} assets)")
        print(f"{'=' * 70}")
        for a in batch:
            print(f"  [{get_robot_name(a)}] {a.name}")

        with tempfile.TemporaryDirectory(
            prefix=f"batch{batch_num}_",
            dir="/tmp",
        ) as tmpdir:
            for a in batch:
                os.symlink(a.resolve(), Path(tmpdir) / a.name)

            cmd = [
                "conda", "run", "-n", "trellis2",
                "python", "-m", "pbr_texture_pipeline.batch",
                "--assets", f"{tmpdir}/*/model.urdf",
                "--jobs-root", JOBS_ROOT,
                "--stages", "render,vlm,imageedit,texture",
                "--resume",
            ]
            print(f"Running batch {batch_num}...")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"BATCH {batch_num} exited with rc={result.returncode}")

    print(f"\nAll {total_batches} batches completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

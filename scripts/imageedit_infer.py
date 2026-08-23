"""Qwen-Image-Edit enhancement (Stage E).

Runs inside the trellis2 conda env (or qwen_edit) via subprocess.
Uses QwenImageEditPlusPipeline with native ControlNet depth conditioning.

CLI:
    python scripts/imageedit_infer.py --model <hf_id> --items-file <path>

Items file: [{"source": "...", "depth": "...", "materials": "...", "output": "..."}, ...]

Emits [PBR_RESULT] JSON lines per the adapter contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULT_MARKER = "[PBR_RESULT]"

PROMPT_TEMPLATE = (
    "Preserve the object's geometry, proportions, colors, patterns, and design. "
    "Materials: {MATERIAL_DESCRIPTION}. "
    "Enhance these materials with realistic surface properties and fine texture detail "
    "while maintaining the original appearance. "
    "Do not add, remove, reshape, or redesign any components. "
    "Keep the background clean and empty."
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Qwen-Image-Edit enhancement (Stage E)")
    ap.add_argument("--model", required=True, help="HuggingFace model id")
    ap.add_argument("--items-file", required=True,
                    help="JSON list of {source, depth, materials, output}")
    ap.add_argument("--num-inference-steps", type=int, default=40)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--true-cfg-scale", type=float, default=4.0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with open(args.items_file) as f:
        items = json.load(f)
    if not items:
        print("[imageedit] no items to process")
        return 0

    import torch
    from PIL import Image
    from diffusers import QwenImageEditPlusPipeline

    print(f"[imageedit] loading model {args.model} ...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="balanced")
    pipeline.set_progress_bar_config(disable=None)

    for i, item in enumerate(items):
        source_img = Image.open(item["source"]).convert("RGB")
        depth_img = Image.open(item["depth"]).convert("RGB")
        materials = item["materials"]

        prompt = PROMPT_TEMPLATE.replace("{MATERIAL_DESCRIPTION}", materials)

        print(f"[imageedit] [{i+1}/{len(items)}] {Path(item['source']).name} ...")
        with torch.inference_mode():
            output = pipeline(
                image=[source_img, depth_img],
                prompt=prompt,
                negative_prompt=" ",
                guidance_scale=args.guidance_scale,
                true_cfg_scale=args.true_cfg_scale,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=1,
                generator=torch.manual_seed(42),
            )
        out_img = output.images[0]
        out_path = Path(item["output"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path)

        result = {"ok": True, "source": item["source"], "output": str(out_path)}
        print(f"{RESULT_MARKER} {json.dumps(result)}", flush=True)

    print(f"[imageedit] done: {len(items)} items processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

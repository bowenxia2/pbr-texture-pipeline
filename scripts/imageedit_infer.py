"""Qwen-Image-Edit enhancement (Stage E).

Runs inside the trellis2 conda env (or qwen_edit) via subprocess.
Uses QwenImageEditPlusPipeline with Canny edge ControlNet conditioning.

CLI:
    python scripts/imageedit_infer.py --model <hf_id> --items-file <path>

Items file: [{"source": "...", "canny": "...", "materials": "...", "output": "...",
              "classification": "edit"|"generate", "category": "..."}, ...]

classification (optional, default "edit") selects the prompt template:
  - "edit": preserve existing textures and enhance materials.
  - "generate": replace blank/flat surfaces with realistic textures.
category (optional, default "object") provides context for the generate prompt.

Emits [PBR_RESULT] JSON lines per the adapter contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULT_MARKER = "[PBR_RESULT]"

NEGATIVE_PROMPT = (
    "text, writing, letters, numbers, logos, labels, symbols, brand names, "
    "buttons, dials, gauges, displays, markings, icons, stickers, decals, "
    "watermarks, scratches, noise, spots, blurry, low quality"
)

PROMPT_TEMPLATE_EDIT = (
    "Preserve the object's geometry, proportions, colors, patterns, and design. "
    "The object's materials are: {MATERIAL_DESCRIPTION}. "
    "Apply each material to its described location with realistic surface properties "
    "while maintaining the original appearance. "
    "CRITICAL: Every surface must be pure material texture only. "
    "Completely remove all text, letters, numbers, logos, icons, labels, symbols, "
    "brand names, warning signs, dials, gauges, readouts, stickers, decals, "
    "screen content, button markings, and any other fine-grained surface details. "
    "Seamlessly replace each removed detail with the natural texture of the "
    "surrounding material so the surface looks continuous and uniform, "
    "as if the detail was never there. "
    "Surfaces should show only material properties like grain, roughness, "
    "reflections, and color variation - never any printed or engraved markings. "
    "Do not add, remove, reshape, or redesign any components. "
    "Keep the background clean and empty. "
    "Soft ambient lighting with very subtle shadows. "
    "No directional light source or specular highlights."
)

PROMPT_TEMPLATE_GENERATE = (
    "This is a {CATEGORY} with blank or flat-colored surfaces. "
    "Replace all flat, uniform surfaces with realistic material textures. "
    "The object's materials should be: {MATERIAL_DESCRIPTION}. "
    "Apply each material to its described location with realistic surface properties "
    "including grain, roughness, reflections, and natural color variation. "
    "CRITICAL: Every surface must be pure material texture only. "
    "Do not generate any text, letters, numbers, logos, icons, labels, symbols, "
    "brand names, warning signs, dials, gauges, readouts, stickers, decals, "
    "screen content, button markings, watermarks, or any other fine-grained surface details. "
    "All surfaces should show only natural material properties like grain, roughness, "
    "reflections, and color variation - never any printed or engraved markings. "
    "Do not add, remove, reshape, or redesign any components. "
    "Keep the background clean and empty. "
    "Soft ambient lighting with very subtle shadows. "
    "No directional light source or specular highlights."
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Qwen-Image-Edit enhancement (Stage E)")
    ap.add_argument("--model", required=True, help="HuggingFace model id")
    ap.add_argument("--items-file", required=True,
                    help="JSON list of {source, canny, materials, output}")
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
        canny_img = Image.open(item["canny"]).convert("RGB")
        materials = item["materials"]
        classification = item.get("classification", "edit")
        category = item.get("category", "object")

        if classification == "generate":
            prompt = PROMPT_TEMPLATE_GENERATE.replace("{CATEGORY}", category)
            prompt = prompt.replace("{MATERIAL_DESCRIPTION}", materials)
        else:
            prompt = PROMPT_TEMPLATE_EDIT.replace("{MATERIAL_DESCRIPTION}", materials)

        print(f"[imageedit] [{i+1}/{len(items)}] {Path(item['source']).name} ...")
        with torch.inference_mode():
            output = pipeline(
                image=[source_img, canny_img],
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
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

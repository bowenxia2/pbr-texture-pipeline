"""Qwen-Image + ControlNet-Union text-to-image generation (Stage E generate path).

Runs inside the trellis2 conda env via subprocess.
Uses QwenImageControlNetPipeline with canny edge conditioning for
geometry-controlled image generation from scratch.

CLI:
    python scripts/imagegen_infer.py --model <base_hf_id> --controlnet <cn_hf_id> --items-file <path>

Items file: [{"canny": "...", "materials": "...", "output": "...", "category": "..."}, ...]

Emits [PBR_RESULT] JSON lines per the adapter contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULT_MARKER = "[PBR_RESULT]"

PROMPT_TEMPLATE = (
    "A photorealistic {CATEGORY}. "
    "Materials: {MATERIAL_DESCRIPTION}. "
    "Realistic surface properties and fine texture detail. "
    "Centered on a clean empty white background, studio lighting. "
    "No text, no labels, no annotations, no watermarks."
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Qwen-Image + ControlNet generation (Stage E)")
    ap.add_argument("--model", required=True, help="HuggingFace base model id (Qwen-Image)")
    ap.add_argument("--controlnet", required=True, help="HuggingFace ControlNet model id")
    ap.add_argument("--items-file", required=True,
                    help="JSON list of {canny, materials, output, category}")
    ap.add_argument("--num-inference-steps", type=int, default=30)
    ap.add_argument("--true-cfg-scale", type=float, default=4.0)
    ap.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with open(args.items_file) as f:
        items = json.load(f)
    if not items:
        print("[imagegen] no items to process")
        return 0

    import torch
    from PIL import Image
    from diffusers import QwenImageControlNetPipeline, QwenImageControlNetModel
    from diffusers import QwenImageTransformer2DModel

    print(f"[imagegen] loading base model {args.model} + controlnet {args.controlnet} ...")
    controlnet = QwenImageControlNetModel.from_pretrained(
        args.controlnet, torch_dtype=torch.bfloat16)
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipeline = QwenImageControlNetPipeline.from_pretrained(
        args.model, controlnet=controlnet, transformer=transformer,
        torch_dtype=torch.bfloat16)
    pipeline.enable_model_cpu_offload()

    for i, item in enumerate(items):
        canny_img = Image.open(item["canny"]).convert("RGB")
        materials = item["materials"]
        category = item.get("category", "object")

        prompt = PROMPT_TEMPLATE.replace("{CATEGORY}", category)
        prompt = prompt.replace("{MATERIAL_DESCRIPTION}", materials)

        print(f"[imagegen] [{i+1}/{len(items)}] {Path(item['canny']).name} ...")
        with torch.inference_mode():
            output = pipeline(
                prompt=prompt,
                negative_prompt=" ",
                control_image=canny_img,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                width=canny_img.size[0],
                height=canny_img.size[1],
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.true_cfg_scale,
                generator=torch.Generator(device="cpu").manual_seed(42),
            )
        out_img = output.images[0]
        out_path = Path(item["output"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path)

        result = {"ok": True, "canny": item["canny"], "output": str(out_path)}
        print(f"{RESULT_MARKER} {json.dumps(result)}", flush=True)

    print(f"[imagegen] done: {len(items)} items processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

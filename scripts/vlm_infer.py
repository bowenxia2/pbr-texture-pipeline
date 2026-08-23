"""VLM material analysis (Stage V).

Runs inside the vlm conda env via subprocess. Uses transformers
AutoModelForImageTextToText with the Qwen2.5-VL processor for
image-conditioned material description.

CLI:
    python scripts/vlm_infer.py --model <hf_id> --items-file <path>

Items file: [{"image": "...", "output": "..."}, ...]

Emits [PBR_RESULT] JSON lines per the adapter contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULT_MARKER = "[PBR_RESULT]"

SYSTEM_PROMPT = (
    "Analyze the object's visible materials and surface appearance. "
    "Identify the main material of each visually distinct component, "
    "including material type, color, and important surface finish when visible. "
    "Preserve the appearance implied by the image rather than redesigning it. "
    "Ignore the background, lighting, camera angle, and object geometry. "
    "Do not invent highly specific materials when the image does not provide enough evidence. "
    'Return only a concise semicolon-separated material description, for example: '
    '"dark brown wood grain; brushed steel hardware; matte black plastic frame".'
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="VLM material analysis (Stage V)")
    ap.add_argument("--model", required=True, help="HuggingFace model id")
    ap.add_argument("--items-file", required=True, help="JSON list of {image, output}")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.1)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with open(args.items_file) as f:
        items = json.load(f)
    if not items:
        print("[vlm] no items to process")
        return 0

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"[vlm] loading model {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"[vlm] model loaded, processing {len(items)} items ...")
    for i, item in enumerate(items):
        img = Image.open(item["image"]).convert("RGB")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe the materials visible in this object."},
            ]},
        ]

        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt_text], images=[img],
            return_tensors="pt", padding=True,
        ).to(model.device)

        print(f"[vlm] [{i+1}/{len(items)}] {Path(item['image']).name} ...")
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
            )

        gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        out_path = Path(item["output"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        result = {"ok": True, "image": item["image"], "output": str(out_path),
                  "materials": text}
        print(f"{RESULT_MARKER} {json.dumps(result)}", flush=True)

    print(f"[vlm] done: {len(items)} items processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

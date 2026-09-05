"""VLM material analysis + texture quality classification (Stage V).

Runs inside the vlm conda env via subprocess. Uses transformers
AutoModelForImageTextToText with the Qwen2.5-VL processor for
image-conditioned material description and texture quality routing.

CLI:
    python scripts/vlm_infer.py --model <hf_id> --items-file <path>

Items file: [{"image": "...", "output": "...",
              "classification_output": "...", "category": "..."}, ...]

When classification_output is present, the script first classifies texture
quality (edit vs generate), then uses a prompt variant matched to the
classification for the material description. When absent, the script runs
the original materials-only prompt (backward compatible).

Emits [PBR_RESULT] JSON lines per the adapter contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULT_MARKER = "[PBR_RESULT]"

CLASSIFICATION_PROMPT = (
    "Look at this rendered image of a 3D object. "
    "Assess whether the object has meaningful visible textures and materials "
    "(distinct colors, patterns, surface finishes on its parts) "
    "or whether it appears blank, untextured, or has only uniform flat colors "
    "(single solid color per part, no surface detail, no patterns). "
    "Respond with exactly one word: EDIT if the object has meaningful existing "
    "textures worth preserving, or GENERATE if the textures are blank/basic "
    "and should be created from scratch."
)

SYSTEM_PROMPT_EDIT = (
    "Analyze the object's visible materials and surface appearance. "
    "Identify the main material of each visually distinct component, "
    "including material type, color, and important surface finish when visible. "
    "For each material, briefly note which part of the object it appears on. "
    "When the same material type appears with slight color or shade variations "
    "across different parts, describe each variant with its location so they "
    "can be distinguished. "
    "Preserve the appearance implied by the image rather than redesigning it. "
    "Ignore the background, lighting, camera angle, and object geometry. "
    "Do not invent highly specific materials when the image does not provide enough evidence. "
    'Return only a concise semicolon-separated list where each entry names the material '
    'and its location, for example: '
    '"dark brown wood grain on the tabletop; light oak wood on the legs; '
    'brushed steel on the frame joints; matte black plastic on the casters".'
)

SYSTEM_PROMPT_GENERATE = (
    'This object has blank or basic textures. Based on its shape, structure, '
    'and the category "{CATEGORY}", infer what realistic materials each '
    "visible exterior component would plausibly be made of. "
    "Focus only on parts you can see in the image - do not invent internal "
    "components, hidden parts, or anything not structurally visible. "
    "For each visually distinct structural component, assign a material type, "
    "color, and surface finish that would be realistic for this kind of object. "
    "Use your knowledge of real-world objects in this category to make plausible choices. "
    "Do not describe what you see (the surfaces are blank); instead describe "
    "what the materials SHOULD be. "
    "Ignore the background, lighting, and camera angle. "
    "Keep the list concise with at most 10 entries. "
    'Return only a semicolon-separated list where each entry names the material '
    'and its location, for example: '
    '"brushed stainless steel on the body; matte black rubber on the handle grip; '
    'tempered glass on the door panel; chrome-plated metal on the hinges".'
)


_DETAIL_KEYWORDS = {
    "button", "buttons", "dial", "dials", "gauge", "gauges", "display",
    "displays", "screen", "screens", "readout", "readouts", "knob", "knobs",
    "switch", "switches", "indicator", "indicators", "led", "leds", "icon",
    "icons", "label", "labels", "logo", "logos", "text", "lettering",
    "sticker", "stickers", "decal", "decals", "marking", "markings",
    "writing", "stamp", "stamps",
}


def _filter_materials(text: str) -> str:
    """Remove semicolon-separated entries that reference fine-grained surface details
    (buttons, displays, dials, etc.) which conflict with the edit model's removal prompt.
    Also deduplicates repeated entries."""
    entries = [e.strip() for e in text.split(";") if e.strip()]
    seen = set()
    filtered = []
    for entry in entries:
        lower = entry.lower()
        if lower in seen:
            continue
        seen.add(lower)
        words = set(lower.split())
        if words & _DETAIL_KEYWORDS:
            continue
        filtered.append(entry)
    return "; ".join(filtered) if filtered else text


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="VLM material analysis (Stage V)")
    ap.add_argument("--model", required=True, help="HuggingFace model id")
    ap.add_argument("--items-file", required=True, help="JSON list of {image, output}")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.1)
    return ap


def _generate(model, processor, img, system_prompt, user_text, max_tokens, temperature):
    """Single VLM generation call."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": user_text},
        ]},
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt_text], images=[img],
        return_tensors="pt", padding=True,
    ).to(model.device)

    import torch
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


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
        do_classify = "classification_output" in item
        category = item.get("category", "object")

        print(f"[vlm] [{i+1}/{len(items)}] {Path(item['image']).name} ...")

        classification = "edit"
        if do_classify:
            cls_text = _generate(
                model, processor, img,
                "You are a texture quality assessor for 3D objects.",
                CLASSIFICATION_PROMPT,
                max_tokens=5, temperature=0.0,
            )
            classification = "generate" if "GENERATE" in cls_text.upper() else "edit"
            cls_path = Path(item["classification_output"])
            cls_path.parent.mkdir(parents=True, exist_ok=True)
            cls_path.write_text(classification)
            print(f"[vlm]   classification: {classification} (raw: {cls_text!r})")

        if classification == "generate":
            system_prompt = SYSTEM_PROMPT_GENERATE.replace("{CATEGORY}", category)
        else:
            system_prompt = SYSTEM_PROMPT_EDIT

        raw_text = _generate(
            model, processor, img,
            system_prompt,
            "Describe the materials visible in this object.",
            max_tokens=args.max_tokens, temperature=args.temperature,
        )
        text = _filter_materials(raw_text)
        if text != raw_text:
            print(f"[vlm]   filtered: {raw_text!r} -> {text!r}")

        out_path = Path(item["output"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        result = {"ok": True, "image": item["image"], "output": str(out_path),
                  "materials": text, "classification": classification}
        print(f"{RESULT_MARKER} {json.dumps(result)}", flush=True)

    print(f"[vlm] done: {len(items)} items processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

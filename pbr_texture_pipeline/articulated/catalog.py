"""Category-aware material catalog for articulated assets (port of trellis_pbr/material_catalog.py).

The catalog constrains Stage P's choices to physically-plausible materials per (category,
semantic-label) and provides the coercion fallback when the VLM's plan misses a label.
The plan's swatch_prompt strings and this module's prompt templates are descriptive material
metadata (kept for the plan schema and PBR hints); nothing renders swatch images anymore.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List

# --- swatch prompt templates -------------------------------------------------
SWATCH_PROMPT = (
    "seamless tileable {material} surface texture, photorealistic material sample, "
    "flat top-down view, even soft studio lighting, no objects, no shadows, "
    "uniform, high detail, 4k material swatch"
)
SWATCH_NEGATIVE = (
    "object, 3d shape, perspective, depth, vignette, text, watermark, logo, "
    "people, hands, drop shadow, frame, border, blurry"
)

# --- material definitions: friendly name -> descriptive phrase for the prompt --
MATERIALS: Dict[str, str] = {
    "oak_wood": "light oak wood grain",
    "walnut_wood": "dark walnut wood grain",
    "mahogany_wood": "rich reddish mahogany wood grain",
    "white_painted_wood": "smooth white painted wood",
    "black_painted_wood": "smooth matte black painted wood",
    "brushed_steel": "brushed stainless steel metal",
    "polished_chrome": "polished mirror chrome metal",
    "matte_black_metal": "matte black powder-coated metal",
    "brass": "polished golden brass metal",
    "brushed_aluminum": "brushed aluminum metal",
    "clear_glass": "clear transparent glass",
    "frosted_glass": "frosted translucent glass",
    "green_glass": "green tinted glass",
    "blue_plastic": "glossy blue plastic",
    "black_plastic": "matte black plastic",
    "white_plastic": "glossy white plastic",
    "cork": "natural cork material",
    "ceramic": "glossy white ceramic glaze",
    "leather": "brown leather upholstery",
    "fabric": "grey woven fabric upholstery",
}

# --- category / semantic-label -> sensible material options -------------------
# Keyed by lowercase substring of the semantic label; falls back to category default.
_LABEL_RULES = {
    # doors
    "door_frame": ["oak_wood", "walnut_wood", "white_painted_wood",
                   "brushed_steel", "matte_black_metal"],
    "rotation_door": ["oak_wood", "walnut_wood", "white_painted_wood",
                      "frosted_glass", "clear_glass", "brushed_steel"],
    "door": ["oak_wood", "walnut_wood", "white_painted_wood", "frosted_glass"],
    "glass": ["frosted_glass", "clear_glass", "green_glass"],
    # bottles
    "bottle_body": ["clear_glass", "frosted_glass", "green_glass",
                    "blue_plastic", "brushed_aluminum", "ceramic"],
    "rotation_lid": ["black_plastic", "brushed_aluminum", "brass", "cork"],
    "lid": ["black_plastic", "brushed_aluminum", "brass", "cork"],
    # furniture / storage
    "handle": ["brushed_steel", "brass", "matte_black_metal", "polished_chrome"],
    "drawer": ["oak_wood", "walnut_wood", "white_painted_wood", "brushed_steel"],
    "cabinet": ["oak_wood", "walnut_wood", "white_painted_wood"],
    "shelf": ["oak_wood", "walnut_wood", "white_painted_wood", "brushed_steel"],
    "wheel": ["black_plastic", "matte_black_metal"],
    "caster": ["black_plastic", "matte_black_metal"],
    "frame": ["brushed_steel", "matte_black_metal", "oak_wood"],
    "tabletop": ["oak_wood", "walnut_wood", "white_painted_wood", "clear_glass"],
    "leg": ["brushed_steel", "matte_black_metal", "oak_wood", "walnut_wood"],
    "button": ["black_plastic", "white_plastic", "brushed_aluminum"],
    "knob": ["brushed_steel", "brass", "black_plastic"],
    "seat": ["leather", "fabric", "black_plastic"],
    "back": ["leather", "fabric", "black_plastic"],
}

_CATEGORY_DEFAULTS = {
    "Door": ["oak_wood", "walnut_wood", "white_painted_wood",
             "brushed_steel", "frosted_glass"],
    "Bottle": ["clear_glass", "frosted_glass", "blue_plastic",
               "brushed_aluminum", "ceramic"],
    "Table": ["oak_wood", "walnut_wood", "white_painted_wood", "brushed_steel"],
    "Cart": ["brushed_steel", "matte_black_metal", "black_plastic"],
    "Dishwasher": ["brushed_steel", "white_plastic", "matte_black_metal"],
    "StorageFurniture": ["oak_wood", "walnut_wood", "white_painted_wood"],
}

_FALLBACK = ["oak_wood", "brushed_steel", "white_plastic", "matte_black_metal"]

# Heuristic threshold: strings longer than this (or containing the swatch boilerplate)
# are treated as full swatch prompts rather than bare material names.
_PROMPT_LEN_THRESHOLD = 60


def options_for(category: str, semantic_label: str) -> List[str]:
    """Return the sensible material options for a part given its category + label."""
    label = (semantic_label or "").lower()
    for key, opts in _LABEL_RULES.items():
        if key in label:
            return opts
    return _CATEGORY_DEFAULTS.get(category, _FALLBACK)


def looks_like_prompt(s: str) -> bool:
    """True if `s` is already a full swatch prompt rather than a bare material name."""
    return len(s) > _PROMPT_LEN_THRESHOLD or "seamless tileable" in s.lower()


def swatch_prompt(material: str) -> str:
    """Resolve a material name (catalog key or free text) to a swatch prompt."""
    if looks_like_prompt(material):
        return material
    phrase = MATERIALS.get(material, material.replace("_", " "))
    return SWATCH_PROMPT.format(material=phrase)


# --- PBR hints for the constant-material fallback (tiny groups skip the backends) ------------
# base_color RGB 0-255, metallic, roughness. Approximate by design: tiny groups are handles,
# pegs, and slivers; plausibility beats precision.
PBR_HINTS: Dict[str, tuple] = {
    "oak_wood": ((196, 160, 110), 0.0, 0.7),
    "walnut_wood": ((92, 60, 38), 0.0, 0.65),
    "mahogany_wood": ((122, 52, 38), 0.0, 0.6),
    "white_painted_wood": ((240, 240, 236), 0.0, 0.5),
    "black_painted_wood": ((28, 28, 28), 0.0, 0.5),
    "brushed_steel": ((150, 152, 155), 1.0, 0.4),
    "polished_chrome": ((200, 202, 205), 1.0, 0.1),
    "matte_black_metal": ((30, 30, 32), 1.0, 0.8),
    "brass": ((205, 165, 80), 1.0, 0.35),
    "brushed_aluminum": ((170, 172, 176), 1.0, 0.45),
    "clear_glass": ((220, 228, 232), 0.0, 0.05),
    "frosted_glass": ((222, 228, 232), 0.0, 0.4),
    "green_glass": ((130, 180, 150), 0.0, 0.1),
    "blue_plastic": ((40, 90, 200), 0.0, 0.3),
    "black_plastic": ((25, 25, 27), 0.0, 0.6),
    "white_plastic": ((238, 238, 240), 0.0, 0.35),
    "cork": ((190, 150, 100), 0.0, 0.85),
    "ceramic": ((242, 242, 240), 0.0, 0.2),
    "leather": ((110, 70, 45), 0.0, 0.6),
    "fabric": ((140, 140, 145), 0.0, 0.95),
}
_PBR_DEFAULT = ((180, 180, 180), 0.0, 0.8)


def pbr_hint(material: str) -> tuple:
    """(base_color RGB 0-255, metallic, roughness) for a material name; sane default otherwise."""
    return PBR_HINTS.get(material, _PBR_DEFAULT)


def slug(material: str) -> str:
    """Filesystem-safe cache key. Truncates long prompts and appends a short hash so distinct
    prompts never collide while filenames stay sane."""
    base = re.sub(r"[^a-z0-9]+", "_", material.lower()).strip("_")
    if len(base) <= 60:
        return base
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:8]
    return f"{base[:60].rstrip('_')}_{digest}"

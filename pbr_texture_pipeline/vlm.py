"""Stage V: automated caption -> appearance spec (Qwen3.6-35B-A3B AWQ via vLLM).

Two-turn automated flow (2026-07-11, replaces the Qwen2.5-VL-7B human chat):
  1. Caption turn: the VLM sees the contact sheet and writes a geometry/semantics caption
     (category, parts, notable shapes - no invented materials or colors).
  2. Spec turn: a fresh conversation gets the contact sheet + that caption and emits the
     [SPEC]...[/SPEC] appearance JSON (PRD section 4 schema, unchanged).
The tag-regex structured extraction, force-finalize retry, and template-fallback philosophy
survive from the trellis_pbr port; interactive chat (regenerate/finalize) still works on top
of the spec conversation, so the human keeps the last word in the app.

Runs in the dedicated `vlm` env (vLLM; the AWQ MoE checkpoint cannot load under the trellis2
env's torch 2.6 + transformers stack). ~20 GB weights on GPU 0; vLLM pre-allocates its KV/state
pool at `vlm.gpu_memory_utilization` of the card, so unloading means dropping the whole engine.
"""
from __future__ import annotations

import ctypes
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

from pbr_texture_pipeline.config import load_config

_CFG = load_config()
DEFAULT_MODEL_ID = _CFG.model("vlm")

_llm = None

# Extraction / retry knobs (ported philosophy).
_SPEC_RE = re.compile(r"\[SPEC\]\s*(.+?)\s*\[/SPEC\]", re.DOTALL)
_VERDICT_RE = re.compile(r"\[VERDICT\]\s*(.+?)\s*\[/VERDICT\]", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_MAX_FORCE_ATTEMPTS = 3

# Fixed prompt discipline (PRD section 4, behavior 3): every ref_prompt ends with this.
BOILERPLATE = "single object, centered, plain neutral gray background, soft even studio lighting"
DEFAULT_NEGATIVE = ("cartoon, painting, illustration, text, watermark, cluttered background, "
                    "harsh shadows, strong reflections, people, multiple objects, "
                    "extra items, accessories, props, loose parts, surrounding objects, "
                    "contents, decorations not part of the object")

CAPTION_MAX_TOKENS = int(_CFG.get("vlm.caption_max_new_tokens", 512))
SPEC_MAX_TOKENS = int(_CFG.get("vlm.spec_max_new_tokens", 1024))


# --- model load/unload --------------------------------------------------------
def _bootstrap_runtime_env() -> None:
    """Prepare the process environment for vLLM on this cluster, before importing it.

    Everything set via os.environ also propagates to the EngineCore / registry subprocesses
    vLLM spawns. Three concerns:

    1. FFmpeg for torchcodec (imported unconditionally by vLLM's video path): it dlopens
       libavutil/... by soname, but a conda env's lib/ dir is not on the loader search path.
       For THIS process LD_LIBRARY_PATH is already fixed at exec, so load the libs
       RTLD_GLOBAL (dependency order); export LD_LIBRARY_PATH for the children.
    2. No JIT against /usr/local/cuda (absent on this cluster): disable the flashinfer
       sampler (Stage V decodes greedily, so it buys nothing) and point CUDA_HOME at the
       env's pip-provided CUDA toolkit for any remaining kernel compilation.
    3. Never write caches under /home: pin the flashinfer/triton/inductor/vLLM cache roots
       next to the shared HF cache.
    """
    import os
    import sysconfig

    libdir = Path(sys.prefix) / "lib"
    prev = os.environ.get("LD_LIBRARY_PATH", "")
    if str(libdir) not in prev.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{libdir}:{prev}" if prev else str(libdir)
    for stem in ("libavutil", "libswresample", "libswscale", "libavcodec",
                 "libavformat", "libavfilter", "libavdevice"):
        for lib in sorted(libdir.glob(stem + ".so.*")):
            if re.fullmatch(r"[^.]+\.so\.\d+", lib.name):
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    cuda_home = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
    if (cuda_home / "bin" / "nvcc").is_file() and not os.environ.get("CUDA_HOME"):
        os.environ["CUDA_HOME"] = str(cuda_home)
        os.environ["PATH"] = f"{cuda_home / 'bin'}:{os.environ.get('PATH', '')}"

    cache_root = Path(_CFG.hf_cache).parent          # shared cache dir (parent of hf_cache)
    # flashinfer appends ".cache/flashinfer" to its base itself.
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(cache_root.parent))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "torchinductor"))
    os.environ.setdefault("VLLM_CACHE_ROOT", str(cache_root / "vllm"))


def load_model(model_id: str = DEFAULT_MODEL_ID):
    """Load (or return cached) vLLM engine for the multimodal Qwen3.6 checkpoint. Idempotent."""
    global _llm
    if _llm is not None:
        return _llm
    _bootstrap_runtime_env()
    from vllm import LLM

    print(f"[vlm] loading {model_id} via vLLM ...")
    _llm = LLM(
        model=model_id,
        max_model_len=int(_CFG.get("vlm.max_model_len", 16384)),
        gpu_memory_utilization=float(_CFG.get("vlm.gpu_memory_utilization", 0.6)),
        # Stage J needs 3 images (two judge sheets + reference); Stage V uses 1.
        limit_mm_per_prompt={"image": int(_CFG.get("vlm.limit_mm_per_prompt_images", 3))},
        # Contact sheets are passed as file:// URLs from the job dirs (local, trusted).
        allowed_local_media_path="/",
        seed=0,
    )
    print("[vlm] model ready")
    return _llm


def unload_model():
    """Free the VLM from VRAM. vLLM pre-allocates its pool, so drop the entire engine."""
    global _llm
    if _llm is not None:
        del _llm
        _llm = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


# --- generation turn ----------------------------------------------------------
def _to_vllm_messages(messages: list[dict]) -> list[dict]:
    """Wire format ({'type':'image','image':<path>}) -> vLLM chat format (file:// image_url).

    The wire format is kept for transcript.json continuity and because the orchestrator/app
    round-trip messages as plain JSON.
    """
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "image":
                    parts.append({"type": "image_url",
                                  "image_url": {"url": "file://" + str(c["image"])}})
                else:
                    parts.append(c)
            content = parts
        out.append({"role": m["role"], "content": content})
    return out


def _generate(messages: list[dict], max_new_tokens: int = SPEC_MAX_TOKENS) -> str:
    """One deterministic generation turn over OpenAI-style messages; content may embed images
    as {"type":"image","image":<path>}. Returns the new assistant text."""
    if _llm is None:
        load_model()  # lazy (re)load: covers stage-exclusion eviction
    from vllm import SamplingParams

    outs = _llm.chat(
        _to_vllm_messages(messages),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=max_new_tokens),
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )
    text = outs[0].outputs[0].text
    # Thinking is disabled above, but strip any stray block defensively.
    return _THINK_RE.sub("", text).strip()


def _user(text: str, image: Optional[str] = None) -> dict:
    """Build a user message, optionally with a leading image (contact sheet)."""
    content: list[dict] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def _strip_spec_tag(text: str) -> str:
    return _SPEC_RE.sub("", text).strip()


# --- spec schema + extraction (PRD section 4) ----------------------------------
_SPEC_FIELDS = ("category", "category_confidence", "front_view_index",
                "materials", "style", "ref_prompt", "negative_prompt")


def _coerce_spec(raw: dict, category_hint: str = "object") -> dict:
    """Validate/fill a parsed spec dict against the PRD section-4 schema."""
    spec = {
        "category": raw.get("category") or category_hint,
        "category_confidence": raw.get("category_confidence", "medium"),
        "front_view_index": int(raw.get("front_view_index", 0) or 0),
        "materials": raw.get("materials", []) or [],
        "style": raw.get("style", "realistic"),
        "ref_prompt": (raw.get("ref_prompt") or "").strip(),
        "negative_prompt": (raw.get("negative_prompt") or DEFAULT_NEGATIVE).strip(),
    }
    if spec["category_confidence"] not in ("high", "medium", "low"):
        spec["category_confidence"] = "medium"
    spec["front_view_index"] = max(0, min(7, spec["front_view_index"]))
    if spec["ref_prompt"] and BOILERPLATE not in spec["ref_prompt"]:
        spec["ref_prompt"] = spec["ref_prompt"].rstrip(" .,;") + ", " + BOILERPLATE
    return spec


def extract_spec(text: str, category_hint: str = "object") -> Optional[dict]:
    """Parse a [SPEC]{json}[/SPEC] block. Returns a coerced spec dict, or None."""
    m = _SPEC_RE.search(text)
    if not m:
        return None
    blob = m.group(1).strip()
    # Tolerate markdown code fences inside the tag.
    blob = re.sub(r"^```(?:json)?|```$", "", blob, flags=re.MULTILINE).strip()
    try:
        raw = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    return _coerce_spec(raw, category_hint)


def template_fallback_spec(category: str, last_user_text: str = "",
                           material_hint: str = "") -> dict:
    """Template fallback when the VLM never emits a valid spec (PRD sections 4, risk 4)."""
    cat = category or "object"
    extra = f" {material_hint}" if material_hint else ""
    prompt = (f"a photograph of a {cat} made of typical realistic materials{extra}, "
              f"product photography, {BOILERPLATE}")
    return _coerce_spec({
        "category": cat,
        "category_confidence": "low",
        "front_view_index": 0,
        "materials": [],
        "style": "realistic",
        "ref_prompt": prompt,
        "negative_prompt": DEFAULT_NEGATIVE,
    }, cat)


# --- caption turn ---------------------------------------------------------------
def caption_system_prompt() -> str:
    return (
        "You are a meticulous 3D asset cataloguer. You are shown clay renders of an untextured "
        "3D object from 8 angles arranged as a 2x4 contact sheet (panels indexed 0-7 row-major: "
        "top row 0,1,2,3; bottom row 4,5,6,7).\n\n"
        "Write a detailed caption of the OBJECT ITSELF: what it is, its distinct parts, their "
        "shapes and proportions, and any notable geometric features. Also state which panel "
        "index most likely shows the object's front.\n\n"
        "When the user message includes dataset metadata for the object, that metadata is "
        "authoritative ground truth: describe the object as the stated category, interpret "
        "ambiguous geometry in that light, and never contradict it.\n\n"
        "The renders are untextured clay: do NOT invent materials, colors, or surface finishes, "
        "and do not describe the gray clay shading or the background. 4-8 sentences, no lists."
    )


def caption_turn(contact_sheet_path: str, metadata_note: str = "") -> str:
    """One-shot caption conversation over the contact sheet. Returns the caption text."""
    if _llm is None:
        load_model()
    text = "Caption this object now."
    if metadata_note:
        text = metadata_note + "\n\n" + text
    messages = [
        {"role": "system", "content": caption_system_prompt()},
        _user(text, image=contact_sheet_path),
    ]
    return _generate(messages, max_new_tokens=CAPTION_MAX_TOKENS)


# --- spec-turn system prompt: four enforced behaviors --------------------------
def system_prompt() -> str:
    example = {
        "category": "chair",
        "category_confidence": "high",
        "front_view_index": 0,
        "materials": [
            {"region": "seat/back", "material": "walnut wood, satin finish"},
            {"region": "legs", "material": "brushed steel"},
        ],
        "style": "realistic, contemporary",
        "ref_prompt": ("a photograph of a mid-century walnut wood chair with brushed steel "
                        f"legs, satin finish, product photography, {BOILERPLATE}, 8k, photorealistic"),
        "negative_prompt": DEFAULT_NEGATIVE,
    }
    return (
        "You are an appearance advisor for 3D object texturing. You are shown clay renders of "
        "an untextured 3D object from 8 angles arranged as a 2x4 contact sheet (panels indexed "
        "0-7 row-major: top row 0,1,2,3; bottom row 4,5,6,7), plus a caption of the object.\n\n"
        "1. CATEGORY: First state what the object is and which panel index shows its front. "
        "When the user message states a known category from dataset metadata, treat it as "
        "ground truth: do not re-classify the object, and set \"category\" to exactly that "
        "value.\n"
        "2. REALISM: Propose only materials the object category is actually manufactured from. "
        "If the user requests an implausible material (e.g. a brick chair, a glass hammer "
        "handle), refuse in ONE sentence and offer plausible alternatives. Allow a stylized or "
        "fictional finish ONLY if the user explicitly insists after your warning.\n"
        "3. PROMPT DISCIPLINE: the ref_prompt you emit MUST be a detailed description of a "
        "plausible textured appearance - name a concrete material and finish for every "
        "distinct region of the object - and it MUST end with this exact boilerplate so "
        "downstream background removal and PBR decomposition work: \"" + BOILERPLATE + "\".\n"
        "4. OBJECT ONLY: the ref_prompt describes ONLY the surface texture of the object's "
        "existing geometry - its materials, colors, and finishes. Do NOT add items, props, "
        "accessories, contents, or decorations that are not part of the mesh (e.g. no coffee "
        "beans on a coffee machine, no food in an oven, no clothes in a washing machine, no "
        "books on a shelf). The image will be used as a texturing reference, not a lifestyle "
        "scene. Include the negative_prompt terms that suppress such additions.\n\n"
        "When you have a final appearance, emit a single block exactly like this (valid JSON "
        "between the tags):\n"
        "[SPEC]\n" + json.dumps(example, indent=2) + "\n[/SPEC]\n"
        "front_view_index is the contact-sheet panel (0-7) showing the object's front."
    )


# --- transcript persistence ----------------------------------------------------
def save_transcript(jobdir, messages: list[dict]) -> None:
    jobdir.write_json(jobdir.transcript(), {"messages": messages})


def save_spec(jobdir, spec: dict) -> None:
    jobdir.write_json(jobdir.spec(), spec)


def save_caption(jobdir, caption: str, model_id: str = DEFAULT_MODEL_ID) -> None:
    jobdir.write_json(jobdir.caption(), {"caption": caption, "model": model_id})


# --- articulated context (WS3): articulation summary + metadata -----------------
def normalize_category(raw: str) -> str:
    """Normalize a dataset category for prompt injection and comparison: split CamelCase
    ("StorageFurniture" -> "storage furniture"), lowercase, preserve all-caps tokens ("USB").
    Used everywhere the metadata category is injected or compared, so "Dishwasher" vs
    "dishwasher" never counts as a mismatch."""
    if not raw:
        return ""
    tokens = []
    for word in re.split(r"[\s_\-/]+", str(raw).strip()):
        for tok in re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]+|[A-Z]+|[a-z]+|\d+", word):
            tokens.append(tok if tok.isupper() and len(tok) > 1 else tok.lower())
    return " ".join(tokens)


def articulation_summary(job) -> str:
    """One-paragraph articulation summary for the spec turn of a urdf job: category (when
    known, stated as ground truth), link/label/motion list. Built from job.state['asset']
    (Stage R fills it)."""
    asset = job.state.get("asset") or {}
    groups = asset.get("groups", [])
    if not groups:
        return ""
    by_label: dict[str, dict] = {}
    for g in groups:
        e = by_label.setdefault(g["label"], {"n": 0, "motions": set()})
        e["n"] += 1
        e["motions"].add(g.get("motion", "static"))
    parts = [f"{label} x{e['n']} ({'/'.join(sorted(e['motions']))})"
             for label, e in sorted(by_label.items())]
    cat_line = ""
    if asset.get("category"):
        cat = normalize_category(asset["category"])
        cat_line = (f"This object's category is known from dataset metadata and is ground "
                    f"truth: it is a {cat}. Set the spec's \"category\" field to exactly "
                    f"\"{cat}\" and propose materials appropriate for a {cat}. ")
    return (f"{cat_line}This is an articulated object with movable parts. "
            f"Its parts (semantic label x count (motion type)): " + ", ".join(parts) + ". "
            "The renders show the object assembled at rest pose.")


def metadata_note(asset: dict) -> str:
    """The authoritative asset-metadata paragraph given to both Stage V turns of a urdf job:
    normalized dataset category plus the union of semantics.txt labels and group labels.
    Empty when the asset has no metadata category."""
    cat = normalize_category(asset.get("category") or "")
    if not cat:
        return ""
    labels = sorted(set(asset.get("semantic_labels") or [])
                    | {g["label"] for g in asset.get("groups", [])})
    note = f"Dataset metadata for this object (authoritative ground truth): category: {cat}."
    if labels:
        note += "\nPart labels from the dataset: " + ", ".join(labels) + "."
    note += f"\nDescribe and treat this object as a {cat}; do not re-classify it."
    return note


def articulated_context(job) -> dict:
    """The articulated Stage V context for a job, shared by run_auto and the worker's open
    op: articulation summary, the authoritative metadata note, and the normalized metadata
    category."""
    ctx = {"articulation_note": "", "metadata_note": "", "category": None}
    ctx["articulation_note"] = articulation_summary(job)
    asset = job.state.get("asset") or {}
    if asset.get("category"):
        ctx["category"] = normalize_category(asset["category"])
        ctx["metadata_note"] = metadata_note(asset)
    return ctx


def apply_category_metadata(spec: dict, category: str) -> tuple[dict, bool]:
    """Force the authoritative metadata category onto a spec (urdf jobs).

    Returns (spec, mismatch). Substring match on the normalized strings counts as agreement
    ("dishwasher" accepts "built-in dishwasher"). On mismatch the VLM's answer is kept as
    spec["vlm_category"]; spec["category"] is always set to the normalized metadata category
    with category_source "metadata", so a wrong VLM classification can never persist."""
    cat = normalize_category(category)
    if not cat or not isinstance(spec, dict):
        return spec, False
    got = normalize_category(str(spec.get("category") or "")).lower()
    want = cat.lower()
    mismatch = not (got and (got in want or want in got))
    if mismatch:
        spec["vlm_category"] = spec.get("category")
    spec["category"] = cat
    spec["category_source"] = "metadata"
    return spec, mismatch


# --- chat drivers ----------------------------------------------------------------
def opening_turn(contact_sheet_path: str, material_hint: str = "",
                 articulation_note: str = "",
                 metadata_note: str = "") \
        -> tuple[list[dict], str, Optional[dict], str]:
    """Automated opening: caption the mesh, then propose a spec from views + caption.

    Articulated jobs (WS3) may add an `articulation_note` paragraph and `metadata_note`
    (the authoritative dataset-metadata paragraph, given to both the caption turn and the
    spec turn). The [SPEC] flow itself is unchanged.

    Returns (messages, assistant_reply, spec_or_none, caption). `messages` is the spec
    conversation (the caption is embedded in its opening user turn, so the transcript is
    self-contained); the caption conversation itself is not retained.
    """
    if _llm is None:
        load_model()
    caption = caption_turn(contact_sheet_path, metadata_note=metadata_note)
    hint = f"\nBatch material steer to honor if plausible: {material_hint}." if material_hint else ""
    text = ""
    if metadata_note:
        text += metadata_note + "\n\n"
    if articulation_note:
        text += articulation_note + "\n\n"
    text += "Caption of this object (from the same renders):\n" + caption + "\n\n"
    text += ("Propose a detailed, plausible appearance spec now. "
             "Emit the [SPEC]...[/SPEC] block." + hint)
    content: list[dict] = [{"type": "image", "image": contact_sheet_path}]
    content.append({"type": "text", "text": text})
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": content},
    ]
    reply = _generate(messages)
    messages.append({"role": "assistant", "content": reply})
    spec = extract_spec(reply)
    return messages, reply, spec, caption


def regenerate(messages: list[dict], user_text: str) -> tuple[list[dict], str, Optional[dict]]:
    """Interactive: append a user message, generate, try to extract an updated spec."""
    messages = list(messages) + [_user(user_text)]
    reply = _generate(messages)
    messages.append({"role": "assistant", "content": reply})
    return messages, reply, extract_spec(reply)


def force_finalize(messages: list[dict]) -> tuple[list[dict], str, Optional[dict]]:
    """Send the force-finalize turn (Finalize button / auto retry)."""
    messages = list(messages) + [_user(
        "Finalize now: output the [SPEC]...[/SPEC] block with valid JSON for the appearance "
        "we agreed on. If your previous JSON was invalid, re-emit it correctly.")]
    reply = _generate(messages)
    messages.append({"role": "assistant", "content": reply})
    return messages, reply, extract_spec(reply)


def run_auto(jobdir, contact_sheet_path: str, category_hint: str = "object",
             material_hint: str = "") -> dict:
    """Automated Stage V: caption -> spec -> force-finalize retry -> template fallback.

    urdf jobs automatically gain the articulation summary (WS3) and, when the asset has a
    metadata category, the authoritative metadata note: the emitted
    spec's category is force-set to the metadata category (one corrective retry when the
    VLM disagreed and its ref_prompt does not mention the category), with the VLM's answer
    kept as spec["vlm_category"] on mismatch.
    Returns {"spec", "caption", "fallback"} and writes caption.json + spec.json +
    transcript.json.
    """
    ctx = articulated_context(jobdir)
    meta_cat = ctx["category"]
    if meta_cat:
        category_hint = meta_cat
    messages, _reply, spec, caption = opening_turn(
        contact_sheet_path, material_hint,
        articulation_note=ctx["articulation_note"],
        metadata_note=ctx["metadata_note"])
    save_caption(jobdir, caption)
    fallback = False

    if spec is None or not spec.get("ref_prompt"):
        messages, _reply, spec = force_finalize(messages)

    if spec is None or not spec.get("ref_prompt"):
        cat = meta_cat or (spec.get("category", category_hint) if spec else category_hint)
        spec = template_fallback_spec(cat, material_hint=material_hint)
        fallback = True

    if meta_cat:
        spec, mismatch = apply_category_metadata(spec, meta_cat)
        if mismatch:
            orig = spec.get("vlm_category")
            print(f"[vlm] spec category {orig!r} contradicts metadata category "
                  f"{meta_cat!r}; forcing metadata")
            if not fallback and meta_cat.lower() not in (spec.get("ref_prompt") or "").lower():
                messages, _reply, retry_spec = regenerate(messages, (
                    f"Dataset metadata says this object is a {meta_cat}, "
                    f"not a {orig or 'different object'}. "
                    f"Re-emit the [SPEC]...[/SPEC] block for a {meta_cat}."))
                if retry_spec is not None and retry_spec.get("ref_prompt"):
                    retry_spec, _ = apply_category_metadata(retry_spec, meta_cat)
                    retry_spec["vlm_category"] = orig
                    spec = retry_spec

    save_spec(jobdir, spec)
    save_transcript(jobdir, messages)
    return {"spec": spec, "caption": caption, "fallback": fallback}


# Back-compat alias (pre-caption name).
run_batch = run_auto


# --- Stage J: A/B judging of textured outputs -----------------------------------
JUDGE_MAX_TOKENS = int(_CFG.get("judge.max_new_tokens", 1024))

_CONFIDENCES = ("high", "medium", "low")
_CRITERIA_KEYS = ("seams_artifacts", "sharpness_detail", "lighting_neutrality",
                  "cross_view_coherence", "material_plausibility", "reference_fidelity")


def extract_verdict(text: str) -> Optional[dict]:
    """Parse a [VERDICT]{json}[/VERDICT] block. Returns a coerced verdict dict, or None.

    `winner` must be exactly "A" or "B"; a tie, both, or anything else is invalid (-> retry /
    fallback upstream). Missing criteria keys are tolerated; confidence is coerced.
    """
    m = _VERDICT_RE.search(text)
    if not m:
        return None
    blob = m.group(1).strip()
    blob = re.sub(r"^```(?:json)?|```$", "", blob, flags=re.MULTILINE).strip()
    try:
        raw = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    winner = raw.get("winner")
    if not isinstance(winner, str) or winner.strip().upper() not in ("A", "B"):
        return None
    confidence = raw.get("confidence", "medium")
    if confidence not in _CONFIDENCES:
        confidence = "medium"
    criteria_raw = raw.get("criteria") if isinstance(raw.get("criteria"), dict) else {}
    criteria = {}
    for key in _CRITERIA_KEYS:
        c = criteria_raw.get(key)
        if isinstance(c, dict):
            criteria[key] = {"better": c.get("better", "tie"), "note": str(c.get("note", ""))}
    return {
        "winner": winner.strip().upper(),
        "confidence": confidence,
        "criteria": criteria,
        "reasoning": str(raw.get("reasoning", "")),
    }


def judge_system_prompt() -> str:
    return (
        "You are a strict, impartial visual-quality judge for 3D texturing. Two different "
        "systems textured the SAME 3D mesh. You will see three images:\n\n"
        "- Image 1 - OUTPUT A: four renders of A's textured mesh in a 2x2 grid (top-left: "
        "front, top-right: right side, bottom-left: back, bottom-right: left side).\n"
        "- Image 2 - OUTPUT B: the same four views of B's textured mesh.\n"
        "- Image 3 - the reference image that guided both systems. The reference is CONTEXT "
        "ONLY, not ground truth: it may itself be flawed, and matching it does NOT make an "
        "output good.\n\n"
        "Pick the output that is the better standalone 3D asset. Judge visual quality, "
        "plausibility, and realism of the textured mesh itself. Criteria, in strict priority "
        "order:\n\n"
        "1. SEAMS AND PROJECTION ARTIFACTS: visible texture seams, projection smears or "
        "ghosting, misaligned or stretched texture, features painted across the wrong "
        "geometry.\n"
        "2. SHARPNESS AND DETAIL: blur, mushy low-resolution regions, noise, smeared detail.\n"
        "3. LIGHTING NEUTRALITY: baked-in highlights, shadows, or shading gradients that "
        "should not be part of the surface color; the texture should look like flat albedo, "
        "evenly lit.\n"
        "4. CROSS-VIEW COHERENCE: consistent color and material between front, sides, and "
        "back. Pay special attention to the BACK and SIDE views - a beautiful front with a "
        "discolored, blank, or hallucinated back loses.\n"
        "5. MATERIAL PLAUSIBILITY: are these materials and finishes something this category "
        "of object is actually made of, applied to the right regions?\n"
        "6. REFERENCE RESEMBLANCE (lowest priority): use only as a tiebreaker when 1-5 are "
        "comparable. Never reward copying the reference's lighting, shadows, or background.\n\n"
        "Do not reward higher saturation, contrast, or busier detail for its own sake. The "
        "A/B labels were assigned at random and mean nothing - never favor A because it is "
        "first.\n\n"
        "You MUST declare exactly one winner, \"A\" or \"B\". A tie is not a valid answer: if "
        "the outputs seem equal, pick the one with fewer artifacts under criterion 1, then 2, "
        "and so on.\n\n"
        "After comparing, emit a single block exactly like this (valid JSON between the "
        "tags):\n\n"
        "[VERDICT]\n"
        "{\n"
        "  \"winner\": \"A\",\n"
        "  \"confidence\": \"high\",\n"
        "  \"criteria\": {\n"
        "    \"seams_artifacts\":      {\"better\": \"A\", \"note\": \"...\"},\n"
        "    \"sharpness_detail\":     {\"better\": \"A\", \"note\": \"...\"},\n"
        "    \"lighting_neutrality\":  {\"better\": \"B\", \"note\": \"...\"},\n"
        "    \"cross_view_coherence\": {\"better\": \"A\", \"note\": \"...\"},\n"
        "    \"material_plausibility\":{\"better\": \"tie\", \"note\": \"...\"},\n"
        "    \"reference_fidelity\":   {\"better\": \"B\", \"note\": \"...\"}\n"
        "  },\n"
        "  \"reasoning\": \"2-4 sentences explaining the decision in priority order.\"\n"
        "}\n"
        "[/VERDICT]\n\n"
        "Per-criterion \"better\" may be \"A\", \"B\", or \"tie\"; the top-level \"winner\" "
        "may only be \"A\" or \"B\"."
    )


# --- Stage P: per-part material plan (articulated jobs, WS3) ---------------------
# A separate [PLAN] block + coercion path so the proven [SPEC] machinery stays untouched.
_PLAN_RE = re.compile(r"\[PLAN\]\s*(.+?)\s*\[/PLAN\]", re.DOTALL)
PLAN_MAX_TOKENS = int(_CFG.get("articulated.plan_max_new_tokens", 4096))

_TABLE_TOP_N = 8   # individually-listed largest groups


def extract_plan(text: str) -> Optional[dict]:
    """Parse a [PLAN]{json}[/PLAN] block. Returns the raw dict (uncoerced), or None."""
    m = _PLAN_RE.search(text)
    if not m:
        return None
    blob = m.group(1).strip()
    blob = re.sub(r"^```(?:json)?|```$", "", blob, flags=re.MULTILINE).strip()
    try:
        raw = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return raw if isinstance(raw, dict) else None


def groups_table(asset_state: dict) -> str:
    """Compressed group listing: one row per semantic label (count, links, summed area share,
    dominant colors, <=3 example group ids) plus individual rows for the largest groups.
    Label-level compression keeps 19898's 79 groups to ~14 rows (~8k prompt tokens)."""
    groups = asset_state.get("groups", [])
    by_label: dict[str, list[dict]] = {}
    for g in groups:
        by_label.setdefault(g["label"], []).append(g)

    lines = ["LABEL | N GROUPS | LINKS | AREA% | MOTION | EXAMPLE GROUP IDS"]
    for label, gs in sorted(by_label.items(), key=lambda kv: -sum(g["area_frac"] for g in kv[1])):
        area = sum(g["area_frac"] for g in gs) * 100
        links = sorted({g["link"] for g in gs})
        motions = sorted({g.get("motion", "static") for g in gs})
        examples = [g["group_id"] for g in gs[:3]]
        lines.append(f"{label} | {len(gs)} | {len(links)} links | {area:.1f}% | "
                     f"{'/'.join(motions)} | {', '.join(examples)}")

    largest = sorted(groups, key=lambda g: -g["area_frac"])[:_TABLE_TOP_N]
    lines.append("")
    lines.append("LARGEST INDIVIDUAL GROUPS (group_id | label | area% | tiny):")
    for g in largest:
        lines.append(f"{g['group_id']} | {g['label']} | {g['area_frac'] * 100:.1f}% | "
                     f"{'tiny' if g.get('tiny') else '-'}")
    return "\n".join(lines)


def plan_system_prompt() -> str:
    example = {
        "materials": {
            "oak_wood": {"description": "light oak wood grain",
                         "base_color": [196, 160, 110], "metallic": 0.0, "roughness": 0.7},
            "brushed_steel": {"description": "brushed stainless steel metal",
                              "base_color": [150, 152, 155], "metallic": 1.0,
                              "roughness": 0.4},
        },
        "assign": {"drawer_front": "oak_wood", "handle": "brushed_steel"},
        "overrides": {"link_2__drawer_front-137": "brushed_steel"},
    }
    return (
        "You are a material planner for per-part 3D texturing of an articulated object. You "
        "will see: (1) a 2x4 clay contact sheet of the whole assembled object and (2) a "
        "generated reference photo of a plausible final appearance. You also get a table of "
        "the object's semantic part groups.\n\n"
        "Design ONE coherent material scheme for the whole object, then assign a material to "
        "every semantic LABEL in the table (per-label, not per-group; use \"overrides\" only "
        "when a specific group_id must differ from its label's material). Reuse the same "
        "material key for parts that should match to keep the object visually consistent.\n\n"
        "Every material entry needs: description (short material description), base_color "
        "(RGB 0-255), metallic (0-1), roughness (0-1). Propose only materials this object "
        "category is actually manufactured from.\n\n"
        "When you are done, emit a single block exactly like this (valid JSON between the "
        "tags):\n"
        "[PLAN]\n" + json.dumps(example, indent=2) + "\n[/PLAN]\n"
        "\"assign\" MUST cover every label in the table. Material keys are short snake_case "
        "names."
    )


def coerce_plan(raw: dict, asset_state: dict, category: str) -> dict:
    """Resolve every group via assign -> overrides -> catalog fallback; normalize materials.

    Missing/unknown labels fall back to catalog.options_for(category, label)[0] and are
    flagged per group.
    """
    from pbr_texture_pipeline.articulated import catalog

    materials_raw = raw.get("materials") if isinstance(raw.get("materials"), dict) else {}
    assign = raw.get("assign") if isinstance(raw.get("assign"), dict) else {}
    overrides = raw.get("overrides") if isinstance(raw.get("overrides"), dict) else {}

    groups_res: dict[str, dict] = {}
    for g in asset_state.get("groups", []):
        gid, label = g["group_id"], g["label"]
        share_key = overrides.get(gid) or assign.get(label)
        fallback = False
        if not isinstance(share_key, str) or not share_key.strip():
            share_key = catalog.options_for(category, label)[0]
            fallback = True
        groups_res[gid] = {"share_key": share_key.strip(), "fallback": fallback}

    materials: dict[str, dict] = {}
    for share_key in sorted({v["share_key"] for v in groups_res.values()}):
        mat = materials_raw.get(share_key)
        if isinstance(mat, str):
            mat = {}
        if not isinstance(mat, dict):
            mat = {}
        description = catalog.material_phrase(share_key)
        base, metallic, rough = catalog.pbr_hint(share_key)
        try:
            base_color = [int(c) for c in (mat.get("base_color") or base)][:3]
        except (TypeError, ValueError):
            base_color = list(base)
        def _num(v, dflt):
            try:
                return min(1.0, max(0.0, float(v)))
            except (TypeError, ValueError):
                return dflt
        materials[share_key] = {
            "description": description,
            "base_color": base_color,
            "metallic": _num(mat.get("metallic"), metallic),
            "roughness": _num(mat.get("roughness"), rough),
        }

    return {
        "materials": materials,
        "assign": {k: v for k, v in assign.items() if isinstance(v, str)},
        "overrides": {k: v for k, v in overrides.items() if isinstance(v, str)},
        "groups": groups_res,
    }


def catalog_fallback_plan(asset_state: dict, category: str) -> dict:
    """Whole-plan fallback when the VLM never emits valid [PLAN] JSON."""
    return coerce_plan({}, asset_state, category)


def run_plan(job, max_new_tokens: int = PLAN_MAX_TOKENS) -> dict:
    """Stage P: one plan call + one force-finalize retry + catalog fallback.

    Images: clay contact sheet + ref/chosen.png (when present).
    Writes vlm/plan.json (+ vlm/plan_transcript.json) and returns {"plan", "fallback"}.
    """
    if _llm is None:
        load_model()
    asset_state = job.state.get("asset")
    if not asset_state:
        raise RuntimeError("job has no 'asset' section; run articulated Stage R first")
    spec = job.read_json(job.spec()) if job.spec().is_file() else {}
    category = asset_state.get("category") or spec.get("category") or "object"

    content: list[dict] = [{"type": "image", "image": str(job.contact_sheet())}]
    img_desc = ["Image 1: clay contact sheet (2x4, whole object at rest pose)."]
    if job.chosen().is_file():
        content.append({"type": "image", "image": str(job.chosen())})
        img_desc.append(f"Image {len(content)}: generated whole-object reference photo "
                        "(anchor the overall look to it).")
    from pbr_texture_pipeline.articulated import catalog

    table = groups_table(asset_state)
    options = {}
    for g in asset_state.get("groups", []):
        options.setdefault(g["label"], catalog.options_for(category, g["label"]))
    options_txt = "\n".join(f"  {label}: {', '.join(opts)}"
                            for label, opts in sorted(options.items()))
    content.append({"type": "text", "text":
                    "\n".join(img_desc) + f"\n\nObject category: {category}.\n"
                    + "\nPART GROUPS:\n" + table
                    + "\n\nSensible catalog options per label (you may also propose other "
                      "plausible materials):\n" + options_txt
                    + "\n\nEmit the [PLAN]...[/PLAN] block now."})

    messages = [{"role": "system", "content": plan_system_prompt()},
                {"role": "user", "content": content}]
    reply = _generate(messages, max_new_tokens=max_new_tokens)
    messages.append({"role": "assistant", "content": reply})
    raw = extract_plan(reply)
    if raw is None:
        messages.append(_user(
            "Finalize now: emit the [PLAN]...[/PLAN] block with valid JSON covering every "
            "label. If your previous JSON was invalid, re-emit it correctly."))
        reply = _generate(messages, max_new_tokens=max_new_tokens)
        messages.append({"role": "assistant", "content": reply})
        raw = extract_plan(reply)

    fallback = raw is None
    plan = coerce_plan(raw, asset_state, category) if raw is not None \
        else catalog_fallback_plan(asset_state, category)
    plan["fallback"] = fallback
    plan["category"] = category
    plan["model"] = DEFAULT_MODEL_ID
    job.write_json(job.plan(), plan)
    job.write_json(job.path("vlm", "plan_transcript.json"), {"messages": messages})
    return {"plan": plan, "fallback": fallback}


# --- targeted-retexture confirmation (PRD.md section 7.7) --------------------
REFINE_MAX_TOKENS = 512
_GRADE_RE = re.compile(r"\[GRADE\]\s*(.+?)\s*\[/GRADE\]", re.DOTALL)
_GRADES = ("good", "blurry", "wrong_material", "missing_texture")


def extract_grade(text: str) -> Optional[dict]:
    """Parse a [GRADE]{json}[/GRADE] block -> {"grade", "reasoning"} or None."""
    m = _GRADE_RE.search(text)
    if not m:
        return None
    blob = re.sub(r"^```(?:json)?|```$", "", m.group(1).strip(), flags=re.MULTILINE).strip()
    try:
        raw = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    grade = str(raw.get("grade", "")).strip().lower().replace(" ", "_")
    if grade not in _GRADES:
        return None
    return {"grade": grade, "reasoning": str(raw.get("reasoning", ""))}


def refine_system_prompt() -> str:
    return (
        "You grade the texture quality of ONE part of a 3D object from close-up renders.\n"
        "The part was flagged by automatic heuristics as possibly badly textured; decide "
        "whether it actually is.\n"
        "Grades: \"good\" (texture is acceptable), \"blurry\" (smeared or detail-free), "
        "\"wrong_material\" (implausible material for this part), \"missing_texture\" "
        "(untextured, flat constant color, or obvious garbage).\n"
        "Answer with one short reasoning sentence, then emit exactly:\n"
        "[GRADE]{\"grade\": \"good|blurry|wrong_material|missing_texture\", "
        "\"reasoning\": \"...\"}[/GRADE]"
    )


def run_refine(job, candidates: list[dict], crops: dict[str, str],
               max_new_tokens: int = REFINE_MAX_TOKENS) -> dict:
    """Grade each retexture candidate group from its crop render (one call per candidate,
    one force-finalize retry). Writes vlm/refine.json and returns its contents."""
    if _llm is None:
        load_model()
    category = (job.state.get("asset", {}).get("category")
                or (job.read_json(job.spec()).get("category")
                    if job.spec().is_file() else None) or "object")
    results: dict[str, dict] = {}
    for cand in candidates:
        gid = cand["group_id"]
        crop = crops.get(gid)
        if not crop:
            results[gid] = {"grade": "ungraded", "reasoning": "no crop render available"}
            continue
        label = gid.split("__", 1)[-1]
        content = [
            {"type": "image", "image": str(crop)},
            {"type": "text", "text":
                f"Object category: {category}. Part group: {label}.\n"
                f"Flagged because: {'; '.join(cand.get('reasons', []))}.\n"
                "Grade this part's texture now and emit the [GRADE] block."},
        ]
        messages = [{"role": "system", "content": refine_system_prompt()},
                    {"role": "user", "content": content}]
        reply = _generate(messages, max_new_tokens=max_new_tokens)
        grade = extract_grade(reply)
        if grade is None:
            messages.append({"role": "assistant", "content": reply})
            messages.append(_user(
                "Finalize now: emit the [GRADE]...[/GRADE] block with valid JSON."))
            reply = _generate(messages, max_new_tokens=max_new_tokens)
            grade = extract_grade(reply)
        results[gid] = grade or {"grade": "ungraded", "reasoning": reply[:300]}
    out = {"results": results, "candidates": candidates, "model": DEFAULT_MODEL_ID}
    job.write_json(job.path("vlm", "refine.json"), out)
    return out


def run_judge(sheet_a: str, sheet_b: str, ref_path: Optional[str], category: str = "object",
              max_new_tokens: int = JUDGE_MAX_TOKENS) -> dict:
    """One deterministic A/B judging call + one force-finalize retry.

    Returns {"verdict": coerced dict or None, "raw": last assistant reply}. Fallback policy
    (default_winner, label->backend mapping) stays with the orchestrator, not here.
    """
    if _llm is None:
        load_model()
    content: list[dict] = [{"type": "image", "image": sheet_a},
                           {"type": "image", "image": sheet_b}]
    if ref_path:
        content.append({"type": "image", "image": ref_path})
    ref_note = ("Image 3 = reference (context only, not ground truth). " if ref_path
                else "No reference image is available. ")
    content.append({"type": "text", "text":
                    f"Object category: {category}. Image 1 = OUTPUT A (front/right/back/left). "
                    f"Image 2 = OUTPUT B (same views). {ref_note}"
                    "Compare them now and emit the [VERDICT] block."})
    messages = [{"role": "system", "content": judge_system_prompt()},
                {"role": "user", "content": content}]
    reply = _generate(messages, max_new_tokens=max_new_tokens)
    verdict = extract_verdict(reply)
    if verdict is None:
        messages.append({"role": "assistant", "content": reply})
        messages.append(_user(
            "Finalize now: emit the [VERDICT]...[/VERDICT] block with valid JSON. "
            "\"winner\" must be exactly \"A\" or \"B\" - a tie or refusal is not a valid "
            "answer. If your previous JSON was invalid, re-emit it correctly."))
        reply = _generate(messages, max_new_tokens=max_new_tokens)
        verdict = extract_verdict(reply)
    return {"verdict": verdict, "raw": reply}

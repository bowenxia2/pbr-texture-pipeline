#!/usr/bin/env python
"""Build a self-contained interactive gallery of textured meshes for a pbr-texture-pipeline jobs root.

Scans <jobs-root> for jobs (dirs with job.json) and emits <jobs-root>/viewer.html: a
responsive grid, one card per mesh, each with a rotatable/zoomable <model-viewer> per backend
(drag to rotate, scroll to zoom), the VLM reference image, category, and IoU/CLIP/LPIPS metrics.

model-viewer.min.js is vendored into <jobs-root>/assets/ so the served page needs no network.
GLB/image URLs are relative to <jobs-root>, so serve from there:

    cd <jobs-root> && python -m http.server 8080
    # then on your laptop:  ssh -L 8080:localhost:8080 <host>  ->  http://localhost:8080/viewer.html

Stdlib only. Re-run any time (idempotent) to pick up newly finished jobs.
"""
import argparse
import html
import json
import os
import sys
import urllib.request
from pathlib import Path

MODEL_VIEWER_URL = (
    "https://cdn.jsdelivr.net/npm/@google/model-viewer@3.5.0/dist/model-viewer.min.js"
)
BACKENDS = ["trellis2"]
GLB_NAMES = ("assembled.glb",)
STAGES = ["render", "vlm", "diffuse", "plan", "texture", "eval", "judge"]


def vendor_model_viewer(jobs_root: Path) -> Path:
    assets = jobs_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    dst = assets / "model-viewer.min.js"
    if dst.is_file() and dst.stat().st_size > 100_000:
        return dst
    print(f"[viewer] downloading model-viewer -> {dst}")
    with urllib.request.urlopen(MODEL_VIEWER_URL, timeout=120) as r:
        data = r.read()
    dst.write_bytes(data)
    print(f"[viewer] vendored {len(data) // 1024} KB")
    return dst


def find_glb(job_dir: Path, backend: str):
    d = job_dir / "textured" / backend
    if not d.is_dir():
        return None
    for name in GLB_NAMES:
        if (d / name).is_file():
            return d / name
    globbed = sorted(d.glob("*.glb"))
    return globbed[0] if globbed else None


def load_json(p: Path):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def rel(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def collect_jobs(jobs_root: Path):
    jobs = []
    for d in sorted(p for p in jobs_root.iterdir() if p.is_dir()):
        state = load_json(d / "job.json")
        if state is None:
            continue
        spec = load_json(d / "vlm" / "spec.json") or {}
        metrics = load_json(d / "eval" / "metrics.json") or {}
        mbackends = metrics.get("backends", {})
        stages = state.get("stages", {})
        statuses = {s: stages.get(s, {}).get("status", "-") for s in STAGES}
        chosen = d / "ref" / "chosen.png"
        verdict = load_json(d / "judge" / "verdict.json") or {}
        winner = (verdict.get("winner")
                  or stages.get("judge", {}).get("params", {}).get("winner"))
        backends = []
        for b in BACKENDS:
            glb = find_glb(d, b)
            cond = d / "previews" / f"{b}_condview.png"
            sheet = d / "previews" / f"{b}_judgesheet.png"
            bm = mbackends.get(b, {})
            if glb is None:
                continue
            # Both outputs survive Stage J; by default the gallery shows only the winner as an
            # interactive viewer and collapses the runner-up to a poster with a link to its GLB,
            # so a browsable jobs root still reads as "one pick per job" while nothing is lost.
            is_runner_up = bool(winner) and b != winner
            poster_img = sheet if sheet.is_file() else (cond if cond.is_file() else None)
            backends.append({
                "name": b,
                "glb": rel(glb, jobs_root),
                "runner_up": is_runner_up and poster_img is not None,
                "winner": bool(winner) and b == winner,
                "poster": rel(poster_img, jobs_root) if poster_img else
                          (rel(cond, jobs_root) if cond.is_file() else None),
                "iou": bm.get("silhouette_iou"),
                "clip": bm.get("clip_prompt_vs_condview"),
                "lpips": bm.get("lpips_ref_vs_condview"),
            })
        jobs.append({
            "job_id": state.get("job_id", d.name),
            "kind": "urdf",
            "category": spec.get("category") or (state.get("asset") or {}).get("category") or "",
            "prompt": spec.get("ref_prompt") or metrics.get("prompt") or "",
            "chosen": rel(chosen, jobs_root) if chosen.is_file() else None,
            "clip_ref": metrics.get("clip_prompt_vs_ref"),
            "statuses": statuses,
            "backends": backends,
            "winner": winner,
            "judge_method": verdict.get("method"),
            "judge_reasoning": verdict.get("reasoning") or verdict.get("reason") or "",
        })
    return jobs


CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  background: #0f1115; color: #e6e8eb; }
@media (prefers-color-scheme: light) { body { background: #f4f5f7; color: #1a1c1f; } }
header { padding: 20px 24px; border-bottom: 1px solid #2a2e37; position: sticky; top: 0;
  background: inherit; z-index: 5; }
@media (prefers-color-scheme: light) { header { border-color: #d8dbe0; } }
h1 { margin: 0 0 4px; font-size: 20px; font-weight: 650; }
.sub { font-size: 13px; opacity: .65; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 18px; padding: 20px 24px; }
.card { background: #171a21; border: 1px solid #262b34; border-radius: 12px; overflow: hidden;
  display: flex; flex-direction: column; }
@media (prefers-color-scheme: light) { .card { background: #fff; border-color: #e2e5ea; } }
.card.empty { opacity: .5; }
.card h2 { margin: 0; padding: 12px 14px 2px; font-size: 16px; font-weight: 620;
  text-transform: capitalize; }
.jid { padding: 0 14px 10px; font-size: 11px; font-family: ui-monospace, Menlo, monospace;
  opacity: .5; word-break: break-all; }
.viewers { display: flex; gap: 8px; padding: 0 10px 10px; flex-wrap: wrap; }
.vwrap { flex: 1 1 150px; min-width: 140px; }
.vlabel { font-size: 11px; opacity: .7; padding: 2px 2px 4px; text-align: center;
  letter-spacing: .02em; }
model-viewer { width: 100%; height: 190px; background: #0b0d11; border-radius: 8px;
  --poster-color: transparent; }
@media (prefers-color-scheme: light) { model-viewer { background: #eef0f3; } }
.refrow { display: flex; gap: 10px; align-items: flex-start; padding: 6px 14px 12px; }
.refrow img { width: 92px; height: 92px; object-fit: cover; border-radius: 8px;
  border: 1px solid #2a2f39; flex: none; }
.refcap { font-size: 11px; opacity: .6; }
table.m { width: 100%; border-collapse: collapse; font-size: 11.5px; margin-top: 4px; }
table.m th, table.m td { padding: 3px 6px; text-align: right; border-top: 1px solid #262b34; }
table.m th:first-child, table.m td:first-child { text-align: left; }
@media (prefers-color-scheme: light) { table.m th, table.m td { border-color: #e6e9ee; } }
table.m thead th { opacity: .6; font-weight: 550; border-top: none; }
.status { font-size: 11px; padding: 4px 14px 12px; opacity: .6; font-family: ui-monospace, monospace; }
.hint { font-size: 11px; opacity: .55; padding: 0 14px 10px; }
.badge { display: inline-block; font-size: 10px; font-weight: 700; letter-spacing: .05em;
  padding: 1px 6px; border-radius: 6px; background: #2e7d32; color: #fff; margin-left: 4px; }
.runnerup { width: 100%; height: 190px; object-fit: contain; background: #0b0d11;
  border-radius: 8px; filter: grayscale(.6) brightness(.75); }
@media (prefers-color-scheme: light) { .runnerup { background: #eef0f3; } }
.runnerupcap { font-size: 10px; opacity: .55; text-align: center; padding-top: 2px;
  font-style: italic; }
.runnerupcap a { color: inherit; }
.verdict { font-size: 11px; opacity: .65; padding: 0 14px 12px; }
"""


def viewer_html(job, title_id):
    parts = []
    cat = html.escape(job["category"] or "(uncategorized)")
    parts.append(f'<div class="card{"" if job["backends"] else " empty"}">')
    parts.append(f"<h2>{cat}</h2>")
    parts.append(f'<div class="jid">{html.escape(job["job_id"])}</div>')

    if job["backends"]:
        parts.append('<div class="viewers">')
        for b in job["backends"]:
            label = html.escape(b["name"])
            if b.get("winner"):
                label += '<span class="badge">WINNER</span>'
            parts.append('<div class="vwrap">')
            parts.append(f'<div class="vlabel">{label}</div>')
            if b.get("runner_up"):
                parts.append(
                    f'<img class="runnerup" src="{html.escape(b["poster"])}"'
                    f' alt="{cat} textured by {b["name"]} (not selected by judge)">'
                    f'<div class="runnerupcap">not selected by judge - '
                    f'<a href="{html.escape(b["glb"])}">view full model</a></div>'
                )
            else:
                poster = f' poster="{html.escape(b["poster"])}"' if b["poster"] else ""
                parts.append(
                    f'<model-viewer src="{html.escape(b["glb"])}"{poster}'
                    ' camera-controls touch-action="pan-y" interaction-prompt="none"'
                    ' shadow-intensity="1" exposure="1" environment-image="neutral"'
                    ' reveal="interaction" loading="lazy"'
                    f' alt="{cat} textured by {b["name"]}"></model-viewer>'
                )
            parts.append("</div>")
        parts.append("</div>")
        parts.append('<div class="hint">Click a model to load it, then drag to rotate / scroll to zoom.</div>')
        links = " &middot; ".join(
            f'<a href="/viewer/{html.escape(job["job_id"])}/{b["name"]}" '
            f'title="needs the pbr-texture-pipeline app running (or scripts/articulated_viewer.py)">'
            f'joints: {b["name"]}</a>' for b in job["backends"])
        parts.append(f'<div class="hint">articulated &middot; {links}</div>')
        if job.get("winner") and job.get("judge_reasoning"):
            snippet = job["judge_reasoning"]
            if len(snippet) > 320:
                snippet = snippet[:317] + "..."
            parts.append(f'<div class="verdict">judge ({html.escape(job.get("judge_method") or "?")}): '
                         f'{html.escape(snippet)}</div>')

        # reference image + metrics table
        parts.append('<div class="refrow">')
        if job["chosen"]:
            parts.append(f'<img src="{html.escape(job["chosen"])}" alt="reference image">')
        parts.append('<div style="flex:1">')
        parts.append('<div class="refcap">generated reference'
                     + (f' &middot; CLIP(prompt,ref) {fmt(job["clip_ref"])}' if job["clip_ref"] is not None else "")
                     + "</div>")
        parts.append('<table class="m"><thead><tr><th>backend</th><th>IoU</th>'
                     '<th>CLIP</th><th>LPIPS</th></tr></thead><tbody>')
        for b in job["backends"]:
            parts.append(
                f'<tr><td>{b["name"]}</td><td>{fmt(b["iou"])}</td>'
                f'<td>{fmt(b["clip"])}</td><td>{fmt(b["lpips"])}</td></tr>'
            )
        parts.append("</tbody></table>")
        parts.append("</div></div>")
    else:
        st = " ".join(f"{k[0].upper()}:{v}" for k, v in job["statuses"].items())
        parts.append(f'<div class="status">not textured &middot; {html.escape(st)}</div>')
    parts.append("</div>")
    return "\n".join(parts)


def build(jobs_root: Path, title: str) -> Path:
    vendor_model_viewer(jobs_root)
    jobs = collect_jobs(jobs_root)
    textured = sum(1 for j in jobs if j["backends"])
    cards = "\n".join(viewer_html(j, i) for i, j in enumerate(jobs))
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script type="module" src="assets/model-viewer.min.js"></script>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="sub">{len(jobs)} meshes &middot; {textured} textured &middot; drag to rotate, scroll to zoom</div>
</header>
<div class="grid">
{cards}
</div>
</body>
</html>
"""
    out = jobs_root / "viewer.html"
    out.write_text(doc)
    print(f"[viewer] wrote {out}  ({len(jobs)} jobs, {textured} textured)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Build an interactive textured-mesh gallery.")
    ap.add_argument("--jobs-root", required=True, help="jobs root to scan (e.g. jobs_static)")
    ap.add_argument("--title", default="pbr-texture-pipeline - textured meshes")
    args = ap.parse_args()
    jobs_root = Path(args.jobs_root).resolve()
    if not jobs_root.is_dir():
        print(f"no such jobs root: {jobs_root}", file=sys.stderr)
        return 1
    build(jobs_root, args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

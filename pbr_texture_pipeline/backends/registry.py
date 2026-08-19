"""Backend registry + subprocess launcher (Stage T, PRD section 7).

Maps a backend name -> {script, env, cwd}. Adapters run via subprocess in their own
conda env with the correct cwd (never in the orchestrator process): free VRAM isolation,
crash containment, and the backend's hard cwd requirement is honored.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

from pbr_texture_pipeline.config import load_config
from pbr_texture_pipeline.backends._adapter_common import RESULT_MARKER

_CFG = load_config()
_HERE = Path(__file__).resolve().parent


def _conda_bin() -> str:
    """Locate the conda executable (for `conda run -n <env>`).

    Checked in order: an explicit `env.conda_bin` override in config.yaml/config.local.yaml,
    the `$CONDA_EXE`/`$MAMBA_EXE` env vars conda/mamba set on activation, then $PATH.
    """
    override = _CFG.get("env.conda_bin")
    if override and Path(override).is_file():
        return override
    for var in ("CONDA_EXE", "MAMBA_EXE"):
        cand = os.environ.get(var)
        if cand and Path(cand).is_file():
            return cand
    found = shutil.which("conda") or shutil.which("mamba")
    if found:
        return found
    raise RuntimeError(
        "conda executable not found. Activate your conda installation, or set "
        "env.conda_bin in config.local.yaml to its path."
    )


# name -> {script, env (conda env), cwd (backend repo dir)}
BACKENDS: dict[str, dict] = {
    "trellis2": {
        "script": str(_HERE / "trellis2_adapter.py"),
        "env": _CFG.backend_env("trellis2"),
        "cwd": str(_CFG.repo("trellis2")),
    },
}


def _subprocess_env(extra_env: Optional[dict] = None) -> dict:
    """Child env pinning the shared HF cache (never write weights under /home).

    `extra_env` overlays additional vars (e.g. CUDA_VISIBLE_DEVICES to pin texturing to GPU 1
    in dual mode, or GPU 0 after evicting the persistent workers in single-GPU mode).
    """
    env = os.environ.copy()
    env.update(_CFG.hf_env())
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    return env


def _parse_results(stdout: str) -> list[dict]:
    out = []
    for line in stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            try:
                out.append(json.loads(line[len(RESULT_MARKER):].strip()))
            except json.JSONDecodeError:
                pass
    return out


def _build_cmd(spec: dict, extra: list[str]) -> list[str]:
    return [_conda_bin(), "run", "-n", spec["env"], "--no-capture-output",
            "python", spec["script"], *extra]


def texture(
    backend: str,
    mesh_path: str,
    image_rgba_path: str,
    out_dir: str,
    seed: int,
    camera_json: Optional[str],
    params: Optional[dict] = None,
    extra_env: Optional[dict] = None,
    on_line: Optional[Callable[[str, bool], None]] = None,
) -> dict:
    """Single-pair adapter call (PRD contract).

    adapter.texture(...) -> {glb_path, logs, returncode, results}

    `on_line(text, carriage)` is invoked for each output line as it is produced (carriage=True
    for \\r-terminated tqdm progress updates), so callers can stream a multi-minute run live.
    """
    spec = BACKENDS[backend]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    extra = ["--mesh", mesh_path, "--image", image_rgba_path, "--out-dir", out_dir,
             "--seed", str(seed)]
    if camera_json:
        extra += ["--camera-json", camera_json]
    extra += _param_args(backend, params or {})
    return _run(spec, extra, extra_env=extra_env, on_line=on_line)


def texture_pairs(backend: str, pairs_file: str, params: Optional[dict] = None,
                  extra_env: Optional[dict] = None,
                  on_line: Optional[Callable[[str, bool], None]] = None) -> dict:
    """Batch: one backend load over many pairs via --pairs-file (pbr_compare sweep pattern)."""
    spec = BACKENDS[backend]
    extra = ["--pairs-file", pairs_file, *_param_args(backend, params or {})]
    return _run(spec, extra, extra_env=extra_env, on_line=on_line)


def _param_args(backend: str, params: dict) -> list[str]:
    """Translate per-backend params dict into adapter CLI flags."""
    args: list[str] = []
    if backend == "trellis2":
        args += ["--resolution", str(params.get("resolution", _CFG.get("texture.trellis2_resolution")))]
        args += ["--texture-size", str(params.get("texture_size", _CFG.get("texture.trellis2_texture_size")))]
    return args


def _run(spec: dict, extra: list[str], extra_env: Optional[dict] = None,
         on_line: Optional[Callable[[str, bool], None]] = None) -> dict:
    cmd = _build_cmd(spec, extra)
    # Stream stdout+stderr line by line (splitting on \r too, so tqdm progress surfaces live)
    # instead of buffering the whole multi-minute run: the caller can render real progress and
    # a slow model load no longer looks like a hang. stderr is merged so it interleaves in order.
    # Read raw bytes (not text mode) so \r survives - universal-newline translation would fold
    # \r into \n and we'd lose the carriage-return vs newline distinction the caller relies on.
    proc = subprocess.Popen(cmd, cwd=spec["cwd"], env=_subprocess_env(extra_env),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    captured: list[str] = []
    buf = bytearray()
    assert proc.stdout is not None

    def _emit(carriage: bool) -> None:
        # \r and \n are single ASCII bytes that never occur inside a UTF-8 multibyte sequence,
        # so decoding each terminator-delimited segment on its own is always safe.
        line = buf.decode("utf-8", "replace")
        if on_line is not None:
            on_line(line, carriage)
        captured.append(line)
        buf.clear()

    while True:
        b = proc.stdout.read(1)
        if not b:
            break
        if b == b"\r" or b == b"\n":
            _emit(b == b"\r")
        else:
            buf += b
    if buf:
        _emit(False)
    proc.wait()
    stdout = "\n".join(captured)
    logs = stdout
    results = _parse_results(stdout)
    last = results[-1] if results else {}
    glb_path = last.get("glb_path")
    ok = bool(last.get("ok") and glb_path and Path(glb_path).exists())
    teardown_crash = ok and proc.returncode != 0
    return {"glb_path": glb_path if ok else None, "ok": ok,
            "teardown_crash": teardown_crash, "logs": logs,
            "returncode": proc.returncode, "results": results}

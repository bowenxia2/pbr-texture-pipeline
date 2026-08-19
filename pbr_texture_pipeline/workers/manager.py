"""GPU memory manager + worker orchestration (PRD section 7).

Owns the two persistent workers and the texturing policy, so the Gradio app just calls
`mgr.render(...)`, `mgr.vlm_open(...)`, `mgr.texture(...)` without touching CUDA or GPU math.

Dual A40 (default): the VLM (Qwen3.6-35B-A3B AWQ via vLLM in its own `vlm` env, ~20 GB weights
plus vLLM's pre-allocated pool) stays resident on GPU 0 so chat/reroll are always live.
The 20B Qwen-Image Stage D model no longer fits beside it, so the imaging worker (renderer +
Qwen-Image via CPU offload, peaking ~40 GB) owns GPU 1. Texturing also runs on GPU 1; because
Stage D and Stage T never overlap for a job, the manager drops the imaging worker's Qwen model
(`unload`) before a texture run frees GPU 1, then it reloads lazily on the next diffuse.

Single GPU (`gpu_mode: single`): stage-exclusion. Before launching a texturing subprocess the
manager tells the imaging worker to drop its heavy models (`unload` op) and shuts the VLM
worker's process down entirely (vLLM pre-allocates its VRAM pool, so in-process unload cannot
reliably return it); both come back lazily on the next render/chat/reroll. The same exclusion
applies between the two workers' heavy models: Qwen-Image ops (diffuse, open_ref) shut the VLM
worker down first, and vlm ops drop the imaging worker's heavy models before starting vLLM.
If config says dual but fewer than 2 GPUs are visible (allocations vary per session), the
manager degrades to single at construction.

Before any texturing run the manager polls `nvidia-smi` for free VRAM on the target GPU and
queues rather than OOMing (the node is shared, no SLURM).
"""
from __future__ import annotations

import subprocess
import time
from typing import Callable, Optional

from pbr_texture_pipeline.backends import registry
from pbr_texture_pipeline.config import load_config
from pbr_texture_pipeline.workers.ipc import WorkerClient

_CFG = load_config()

# Rough peak VRAM a texturing backend needs (PRD section 7: ~21-24 GB).
_TEXTURE_VRAM_MB = 24_000
_VRAM_POLL_S = 5.0
_VRAM_WAIT_TIMEOUT_S = 1800.0


def free_vram_mb(gpu_index: int) -> Optional[int]:
    """Free VRAM (MiB) on a physical GPU via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", str(gpu_index)],
            capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def visible_gpu_count() -> int:
    """Number of GPUs nvidia-smi can see (0 if nvidia-smi is unavailable)."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return 0
        return sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
    except Exception:  # noqa: BLE001
        return 0


class WorkerManager:
    def __init__(self, gpu_mode: Optional[str] = None):
        self.gpu_mode = gpu_mode or _CFG.get("gpu_mode", "dual")
        # Allocations vary per session on this cluster; dual mode with one reachable GPU
        # would pin the imaging worker to a GPU that does not exist.
        if self.gpu_mode == "dual" and visible_gpu_count() < 2:
            print("[manager] gpu_mode is dual but fewer than 2 GPUs are visible; "
                  "degrading to single (stage-exclusion)")
            self.gpu_mode = "single"
        self.conda_bin = registry._conda_bin()
        env_name = _CFG.env_name
        # Dual: VLM stays on GPU 0 (always-live chat); imaging + Qwen-Image Stage D own GPU 1
        # (the 20B model would OOM beside the VLM). Single: both collapse onto GPU 0.
        imaging_gpu = "1" if self.gpu_mode == "dual" else "0"
        hf_env = _CFG.hf_env()  # pin the shared HF cache; never let a worker write under /home
        self.imaging = WorkerClient("pbr_texture_pipeline.workers.imaging_worker", env_name,
                                    cuda_visible=imaging_gpu, conda_bin=self.conda_bin,
                                    extra_env=hf_env, name="imaging")
        self.vlm = WorkerClient("pbr_texture_pipeline.workers.vlm_worker", _CFG.vlm_env_name,
                                cuda_visible="0", conda_bin=self.conda_bin,
                                extra_env=hf_env, name="vlm")
        # In single-GPU mode workers were told to drop their models before the last texture.
        self._evicted = False

    # -- lifecycle --
    def start(self) -> None:
        if not self.imaging.is_alive():
            self.imaging.start()
        if not self.vlm.is_alive():
            self.vlm.start()

    def _ensure_workers(self) -> None:
        """Bring workers up (dual: resident; single: models reload lazily on the next op)."""
        self.start()
        self._evicted = False

    def _ensure_imaging(self, heavy: bool = False) -> None:
        """Bring the imaging worker up for an imaging op.

        `heavy` marks ops that load Qwen-Image (~40 GB peak: diffuse, open_ref). Dual mode
        keeps both workers resident (imaging owns GPU 1, no conflict with the VLM on GPU 0).
        Single GPU: a heavy op cannot share the card with vLLM's pre-allocated pool (~38 GB),
        so the VLM worker is shut down first, the same eviction texturing uses; it reloads
        lazily on the next vlm op.
        """
        if self.gpu_mode == "dual":
            self._ensure_workers()
            return
        if not self.imaging.is_alive():
            self.imaging.start()
        if heavy and self.vlm.is_alive():
            self.vlm.shutdown()
        self._evicted = False

    def _ensure_vlm(self) -> None:
        """Bring the VLM worker up for a vlm op.

        Single GPU: before (re)starting vLLM, drop the imaging worker's heavy models so the
        pre-allocated pool fits beside the lightweight renderer; light imaging ops keep
        working and Qwen-Image reloads on the next diffuse.
        """
        if self.gpu_mode == "dual":
            self._ensure_workers()
            return
        if not self.vlm.is_alive():
            if self.imaging.is_alive():
                try:
                    self.imaging.call("unload", timeout=120)
                except Exception:  # noqa: BLE001
                    pass
            self.vlm.start()
        self._evicted = False

    def shutdown(self) -> None:
        self.imaging.shutdown()
        self.vlm.shutdown()

    # -- imaging ops --
    def render(self, job_root: str) -> dict:
        self._ensure_imaging()
        return self.imaging.call("render", job_root=job_root)

    def rerender(self, job_root: str, front_index: int = 0, yaw_nudge: float = 0.0,
                 pitch_nudge: float = 0.0) -> dict:
        self._ensure_imaging()
        return self.imaging.call("rerender", job_root=job_root, front_index=front_index,
                                 yaw_nudge=yaw_nudge, pitch_nudge=pitch_nudge)

    def diffuse(self, job_root: str, prompt: str, negative: str, base_seed: int,
                n: int = 4, cn_scale: Optional[float] = None,
                canny_scale: Optional[float] = None,
                guidance: Optional[float] = None) -> dict:
        self._ensure_imaging(heavy=True)
        return self.imaging.call("diffuse", job_root=job_root, prompt=prompt, negative=negative,
                                 base_seed=base_seed, n=n, cn_scale=cn_scale,
                                 canny_scale=canny_scale, guidance=guidance)

    def score(self, job_root: str, prompt: str, n: int = 4) -> dict:
        self._ensure_imaging()
        return self.imaging.call("score", job_root=job_root, prompt=prompt, n=n)

    def cutout(self, job_root: str, index: int) -> dict:
        self._ensure_imaging()
        return self.imaging.call("cutout", job_root=job_root, index=index)

    def open_reference(self, job_root: str, base_seed: int, index: int) -> dict:
        """Pass B (open-pose) reference for global-mode urdf jobs; the 20B generation takes
        the same order of time as one diffuse candidate."""
        self._ensure_imaging(heavy=True)
        return self.imaging.call("open_ref", job_root=job_root, base_seed=base_seed,
                                 index=index, timeout=3600)

    def judge_sheets(self, job_root: str, backends: list[str]) -> dict:
        """Stage J phase 1: render judge sheets in the imaging worker (renderer, GPU 1 in dual)."""
        self._ensure_imaging()
        return self.imaging.call("judge_sheets", job_root=job_root, backends=backends)

    # -- vlm ops --
    def vlm_open(self, contact_sheet: str, material_hint: str = "",
                 job_root: Optional[str] = None) -> dict:
        """job_root (urdf jobs) lets the worker attach the articulation summary and metadata
        note to the opening turn (WS3/WS7); flat jobs may omit it."""
        self._ensure_vlm()
        return self.vlm.call("open", contact_sheet=contact_sheet, material_hint=material_hint,
                             job_root=job_root)

    def vlm_regenerate(self, messages: list, user_text: str) -> dict:
        self._ensure_vlm()
        return self.vlm.call("regenerate", messages=messages, user_text=user_text)

    def vlm_finalize(self, messages: list) -> dict:
        self._ensure_vlm()
        return self.vlm.call("finalize", messages=messages)

    def vlm_save(self, job_root: str, spec: dict, messages: list,
                 caption: Optional[str] = None) -> dict:
        self._ensure_vlm()
        return self.vlm.call("save", job_root=job_root, spec=spec, messages=messages,
                             caption=caption)

    def vlm_plan(self, job_root: str) -> dict:
        """Stage P (articulated): per-part material plan in the vlm worker."""
        self._ensure_vlm()
        return self.vlm.call("plan", job_root=job_root)

    def vlm_judge(self, sheet_a: str, sheet_b: str, ref: Optional[str] = None,
                  category: str = "object") -> dict:
        """Stage J phase 2: one A/B verdict call. In single-GPU mode _ensure_vlm restarts
        the vlm worker after a texture eviction (same path chat uses); dual mode has the VLM
        resident on GPU 0, no conflict with texturing on GPU 1."""
        self._ensure_vlm()
        return self.vlm.call("judge", sheet_a=sheet_a, sheet_b=sheet_b, ref=ref,
                             category=category)

    # -- texturing (VRAM-aware) --
    def _texture_gpu(self) -> int:
        return 1 if self.gpu_mode == "dual" else 0

    def _evict_for_texture(self) -> None:
        """Free the texturing GPU before a backend runs.

        Single-GPU: drop the imaging worker's models and kill the VLM worker process (vLLM's
        pre-allocated pool only reliably frees on process exit; it restarts lazily on the next
        chat op). Dual: texturing shares GPU 1 with the imaging worker, so drop only its
        Qwen-Image Stage D model; the VLM keeps GPU 0 and chat stays live.
        """
        if self.imaging.is_alive():
            try:
                self.imaging.call("unload", timeout=120)
            except Exception:  # noqa: BLE001
                pass
        if self.gpu_mode == "single" and self.vlm.is_alive():
            self.vlm.shutdown()
        self._evicted = True

    def _wait_for_vram(self, gpu: int, need_mb: int = _TEXTURE_VRAM_MB) -> None:
        """Poll until the target GPU has enough free VRAM, or proceed if nvidia-smi is absent."""
        deadline = None
        while True:
            free = free_vram_mb(gpu)
            if free is None or free >= need_mb:
                return
            if deadline is None:
                deadline = time.monotonic() + _VRAM_WAIT_TIMEOUT_S
            if time.monotonic() > deadline:
                print(f"[manager] proceeding after VRAM wait timeout (free={free} MiB on GPU {gpu})")
                return
            print(f"[manager] GPU {gpu} free={free} MiB < {need_mb}; queuing {_VRAM_POLL_S}s ...")
            time.sleep(_VRAM_POLL_S)

    def texture(self, backend: str, mesh_path: str, image_rgba_path: str, out_dir: str,
                seed: int, camera_json: Optional[str], params: Optional[dict] = None,
                on_line: Optional[Callable[[str, bool], None]] = None) -> dict:
        """Run a texturing backend as a subprocess with VRAM checks + GPU pinning.

        `on_line` (if given) streams each subprocess output line to the caller as it is produced.
        """
        self._evict_for_texture()
        gpu = self._texture_gpu()
        self._wait_for_vram(gpu)
        extra_env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
        res = registry.texture(backend, mesh_path, image_rgba_path, out_dir, seed,
                               camera_json, params=params, extra_env=extra_env, on_line=on_line)
        # Workers reload lazily on the next call; nothing to restart here.
        return res

    def texture_pairs(self, backend: str, pairs_file: str,
                      params: Optional[dict] = None,
                      on_line: Optional[Callable[[str, bool], None]] = None) -> dict:
        """Articulated Stage T: one backend load over many per-group pairs (--pairs-file),
        with the same eviction + VRAM queueing as single-pair texturing."""
        self._evict_for_texture()
        gpu = self._texture_gpu()
        self._wait_for_vram(gpu)
        extra_env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
        return registry.texture_pairs(backend, pairs_file, params=params,
                                      extra_env=extra_env, on_line=on_line)

"""Config loader for pbr_texture_pipeline.

Every module reads paths and defaults through this loader so nothing is hardcoded
(PRD risk 8, env drift). Loads config.yaml from the project root by default.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

# pbr_texture_pipeline/pbr_texture_pipeline/config.py -> project root is two parents up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
# Machine-specific overrides (gitignored): conda/HF paths, real backend repo locations if they
# live outside this checkout. See config.local.yaml.example.
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config.local.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively overlay `override` onto `base`, mutating and returning `base`."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_path(value: str) -> Path:
    """Expand `~` and resolve relative paths against the project root (not cwd)."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


class Config:
    """Thin dict wrapper with dotted-path access and a few resolved helpers."""

    def __init__(self, data: dict[str, Any], path: Path):
        self._data = data
        self.path = path

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a value by dotted path, e.g. cfg.get('diffusion.cn_scale')."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # --- resolved conveniences -------------------------------------------
    @property
    def python(self) -> str:
        return self._data["env"]["python"]

    @property
    def env_name(self) -> str:
        return self._data["env"]["name"]

    @property
    def vlm_env_name(self) -> str:
        """Conda env the VLM worker runs in (vLLM stack; falls back to the primary env)."""
        return self._data["env"].get("vlm_name", self.env_name)

    @property
    def hf_cache(self) -> str:
        return str(_resolve_path(self._data["env"]["hf_cache"]))

    @property
    def torch_cache(self) -> str:
        return str(_resolve_path(self._data["env"]["torch_cache"]))

    def repo(self, name: str) -> Path:
        return _resolve_path(self._data["repos"][name])

    def backend_env(self, backend: str) -> str:
        """Conda env name a backend adapter subprocess runs in (Stage T)."""
        return self._data["backend_envs"][backend]

    def model(self, name: str) -> Any:
        return self._data["models"][name]

    def hf_env(self) -> dict[str, str]:
        """Environment overlay that pins the HF cache to the shared gscratch path.

        Pass to subprocess env so backends never write weights under /home.
        """
        cache = self.hf_cache
        return {
            "HF_HOME": cache,
            "HUGGINGFACE_HUB_CACHE": os.path.join(cache, "hub"),
            "HF_HUB_CACHE": os.path.join(cache, "hub"),
        }

    def as_dict(self) -> dict[str, Any]:
        return self._data


@functools.lru_cache(maxsize=None)
def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and cache the pbr-texture-pipeline config. Override path with PBR_CONFIG env var.

    If `config.local.yaml` exists next to the base config, it is deep-merged on top: it holds
    machine-specific values (conda paths, HF cache location, backend repo dirs if they live
    outside this checkout) that must never be committed. See config.local.yaml.example.
    """
    if path is None:
        path = os.environ.get("PBR_CONFIG", DEFAULT_CONFIG_PATH)
    path = Path(path).resolve()
    with open(path) as f:
        data = yaml.safe_load(f)
    local_path = path.parent / "config.local.yaml"
    if local_path.is_file():
        with open(local_path) as f:
            local_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, local_data)
    return Config(data, path)

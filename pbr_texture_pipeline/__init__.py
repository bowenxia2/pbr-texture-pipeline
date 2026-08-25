"""pbr-texture-pipeline: VLM-guided texturing of untextured articulated 3D assets.

Given a URDF-described articulated object, generate a category-aware, realistic,
pose-matched reference image, then drive the TRELLIS.2 texturing backend with it.
See PRD.md for the full spec.
"""

import os as _os

from pbr_texture_pipeline.config import load_config

# Pin every cache off /home before any torch / HF import happens (memory rule: never write
# under /home). Covers torch.hub (LPIPS AlexNet), HF hub, and transformers downloads.
_cfg = load_config()
_os.environ.setdefault("TORCH_HOME", _cfg.torch_cache)
for _k, _v in _cfg.hf_env().items():
    _os.environ.setdefault(_k, _v)

__all__ = ["load_config"]
__version__ = "0.1.0"

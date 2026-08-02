#!/usr/bin/env python
"""Orient-Anything-V2 batch inference (Stage R front detection, PRD_articulated_v2 section 6).

Runs inside the orianyv2 conda env with the Orient-Anything-V2 repo importable from --repo.
Follows the backend-adapter script contract: no pbr-texture-pipeline imports, load the model once,
process every image, and print one machine-readable result line:

  [PBR_RESULT] {"ok": true, "results": [{"image": ..., "az": ..., "el": ..., "ro": ...,
                                              "alpha": ..., "ok": true}, ...]}

az is degrees in [0, 360) (0 = object front faces the camera), el in [-90, 90),
ro in [-180, 180), alpha the symmetry estimate. Invoked by pbr_texture_pipeline.articulated.orient.
"""
from __future__ import annotations

import argparse
import json
import sys

RESULT_MARKER = "[PBR_RESULT]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="Orient-Anything-V2 repo directory")
    ap.add_argument("--ckpt", required=True, help="rotmod checkpoint path (.pt)")
    ap.add_argument("--images", nargs="+", required=True, help="input images, processed in order")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import torch
    from PIL import Image
    from utils.app_utils import inf_single_case
    from vision_tower import VGGT_OriAny_Ref

    use_cuda = torch.cuda.is_available()
    dtype = (torch.bfloat16 if use_cuda and torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)
    model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    model = model.to("cuda" if use_cuda else "cpu")
    print(f"[orient_infer] model loaded ({'cuda' if use_cuda else 'cpu'}, {dtype})", flush=True)

    results = []
    for path in args.images:
        try:
            pil = Image.open(path).convert("RGB")
            ans = inf_single_case(model, pil, None)
            results.append({
                "image": path, "ok": True,
                "az": float(ans["ref_az_pred"]), "el": float(ans["ref_el_pred"]),
                "ro": float(ans["ref_ro_pred"]), "alpha": float(ans["ref_alpha_pred"]),
            })
        except Exception as e:  # noqa: BLE001
            results.append({"image": path, "ok": False, "error": str(e)})
    print(f"{RESULT_MARKER} {json.dumps({'ok': True, 'results': results})}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Standalone joint-slider viewer for an articulated pbr-texture-pipeline job (WS6).

Thin stdlib http.server wrapper around pbr_texture_pipeline.articulated.viewer (the same core the Gradio
app mounts as FastAPI routes). CUDA-free; runs in any env with pbr-texture-pipeline importable.

    python scripts/articulated_viewer.py jobs/<job_id> --backend trellis2 --port 8090
    # then: ssh -L 8090:localhost:8090 <host>  ->  http://localhost:8090/
"""
from __future__ import annotations

import argparse
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbr_texture_pipeline.articulated import viewer as V  # noqa: E402
from pbr_texture_pipeline.jobdir import JobDir  # noqa: E402

_ctx: dict = {}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0].lstrip("/")
        if path in ("", "index.html", _ctx["backend"]):
            self._send(_ctx["html"].encode(), "text/html; charset=utf-8")
            return
        for kind in ("asset", "job"):
            prefix = kind + "/"
            if path.startswith(prefix):
                p = V.resolve_file(_ctx["job"], kind, path[len(prefix):])
                if p is None:
                    self.send_error(404)
                else:
                    self._send(p.read_bytes(), V.mime_for(p))
                return
        self.send_error(404)

    def _send(self, data: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Joint-slider viewer for an articulated job.")
    ap.add_argument("job_dir", help="path to the job dir, e.g. jobs/8930_door_urdf_...")
    ap.add_argument("--backend", default="trellis2")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    job = JobDir.load(args.job_dir)
    if not job.textured_urdf(args.backend).is_file():
        print(f"error: no textured URDF for backend {args.backend} "
              f"({job.textured_urdf(args.backend)})")
        return 1

    scene = V.build_scene_data(job, args.backend)
    _ctx.update({"job": job, "backend": args.backend, "html": V.make_html(scene)})

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("", args.port))
        except OSError:
            print(f"error: port {args.port} is already in use; try --port <N>")
            return 1

    host = socket.gethostname()
    movable = [n for n, j in scene["joints"].items() if j["type"] != "fixed"]
    print(f"job     : {job.job_id} ({scene['category']}, backend {args.backend})")
    print(f"links   : {len(scene['links'])}   joints: {len(scene['joints'])}   "
          f"movable: {', '.join(movable) or '-'}")
    print(f"local   : http://localhost:{args.port}")
    print(f"ssh     : ssh -L {args.port}:localhost:{args.port} {host}")
    print("Ctrl-C to stop.")
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

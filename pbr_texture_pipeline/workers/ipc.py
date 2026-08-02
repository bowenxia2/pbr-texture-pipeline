"""JSON-lines IPC between the Gradio orchestrator and persistent GPU workers (PRD section 7).

The orchestrator holds zero CUDA. Each worker is a subprocess in a specific conda env with a
pinned GPU (CUDA_VISIBLE_DEVICES). Protocol: one request `{op, args, id}` per line in, one
response `{ok, id, ...}` per line out.

Channel hygiene: model libraries print freely to stdout, which would corrupt a JSON-lines
protocol on the same stream. So a worker reserves the *real* stdout for protocol lines only
(each tagged with MARKER) and redirects Python-level `sys.stdout` to stderr for the duration
of the process; the client parses MARKER lines from the child's stdout and drains stderr into
a rolling log buffer it can stream into the UI.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections import deque
from typing import Any, Callable, Optional

MARKER = "@@BTX@@"          # protocol-line tag on the worker's real stdout


# --- worker side -------------------------------------------------------------
def serve(handlers: dict[str, Callable[[dict], dict]], name: str = "worker") -> None:
    """Run a worker: read request lines from stdin, dispatch to `handlers`, write responses.

    Reserved ops: `ping` -> {ok, ready:True}; `shutdown` -> clean exit. A handler returning a
    dict is wrapped as {ok:True, id, **result}; an exception becomes {ok:False, id, error}.
    """
    real_stdout = sys.stdout
    sys.stdout = sys.stderr        # library prints go to stderr; protocol owns real stdout

    def _emit(obj: dict) -> None:
        real_stdout.write(MARKER + " " + json.dumps(obj) + "\n")
        real_stdout.flush()

    _emit({"ok": True, "event": "ready", "name": name})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = req.get("id")
        op = req.get("op")
        args = req.get("args", {}) or {}
        if op == "shutdown":
            _emit({"ok": True, "id": rid, "event": "bye"})
            return
        if op == "ping":
            _emit({"ok": True, "id": rid, "ready": True})
            continue
        handler = handlers.get(op)
        if handler is None:
            _emit({"ok": False, "id": rid, "error": f"unknown op {op!r}"})
            continue
        try:
            result = handler(args) or {}
            _emit({"ok": True, "id": rid, **result})
        except Exception as e:  # noqa: BLE001
            import traceback
            _emit({"ok": False, "id": rid, "error": str(e),
                   "trace": traceback.format_exc()[-2000:]})


# --- orchestrator side -------------------------------------------------------
class WorkerError(RuntimeError):
    pass


class WorkerClient:
    """Manages one worker subprocess and does synchronous request/response calls.

    Thread-safe (a lock serializes calls) so concurrent Gradio callbacks can share one worker.
    """

    def __init__(self, module: str, env_name: str, cuda_visible: str,
                 conda_bin: str, extra_env: Optional[dict] = None,
                 log_lines: int = 400, name: Optional[str] = None):
        self.module = module
        self.env_name = env_name
        self.cuda_visible = cuda_visible
        self.conda_bin = conda_bin
        self.extra_env = extra_env or {}
        self.name = name or module.rsplit(".", 1)[-1]
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._log: deque[str] = deque(maxlen=log_lines)
        self._stderr_thread: Optional[threading.Thread] = None
        self._counter = 0

    # -- lifecycle --
    def start(self, ready_timeout: float = 600.0) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env.update(self.extra_env)
        cmd = [self.conda_bin, "run", "-n", self.env_name, "--no-capture-output",
               "python", "-u", "-m", self.module]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        ev = self._read_protocol(ready_timeout)
        if not (ev and ev.get("event") == "ready"):
            raise WorkerError(f"{self.name}: did not report ready ({ev})")

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def shutdown(self) -> None:
        if not self.is_alive():
            self.proc = None
            return
        try:
            self._write({"op": "shutdown"})
            self.proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass
        self.proc = None

    # -- calls --
    def call(self, op: str, timeout: float = 1800.0, **args) -> dict:
        with self._lock:
            if not self.is_alive():
                raise WorkerError(f"{self.name}: worker is not running")
            self._counter += 1
            rid = self._counter
            self._write({"op": op, "args": args, "id": rid})
            while True:
                resp = self._read_protocol(timeout)
                if resp is None:
                    raise WorkerError(f"{self.name}: no response to {op!r} "
                                      f"(worker died?)\n{self.tail_log()}")
                if resp.get("event") in ("ready", "bye"):
                    continue
                if resp.get("id") != rid:
                    continue
                if not resp.get("ok"):
                    raise WorkerError(f"{self.name}.{op} failed: {resp.get('error')}")
                return resp

    def ping(self, timeout: float = 30.0) -> bool:
        try:
            return bool(self.call("ping", timeout=timeout).get("ready"))
        except Exception:  # noqa: BLE001
            return False

    # -- internals --
    def _write(self, obj: dict) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _read_protocol(self, timeout: float) -> Optional[dict]:
        """Read the next MARKER-tagged line from the child's stdout (blocking up to timeout)."""
        assert self.proc and self.proc.stdout
        # Popen stdout has no per-read timeout; rely on a watchdog thread to kill on hard hangs.
        deadline = threading.Event()
        timer = threading.Timer(timeout, deadline.set)
        timer.start()
        try:
            while not deadline.is_set():
                line = self.proc.stdout.readline()
                if line == "":            # EOF: worker exited
                    return None
                line = line.rstrip("\n")
                if line.startswith(MARKER):
                    return json.loads(line[len(MARKER):].strip())
                if line:
                    self._log.append(line)
            return None
        finally:
            timer.cancel()

    def _drain_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            self._log.append(line.rstrip("\n"))

    def tail_log(self, n: int = 40) -> str:
        lines = list(self._log)[-n:]
        return "\n".join(lines)

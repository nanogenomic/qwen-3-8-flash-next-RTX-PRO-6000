#!/usr/bin/env python3
# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Minimal harness for the two dashboards: poll providers, serve the wall.

    ./serve.py                            # http://127.0.0.1:8770
    ./serve.py --port 9000 --host 0.0.0.0
    ./serve.py --providers gpu,gpu_fleet   # just the GPU side

Each provider runs on its own cadence in its own thread. The browser gets the
whole composed state as JSON on `/api/state` and as a Server-Sent Events stream
on `/events`; the renderers in `static/views.js` draw from that and nothing
else. One pane is pinned to the highest-priority panel and the other rotates
through the rest, which is what the panel `priority`/`group`/`rank` fields are
for.

⛔ ONE PANE MEANS NO ROTATION. `panes` defaults to 2. If you set it to 1 and
leave `pin_primary` on, the single pane is pinned and nothing on it ever
changes; with one pane, `pin_primary` is forced off here.

Config: `config.json` next to this file overrides DEFAULTS, and a handful of
env vars (TOKENS_FILE, TPUT_SAMPLES_DIR, GPU_FLEET, DASH_PORT) override the
paths that usually differ per machine. Nothing is read from anywhere else.
"""
from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

DEFAULTS: dict = {
    "host": "127.0.0.1",
    "port": int(os.environ.get("DASH_PORT", 8770)),
    # Which providers this instance runs.
    "providers": ["gpu", "gpu_fleet", "tokens"],
    # How many panes the browser lays out side by side.
    "panes": 2,
    # Pane 0 holds the highest-priority panel and never rotates.
    "pin_primary": True,
    "rotate_seconds": 95,
    "tick_seconds": 2.0,
    "gpu": {
        "interval": 2.0,
        "history": 300,
        "mounts": ["/"],
    },
    "gpu_fleet": {
        # '<label>=local' or '<label>=ssh:<host>', comma separated. Default is
        # this host alone; see tools/fleet_probe.py.
        "fleet": os.environ.get("GPU_FLEET", ""),
        "interval": 45.0,
    },
    "tokens": {
        "file": os.environ.get("TOKENS_FILE", str(HERE / "var" / "tokens.json")),
        "stale_after_s": 900,
        "interval": 10.0,
        "tput_samples_dir": os.environ.get(
            "TPUT_SAMPLES_DIR", str(HERE / "var" / "samples")),
        "tput_stale_after_s": 90,
    },
}


def load_config(path: Path | None = None) -> dict:
    def merge(base: dict, over: dict) -> dict:
        out = copy.deepcopy(base)
        for k, v in (over or {}).items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(
                out.get(k), dict) else v
        return out

    p = path or (HERE / "config.json")
    user = {}
    if p.exists():
        try:
            user = json.loads(p.read_text())
        except Exception as exc:
            print(f"config {p} ignored: {exc}")
    return merge(DEFAULTS, user)


def build_providers(cfg: dict):
    from providers.gpu import GpuProvider
    from providers.gpu_fleet import GpuFleetProvider
    from providers.tokens import TokensProvider

    catalogue = {
        "gpu": GpuProvider,
        "gpu_fleet": GpuFleetProvider,
        "tokens": TokensProvider,
    }
    out = []
    for name in cfg["providers"]:
        klass = catalogue.get(name)
        if klass is None:
            print(f"unknown provider {name!r}; known: {', '.join(catalogue)}")
            continue
        out.append(klass(cfg))
    return out


class Registry:
    """Polls providers, keeps the latest panels, composes the wall state."""

    def __init__(self, cfg: dict, providers: list):
        self.cfg = cfg
        self.providers = providers
        self.panels: dict[str, dict] = {}
        self.order: list[str] = []
        self.errors: dict[str, str] = {}
        self.lock = threading.Lock()
        self.rotate_at = time.time() + cfg["rotate_seconds"]
        self.rotor = 0
        self._stop = threading.Event()

    def start(self):
        for p in self.providers:
            threading.Thread(target=self._loop, args=(p,), daemon=True).start()

    def _loop(self, p):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                panels = [x.to_dict() for x in p.poll()]
                with self.lock:
                    # Drop this provider's previous panels, then re-add: a
                    # provider that stops emitting a panel means that thing is
                    # no longer running, and the wall must stop showing it.
                    self.panels = {k: v for k, v in self.panels.items()
                                   if v["provider"] != p.id}
                    for panel in panels:
                        self.panels[panel["key"]] = panel
                    self.order = [k for k in self.order if k in self.panels]
                    self.order += [k for k in self.panels if k not in self.order]
                    self.errors.pop(p.id, None)
            except Exception as exc:
                with self.lock:
                    self.errors[p.id] = f"{type(exc).__name__}: {exc}"[:300]
            self._stop.wait(max(0.25, p.interval - (time.time() - t0)))

    def ribbon(self) -> dict:
        """The always-on status strip, from the gpu provider's panel if present."""
        with self.lock:
            host = next((v for v in self.panels.values() if v["view"] == "gpu.host"), None)
        if not host:
            return {}
        d = host["data"]
        gpus = d.get("gpus") or []
        g0 = gpus[0] if gpus else {}
        h = d.get("history", {})
        g0h = (h.get("gpu") or {}).get(str(g0.get("index", 0)), {})
        return {
            "host": d.get("host"),
            "gpu_util": g0.get("utilization_gpu"),
            "gpu_mem_used": sum(g.get("memory_used") or 0 for g in gpus) or None,
            "gpu_mem_total": sum(g.get("memory_total") or 0 for g in gpus) or None,
            "gpu_temp": g0.get("temperature_gpu"),
            "gpu_power": sum(g.get("power_draw") or 0 for g in gpus) or None,
            "gpu_power_limit": sum(g.get("power_limit") or 0 for g in gpus) or None,
            "cpu_pct": d.get("cpu_pct"),
            "ram_used": d.get("ram_used"),
            "ram_total": d.get("ram_total"),
            "net_rx": d.get("net_rx"),
            "net_tx": d.get("net_tx"),
            "history": {"gpu_util": g0h.get("util"), "gpu_mem": g0h.get("mem"),
                        "cpu": h.get("cpu")},
        }

    def state(self) -> dict:
        now = time.time()
        n_panes = max(1, int(self.cfg["panes"]))
        pin = bool(self.cfg["pin_primary"]) and n_panes > 1
        with self.lock:
            panels = [self.panels[k] for k in self.order if k in self.panels]
            errors = dict(self.errors)
        panels.sort(key=lambda p: (-p["priority"], p["group"], p["rank"]))
        if not panels:
            panels = [{"provider": "-", "view": "__fallback", "title": "no panels yet",
                       "subtitle": "; ".join(f"{k}: {v}" for k, v in errors.items()),
                       "priority": 0, "key": "-", "group": "-", "rank": 0,
                       "state": "error" if errors else "idle", "data": {}, "ts": now}]
        rotating = panels[1:] if pin and len(panels) > 1 else panels
        if now >= self.rotate_at and rotating:
            self.rotor = (self.rotor + 1) % len(rotating)
            self.rotate_at = now + self.cfg["rotate_seconds"]
        groups: dict[str, list] = {}
        for p in panels:
            groups.setdefault(p["group"], []).append(p["key"])
        assigned = []
        for i in range(n_panes):
            if pin and i == 0:
                panel, rot = panels[0], False
            else:
                if not rotating:
                    continue
                k = (self.rotor + (i - (1 if pin else 0))) % len(rotating)
                panel, rot = rotating[k], len(rotating) > 1
            assigned.append({
                "pane": {"id": i, "label": f"pane {i}"},
                "panel": panel,
                "rotating": rot,
                "rotate_seconds": self.cfg["rotate_seconds"],
                "rotate_in": max(0.0, self.rotate_at - now) if rot else 0.0,
            })
        return {
            "now": now,
            "ribbon": self.ribbon(),
            "panes": assigned,
            "groups": [{"group": g, "panels": ks} for g, ks in groups.items()],
            "primary_group": panels[0]["group"],
            "frozen": False,
            "picker": {"open": False, "items": [], "index": 0},
            "errors": errors,
        }


class Handler(BaseHTTPRequestHandler):
    registry: Registry = None        # set in main()
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):       # keep the console for provider errors
        pass

    def _send(self, code, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._file(STATIC / "index.html")
        if path == "/api/state":
            body = json.dumps(self.registry.state(), default=str).encode()
            return self._send(200, body, "application/json")
        if path == "/events":
            return self._events()
        if path.startswith("/static/"):
            # Resolve inside STATIC and refuse anything that escapes it.
            target = (STATIC / path[len("/static/"):]).resolve()
            if STATIC.resolve() in target.parents and target.is_file():
                return self._file(target)
            return self._send(404, b"not found", "text/plain")
        return self._send(404, b"not found", "text/plain")

    def _file(self, p: Path):
        try:
            body = p.read_bytes()
        except OSError:
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, body, ctype)

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        tick = float(self.registry.cfg["tick_seconds"])
        try:
            while True:
                payload = json.dumps(self.registry.state(), default=str)
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
                time.sleep(tick)
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--providers", default=None,
                    help="comma separated subset, e.g. gpu,gpu_fleet")
    ap.add_argument("--panes", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.host:
        cfg["host"] = a.host
    if a.port:
        cfg["port"] = a.port
    if a.panes:
        cfg["panes"] = a.panes
    if a.providers:
        cfg["providers"] = [x.strip() for x in a.providers.split(",") if x.strip()]

    os.chdir(HERE)                       # so `providers` imports resolve
    import sys
    sys.path.insert(0, str(HERE))

    reg = Registry(cfg, build_providers(cfg))
    reg.start()
    Handler.registry = reg
    srv = ThreadingHTTPServer((cfg["host"], cfg["port"]), Handler)
    srv.daemon_threads = True
    print(f"dashboards on http://{cfg['host']}:{cfg['port']}  "
          f"providers: {', '.join(cfg['providers'])}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fleet accelerator inventory panel.

A thin wrapper around `tools/fleet_probe.py`: the probe does the enumeration,
this turns it into a Panel and keeps the probe OFF the poll thread's critical
path by caching the last result for `interval` seconds.

The probe may shell out over ssh, so this provider's interval is deliberately
slow (45 s by default). Nothing here retries, smooths, or remembers a reading
across a failure: a host that stopped answering shows as UNREACHABLE on the
very next poll, which is the whole point of the probe.
"""
from __future__ import annotations

import os
import sys
import time

from .base import Panel, Provider

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))
from fleet_probe import probe_fleet  # noqa: E402


class GpuFleetProvider(Provider):
    id = "gpu_fleet"
    title = "GPU fleet"
    interval = 45.0

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._spec = self.conf.get("fleet") or os.environ.get("GPU_FLEET")

    def poll(self) -> list[Panel]:
        started = time.time()
        try:
            d = probe_fleet(self._spec)
        except Exception as exc:                 # the panel must survive a probe bug
            return [Panel(
                provider=self.id, view="gpu.fleet", title="GPU fleet",
                subtitle="probe failed", priority=14.0, key="gpu_fleet:inventory",
                group="gpu", rank=1, state="error",
                data={"error": f"{type(exc).__name__}: {exc}"[:300],
                      "probed_at": started})]
        accel = d.get("accelerators", [])
        live = [a for a in accel if a.get("status") == "live"]
        down = [a for a in accel if a.get("status") == "UNREACHABLE"]
        nogpu = [a for a in accel if a.get("status") == "no-gpu"]
        hosts = d.get("hosts", [])
        ok_hosts = sum(1 for h in hosts if h.get("ok"))
        sub = f"{len(live)} GPU(s) live on {ok_hosts}/{len(hosts)} host(s)"
        if nogpu:
            sub += f" · {len(nogpu)} host(s) with no CUDA device"
        if down:
            sub += f" · {len(down)} host(s) UNREACHABLE"
        return [Panel(
            provider=self.id, view="gpu.fleet", title="GPU fleet",
            subtitle=sub, priority=14.0, key="gpu_fleet:inventory",
            group="gpu", rank=1,
            state="error" if down else ("live" if live else "idle"),
            data={
                "probed_at": d.get("probed_at"),
                "probe_s": time.time() - started,
                "hosts": hosts,
                "accelerators": accel,
                "n_live": len(live), "n_unreachable": len(down), "n_nogpu": len(nogpu),
                "total_vram_mb": sum(a.get("total_vram_mb") or 0 for a in live) or None,
                "used_vram_mb": sum(a.get("observed_used_vram_mb") or 0 for a in live) or None,
            })]

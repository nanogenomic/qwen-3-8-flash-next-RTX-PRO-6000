# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host + GPU telemetry, with the card set DISCOVERED on every poll.

Everything here is measured. `nvidia-smi` is the only GPU source; if it fails
the fields are reported absent rather than zero-filled.

⛔ NOTHING IN THIS FILE KNOWS HOW MANY CARDS THE BOX HAS. The card set comes
from `nvidia-smi --query-gpu` each poll, per-card history is created lazily the
first time an index is seen, and a card that disappears from a poll gets a
`None` sample rather than being dropped or carried forward. Two consequences
that were both learned the hard way:

  * a card installed (or a driver reset) while this is running appears on the
    wall without a restart and without an edit;
  * a hot-added card's history is LEFT-PADDED to the timeline that already
    exists. Callers zip these series together, and unequal lengths left-align,
    which pairs the new card's NEWEST sample against the incumbent's OLDEST.

The per-process table joins `--query-compute-apps` by GPU UUID, not by index,
because indices are an enumeration order and UUIDs are an identity.
"""
from __future__ import annotations

import collections
import os
import shutil
import subprocess
import time

import psutil

from .base import Panel, Provider

_GPU_FIELDS = [
    "index", "uuid", "name", "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total", "temperature.gpu",
    "power.draw", "power.limit", "clocks.sm", "clocks.mem", "fan.speed",
]


def _nvidia_smi(args: list[str], timeout: float = 4.0) -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _num(tok: str):
    tok = tok.strip()
    if tok in ("", "N/A", "[N/A]", "[Not Supported]"):
        return None
    try:
        return float(tok)
    except ValueError:
        return tok


class GpuProvider(Provider):
    id = "gpu"
    title = "Host"
    interval = 2.0

    # Mounts to report free space for. Override in config.json.
    DEFAULT_MOUNTS = ("/",)

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._hist_n = int(self.conf.get("history", 300))
        self._mounts = tuple(self.conf.get("mounts") or self.DEFAULT_MOUNTS)
        n = self._hist_n
        self.hist = {
            "t": collections.deque(maxlen=n),
            "cpu": collections.deque(maxlen=n),
            "ram": collections.deque(maxlen=n),
        }
        # Per-GPU history, keyed by nvidia-smi index, built lazily so a card
        # installed later (or one that drops out of a poll) never forces a
        # fixed GPU count.
        self.gpu_hist: dict[int, dict[str, collections.deque]] = {}
        self._net0 = psutil.net_io_counters()
        self._net_t = time.time()
        psutil.cpu_percent(interval=None)

    def _gpu_hist_for(self, idx: int) -> dict[str, collections.deque]:
        h = self.gpu_hist.get(idx)
        if h is None:
            # Left-pad to the timeline that already exists, so EVERY per-GPU
            # deque stays the same length as hist["t"]. hist["t"] has already
            # taken this tick when we get here, hence the -1.
            pad = [None] * max(0, len(self.hist["t"]) - 1)
            h = {k: collections.deque(pad, maxlen=self._hist_n)
                 for k in ("util", "mem", "temp", "power")}
            self.gpu_hist[idx] = h
        return h

    # -- GPU ------------------------------------------------------------
    def _gpus(self) -> list[dict]:
        raw = _nvidia_smi(
            ["--query-gpu=" + ",".join(_GPU_FIELDS), "--format=csv,noheader,nounits"]
        )
        if raw is None:
            return []
        gpus = []
        for line in raw.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(_GPU_FIELDS):
                continue
            g = {f.replace(".", "_"): _num(p) for f, p in zip(_GPU_FIELDS, parts)}
            # ⛔ _num() floats EVERY numeric field, so index arrives as 0.0.
            # The per-GPU history below is keyed str(int(index)) -> "0", while a
            # consumer doing str(g["index"]) gets "0.0" and misses every lookup.
            # That silently blanked the GPU sparklines on a live wall. Coerce
            # once, here, so there is ONE source of truth.
            if g.get("index") is not None:
                g["index"] = int(g["index"])
            gpus.append(g)
        return gpus

    def _gpu_procs(self, uuid_to_index: dict[str, int] | None = None) -> list[dict]:
        uuid_to_index = uuid_to_index or {}
        raw = _nvidia_smi(
            ["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]
        )
        procs: list[dict] = []
        if raw is None:
            return procs
        for line in raw.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                pid = int(parts[1])
                mem = float(parts[2])
            except ValueError:
                continue
            entry = {
                "pid": pid, "mem_mib": mem, "label": f"pid {pid}", "kind": "other",
                "gpu": uuid_to_index.get(parts[0]),
            }
            try:
                p = psutil.Process(pid)
                cmd = p.cmdline()
                entry["cmd"] = " ".join(cmd)[:400]
                entry["started"] = p.create_time()
                entry["cpu"] = p.cpu_percent(None)
                entry["rss_mib"] = p.memory_info().rss / 1048576.0
                entry.update(classify(cmd, p))
            except Exception:
                # A process owned by another user, or one that exited between
                # the two nvidia-smi calls. The VRAM row is still true.
                pass
            procs.append(entry)
        procs.sort(key=lambda e: -e["mem_mib"])
        return procs

    # -- poll -----------------------------------------------------------
    def poll(self) -> list[Panel]:
        now = time.time()
        gpus = self._gpus()
        uuid_to_index = {g["uuid"]: g["index"] for g in gpus
                         if g.get("uuid") and g.get("index") is not None}
        procs = self._gpu_procs(uuid_to_index)

        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        try:
            load = os.getloadavg()
        except OSError:
            load = (0.0, 0.0, 0.0)

        net = psutil.net_io_counters()
        dt = max(1e-3, now - self._net_t)
        net_rx = (net.bytes_recv - self._net0.bytes_recv) / dt
        net_tx = (net.bytes_sent - self._net0.bytes_sent) / dt
        self._net0, self._net_t = net, now

        self.hist["t"].append(now)
        self.hist["cpu"].append(cpu)
        self.hist["ram"].append(vm.percent)
        seen = set()
        for g in gpus:
            idx = int(g.get("index") if g.get("index") is not None else 0)
            gh = self._gpu_hist_for(idx)
            seen.add(idx)
            gh["util"].append(g.get("utilization_gpu"))
            mtot = g.get("memory_total")
            gh["mem"].append((g.get("memory_used") or 0) / mtot * 100 if mtot else None)
            gh["temp"].append(g.get("temperature_gpu"))
            gh["power"].append(g.get("power_draw"))
        # A card known from an earlier poll but MISSING from this one still has
        # to advance, or its series drifts one sample out of step with the other
        # card's for the rest of the process's life and nothing can detect it.
        # None records "no reading", which is the truth.
        for idx, gh in self.gpu_hist.items():
            if idx not in seen:
                for q in gh.values():
                    q.append(None)

        disks = []
        for mp in self._mounts:
            try:
                u = psutil.disk_usage(mp)
                disks.append({"mount": mp, "used": u.used, "total": u.total, "pct": u.percent})
            except Exception:
                continue

        data = {
            "host": os.uname().nodename,
            "gpus": gpus,
            "gpu_procs": procs[:12],
            "cpu_pct": cpu,
            "cpu_count": psutil.cpu_count(),
            "load": list(load),
            "ram_used": vm.used,
            "ram_total": vm.total,
            "ram_pct": vm.percent,
            "net_rx": net_rx,
            "net_tx": net_tx,
            "disks": disks,
            "uptime": now - psutil.boot_time(),
            "history": {
                **{k: list(v) for k, v in self.hist.items()},
                "gpu": {str(idx): {k: list(v) for k, v in gh.items()}
                        for idx, gh in self.gpu_hist.items()},
            },
        }
        if not gpus:
            sub = "GPU telemetry unavailable"
        elif len(gpus) == 1:
            g0 = gpus[0]
            sub = f"{g0.get('name', 'GPU')} · {int(g0.get('utilization_gpu') or 0)}% · " \
                  f"{int((g0.get('memory_used') or 0) / 1024)}/" \
                  f"{int((g0.get('memory_total') or 0) / 1024)} GiB"
        else:
            util_bits = " / ".join(f"{int(g.get('utilization_gpu') or 0)}%" for g in gpus)
            mem_used = sum(g.get("memory_used") or 0 for g in gpus)
            mem_tot = sum(g.get("memory_total") or 0 for g in gpus)
            sub = f"{len(gpus)}x {gpus[0].get('name', 'GPU')} · {util_bits} · " \
                  f"{int(mem_used / 1024)}/{int(mem_tot / 1024)} GiB"
        return [
            Panel(
                provider=self.id,
                view="gpu.host",
                title=os.uname().nodename,
                subtitle=sub,
                priority=20.0,
                key="gpu:host",
                group="gpu",
                rank=0,
                state="live" if gpus else "error",
                data=data,
            )
        ]


# Substring -> label for the GPU process table. Purely cosmetic: an unmatched
# process is still listed, under its argv[0]. Add your own workloads here.
WORKLOAD_LABELS: tuple[tuple[str, str, str], ...] = (
    ("sglang", "serving", "SGLang"),
    ("vllm", "serving", "vLLM"),
    ("trtllm", "serving", "TensorRT-LLM"),
    ("torchrun", "training", "Training"),
    ("train", "training", "Training"),
)


def classify(cmd: list[str], proc) -> dict:
    """Name the workload behind a GPU process from its argv and cwd."""
    joined = " ".join(cmd).lower()
    kind, label = "other", os.path.basename(cmd[0] if cmd else "?")
    try:
        cwd = proc.cwd()
    except Exception:
        cwd = ""
    for needle, k, lbl in WORKLOAD_LABELS:
        if needle in joined:
            kind, label = k, lbl
            break
    return {"kind": kind, "label": label, "cwd": cwd}

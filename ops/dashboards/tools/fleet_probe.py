#!/usr/bin/env python3
# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Live accelerator inventory for a set of hosts, from nvidia-smi.

WHY THIS EXISTS. The inventory it replaced was a database table, written by
whichever agent happened to run a probe, on whichever host that agent ran on.
Exactly one host ever wrote to it, and it stopped writing months before anyone
looked, so a straight SELECT reported a ONE-GPU fleet frozen at the last
heartbeat -- while the card doing essentially all of the work was absent. A
dashboard was rendering that as the live fleet.

⛔ THIS MODULE NEVER INVENTS A CARD. Every field comes from an nvidia-smi line
read seconds ago. A host that does not answer produces an UNREACHABLE row
carrying the attempt timestamp and the error, never a zero-GPU host and never a
remembered reading. **An absent host is not a host with no GPUs** -- and a host
whose nvidia-smi answers but lists nothing IS a real observation of a GPU-less
host, reported as `no-gpu`, which is a third state.

READ-ONLY EVERYWHERE. The remote arm is `ssh <host> nvidia-smi --query-gpu`; it
touches no job, allocates no memory, and changes no GPU state.

FLEET SPEC. `GPU_FLEET`, comma separated, each entry `<label>=<transport>` where
transport is `local` or `ssh:<ssh-host>`. Default: this host, locally, only.

    GPU_FLEET='box=local'                       # just me (default)
    GPU_FLEET='box=local,rig2=ssh:rig2'         # me plus one over ssh
    GPU_FLEET='a=ssh:node-a,b=ssh:node-b'       # neither of them is me

Run it directly to print the inventory as JSON.
"""
from __future__ import annotations

import concurrent.futures
import os
import platform
import subprocess
import time

QUERY = ("index,name,uuid,memory.total,memory.used,"
         "utilization.gpu,temperature.gpu")
SMI = ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"]

DEFAULT_FLEET = f"{platform.node()}=local"

# ssh must never prompt and never hang the caller's poll.
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
            "-o", "StrictHostKeyChecking=accept-new"]


def parse_fleet(spec: str | None = None) -> list[tuple[str, str]]:
    """[(label, transport)] from the fleet spec. Malformed entries are dropped."""
    out: list[tuple[str, str]] = []
    for chunk in (spec or os.environ.get("GPU_FLEET") or DEFAULT_FLEET).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        label, _, transport = chunk.partition("=")
        label, transport = label.strip(), (transport or "local").strip()
        if label:
            out.append((label, transport))
    return out


def _argv(transport: str) -> list[str]:
    if transport == "local":
        return SMI
    if transport.startswith("ssh:"):
        return ["ssh", *SSH_OPTS, transport[4:], " ".join(SMI)]
    raise ValueError(f"unsupported transport {transport!r}")


def _row(host: str, index: str, fields: list[str], observed_at: float) -> dict:
    name, uuid, total, used, util, temp = fields

    def num(v, cast=float):
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None

    return {
        "id": f"{host}_gpu_{index}",
        "host": host,
        "index": num(index, int),
        "name": name,
        "uuid": uuid,
        "total_vram_mb": num(total, int),
        "observed_used_vram_mb": num(used, int),
        "utilization": num(util),
        "temperature_c": num(temp),
        "source": "nvidia-smi",
        "reachable": True,
        "status": "live",
        "observed_at": observed_at,
        "age_s": 0.0,
    }


def probe_host(host: str, transport: str, timeout: float = 12.0) -> dict:
    """One host. Returns {host, transport, ok, observed_at, accelerators|error}."""
    started = time.time()
    try:
        proc = subprocess.run(_argv(transport), capture_output=True, text=True,
                              timeout=timeout)
    except Exception as exc:                       # timeout, no ssh binary, ...
        return {"host": host, "transport": transport, "ok": False,
                "observed_at": started,
                "error": f"{type(exc).__name__}: {exc}"[:200],
                "accelerators": []}
    observed_at = time.time()
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"host": host, "transport": transport, "ok": False,
                "observed_at": observed_at,
                "error": (err[-1] if err else f"exit {proc.returncode}")[:200],
                "accelerators": []}
    accelerators = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        accelerators.append(_row(host, parts[0], parts[1:], observed_at))
    if not accelerators:
        # nvidia-smi answered but listed nothing. That IS a real observation of
        # a GPU-less host -- distinct from UNREACHABLE, and reported as such.
        return {"host": host, "transport": transport, "ok": True,
                "observed_at": observed_at, "accelerators": [],
                "note": "no CUDA devices"}
    return {"host": host, "transport": transport, "ok": True,
            "observed_at": observed_at, "accelerators": accelerators}


def unreachable_row(result: dict) -> dict:
    """A placeholder that states the absence. No name, no VRAM, no utilisation."""
    return {
        "id": f"{result['host']}_unreachable",
        "host": result["host"],
        "index": None,
        "name": f"{result['host']} — UNREACHABLE",
        "uuid": None,
        "total_vram_mb": None,
        "observed_used_vram_mb": None,
        "utilization": None,
        "temperature_c": None,
        "source": "nvidia-smi",
        "reachable": False,
        "status": "UNREACHABLE",
        "observed_at": result.get("observed_at"),
        "error": result.get("error", "no response"),
    }


def probe_fleet(spec: str | None = None) -> dict:
    """Probe every host in parallel. Never raises; unreachable hosts are reported."""
    fleet = parse_fleet(spec)
    if not fleet:
        return {"probed_at": time.time(), "hosts": [], "accelerators": []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(fleet))) as pool:
        results = list(pool.map(lambda hp: probe_host(*hp), fleet))
    accelerators: list[dict] = []
    for result in results:
        if result["ok"]:
            accelerators.extend(result["accelerators"])
            if not result["accelerators"]:
                accelerators.append({
                    "id": f"{result['host']}_none", "host": result["host"],
                    "index": None,
                    "name": f"{result['host']} — no CUDA device", "uuid": None,
                    "total_vram_mb": None, "observed_used_vram_mb": None,
                    "utilization": None, "temperature_c": None,
                    "source": "nvidia-smi", "reachable": True, "status": "no-gpu",
                    "observed_at": result["observed_at"],
                })
        else:
            accelerators.append(unreachable_row(result))
    return {
        "probed_at": time.time(),
        "hosts": [{k: v for k, v in r.items() if k != "accelerators"} | {
            "n_accelerators": len(r["accelerators"])} for r in results],
        "accelerators": accelerators,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(probe_fleet(), indent=2, default=str))

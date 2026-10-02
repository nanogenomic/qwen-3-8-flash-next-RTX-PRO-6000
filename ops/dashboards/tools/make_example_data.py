#!/usr/bin/env python3
# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate OBVIOUSLY SYNTHETIC inputs, so the dashboards can be looked at
before a real ledger or sampler exists.

⛔ EVERY NUMBER THIS WRITES IS FABRICATED. The snapshot carries
`"synthetic": true` and the views render a SYNTHETIC DATA banner from it, which
is the only reason this script is allowed to exist in a tree whose entire point
is measured numbers. Never hand a chart made from this to anyone, and never
commit its output over a real snapshot.

    make_example_data.py --out-dir ./var          # tokens.json + samples/

The GPU dashboard has no synthetic mode and never will: it reads nvidia-smi, and
on a machine with no NVIDIA GPU it renders "GPU telemetry unavailable", which is
the truth.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import date, timedelta

FIELDS = ("input", "cached", "output", "calls")
FAMILIES = [("flashnext", "Flash-Next (SYNTHETIC)", ["example/flash-next-synthetic"]),
            ("q27b", "27B (SYNTHETIC)", ["example/27b-synthetic"])]
PROFILES = ["profile-1", "profile-2", "profile-3"]


def tokens_snapshot(days_back: int, seed: int) -> dict:
    rng = random.Random(seed)
    today = date.today()
    days: dict[str, dict] = {}
    profiles: dict[str, dict] = {p: {} for p in PROFILES}
    last = 0.0
    for i in range(days_back, -1, -1):
        d = today - timedelta(days=i)
        iso = d.isoformat()
        # a weekly rhythm plus a slow ramp, so the calendar has shape
        week = 0.45 if d.weekday() >= 5 else 1.0
        ramp = 0.3 + 0.7 * (days_back - i) / max(1, days_back)
        if rng.random() < 0.12:            # idle days exist
            continue
        row = {}
        for key, _, _ in FAMILIES:
            # 27B reports no cache on the reference deployment; keep that shape
            cache_reported = key != "q27b"
            calls = int(rng.uniform(40, 900) * week * ramp)
            inp = int(calls * rng.uniform(900, 4200))
            cached = int(inp * rng.uniform(0.35, 0.8)) if cache_reported else 0
            out = int(inp * rng.uniform(0.012, 0.035))
            row[key] = {"input": inp, "cached": cached, "output": out,
                        "calls": calls, "sessions": max(1, calls // 60),
                        "cache_reported": cache_reported}
            pf = profiles[rng.choice(PROFILES)].setdefault(key, {
                "input": 0, "cached": 0, "output": 0, "calls": 0,
                "sessions": 0, "cache_reported": cache_reported})
            for f, v in zip(FIELDS, (inp, cached, out, calls)):
                pf[f] += v
            pf["sessions"] += 1
        days[iso] = row
        last = time.mktime(d.timetuple()) + rng.uniform(0, 86000)
    return {
        "generated_at": time.time(),
        "synthetic": True,
        "tz": time.strftime("%Z"),
        "today": today.isoformat(),
        "source": "SYNTHETIC EXAMPLE — not a measurement",
        "attribution": "synthetic; no sessions were read",
        "semantics": {"input": "uncached prompt", "cached": "prompt cache read",
                      "output": "completion"},
        "families": [{"key": k, "label": lbl, "models": m} for k, lbl, m in FAMILIES]
                    + [{"key": "other", "label": "Other", "models": []}],
        "last_usage_at": last or None,
        "days": days,
        "profiles": profiles,
        "sources": [{"db": "SYNTHETIC", "profile": p, "rows": 0} for p in PROFILES],
    }


def sampler_rings(out_dir: str, hours: float, seed: int) -> list[str]:
    """Fabricate the VRAM/KV sampler's dated JSONL ring, 20 s cadence."""
    rng = random.Random(seed + 1)
    os.makedirs(out_dir, exist_ok=True)
    now = time.time()
    t = now - hours * 3600
    dec = pre = 0
    boot = 1000.0
    files: dict[str, list[str]] = {}
    while t <= now:
        # bursty: ~35% of the time a request is in flight
        busy = rng.random() < 0.35
        rate = rng.gauss(175, 22) if busy else 0.0
        rate = max(0.0, rate)
        if rng.random() < 0.004:            # an engine restart, occasionally
            boot += 1
            dec = pre = 0
        dec += int(rate * 20)
        pre += int(rate * 20 * rng.uniform(2, 9)) if busy else 0
        line = json.dumps({
            "ts_epoch": round(t, 1),
            "synthetic": True,
            "unit": {"main_start_monotonic": boot, "main_pid": 4242},
            "backend": {
                "ok": True,
                "running_reqs": (rng.randint(1, 3) if busy else 0),
                "queue_reqs": (rng.randint(0, 2) if busy and rng.random() < 0.2 else 0),
                "token_usage": round(rng.uniform(0.05, 0.62), 3),
                "tput": {"decode_tok_total": dec, "prefill_tok_total": pre,
                         "gen_throughput": round(rate * rng.uniform(0.9, 1.35), 1)
                                           if busy else 0.0},
            },
        }, separators=(",", ":"))
        day = time.strftime("%Y%m%d", time.gmtime(t))
        files.setdefault(day, []).append(line)
        t += 20
    written = []
    for day, lines in files.items():
        p = os.path.join(out_dir, f"vram-{day}.jsonl")
        with open(p, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        written.append(p)
    # the sampler also keeps a latest.json; mirror that shape
    with open(os.path.join(out_dir, "latest.json"), "w") as fh:
        json.dump(json.loads(lines[-1]), fh, indent=2)
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="./var")
    ap.add_argument("--days", type=int, default=70, help="days of token history")
    ap.add_argument("--hours", type=float, default=30.0, help="hours of sampler ring")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tok = os.path.join(a.out_dir, "tokens.json")
    with open(tok, "w") as fh:
        json.dump(tokens_snapshot(a.days, a.seed), fh, separators=(",", ":"))
    rings = sampler_rings(os.path.join(a.out_dir, "samples"), a.hours, a.seed)
    print(f"SYNTHETIC: {tok} ({a.days} d) + {len(rings)} sampler ring file(s) "
          f"in {os.path.join(a.out_dir, 'samples')}")
    print("These are fabricated numbers. The snapshot is flagged synthetic and "
          "the dashboard says so on screen.")
    print(f"math check: {math.floor(a.hours * 180)} samples at 20 s cadence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

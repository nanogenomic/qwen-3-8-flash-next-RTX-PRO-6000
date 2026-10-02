# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Agent token consumption per model family -- daily and monthly.

The agent framework's usage ledger lives on the CLIENT host; this dashboard runs
on the SERVING host. `tools/tokens_export.py` reads the ledger where it lives
and pushes a snapshot JSON here; this provider shapes that snapshot for the view
and ALWAYS carries its age, so a stalled exporter shows as `stale` rather than
as a quiet day. That distinction is the only reason the age is in the payload.

Config (`tokens` section of config.json, or the matching env var):

    file              path to the snapshot JSON          TOKENS_FILE
    stale_after_s     age at which the panel goes stale  (default 900)
    tput_samples_dir  the sampler ring directory         TPUT_SAMPLES_DIR
    tput_stale_after_s                                   (default 90)
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta

from .base import Panel, Provider
from .tput import TputSeries

FIELDS = ("input", "cached", "output", "calls")


class TokensProvider(Provider):
    id = "tokens"
    title = "Agent tokens"
    interval = 10.0

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # Live tok/s from the sampler ring -- tailed from disk, no request to
        # the engine. See providers/tput.py and ../../observability.md.
        self._tput = TputSeries(self.conf.get(
            "tput_samples_dir",
            os.environ.get("TPUT_SAMPLES_DIR", "./var/samples")))

    def _tput_data(self) -> dict:
        try:
            t = self._tput.snapshot()
        except Exception as exc:  # the calendar must survive a bad sample file
            return {"ok": False, "reason": f"tput: {str(exc)[:160]}"}
        t["stale_after_s"] = float(self.conf.get("tput_stale_after_s", 90))
        return t

    def _panel(self, state, subtitle, data):
        return Panel(
            provider=self.id, view="tokens.usage", title="Agent token usage",
            subtitle=subtitle, priority=18.0, key="tokens:overview",
            group="tokens", rank=0, state=state, data=data)

    def poll(self) -> list[Panel]:
        path = self.conf.get("file", os.environ.get(
            "TOKENS_FILE", "./var/tokens.json"))
        stale_after = float(self.conf.get("stale_after_s", 900))
        try:
            with open(path) as fh:
                d = json.load(fh)
        except FileNotFoundError:
            return [self._panel("error", "no snapshot yet",
                                {"error": f"{path} missing — run "
                                          "tools/tokens_export.py on the client host",
                                 "tput": self._tput_data()})]
        except Exception as exc:
            return [self._panel("error", "snapshot unreadable",
                                {"error": str(exc)[:200], "tput": self._tput_data()})]

        age = time.time() - float(d.get("generated_at") or 0)
        days = d.get("days", {})
        fams = [f for f in d.get("families", []) if f["key"] != "other"]
        keys = [f["key"] for f in fams]
        today = date.fromisoformat(d.get("today") or date.today().isoformat())

        # Contribution-graph layout (Sunday-first week columns), but only as far
        # back as the first recorded day -- a year of empty cells before the
        # ledger existed reads as a year of idleness. Floor of 4 weeks so a new
        # ledger still draws as a calendar.
        active = sorted(k for k, v in days.items() if any(v.get(f) for f in keys))
        first = date.fromisoformat(active[0]) if active else today
        first = min(first, today - timedelta(weeks=4))
        start = first - timedelta(days=(first.weekday() + 1) % 7)
        cal = []
        cur = start
        while cur <= today:
            iso = cur.isoformat()
            row = {"d": iso}
            for k in keys:
                v = days.get(iso, {}).get(k)
                row[k] = ([v[f] for f in FIELDS] + [int(bool(v["cache_reported"]))]
                          if v else None)
            cal.append(row)
            cur += timedelta(days=1)

        months: dict[str, dict] = {}
        for iso, fd in days.items():
            m = months.setdefault(iso[:7], {k: [0, 0, 0, 0] for k in keys})
            for k in keys:
                v = fd.get(k)
                if v:
                    for i, f in enumerate(FIELDS):
                        m[k][i] += v[f]
        profiles = {p: {k: [v[f] for f in FIELDS] for k, v in fd.items() if k in keys}
                    for p, fd in d.get("profiles", {}).items()}

        tv = days.get(today.isoformat(), {})
        tot_today = sum(tv.get(k, {}).get(f, 0)
                        for k in keys for f in ("input", "cached", "output"))
        state = "stale" if age > stale_after else ("live" if tot_today else "idle")
        return [self._panel(state, f"{d.get('source', '')} · snapshot {int(age)}s old", {
            "age_s": age, "stale": age > stale_after, "tz": d.get("tz"),
            "today": today.isoformat(), "families": fams, "fields": list(FIELDS),
            "calendar": cal, "first_day": active[0] if active else None,
            "months": dict(sorted(months.items())),
            "profiles": profiles, "attribution": d.get("attribution"),
            "semantics": d.get("semantics"), "last_usage_at": d.get("last_usage_at"),
            "sources": d.get("sources", []),
            "synthetic": bool(d.get("synthetic")),
            "tput": self._tput_data(),
        })]

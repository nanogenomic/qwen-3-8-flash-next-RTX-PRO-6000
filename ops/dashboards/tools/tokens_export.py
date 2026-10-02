#!/usr/bin/env python3
# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Agent token-consumption snapshot, for the token-usage dashboard.

Runs on the CLIENT host (where the agent sessions live), writes a JSON snapshot,
and optionally rsyncs it to the SERVING host (where the dashboard runs).

SOURCE OF TRUTH: the `session_model_usage` table in every agent state DB under
`$HERMES_HOME` (`state.db`, plus `profiles/*/state.db` -- see
[client rule 7](../../../clients/README.md#7-isolate-test-state-from-live-state)
for why test state is a separate home). Each row is one (session, model) pair
with cumulative token counts and a `first_seen..last_seen` span:

    CREATE TABLE session_model_usage (
      session_id TEXT, model TEXT, api_call_count INTEGER,
      input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER,
      first_seen REAL, last_seen REAL);
    CREATE TABLE messages (session_id TEXT, role TEXT, timestamp REAL, ...);

Any orchestrator that records those columns works; point `--db` at it. Nothing
is written -- every DB is opened `mode=ro`.

⛔ TOKEN SEMANTICS. `input_tokens` is the prompt tokens NOT served from cache
(prompt_total minus cached), `cache_read_tokens` is the prefix-cache share, and
`output_tokens` is the completion. An engine that never sends
`prompt_tokens_details.cached_tokens` reports cache 0 for every row; that is
**"cache not reported", not "0% hit rate"**, and the snapshot flags it per
model-family per day so the dashboard can render `n/r` instead of a lie.

⛔ A MULTI-DAY SESSION IS NOT A FLAT DAY. A row whose span sits inside one day
goes to that day. A row spanning days is split in proportion to the session's
assistant turns (`messages.role='assistant'`) timestamped inside the row's
span -- each turn is one API call, so this tracks where the tokens were actually
spent. On the reference deployment one long session carried 1.4B tokens over 14
days; a uniform split painted that as flat daily use. Uniform is the FALLBACK,
used only when the session has no turns inside the window.

Usage:

    tokens_export.py --out /tmp/tokens.json
    tokens_export.py --out /tmp/tokens.json --push user@serving-host:/srv/dash/var/tokens.json
    HERMES_HOME=~/.myagent tokens_export.py --out /tmp/tokens.json

Model families are matched by substring on the SERVED model name, lowest match
first, and anything unmatched lands in `other`. Override with --families:

    --families 'flashnext=Flash-Next:flash-next,q27b=27B:27b'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime, timedelta

AGENT_HOME = os.path.expanduser(os.environ.get("HERMES_HOME", "~/.hermes"))

# (family key, display label, substring matched against the served model name)
DEFAULT_FAMILIES = [
    ("flashnext", "Flash-Next", "flash-next"),
    ("q27b", "27B", "27b"),
]
FIELDS = ("input", "cached", "output", "calls")


def parse_families(spec: str | None):
    """'key=Label:substr,key2=Label2:substr2' -> [(key, label, substr)]."""
    if not spec:
        return list(DEFAULT_FAMILIES)
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, _, rest = chunk.partition("=")
        label, _, substr = rest.partition(":")
        if key and substr:
            out.append((key.strip(), (label or key).strip(), substr.strip().lower()))
    return out or list(DEFAULT_FAMILIES)


def family_of(model: str, families) -> str:
    m = (model or "").lower()
    for key, _, substr in families:
        if substr in m:
            return key
    return "other"


def day_split(t0: float, t1: float):
    """Yield (YYYY-MM-DD, fraction) covering [t0, t1] in local time."""
    if not t0 and not t1:
        return
    t0 = t0 or t1
    t1 = max(t1 or t0, t0)
    if t1 - t0 < 1:
        yield datetime.fromtimestamp(t1).strftime("%Y-%m-%d"), 1.0
        return
    span = t1 - t0
    cur = datetime.fromtimestamp(t0)
    end = datetime.fromtimestamp(t1)
    while cur < end:
        nxt = min(datetime(cur.year, cur.month, cur.day) + timedelta(days=1), end)
        yield cur.strftime("%Y-%m-%d"), (nxt - cur).total_seconds() / span
        cur = nxt


def turn_split(con, sid: str, t0: float, t1: float):
    """Split by assistant-turn timestamps; falls back to day_split."""
    d0 = datetime.fromtimestamp(t0 or t1).date()
    d1 = datetime.fromtimestamp(t1 or t0).date()
    if d0 == d1:
        yield d1.strftime("%Y-%m-%d"), 1.0
        return
    counts: dict[str, int] = {}
    try:
        rows = con.execute(
            "select timestamp from messages where session_id=? and role='assistant' "
            "and timestamp between ? and ?", (sid, t0 - 1, t1 + 1))
    except sqlite3.Error:
        # No `messages` table in this ledger: time-proportional is all we have.
        yield from day_split(t0, t1)
        return
    for (ts,) in rows:
        day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        counts[day] = counts.get(day, 0) + 1
    n = sum(counts.values())
    if not n:
        yield from day_split(t0, t1)
        return
    for day, c in sorted(counts.items()):
        yield day, c / n


def empty() -> dict:
    return {k: 0 for k in FIELDS} | {"sessions": 0, "cache_reported": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/agent-tokens.json")
    ap.add_argument("--db", action="append", default=[],
                    help="explicit ledger path; repeatable. Default: "
                         "$HERMES_HOME/state.db + $HERMES_HOME/profiles/*/state.db")
    ap.add_argument("--families", default=os.environ.get("TOKEN_FAMILIES"),
                    help="key=Label:substr,... (default: flash-next and 27b)")
    ap.add_argument("--anonymise-profiles", action="store_true",
                    help="report profiles as profile-1, profile-2, ... instead of "
                         "their directory names")
    ap.add_argument("--push", default=os.environ.get("TOKENS_PUSH", ""),
                    help="rsync destination for the snapshot, e.g. "
                         "user@serving-host:/srv/dash/var/tokens.json")
    a = ap.parse_args()

    families = parse_families(a.families)
    dbs = a.db or ([os.path.join(AGENT_HOME, "state.db")] + sorted(
        glob.glob(os.path.join(AGENT_HOME, "profiles", "*", "state.db"))))
    days: dict[str, dict[str, dict]] = {}
    profiles: dict[str, dict[str, dict]] = {}
    raw_models: dict[str, set] = {}
    sources = []
    alias: dict[str, str] = {}
    last_seen_max = 0.0
    for db in dbs:
        prof = ("default" if os.path.dirname(db) == AGENT_HOME
                else os.path.basename(os.path.dirname(db)))
        if a.anonymise_profiles:
            prof = alias.setdefault(prof, f"profile-{len(alias) + 1}")
        # The DB PATH is only reported when it is not being anonymised: on a
        # shared dashboard it is a filesystem layout nobody needs.
        src_id = prof if a.anonymise_profiles else db
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            rows = con.execute(
                "select session_id, model, api_call_count, input_tokens, output_tokens, "
                "cache_read_tokens, first_seen, last_seen from session_model_usage"
            ).fetchall()
        except sqlite3.Error as e:
            sources.append({"db": src_id, "profile": prof, "error": str(e)[:160]})
            continue
        sources.append({"db": src_id, "profile": prof, "rows": len(rows)})
        for sid, model, calls, inp, out, cache, fs, ls in rows:
            fam = family_of(model, families)
            raw_models.setdefault(fam, set()).add(model)
            last_seen_max = max(last_seen_max, ls or 0)
            pt = profiles.setdefault(prof, {}).setdefault(fam, empty())
            for k, v in zip(FIELDS, (inp, cache, out, calls)):
                pt[k] += int(v or 0)
            pt["sessions"] += 1
            pt["cache_reported"] |= bool(cache)
            for i, (day, frac) in enumerate(turn_split(con, sid, fs or 0, ls or 0)):
                d = days.setdefault(day, {}).setdefault(fam, empty())
                for k, v in zip(FIELDS, (inp, cache, out, calls)):
                    d[k] += (v or 0) * frac
                d["sessions"] += 1 if i == 0 else 0
                d["cache_reported"] |= bool(cache)
        con.close()

    for fams in days.values():
        for d in fams.values():
            for k in FIELDS:
                d[k] = int(round(d[k]))

    # ⛔ NO session_id REACHES THE SNAPSHOT. Sessions are read, attributed to a
    # day, and counted; the identifiers stay in the ledger.
    data = {
        "generated_at": time.time(),
        "tz": time.strftime("%Z"),
        "today": datetime.now().strftime("%Y-%m-%d"),
        "source": "agent session_model_usage",
        "attribution": "multi-day sessions split by assistant-turn timestamps",
        "semantics": {"input": "uncached prompt", "cached": "prompt cache read",
                      "output": "completion"},
        "families": [{"key": k, "label": lbl, "models": sorted(raw_models.get(k, []))}
                     for k, lbl, _ in families]
                    + [{"key": "other", "label": "Other",
                        "models": sorted(raw_models.get("other", []))}],
        "last_usage_at": last_seen_max or None,
        "days": days,
        "profiles": profiles,
        "sources": sources,
    }
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    os.replace(tmp, a.out)
    if a.push:
        r = subprocess.run(["rsync", "-a", "--timeout=25", a.out, a.push],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("push failed:", r.stderr.strip()[:200])
            return 2
    print(f"tokens: {len(days)} days, "
          f"{sum(s.get('rows', 0) for s in sources)} rows from {len(dbs)} ledger(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

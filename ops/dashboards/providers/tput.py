# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Live generation throughput (tok/s), read off the VRAM/KV sampler's ring.

NOT a poller. It sends NO request to the engine. The sampler described in
`../../observability.md` already fetches the engine's `/metrics` every 20 s and
records the cumulative token counters it finds there under `backend.tput` in a
dated JSONL ring (`vram-YYYYMMDD.jsonl`). This module TAILS those files by byte
offset and shapes them into per-window series.

Signal, in order of how much it can be trusted:

  rate    difference of `realtime_tokens_total{mode=decode}` between consecutive
          samples / elapsed wall time. The engine bumps that counter on EVERY
          decode iteration, so this is the exact mean generated tok/s over the
          sample interval (and, summed, over any bucket). Primary series.
  smooth  trailing 5-min mean of the same counter (sum of tokens / sum of time).
  gauge   the engine's instantaneous `gen_throughput` gauge at the sample
          instant: aggregate over running streams, spiky, reads 0 between
          requests and decays to 0 after ~30 s idle. Drawn as faint dots for
          reference only -- it is speed WHILE decoding, not throughput.

⛔ COUNTER RESETS ARE NOT NEGATIVE THROUGHPUT. An engine restart gives a new
`main_start_monotonic`, and a counter can also simply go backwards. The interval
is DROPPED, never differenced across; the restart is reported as a marker so the
discontinuity is visible rather than smoothed away.

⛔ THE FIRST REQUEST AFTER A BOOT IS NOT A THROUGHPUT MEASUREMENT. The interval
in which the decode counter first leaves zero is flagged `warm`: on the
reference deployment the first request after a boot measured ~4.5x slower than
steady state (39.6 vs 173-183 tok/s). It stays VISIBLE, and is excluded from the
y-axis scale and from the peak/p95 stats.

Sample schema this expects, per JSONL line (absent keys degrade to gaps, never
to zeroes):

    {"ts_epoch": 1.7e9,
     "unit":    {"main_start_monotonic": ..., "main_pid": ...},
     "backend": {"ok": true, "running_reqs": 2, "queue_reqs": 0,
                 "token_usage": 0.41,
                 "tput": {"decode_tok_total": ..., "prefill_tok_total": ...,
                          "gen_throughput": ...}}}
"""
from __future__ import annotations

import glob
import json
import os
import time
from bisect import bisect_left

SMOOTH_S = 300.0
WARM_FROM_TOK = 64         # decode counter at/below this = nothing served yet
BUSY_TOK_S = 1.0            # an interval under 1 tok/s counts as idle
MAX_GAP_S = 90.0            # > 4 missed 20 s samples => gap, not an interval

# (key, span seconds, bucket seconds; 0 = raw 20 s samples)
WINDOWS = [
    ("15m", 900, 0),
    ("1h", 3600, 0),
    ("6h", 6 * 3600, 60),
    ("24h", 86400, 300),
    ("7d", 7 * 86400, 900),
]


def _r(v, nd=1):
    return None if v is None else round(v, nd)


class TputSeries:
    def __init__(self, samples_dir: str, retain_s: float = 7 * 86400 + 3600):
        self.dir = samples_dir
        self.retain_s = retain_s
        self.offsets: dict[str, int] = {}
        # samples: (t, boot, dec, pre, gauge, run, queue, usage, ok)
        self.samples: list[tuple] = []

    # ------------------------------------------------------------------ ingest
    def _ingest(self, now: float) -> None:
        cutoff = now - self.retain_s
        cut_day = time.strftime("%Y%m%d", time.gmtime(cutoff - 86400))
        last_t = self.samples[-1][0] if self.samples else 0.0
        for path in sorted(glob.glob(os.path.join(self.dir, "vram-*.jsonl"))):
            day = os.path.basename(path)[5:13]
            if day < cut_day:
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            off = self.offsets.get(path, 0)
            if size < off:          # truncated / replaced: re-read, ts dedupes
                off = 0
            if size == off:
                continue
            with open(path, "rb") as fh:
                fh.seek(off)
                buf = fh.read(size - off)
            end = buf.rfind(b"\n")
            if end < 0:
                continue
            self.offsets[path] = off + end + 1
            for line in buf[:end].split(b"\n"):
                try:
                    s = json.loads(line)
                except Exception:
                    continue
                t = s.get("ts_epoch")
                if not isinstance(t, (int, float)) or t <= last_t:
                    continue
                b = s.get("backend") or {}
                tp = b.get("tput") or {}
                u = s.get("unit") or {}
                self.samples.append((
                    float(t),
                    u.get("main_start_monotonic") or u.get("main_pid"),
                    tp.get("decode_tok_total"), tp.get("prefill_tok_total"),
                    tp.get("gen_throughput"),
                    b.get("running_reqs"), b.get("queue_reqs"), b.get("token_usage"),
                    bool(b.get("ok")),
                ))
                last_t = float(t)
        if self.samples and self.samples[0][0] < cutoff:
            i = bisect_left([x[0] for x in self.samples], cutoff)
            del self.samples[:i]

    # --------------------------------------------------------------- intervals
    def _intervals(self):
        """Per-sample derived rows: (t, dt|None, ddec, dpre, warm, reset)."""
        rows = []
        prev = None
        for s in self.samples:
            t, boot, dec, pre = s[0], s[1], s[2], s[3]
            dt = ddec = dpre = None
            warm = reset = False
            if prev is not None and boot is not None and prev[1] is not None and boot != prev[1]:
                reset = True
            elif prev is not None and dec is not None and prev[2] is not None:
                gap = t - prev[0]
                if dec < prev[2]:
                    reset = True
                elif 0 < gap <= MAX_GAP_S:
                    dt, ddec = gap, dec - prev[2]
                    if pre is not None and prev[3] is not None and pre >= prev[3]:
                        dpre = pre - prev[3]
                    # counter leaves ~0 => first request since this backend booted
                    warm = ddec > 0 and prev[2] <= WARM_FROM_TOK
            rows.append((t, dt, ddec, dpre, warm, reset))
            prev = s
        return rows

    # ------------------------------------------------------------------ shape
    def snapshot(self, now: float | None = None) -> dict:
        now = now or time.time()
        self._ingest(now)
        S = self.samples
        if not S:
            return {"ok": False, "reason": f"no sampler rings in {self.dir}"}
        rows = self._intervals()
        ts = [r[0] for r in rows]
        first_tput = next((s[0] for s in S if s[2] is not None), None)

        def smooth_at(i):
            tok = sec = 0.0
            t0 = rows[i][0] - SMOOTH_S
            j = i
            while j >= 0 and rows[j][0] > t0:
                if rows[j][1]:
                    tok += rows[j][2]
                    sec += rows[j][1]
                j -= 1
            return tok / sec if sec >= 60 else None

        out = {"ok": True, "now": now, "first_sample": S[0][0], "first_tput": first_tput,
               "latest_t": S[-1][0], "age_s": now - S[-1][0], "windows": {},
               "resets": [r[0] for r in rows if r[5]]}

        # current readings
        last = rows[-1]
        s_last = S[-1]
        out["current"] = {
            "rate": _r(last[2] / last[1]) if last[1] else None,
            "smooth": _r(smooth_at(len(rows) - 1)),
            "gauge": _r(s_last[4]),
            "running": s_last[5], "queued": s_last[6],
            "usage_pct": _r(s_last[7] * 100 if s_last[7] is not None else None),
            "backend_ok": s_last[8],
        }

        for key, span, bucket in WINDOWS:
            lo = now - span
            i0 = bisect_left(ts, lo)
            win = {"span": span, "bucket": bucket or 20}
            tok = sec = 0.0
            busy_tok = busy_sec = 0.0
            peaks = []
            for i in range(i0, len(rows)):
                t, dt, ddec = rows[i][0], rows[i][1], rows[i][2]
                if dt:
                    tok += ddec
                    sec += dt
                    if ddec >= BUSY_TOK_S * dt:
                        busy_tok += ddec
                        busy_sec += dt
                    if not rows[i][4]:
                        peaks.append(ddec / dt)
            peaks.sort()
            win["stats"] = {
                "tokens": int(tok), "covered_s": sec,
                "mean": _r(tok / sec) if sec else None,
                "busy_mean": _r(busy_tok / busy_sec) if busy_sec else None,
                "busy_frac": _r(busy_sec / sec, 3) if sec else None,
                "peak": _r(peaks[-1]) if peaks else None,
                "p95": _r(peaks[int(0.95 * (len(peaks) - 1))]) if peaks else None,
            }
            if not bucket:
                # raw 20 s rows: [t, rate, smooth, gauge, prefill_rate, run, queue, usage%, warm]
                pts = []
                for i in range(i0, len(rows)):
                    t, dt, ddec, dpre, warm, _ = rows[i]
                    s = S[i]
                    pts.append([
                        round(t, 1),
                        _r(ddec / dt) if dt else None,
                        _r(smooth_at(i)),
                        _r(s[4]),
                        _r(dpre / dt) if dt and dpre is not None else None,
                        s[5], s[6],
                        _r(s[7] * 100 if s[7] is not None else None),
                        1 if warm else 0,
                    ])
                win["cols"] = ["t", "rate", "smooth", "gauge", "prefill", "run", "queue", "usage", "warm"]
                win["pts"] = pts
            else:
                # buckets: [t0, mean, peak, gauge_mean, prefill_mean, run_mean, queue_max, usage_mean, cover]
                b0 = lo - (lo % bucket)
                nb = int((now - b0) // bucket) + 1
                acc = [[0.0, 0.0, None, 0.0, 0, 0.0, 0.0, 0.0, 0, None, 0.0, 0, 0] for _ in range(nb)]
                # tok, sec, peak, gauge_sum, gauge_n, pre_tok, pre_sec, run_sum, run_n, q_max, use_sum, use_n, warm
                for i in range(i0, len(rows)):
                    t, dt, ddec, dpre, warm, _ = rows[i]
                    k = int((t - b0) // bucket)
                    if not 0 <= k < nb:
                        continue
                    a = acc[k]
                    s = S[i]
                    if dt:
                        a[0] += ddec
                        a[1] += dt
                        if warm:
                            a[12] = 1
                        else:
                            r = ddec / dt
                            a[2] = r if a[2] is None else max(a[2], r)
                        if dpre is not None:
                            a[5] += dpre
                            a[6] += dt
                    if s[4] is not None:
                        a[3] += s[4]
                        a[4] += 1
                    if s[5] is not None:
                        a[7] += s[5]
                        a[8] += 1
                    if s[6] is not None:
                        a[9] = s[6] if a[9] is None else max(a[9], s[6])
                    if s[7] is not None:
                        a[10] += s[7] * 100
                        a[11] += 1
                pts = []
                for k, a in enumerate(acc):
                    if not (a[1] or a[4] or a[8]):
                        continue
                    pts.append([
                        round(b0 + k * bucket, 1),
                        _r(a[0] / a[1]) if a[1] else None,
                        _r(a[2]),
                        _r(a[3] / a[4]) if a[4] else None,
                        _r(a[5] / a[6]) if a[6] else None,
                        _r(a[7] / a[8], 2) if a[8] else None,
                        a[9],
                        _r(a[10] / a[11]) if a[11] else None,
                        _r(min(1.0, a[1] / bucket), 2),
                        a[12],
                    ])
                win["cols"] = ["t", "rate", "peak", "gauge", "prefill", "run", "queue", "usage", "cover", "warm"]
                win["pts"] = pts
            out["windows"][key] = win
        return out

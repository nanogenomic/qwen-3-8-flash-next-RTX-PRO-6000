# Two dashboards: what the cards are doing, and what the agents spent

**Published as runnable code, not as screenshots.** Both dashboards are extracted from a
wall that runs beside this engine; everything specific to that deployment — hostnames,
paths, unit names, profile names, model lists — is a parameter here, with a default that
works on one machine with one card and no configuration at all.

Two dashboards, one harness:

| | what it answers | reads |
|---|---|---|
| **GPU** | which cards exist right now, what is resident on them, and which hosts did not answer | `nvidia-smi`, every poll |
| **Token usage** | what the agent sessions actually spent, per model family, per day, and how fast the engine generated it | an agent framework's `session_model_usage` ledger + the [sampler ring](../observability.md) |

```bash
pip install -r requirements.txt
python3 tools/make_example_data.py --out-dir ./var    # SYNTHETIC, see below
./serve.py                                            # http://127.0.0.1:8770
```

With no NVIDIA GPU present the GPU dashboard says **"GPU telemetry unavailable"**. That is
the point of the whole directory, and the next section is why.

---

## The rule both dashboards are built around

**⛔ A DASHBOARD THAT FILLS A GAP WITH A ZERO IS WORSE THAN NO DASHBOARD.** Every design
decision below exists because the obvious version of it shipped a number that looked fine
and was wrong:

| the obvious version | what it claimed | what was true |
|---|---|---|
| inventory read from a database table | a **one-GPU fleet** | one agent had ever written to that table, from one host, and had stopped months earlier. The card doing nearly all the work was simply absent |
| a host that did not answer, omitted | that host has no GPUs | nobody looked. **Absent is not empty** — `UNREACHABLE` is now a row, carrying the error |
| cache hit rate `0%` | the prefix cache is useless | the engine never sent `prompt_tokens_details.cached_tokens` at all. Rendered `n/r` now — *not reported*, not zero |
| a multi-day session, spread evenly over its days | a flat workload | one session carried 1.4B tokens over 14 days. Split by assistant-turn timestamps now; uniform only as a fallback |
| last reading carried forward while the exporter was dead | a quiet day | the snapshot was 14 hours stale. Every panel carries its own age and flips to `STALE` |
| the first tok/s reading after a restart | a throughput regression | first request after a boot measured **~4.5× slower** (39.6 vs 173–183 tok/s). Flagged `warm`, kept visible, excluded from the scale and from peak/p95 |
| a card that vanished mid-run, dropped from its history | nothing | its series silently drifted one sample out of step with the other card's, permanently. A missing poll appends `None` |

Both dashboards render an absent value as an em dash, a shaded no-data band, `n/r`, or the
word `STALE`. Never as `0`.

---

## 1. The GPU dashboard

Two views, both of which discover the hardware rather than being told about it.

### `gpu.host` — the local cards

`providers/gpu.py`. One `nvidia-smi --query-gpu` per poll (2 s default) plus
`--query-compute-apps` for what is resident.

**Nothing in that file knows how many cards the box has.** The card set comes from the
driver every poll; per-card history is created lazily the first time an index is seen; a
card that disappears gets a `None` sample. A card installed, reset, or lost while the
dashboard is running appears or disappears on the wall without an edit and without a
restart. The browser rebuilds the per-GPU DOM when, and only when, the **count** changes.

Two details that are easy to get wrong and silent when you do:

- **a hot-added card's history is left-padded** to the timeline that already exists.
  Callers zip these series together, and unequal lengths left-align — pairing the new
  card's *newest* sample against the incumbent's *oldest*.
- **`index` is coerced to `int` exactly once, at the source.** The CSV parser floats every
  numeric field, so the index arrives as `0.0`; history is keyed `"0"`, a consumer doing
  `str(g.index)` asks for `"0.0"`, and every sparkline silently goes blank.

The process table joins by **GPU UUID, not index** — an index is an enumeration order, a
UUID is an identity. Label your own workloads by editing `WORKLOAD_LABELS` in
`providers/gpu.py`; an unmatched process is still listed, under its `argv[0]`.

### `gpu.fleet` — every host you name

`tools/fleet_probe.py` (standalone, runnable on its own) and the thin provider around it.
`nvidia-smi` locally and over `ssh`, in parallel, read-only — it allocates nothing, touches
no job, and changes no GPU state.

```bash
GPU_FLEET='box=local' ./tools/fleet_probe.py                  # just this machine (default)
GPU_FLEET='box=local,rig2=ssh:rig2' ./tools/fleet_probe.py    # plus one over ssh
GPU_FLEET='a=ssh:node-a,b=ssh:node-b' ./tools/fleet_probe.py  # neither of them is here
```

**Three states, not two.** `live` answered. `UNREACHABLE` did not, and carries the error
and the attempt timestamp. `no-gpu` is a host whose `nvidia-smi` answered and listed
nothing — a real observation, and a different claim. Collapsing the last two is exactly
how an inventory comes to describe a one-GPU fleet.

### `tools/gpu_hwstate.sh` — below `nvidia-smi`

Not part of either dashboard; the PCI-level companion to them, and the one place here that
goes to the device itself. Two settings change under a reboot and **both** move every
benchmark number on the card:

- **Resizable BAR.** No `--query-gpu` field exists. BAR1 has to be parsed out of
  `nvidia-smi -q -d MEMORY`, and the *capability* only exists in `lspci` — `nvidia-smi`
  shows the window it got, not whether a larger one was on offer.
- **ECC.** No overhead field exists either. It shows up as the gap between `memory.total`
  and the driver-reserved figure, so both are recorded and the overhead is **measured**
  rather than assumed to be "about 6%".

```bash
./tools/gpu_hwstate.sh 0 > hwstate.json   # run before a benchmark, store beside the result
```

Without enough privilege for the capability block the `lspci_*` fields come back **empty**,
which is reported as empty rather than guessed.

---

## 2. The token-usage dashboard

`tokens.usage`, fed by `providers/tokens.py`, which reads a snapshot written by
`tools/tokens_export.py`.

**It is a two-host design on purpose.** The agent ledger lives on the client host; the
dashboard runs on the serving host ([`../README.md`](../README.md#scope-stated-honestly) —
the agent sessions do not run on the card). The exporter reads the ledger where it lives
and pushes a snapshot across; the provider always carries the snapshot's **age**, so a dead
exporter reads as `STALE` rather than as a quiet day.

### Pointing it at your own agent framework

`tools/tokens_export.py` opens every SQLite ledger under `$HERMES_HOME` read-only
(`state.db`, plus `profiles/*/state.db` — see [client rule 7](../../clients/README.md#7-isolate-test-state-from-live-state)
for why test state is a separate home) and wants two tables:

```sql
session_model_usage(session_id, model, api_call_count,
                    input_tokens, output_tokens, cache_read_tokens,
                    first_seen, last_seen)
messages(session_id, role, timestamp)     -- optional; day attribution falls back without it
```

Any orchestrator recording those columns works — point `--db` at it.

```bash
# on the client host
./tools/tokens_export.py --out /tmp/tokens.json \
    --push user@serving-host:/srv/dash/var/tokens.json
# model families are matched by substring on the SERVED model name
./tools/tokens_export.py --families 'flashnext=Flash-Next:flash-next,q27b=27B:27b' --out ...
# directory names replaced by profile-1, profile-2, ... in the snapshot
./tools/tokens_export.py --anonymise-profiles --out ...
```

**No `session_id` reaches the snapshot.** Sessions are read, attributed to a day, and
counted; the identifiers stay in the ledger.

### What the three numbers mean

`input` is the prompt **not** served from cache, `cached` is the prefix-cache share,
`output` is the completion. Prompt and output are drawn on **separate charts** — output
runs ~2% of prompt and vanishes on a shared axis.

### The throughput strip

`providers/tput.py` tails the VRAM/KV sampler's dated JSONL ring — the one described in
[`observability.md`](../observability.md) — and **sends no request to the engine**. It wants,
per line:

```json
{"ts_epoch": 1.7e9,
 "unit":    {"main_start_monotonic": 0, "main_pid": 0},
 "backend": {"ok": true, "running_reqs": 2, "queue_reqs": 0, "token_usage": 0.41,
             "tput": {"decode_tok_total": 0, "prefill_tok_total": 0, "gen_throughput": 0}}}
```

Three signals, in order of how much they can be trusted: **rate** (Δ decode counter / Δt —
the exact mean over the interval), **smooth** (trailing 5-min mean of the same counter),
and **gauge** (the engine's instantaneous `gen_throughput`, which is speed *while
decoding*, reads 0 between requests, and is drawn as faint dots only). An engine restart
gives a new `main_start_monotonic`; that interval is **dropped, never differenced across**,
and the restart is drawn as a marker so the discontinuity stays visible.

---

## Running it

```
serve.py            poll providers, serve /api/state (JSON) and /events (SSE)
providers/          gpu.py, gpu_fleet.py, tokens.py, tput.py, base.py
tools/              fleet_probe.py, tokens_export.py, gpu_hwstate.sh, make_example_data.py
static/             index.html, wall.css, charts.js, views.js, wall.js
examples/           a synthetic token snapshot, committed so the page has something to draw
```

```bash
./serve.py --port 9000 --host 0.0.0.0
./serve.py --providers gpu,gpu_fleet --panes 1      # GPU side only, one pane
TOKENS_FILE=examples/tokens.example.json ./serve.py # draw the committed synthetic snapshot
```

Config precedence: `DEFAULTS` in `serve.py` → `config.json` beside it → env
(`TOKENS_FILE`, `TPUT_SAMPLES_DIR`, `GPU_FLEET`, `DASH_PORT`) → CLI flags. Nothing is read
from anywhere else, and both `config.json` and `var/` are gitignored.

**⛔ One pane means no rotation.** `panes` defaults to 2: pane 0 pins the highest-priority
panel and pane 1 cycles the rest. With one pane, `pin_primary` is forced off here — the
alternative is a single pinned pane on which nothing ever changes.

Dependencies: `psutil`, plus the `nvidia-smi` and (for `gpu_hwstate.sh`) `lspci` binaries.
Everything else is the standard library. The server binds `127.0.0.1` by default and has
no authentication — put it behind something before `--host 0.0.0.0`.

## Example data

`tools/make_example_data.py` writes an **obviously synthetic** token snapshot and sampler
ring so the page can be looked at before a real ledger exists. The snapshot carries
`"synthetic": true`, and the view renders a **SYNTHETIC EXAMPLE DATA** banner from that
flag. Every number in it is fabricated. Never hand a chart made from it to anyone, and
never commit its output over a real snapshot — a real one carries session timing and
measured usage.

The GPU dashboard has no synthetic mode and never will. On a machine with no NVIDIA GPU it
says so.

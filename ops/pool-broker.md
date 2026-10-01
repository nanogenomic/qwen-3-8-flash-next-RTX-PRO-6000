# The cross-session KV pool broker

**Published as design, interface and runtime-config schema. The implementation is not shipped**
— it resolves its paths, endpoints and lane set from one deployment's layout, and the two
modules are ~2,600 lines of which the interesting ~200 are reproduced below.

The *policy* this implements is already written up:
[`clients/hermes/docs/shared-pool-grants.md`](../clients/hermes/docs/shared-pool-grants.md)
(share-bounded ceilings, water-filling, the invariant that could not fail) and
[`clients/hermes/docs/prefill-capacity-throttle.md`](../clients/hermes/docs/prefill-capacity-throttle.md)
(the admission gate that replaced a free-VRAM divisor), with the framework-agnostic rules as
[client rules 10, 11 and 15](../clients/README.md#10-bound-the-ceilings-you-grant-not-the-reservations-you-measure).
**Read those for the arithmetic; this page is the mechanism underneath them** — how the
allowance is published and shared, how it is retuned on a live fleet without restarting the
sessions being retuned, and how every reader degrades when it is absent.

---

## What it is, in one paragraph

Several agent sessions, and every subagent they spawn, draw on **one** KV pool and **one** set
of scheduler slots. Before this existed each process sized itself in isolation, so two sessions
could not see each other and each fanned out as if it owned the box. The broker publishes a
single small JSON **allowance** per backend; every process reads it and sizes itself from it.
That is the whole idea, and the rest of this page is the three properties that make it safe to
run against sessions a human is working in.

### No daemon

Any process that needs the allowance and finds it older than `REFRESH_S` (**5 s**) elects itself
refresher by taking a **non-blocking** `flock`, probes the backend, scans the process table for
live main sessions, and rewrites the file atomically. Everybody else just reads the file. So:

- there is no new always-on service to supervise — and the units in
  [`reboot-survival.md`](reboot-survival.md) are the evidence that every always-on service is a
  thing that can be silently dead;
- when nothing is running, nothing is polling;
- the refresh cost is paid by whichever process happened to find the file stale, never by a
  dispatch. A CLI (`--watch`) runs the same refresh in a loop for a live view, but nothing
  depends on it being up.

The liveness authority is the **process table**, not the file: a heartbeat directory records
what each session reports, and a session's pid must still be alive for its heartbeat to count.
Sessions that predate the broker are discovered anyway and counted with `reports=false` — their
window comes from their profile config and their child counts read 0, because they cannot report
them. Counting them is the point: an unaware peer still consumes pool.

---

## 1. The runtime-tunable config pattern

**This is the transferable part of the whole component.** Every throttle parameter resolves at
**call time**, in this order, first hit wins:

```
1. environment variable          — a launch-time pin; still wins
2. $HERMES_HOME/pool-broker.json   "throttle": { "<key>": <value> }
3. the module constant           — the documented default, and what a test monkeypatches
```

**Why it matters.** Every knob used to be `int(os.environ.get(…))` evaluated at **import**.
Retuning one therefore meant relaunching the processes that had already imported the module —
i.e. relaunching the live agent sessions a human is working in, which is the single operation
you cannot keep performing. On the night the admission gate was rewritten, that turned a
one-number fix into a "needs a relaunch" item, which is how a tuning change becomes a change
nobody makes.

With call-time resolution, editing the config file retunes **every live session within ≤ 5 s**,
with no restart, no relaunch, and no signal to deliver.

**Make the file read cheap, not free.** The parse is cached on `(mtime_ns, size)`:

```python
_config_cache = {"key": None, "value": {}}

def _broker_config() -> dict:
    """Re-read whenever the file changes. CALL TIME, NOT IMPORT TIME — that is the
    whole point. A stat is ~1 us; a parse is not, and every tunable lookup comes
    through here."""
    path = hermes_root() / "pool-broker.json"
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except Exception:
        _config_cache["key"], _config_cache["value"] = None, {}
        return {}                                   # absent file -> no config, not an error
    if _config_cache["key"] == key:
        return _config_cache["value"]
    try:
        value = json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        value = {}                                  # malformed -> defaults, never an exception
    if not isinstance(value, dict):
        value = {}
    _config_cache["key"], _config_cache["value"] = key, value
    return value
```

Four properties worth copying, each of which is a bug if you get it wrong:

- **Keyed on `(mtime_ns, size)`, not mtime alone.** Two edits within one mtime granule that
  change the size are both seen; an editor that preserves mtime is still caught by size.
- **A missing file is a valid state**, not an error: it means "no overrides", which is how a
  config-driven component stays optional.
- **A malformed value falls through to the default instead of raising.** Nothing on this path
  may fail a dispatch, and a JSON typo at 03:00 must not be able to stop an agent.
- **A malformed *value* (not file) logs once and uses the default**, coerced to the default's
  own type, so `"total_stream_cap": "three"` degrades instead of exploding:

```python
def tunable(name: str):
    """env > config file > module default. Never raises."""
    default = globals().get(name.upper())        # the UPPERCASE module attr is the one
    env, ckey = _TUNABLE_SPEC[name]              # documented default AND monkeypatchable
    raw = os.environ.get(env)
    if raw in (None, ""):
        block = _broker_config().get("throttle")
        if isinstance(block, dict):
            raw = block.get(ckey)
    if raw in (None, ""):
        return default
    try:
        return type(default)(raw) if default is not None else raw
    except Exception:
        logger.warning("tunable %s=%r is not a %s; using the default %r",
                       name, raw, type(default).__name__, default)
        return default
```

### Publish what resolved, and where it came from

A retune you cannot observe is a retune you will repeat. The allowance carries, for every knob,
the resolved value **and its source**:

```python
def tunables() -> dict:
    out = {}
    for name, (env, ckey) in _TUNABLE_SPEC.items():
        block = _broker_config().get("throttle")
        src = ("env" if os.environ.get(env) else
               "config" if isinstance(block, dict) and block.get(ckey) is not None
               else "default")
        out[name] = {"value": tunable(name), "source": src}
    return out
```

So `--show` answers "is my edit live?" without reading any code, and an operator who pinned a
value in the environment at launch can see that their pin, not the file, is what is in force.

### ⛔ A tunable's own docstring is a snapshot, and it drifts

`[observed in this repository, 2026-10-01]` The live config's self-documenting comment derives
its `total_stream_cap` default from a per-lane decode-share curve it quotes as
**100 / 80.0 / 64.2 / 58.0 %** of solo rate at concurrency 1/2/3/4, from 6,436 warm decode
samples. The queue-filtered extraction in
[BENCHMARKS §3.20](../BENCHMARKS.md#320-throughput-versus-concurrency-warm-segmented--and-the-071-retraction)
— same card, same day, with uncontended samples selected by the engine's own `#queue-req == 0`
rather than by a proxy — gives **100 / 78.3 / 64.5 / 52.0 %** on the on-GPU pool and
**100 / 73.9 / 56.7 %** on the host-resident pool. The *decision* is unchanged (2 is the largest
concurrency at which a lane keeps about two thirds of itself, which is what
[client rule 15](../clients/README.md#15-a-proxy-stops-being-a-proxy-when-the-thing-it-proxied-for-moves)
states), but the numbers in the config comment are from the earlier, weaker segmentation.

The lesson is not "that comment is wrong". It is that **a runtime-tunable file is the one place
an operator reads a derivation under pressure, and it is the place least likely to be
re-measured.** So: put the *derivation* and a pointer to the authoritative measurement in the
config, and keep the measurement itself in one place that gets revised. Treat a number inlined
in a comment as of the date it was written.

---

## 2. The published allowance as the single source of truth

One file, rewritten atomically every ~5 s, versioned, and legible on purpose — every number in
it is checkable against the backend's own `/get_server_info` and `/v1/loads` by eye. The CLI's
rendering is the schema. A real one, from the reference deployment, with pids, session ids and
the backend address replaced:

```
GPU0 pool broker  policy=rule-v1  backend=http://127.0.0.1:30000
  pool       1,310,720 tokens, 190,400 used (15%, normal), 989,248 free after 131,072 cache reserve
  slots      running 1 / mrr 4, queued 0;  stream target min(12, mrr) = 3
  ceiling    540,000 (server --context-length, tightened by max_req_input_len 539,994)
  mains      2 protected, 668,685 tokens reserved at peak
    pid  <pid-a>  -        <session-a>  window 532,480  reserved 453,836  children 0  reports
    pid  <pid-b>  <pane>   <session-b>  window 393,216  reserved 214,849  children 0  pre-broker session (counted, not reporting)
  children   1 stream(s) across all sessions (0 active); initial window <= 327,680, may grow to 327,680; child pool 510,963 tokens
  device VRAM  5,591 MiB free on GPU0 (baseline 5643, sample 9.8s old) -> OK
               5,591 MiB runtime device-free vs a 688 MiB floor (560 MiB largest fatal alloc + 4 x 32 MiB prefill transient) -> above the floor: no VRAM cap (the prefill gate decides)
               admit new children: True
  priority   main=100 child=0  ACTIVE; tagging on; policy fcfs
```

Three things to read off that, because they are the properties, not the numbers:

- **`stream target min(12, mrr) = 3`.** `mrr` is 4 and the module default for
  `total_stream_cap` is 2; the 3 is a live retune, read from the config file at call time (§1)
  with nothing restarted. That is what a working runtime knob looks like in the output.
- **`pre-broker session (counted, not reporting)`.** The second main predates the broker and
  cannot report its children, so its window comes from its profile config and its child count
  reads 0 — but it is **counted against the pool**, because an unaware peer still consumes it.
- **The VRAM line states the whole derivation**, not a verdict. `5,591 MiB` runtime
  device-free against a `560 + 4 × 32 = 688 MiB` floor ⇒ no cap, and the renderer says *which
  component decides instead*. A one-word `OK` would have been unreviewable.

`--json` emits it raw. Fields that matter to a reader other than the renderer:

| field | meaning |
|---|---|
| `version` | schema version. **3** at the time of writing. Bump it when a *reader* must change. |
| `updated` | unix seconds. The only staleness authority (see §3). |
| `pool`, `used_tokens`, `token_usage`, `running`, `waiting`, `max_running_requests`, `max_req_input_len` | straight from the backend, unmodified, so a reader can sanity-check the derived numbers |
| `ceiling` | the server's `--context-length`, tightened by `max_req_input_len` |
| `cache_reserve` | prefix-cache slack never handed out (`cache_reserve_frac`, default **0.10**) |
| `mains[]` | one row per live main session: `window`, `reserved` (peak in-flight footprint), `window_ceiling` (its share-bounded cap), `over_ceiling`, `children_active`, `reports` |
| `child_window_max`, `child_grow_max`, `child_streams_total`, `child_pool_tokens` | what a dispatcher may give a subagent |
| `invariant` | see below |
| `tunables` | every knob, resolved, with its source |
| `vram` | `{state, free_mib, baseline_free_mib, sample_age_s, why}` — or `state: "unknown"` with a `why` |
| `priority` | the values, whether the server honours them, and whether tagging is safe |
| `shadow` | a non-applied policy's output, if one is configured (§4) |

### The invariant has to sum the ceilings you GRANTED

This is [client rule 10](../clients/README.md#10-bound-the-ceilings-you-grant-not-the-reservations-you-measure)
and it is the one field a reader should check before trusting anything else. The published
invariant carries **both** sums, deliberately, because they answer different questions:

```json
"invariant": {
  "mains_reserved": …, "children_reserved": …, "total": …, "limit": …,
  "reservations_ok": true,
  "grants": {
    "mains_granted": …, "children_granted": …, "total": …, "limit": …, "ok": true,
    "window_ceiling_sum": …,
    "held_total": …, "deadlock_risk": false,
    "grandfathered": [ … ]
  },
  "ok": true
}
```

- `reservations_ok` sums **measured demand** scaled to fit the budget. It is useful telemetry
  and it is **mathematically incapable of failing** — which is exactly how the original version
  reported `{ok: true, total: 406137, limit: 406138}` while two sessions had each been granted
  the entire pool. Never gate on it alone.
- `grants.ok` sums the **ceilings actually issued**: `Σ max(main_window, main_grow_max) ≤ pool −
  cache_reserve`. This one can fail, and that is the point of having it.
- `held_total` / `deadlock_risk` / `grandfathered` report sessions that grew **above** their
  share before the share existed or before the pool shrank. That is grandfathered, not granted,
  and it is reported rather than hidden so the stall is visible in the file. Growth is refused
  for those sessions so they compact instead of ratcheting; note the honest limit, logged
  verbatim by the broker: *a session whose resident prompt already exceeds its share's
  compaction trigger cannot shrink itself and needs a relaunch.*

### ⛔ Older peers must be able to read it safely

A broker rollout is a rolling one: peers written before share-bounding exist and keep running
for days. So the **reader** carries the check, not just the writer. A reader that finds a grant
above the pool ceiling, or grants that sum above it, treats that document as **unbounded** —
it renders `no share cap ⚠` rather than a number it knows is wrong, and falls back to its
configured defaults. That single rule is what makes it safe to publish a new allowance format
into a fleet of mixed peers, and it is why `window_ceiling_sum` is in the file as a scalar
rather than left for a reader to recompute. The display path is **read-only**: it never probes
the backend, never requests an allowance, and cannot write (see
[client rule 11](../clients/README.md#11-show-the-share-and-degrade-honestly-when-you-cannot)).

---

## 3. Staleness and fail-open discipline

**Every absence is a distinct, named state. None of them is an error, and none of them blocks
work.**

| signal | fresh | stale threshold | behaviour past it |
|---|---|---|---|
| the allowance document | rewritten every `REFRESH_S` = **5 s** | `STALE_S` = **30 s** | `get_allowance()` returns `None`; every caller uses its configured default — the behaviour the framework had before the broker existed |
| between 5 s and 30 s | — | — | the value is returned **now, from the file**, and a refresh runs on a background thread — so a turn or a dispatch never waits on a probe |
| the same process, repeatedly | — | — | a **1-second in-process memo** in front of the file read, so a hot path that asks ten times a second costs one `read()` |
| the runtime free-VRAM sample | published every **20 s** by the watchdog | TTL **20 s** for reuse, **max age 90 s** | the sample is **discarded**; `vram.state` becomes `"unknown"` with a `why`, and VRAM throttling is off — *loudly*, in the rendered output |
| the backend itself | — | `PROBE_TIMEOUT_S` = **1.5 s** | no allowance is produced at all; consumers fall back to defaults. The VRAM signal is read anyway, because an outage is exactly when you want it |

**Judge the staleness threshold against the publish interval, don't guess it.** The VRAM
sampler writes every 20 s, so **90 s** means "four consecutive misses" — i.e. the publisher is
*gone*, not merely busy. The same reasoning sets the allowance's 30 s against a 5 s refresh.
A threshold chosen as a round number tells you nothing about the publisher's state; a threshold
derived from its cadence tells you it is dead.

**A stale sample must read as absent, not as its last value.** This is the whole reason the
sample is **pulled** with its own `ts_epoch` inside it rather than pushed, or inferred from a
file mtime: the age is a property of the measurement, not of the transport or the filesystem.
`[measured]` The pull costs **19 ms** over a persistent ssh connection, and is paid only by the
process refreshing the shared allowance (every ~20 s) — never by a dispatch, because
dispatchers read the file.

**And read the runtime number, never the boot-time one.** `[measured]` The engine's boot-time
`available_gpu_mem` field reported **3.42 GB** while the card had **287 MiB** free while
serving. A field that is correct at startup and never updated is worse than no field.

**What "fail open" means here, precisely:**

- if the pluggable policy raises, returns a non-dict, or exceeds `ALLOCATE_BUDGET_MS` (**200 ms**),
  the broker logs it and uses the shipped rule policy **for that refresh** — not for good;
- if the broker cannot produce an allowance at all, consumers see `None`;
- no backend configured ⇒ the broker is **disabled** and every caller uses its configured
  default. It never guesses a production address;
- every public function swallows its own exceptions, and every probe is bounded.

**One exception, deliberately.** A test process that tries to write live allowance state raises
a `BaseException` subclass, specifically so it is **not** caught by the fail-open
`except Exception` handlers that surround every write. That is
[client rule 7](../clients/README.md#7-isolate-test-state-from-live-state) made structural: a
single test run had previously advanced a live window override, and a shared-pool module's
hard-coded run directory let a test win the production directory's lock and rewrite the
allowance every live agent sized itself from. Fail-open must not become fail-silent for the one
class of failure where silence is the bug.

---

## 4. The allocator contract, and the shadow slot

The broker gathers every input, calls one `allocate()`, enforces hard rails on the result, and
publishes it. An allocator never touches the backend, the process table, config files or the
allowance file. That separation is what makes a learned allocator testable against live traffic:

```
active policy   HERMES_POOL_POLICY=<module.path>:<attribute>
shadow policy   HERMES_POOL_SHADOW_POLICY=<module.path>:<attribute>
```

The **shadow** policy is called on every refresh with the *same* inputs; its output is recorded
under `allowance["shadow"]` beside what was actually applied, and is **never acted on**. That
is the zero-risk path for evaluating a different allocator on real traffic, and it costs one
extra pure-CPU call per 5 s.

`allocate()` runs inside whichever agent process is refreshing the shared allowance, so it must
be pure CPU: no network, no disk, inside 200 ms.

**Hard rails the broker applies after any policy — a policy cannot violate them:**

- `main_window` and `main_grow_max` clamp to `[MAIN_WINDOW_FLOOR, min(pool_ceiling, session.window_ceiling)]`,
  where `window_ceiling` is the share-bounded cap;
- **`main_grow_max` is deliberately allowed to be *below* `main_window`.** That is how "hold what
  you have, but do not grow" is expressed: the compactor's grow-instead-of-compact path sees
  `current ≥ cap` and compacts. ⛔ The rail this replaced was a "mains are never shrunk" guard
  that did `mg = max(main_grow_max, main_window)` — which *raised* any bounded growth ceiling back
  up to whatever the session had already grown to, so no policy could express "you are above your
  share". That was the mechanical cause of the deadlock in
  [shared-pool-grants.md](../clients/hermes/docs/shared-pool-grants.md). Refusing growth is
  recoverable; a deadlocked pool is not;
- child windows clamp to `[child_floor, min(pool_ceiling, child_window_ceiling)]` with
  `child_grow_max ≥ child_window_max`;
- `child_streams` clamps to `[1, child_streams_budget]` (slots left after one per live main,
  bounded by memory) and is an **upper bound**: at dispatch a session also subtracts the children
  *other* sessions have in flight, so the sessions together never exceed the budget;
- priority values clamp to `[0, 1000]`.

**Two constants that are easy to confuse**, restated here because the distinction has been got
wrong twice: `GUARANTEED_LANE_WINDOW` = **131,072** is the share a lane is guaranteed *before
lending*, **not a floor**. The floor is `MAIN_WINDOW_FLOOR` = **64,000**. Proof they differ: at
five live mains the grant drops to 65,536. Windows are chosen on an **8,192-token grid** —
legible, and cache-friendly against the engine's page size.

---

## 5. The CLI

One command, three modes, no side effects beyond the refresh every reader would have done
anyway:

```
hermes-pool-broker              refresh once and print the allowance
hermes-pool-broker --json       the same document, raw
hermes-pool-broker --watch 5    refresh every 5 s until interrupted
```

Exit **3** when the backend is unreachable, with the VRAM assessment still printed to stderr —
the two signals are independent and an outage is when you want the second one. `--watch` exists
for a live view and nothing depends on it running.

---

## The config file

Annotated schema, with every default: [`examples/pool-broker.json`](examples/pool-broker.json).
The defaults in it are the measured values from this deployment; the file documents its own
keys so the next person retuning does not have to read the module. Remove the `backend` key to
disable the broker entirely — every caller then falls back to its configured defaults, which is
the pre-broker behaviour.

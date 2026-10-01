# Observability and incident capture

**Published as design, interface and the two reading traps.** The sampler and the capture are
~1,700 lines between them and resolve one deployment's unit names, log paths, endpoints and
alert transport; the parts worth copying are below in full.

Two components, deliberately coupled:

- a **VRAM / KV-pool sampler** writing a dated JSONL ring plus a `latest.json`, which is the only
  thing on the box that keeps free-VRAM history;
- a **per-death incident artifact** folded into that same sampler, so one file per lane death
  replaces reconstructing a cause by hand from a multi-hundred-megabyte append-only log.

What made both necessary: a scheduler died because a constrained-decoding request took a **lazy
Triton kernel-load path** with 0.54 GiB of device VRAM free. ~9 minutes of downtime, reconstructed
after the fact from engine logs **because nothing was sampling**.

Contents:

1. [A sampler that cannot die quietly](#1-a-sampler-that-cannot-die-quietly)
2. [Thresholds: why the obvious absolute floors are useless here](#2-thresholds-why-the-obvious-absolute-floors-are-useless-here)
3. [Two traps in reading an engine log](#3-two-traps-in-reading-an-engine-log)
4. [One artifact per death](#4-one-artifact-per-death)
5. [Is the capture itself alive?](#5-is-the-capture-itself-alive)

---

## 1. A sampler that cannot die quietly

**⛔ Not a systemd timer. A long-running service with systemd-native liveness.**

`[measured]` The sampler this replaced was a timer, and at the moment it was replaced it read
`active (elapsed)` with `Trigger: n/a` and had last written a sample **24 h earlier**. Silently
dead, with nobody noticing. Its unit was:

```ini
[Timer]
OnBootSec=30s
OnUnitActiveSec=60s
```

A timer of that shape, once (re)started long after boot, has `OnBootSec` already in the past and
**no last-activation anchor** for `OnUnitActiveSec`, so systemd computes no next elapse and it
never fires again. There is no failure, no alert, and `active (elapsed)` reads like health.

If you must use a timer, anchor it on a calendar (`OnCalendar=*:0/2`, `Persistent=true`) — see
[`reboot-survival.md §5`](reboot-survival.md#5-classifying-a-boot-crash-recurrence-by-the-effective-environment).
For a sampler, don't:

```ini
[Service]
Type=notify
NotifyAccess=main
# The loop pings every ~20 s; 180 s tolerates 9 missed pings before systemd acts.
WatchdogSec=180
Restart=always
RestartSec=10
Nice=10
# Read-only observer. It must never be able to perturb what it watches.
StandardOutput=append:<stack>/ops/vram/watchdog.log
StandardError=append:<stack>/ops/vram/watchdog.log
```

So a **wedged** sampling loop is killed and restarted by systemd rather than going quiet, and
`Restart=always` covers an outright crash. Belt and braces: the script **alerts on its own
startup** if the on-disk heartbeat shows a sampling gap, and says plainly that no data exists for
that window. A gap you are told about is a gap; a gap you are not told about is data you will
later believe is complete.

Three interface modes a sampler should have, because each answers a question someone asks under
pressure:

```
vram-watchdog.py              run the sampling loop (what systemd runs)
vram-watchdog.py --once       take exactly one sample and exit
vram-watchdog.py --selfcheck  exit 0 fresh / 1 stale / 2 never ran
vram-watchdog.py --status     human summary of the latest sample
```

`--selfcheck`'s three-way exit is the one to copy: "stale" and "never ran" are different
incidents and a boolean conflates them.

**Files it maintains**, all gitignored — they are observations, not source:

| file | what |
|---|---|
| `samples/vram-YYYYMMDD.jsonl` | the ring, one object per sample, rolled per calendar day by the writer itself |
| `latest.json` | the most recent sample, rewritten in place: the programmatic consumer |
| `heartbeat.json` | liveness, for `--selfcheck` and for the startup gap alert |
| `baseline.json` | rolling median of *healthy* samples (§2) |
| `alert-state.json`, `alert-spool.jsonl` | de-duplication, per-hour cap, and spool-then-drain so an alert-transport outage does not lose alerts |

**Metrics only.** It never logs prompt, message or completion content. That is a property to
state in the header of anything that samples a model server, because the obvious next feature
request violates it.

### What one sample carries

Per GPU: `total_mib`, `used_mib`, `free_mib`, `util_pct`, `temp_c`. Plus the per-process VRAM map
and a process count — which is what answers *"who is holding the card"* without resolving pids by
hand afterwards. From the backend: `running_reqs`, `queue_reqs`, `token_usage`, `used_tokens`,
`max_total_num_tokens`, `max_running_requests`, the recurrent-state slot count, `page_size`,
`chunked_prefill_size`, `mem_fraction_static`, `pool_free_tokens`, and the **cumulative**
throughput counters. From systemd: `active`, `sub`, `result`, `main_pid`, `n_restarts`,
`main_start_monotonic`. Plus the derived `baseline_free_mib`, a `healthy` boolean and a `flags`
list.

**⛔ Keep the cumulative counters, not just the gauges.** `prefill_tok_total`, `decode_tok_total`,
`requests_total` and the busy-microsecond counters are what let you compute an honest rate over
*any* window afterwards:

```
sustained_decode_tok_per_s = (decode_tok_total[t1] − decode_tok_total[t0]) / (t1 − t0)
```

A cumulative counter integrates every request whether or not a sample landed inside it, so it
does not alias at any cadence. A **20 s point sample of an occupancy gauge does**, and it aliases
*downward*: on this deployment a `running_reqs` gauge averaged **0.26** over three hours, which
the obvious formula turns into "the backend is idle 78 % of the time, add concurrency", while the
cumulative decode counter over the same window says **131.4 tok/s sustained** — a factor of
**six**. That is [client rule 12](../clients/README.md#12-more-client-concurrency-is-net-negative-here-and-the-usual-instrument-hides-it),
and the reason it is repeated here is that **the sampler is where the right instrument has to be
recorded.** You cannot reconstruct a counter difference from gauges after the fact.

---

## 2. Thresholds: why the obvious absolute floors are useless here

**⛔ The measured baseline was 515 MiB free, not the 3.39 GB the engine logs at startup.**

`[measured]` The engine logs `available_gpu_mem=3.39 GB` at boot; `nvidia-smi` reports **515 MiB**
free on the same card, pinned, with **zero variance over a 40 s window**. The ~2.9 GiB gap is
memory torch has **reserved** in its caching allocator but not allocated. torch can reuse that for
tensors; a Triton kernel JIT-loading a CUDA module **cannot** — module load goes through the CUDA
driver, which only sees the 515 MiB. **That gap is the crash mechanism**, and it means absolute
free-VRAM thresholds of 1.5 GiB / 0.8 GiB are already breached at healthy steady state.

`[measured]` The 0.8 GiB "critical" threshold fired on **48 of 48** samples. Paging on it would
make the sampler a permanent siren that everyone mutes — which is how the previous sampler became
useless before it became dead.

So the page rule is **deviation-based**, against a baseline the sampler learns:

```
page when   free_mib  <  max(FREE_FLOOR_MIB, baseline_mib − FREE_DROP_MIB)
            # defaults: FREE_FLOOR_MIB = 256, FREE_DROP_MIB = 192
```

and the absolute thresholds are still **evaluated and recorded in every sample as flags**
(`abs_warn` / `abs_crit`) so a post-mortem has them — they just do not page. Recording a
threshold you do not act on is cheap and keeps the post-mortem honest.

**Only healthy samples feed the baseline**, so a crash-window dip cannot drag it down and mask the
next one.

**⛔ And know what the deviation rule cannot see.** `[measured]` At `mem-fraction-static 0.98` the
steady state was **287 MiB** free, flat for the four minutes before a fatal OOM. The baseline rule
was silent because its rolling median had **learned 287 MiB as normal**
(`threshold = max(256, 287 − 192) = 256`; free 287 ≥ 256, no alert). Baseline-relative detects *a
new consumer stealing VRAM*. It cannot detect *"normal is already one allocation from death"*,
which is what happened. That second question belongs to the admission gate, not the monitor, and
it is why the client-side floor is `largest_fatal_allocation + max_running_requests ×
per_prefill_transient` rather than anything baseline-derived — see
[`clients/hermes/docs/prefill-capacity-throttle.md`](../clients/hermes/docs/prefill-capacity-throttle.md).

**Queue depth pages only when sustained.** Transient queueing is the designed behaviour at a small
`--max-running-requests`: `[measured]` a depth of 1–2 is normal with 2–3 live sessions and is not
an incident. `QUEUE_WARN=3` with `SUSTAIN_N=6` consecutive samples. A monitor that pages on the
designed behaviour trains its audience to ignore it.

---

## 3. Two traps in reading an engine log

Both of these cost real diagnosis time, and both are the kind of thing that makes a correct
investigation reach a wrong conclusion.

### (a) Embedded NUL bytes make `grep -n` stop listing

An engine log contains NUL bytes. `grep -n` on it prints `binary file matches` and **stops
listing**, so a grep for the exception class **silently hides every later match**.

`[measured]` A plain `grep -n torch.OutOfMemoryError` over one **648 MB / 4.56M-line** log
surfaced the two oldest deaths and hid the two recent ones that were actually being investigated —
i.e. it surfaced the two that did not matter while hiding the two that did. Nothing warned.

```sh
# ad-hoc
tr -d '\000' < engine.log | grep -n 'torch.OutOfMemoryError'
```

```python
# in code: read binary, decode permissively — never rely on text mode
with path.open("r", errors="replace") as fh:
    ...
```

Every reader in this layer does the second. `grep -a` also works for the ad-hoc case; the point is
that the **default** is wrong and fails quietly.

### (b) The outermost exception is usually not the cause

So the capture scans the whole window for known culprit signatures and records a **cause chain**,
not just the top frame. Two measured examples, both from the same lane:

**Chain, three deep.** The outermost exception was
`torch.distributed.DistBackendError: NCCL error … unhandled cuda error`, which is useless on its
own. Beneath it:

```
cause chain
  cuda-calloc-failed    Failed to CUDA calloc 536870912 bytes
  triton-jit-late-load  Triton kernel 'apply_token_bitmask_inplace_kernel' device-loaded
                        after serving started (free device mem: 0.54 GiB). Pre-load it
                        during engine init to avoid CUDA OOM.
  nccl-unhandled-cuda   torch.distributed.DistBackendError: NCCL error in: ...
```

The actionable line is the **warning**, three lines earlier than the exception, and it is not an
exception at all. A capture that only parses tracebacks would miss it entirely.

**A traceback that points at the wrong layer.** A CUDA-graph capture failure:

```
RuntimeError: info.status != cudaStreamCaptureStatusInvalidated
INTERNAL ASSERT FAILED at ".../c10/cuda/CUDACachingAllocator.cpp":2213
```

Its deepest frame was a BF16 skinny-GEMM's output allocation (`torch.empty((m, n), …)`) in a
kernel with nothing to do with the fault. The capture had been poisoned **asynchronously, by
another thread** — a stats reporter doing a device-to-host copy plus a stream synchronize on a
30 s tick, which invalidates any in-flight capture because the graph is captured with
`capture_error_mode="global"`. Evidence that settled it: tick timing matched the failure to the
second, and it reproduced **7 / 24** with the thread and **0 / 24** without it, with no engine
involved.

**The generalisation.** A traceback localises the **detection**, not the cause. Whenever a fault
can be introduced by a thread other than the one that raises, the frame you are shown is
evidence about *who noticed*. Ask what else was running on that stream, on that tick.

Once such a cause is known and mitigated by an environment change, the same symptom carries two
different meanings and an alert must say which — that is
[`reboot-survival.md §5`](reboot-survival.md#5-classifying-a-boot-crash-recurrence-by-the-effective-environment).

---

## 4. One artifact per death

`incident_<YYYY-MM-DDTHHMMSS>_<cause>.md` plus a `.json` sidecar, written beside each other so
**`ls` is triage**:

```
incident_2026-09-29T164524_cuda-calloc-failed.md
incident_2026-09-30T164417_cuda-oom-prefill.md
incident_2026-09-30T232858_runtimeerror.md
```

The filename carries the **actionable** cause, not the outermost exception class — note the third,
where no culprit signature matched and the artifact says so rather than inventing a label:
`_no known culprit signature matched; read the traceback below_`. A classifier that always
produces a class is a classifier you cannot trust.

### What one artifact contains

| section | why it is there |
|---|---|
| cause chain | every culprit signature matched, with its evidence line |
| the allocation | what was wanted vs. what the **driver** had free, and which pids held the card |
| pre-death batch table | `pool / recurrent-state / running / queue / pending-token` per batch for the last ~12 batches — the numbers the scheduler actually had |
| free-VRAM history | from the sampler's 20 s ring. **Nothing else on the box keeps this** |
| exception + traceback | bounded (default 120 lines) |
| last N scheduler lines | verbatim, default 40 |
| card state at capture | GPUs and compute apps |
| outage accounting | the systemd side: exit, restart count, `RestartUSec`, `TimeoutStopUSec`, `StartLimit*`, and the journal window |
| server flags in force | **read from `/proc`, not from the unit file** |

**⛔ Read the `pool` column against the allocation. That pairing is the whole point.** A pool well
under 1.0 with a failed allocation means the shortage was **driver-level VRAM outside the KV
pool**, and no KV-pool admission knob can see it. `[measured]` One of these artifacts shows pool
`0.27` in the two batches before the death. Capturing either number alone would have supported the
wrong conclusion, and it did: the first hand-diagnosis of a related incident was *"orphaned
scheduler holding 77 GiB"* when the holder was a healthy, active peer unit — and an auto-reap keyed
on that diagnosis would have killed it.

**⛔ Server flags from `/proc`, never from the unit file.** The unit text says what the *next*
start will use. `/proc/<pid>/cmdline` says what the process that just died was actually running.
After a staged drop-in, a rollback, or a restart from a different build, those differ — and when
they differ is exactly when you are reading the artifact.

### How it runs

**Folded into the sampler service, not given a timer of its own.** The sampler already queries
the unit every 20 s, and the timer it replaced is *why* it is a service (§1). The sampler calls the
capture:

- **on the restart edge** — the moment `NRestarts` increases, while the pre-death samples are
  still in the ring; and
- **every 15 cycles (~5 min)** as a backstop, in case the sampler itself was down when the lane
  died.

**Out-of-process, with a 120 s timeout, and every failure logged and dropped.** A capture problem
must never be able to affect the observer, let alone the lane. It never acts: it opens no
connection to the inference server and never restarts, kills or reconfigures anything — it reads
files and runs `systemctl show` and `nvidia-smi`.

### ⛔ Forward-only reads, or the cost is unbounded

The log is only ever read **forward** from the byte offset of the last scan, persisted in a state
file. A sweep therefore costs `O(bytes appended since the last sweep)`, not `O(648 MB)`. A first
run, or a truncated or rotated log, falls back to a bounded window from EOF.

| env | default | meaning |
|---|---|---|
| `LANE_INCIDENT_ROOT` | `<stack>/ops/incidents` | where artifacts land |
| `LANE_INCIDENT_LOG` | — | scheduler log to scan |
| `LANE_INCIDENT_UNIT` | — | unit to interrogate |
| `LANE_INCIDENT_CONTEXT_LINES` | 40 | scheduler lines kept before the exception |
| `LANE_INCIDENT_TRACEBACK_LINES` | 120 | traceback lines kept |
| `LANE_INCIDENT_TAIL_BYTES` | 32 MiB | first-run / rotated-log window |
| `LANE_INCIDENT_MAX_SCAN_BYTES` | 512 MiB | ceiling on one forward scan |
| `VRAM_WD_INCIDENT_SWEEP_EVERY` | 15 | sampler cycles between backstop sweeps |

Raise `LANE_INCIDENT_TAIL_BYTES` to reach further back when reconstructing an older death —
`[measured]` one of the two above needed ~160 MiB:

```sh
LANE_INCIDENT_TAIL_BYTES=$((160<<20)) lane-incident-capture --backfill 3
lane-incident-capture --dry-run --backfill 1   # what would it write for the last death
lane-incident-capture                          # one idempotent sweep
```

`--dry-run` plus `--backfill` is the pair that makes this safe to develop against a live log:
you can re-derive a historical artifact without writing one, which is also how you discover that
your classifier would have mislabelled it.

---

## 5. Is the capture itself alive?

A silent backstop is indistinguishable from a broken one, and the dead timer in §1 is the
cautionary tale. So the backstop leaves a **timestamp** rather than log spam:

```sh
jq -r .last_sweep <stack>/ops/incidents/state.json    # should be < ~6 min old
```

and `log_offset` advancing between sweeps proves it is really reading the log, not just waking up.
`[measured on deployment]` A sweep at 17:07:33, exactly 15 sampler cycles after the service
started, offset `+263 KB`, no new incidents found — which is the full health check, and it is two
commands.

**The generalisable rule: a component whose job is to notice things must publish a cheap,
positive proof that it ran.** Not a log line that scrolls away, and not the absence of errors. A
timestamp and a monotonically advancing position, both readable in one command, both of which are
*wrong* in a detectable way if it has stopped.

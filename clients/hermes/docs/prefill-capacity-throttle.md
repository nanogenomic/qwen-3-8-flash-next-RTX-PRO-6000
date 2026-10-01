# The night the throttle stopped throttling because the backend got better

The implementation behind [client contract rule 15](../../README.md#15-a-proxy-stops-being-a-proxy-when-the-thing-it-proxied-for-moves),
which corrects [rule 9](../../README.md#9-throttle-fan-out-on-vram-headroom-not-only-on-pool-utilisation).
Written up rather than shipped as a diff — see [the README](../README.md#why-this-round-is-write-ups-and-not-patches).

---

## The shape of it

Rule 9 exists because the backend died of a CUDA OOM while its KV pool was only 73 % used. The
lesson was correct: read device-free VRAM, not just pool utilisation. The client implemented it as
an admission divisor —

```
affordable_children = free_mib // PREFILL_ALLOC_MIB      # 560 MiB
```

— where 560 MiB was not a guess. It is the allocation size from the fatal OOM's own message,
`Tried to allocate 560.00 MiB … 280.94 MiB is free`. At the geometry that rule was written against,
`935 // 560 == 1`, and one child at a time was the right answer.

About thirty hours later the engine side of this repository landed two changes on the same evening:

| change | effect |
|---|---|
| QSA KV pool moved to pinned host RAM ([BENCHMARKS §3.15](../../../BENCHMARKS.md#315-host-resident-qsa-kv-pool--1310720-tokens-on-one-96-gb-card)) | pool 451,264 → 1,310,720 tokens; free device VRAM **935–1,533 → 6,463 MiB** |
| bounded QSA prefill gather ([§3.16](../../../BENCHMARKS.md#316-bounded-qsa-prefill-gather--the-transient-that-scaled-with-context)) | prefill transients **flat at 32 MiB** at every context length, from 1,024 MiB at 512K |

Both moved the divisor the same way. `6_463 // 560` is **11**.

No exception. No warning. No log line saying the gate had changed meaning. The client went from
licensing about one concurrent child to licensing eleven, at the exact moment the backend acquired
the headroom that made the old number wrong — and the eleven were admitted against a backend whose
scheduler still only runs four requests at a time.

## What it cost, measured on the lane

| | observed |
|---|---|
| Main lanes' own turn-average decode | **11–17 tok/s** (solo rate on this build is 158–178) |
| Requests queued | **15** |
| Uncached tokens waiting to be prefilled | **588,268** at the moment the gate was rewritten, and **668,196** read live an hour later — up to **51 %** of the entire 1,310,720-token pool, queued for prefill |
| KV pool utilisation at the same moment | **14–32 %** |
| `cache_hit_rate` during the stall | **0.0** |

The last two rows are the whole problem in miniature. Every pool-derived signal read healthy. The
pool *was* healthy. What was saturated was prefill throughput and prefix-cache residency, and
nothing the client was looking at could see either.

## Why free VRAM had stopped being a signal

Free device VRAM was never the quantity that mattered — it was a **proxy** for prefill working set,
because prefill transients and the KV pool both lived on the device. After the cutover:

- the pool is in host RAM, so it no longer moves device-free VRAM at all;
- transients are flat at 32 MiB regardless of context length, so they barely move it either.

So the reading stopped responding to load. It just sat at ~6,463 MiB and reported capacity.
**Dividing a number that does not respond to pressure is not a throttle; it is a constant.** The
same trap waits in any gate keyed on a resource whose consumer has since been offloaded, cached or
compressed — and offloading is precisely what capacity work does.

## The replacement: gate on what actually binds

The scarce resources, in the order they bind on this deployment:

1. **Prefill throughput**, against `--max-running-requests 4` and `chunked_prefill_size 4096`.
   Every admitted request shares one 4,096-token chunk budget per step, so a main's 300–500K
   re-prefill is serialised behind whatever else is in flight.
2. **Prefix-cache residency.** A child with a distinct prompt evicts a main's cached prefix, and the
   main then pays its whole prompt again next turn. Measured waste per call: **2,622 tokens** with no
   contention, **114,385** against five in-flight peers.
3. **Scheduler slots.** A main that cannot get one waits, however much pool and VRAM are free.
4. **Pool memory.** Last. At 14–32 % used it was never the constraint.

Three conjunctive legs, each read live:

**(A) Slots, one reserved per live main before any child is counted.** Every main needs a slot for
its own control-plane traffic — compaction, titling, the goal judge — so children are counted
against what is left once the mains are seated. Total streams are capped at **2**, derived from the
measured concurrency curve rather than chosen: a lane keeps **100 % / 74 % / 57 %** of its solo
decode rate at concurrency 1 / 2 / 3, so 2 is the largest concurrency at which it keeps two thirds
of itself. The cap is a runtime key, because raising it buys aggregate throughput at a lane's
expense and that is a product decision, not a measurement.

There is **no interior aggregate maximum** below `--max-running-requests 4` to cap at — aggregate
rises monotonically through concurrency 3 on both pool geometries
([§3.20](../../../BENCHMARKS.md#320-throughput-versus-concurrency-warm-segmented--and-the-071-retraction)).
That is why the cap keys on a lane's own rate instead.

**(B) Prefill backlog against a drain budget.** `num_waiting_uncached_tokens` divided by a prefill
rate the client calibrates from the server's own cumulative counters on `/v1/loads`:

```
rate = total_prefill_uncached_tokens / total_prefill_busy_us
```

Measured **8,481 tok/s**, and **8,377 tok/s** read again an hour later — 28,285,018 tokens over
3,375.99 s. Checked against a fixed drain budget (here the control-plane judge timeout, 30 s).
Within the remaining headroom each child is charged the eviction waste it causes, interpolated
between the two measured anchors above. Self-calibration is the point: the divisor survives a model
swap, a quantisation change or a different card with nobody editing a constant. Log-derived
chunk-saturated prefill rates on the same lane — median 8,958 tok/s, p25–p75 7,257–10,191 — agree
with the counter form to within 6 %.

**(C) Pool residency**, with backlog tokens charged against the pool before children. It correctly
does not bind at 14–32 %; it is in the gate so that it binds when it should.

### VRAM is demoted to a floor that should almost never fire

```
floor = PREFILL_ALLOC_MIB + max_running_requests × PREFILL_TRANSIENT_MIB
      = 560 + 4 × 32 = 688 MiB
```

"The card can still absorb the largest allocation measured to be fatal, plus one measured transient
per slot the scheduler can fill." Below it, admit nothing and let in-flight work drain — the lane
died twice in one day on VRAM exhaustion and that path stays armed. Above it, VRAM contributes **no
cap at all**.

The test a threshold has to pass is that it separates the geometries that matter:

| geometry | free VRAM | verdict |
|---|---|---|
| the fatal OOM, `--mem-fraction-static 0.98` | 287 MiB | **critical** ✓ |
| pre-cutover steady state | 935 MiB | clear ✓ |
| post-cutover | 6,463 MiB | clear ✓ |

⛔ **An absolute floor from the monitoring side would not have worked.** The watchdog's own
suggested floors — 1.5 GiB warn, 0.8 GiB critical — were *permanently* breached at 0.98 (the warn
floor is still always breached at 0.97), so "admit no child below 0.8 GiB" means "never admit a
child". Its baseline-relative rule was silent for the opposite reason: the baseline is a rolling
median of healthy samples, so it had **learned 287 MiB as normal**. Baseline-relative detects a new
consumer stealing VRAM. It cannot detect that normal is already one allocation from death.

⛔ **It must read runtime free VRAM, never boot-time `available_gpu_mem`** — that field was
measured reporting 3.42 GB while the card had 287 MiB free while serving. The sample used here is
the monitor's own 20-second runtime sample, pulled over a multiplexed SSH channel (19 ms per read,
paid only by the process refreshing the shared allowance, never by a dispatch), with a 90-second
staleness check that **fails open** rather than trusting an old number.

## Verified effect

At the live geometry that had yielded eleven children — 588,268 uncached tokens waiting (≈ 69 s of
backlog), 2 mains, 3 running, 5,651 MiB free — the replacement gate yields **0** child streams. No
main's window, growth ceiling or request priority is touched by any of it; the share-bounded
per-main grants of [rule 10](../../README.md#10-bound-the-ceilings-you-grant-not-the-reservations-you-measure)
are unchanged.

## Two things that generalise past this gate

- **Every threshold resolves at call time** — environment variable, then a config file, then the
  module default — so retuning a throttle does not mean relaunching the sessions being throttled.
  The tunables are published back to clients with each allowance refresh.
- **Refuse the second concurrent child outright rather than admitting six and self-clamping.** A
  gate that admits and then recovers has already paid the prefill, the eviction and the queue. The
  state this was written to prevent — three children on three of four slots with fifteen queued —
  is reachable by a gate that is merely *eventually* correct.

## What this does not establish

- The concurrency curve the stream cap is derived from is **not a controlled A/B** between the two
  pool geometries: different boots, different hours, different pool sizes, different client mixes,
  and the host-KV concurrency-3 cell has n = 71. See §3.20 and §6.
- The eviction-waste model interpolates between **two** measured anchors. It is a model, not a
  curve, and rule 12 says to re-measure it on your own prefix sizes.
- The drain budget is set to one framework's judge timeout. That is a defensible default, not a
  derived quantity.

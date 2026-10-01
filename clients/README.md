# The client contract

What any agent orchestrator — any client framework, on any engine — has to do to use a
shared, self-hosted backend like this one well.

The engine work in this repository makes the backend faster and larger. It cannot make a
careless client efficient. Every rule below is here because breaking it cost something
measurable on the reference deployment, and each one comes with that evidence. None of it
is specific to one framework; wherever an example names a framework's setting, the rule
underneath applies to every orchestrator.

A **reference implementation** for Hermes Agent lives in [`hermes/`](hermes/). It is MIT
licensed (unlike the rest of this repository, which is Apache-2.0). Two rules ship there as
applicable patches — rule 3 (patch 0002) and own-versus-shared status accounting (patch 0001).
Rules 10–14 were implemented on that framework too, but are published here as **write-ups
rather than diffs**; its README says why.

> **These rules assume a backend that is up, and a monitor you can believe.** Neither is free.
> The control-plane layer that keeps them true — the cross-session pool broker that publishes
> the allowance rules 10 and 11 read, reboot survival, the sampler whose numbers rules 9 and 15
> gate on, log rotation, and how a serving config change lands without restarting what it
> configures — is [**`../ops/`**](../ops/README.md). Rules 9, 10, 11, 14, 15 and 16 each have a
> counterpart there; the eleven ops lessons at the top of that README are the systemd, tmux and
> log-reading failures underneath them, and every one is silent.

Contents:

1. [Priority-tag control-plane calls](#1-priority-tag-control-plane-calls)
2. [Prioritise the main agent only when a human is waiting](#2-prioritise-the-main-agent-only-when-a-human-is-waiting)
3. [Size subagent context explicitly — never inherit it](#3-size-subagent-context-explicitly--never-inherit-it)
4. [Never advertise a window larger than the pool, or smaller than it grew to](#4-never-advertise-a-window-larger-than-the-pool-or-smaller-than-it-grew-to)
5. [Grow instead of compact when the pool allows](#5-grow-instead-of-compact-when-the-pool-allows)
6. [Hybrid models: warm conversations need state slots too](#6-hybrid-models-warm-conversations-need-state-slots-too)
7. [Isolate test state from live state](#7-isolate-test-state-from-live-state)
8. [Set the compaction threshold explicitly — beware step functions](#8-set-the-compaction-threshold-explicitly--beware-step-functions)
9. [Throttle fan-out on VRAM headroom, not only on pool utilisation](#9-throttle-fan-out-on-vram-headroom-not-only-on-pool-utilisation)
10. [Bound the ceilings you GRANT, not the reservations you MEASURE](#10-bound-the-ceilings-you-grant-not-the-reservations-you-measure)
11. [Show the share, and degrade honestly when you cannot](#11-show-the-share-and-degrade-honestly-when-you-cannot)
12. [More client concurrency is net-negative here, and the usual instrument hides it](#12-more-client-concurrency-is-net-negative-here-and-the-usual-instrument-hides-it)
13. [Give background work an explicit priority, and stop paying for calls nobody reads](#13-give-background-work-an-explicit-priority-and-stop-paying-for-calls-nobody-reads)
14. [Classify an interrupt by provenance, or your autonomous loop will stop on its own](#14-classify-an-interrupt-by-provenance-or-your-autonomous-loop-will-stop-on-its-own)
15. [A proxy stops being a proxy when the thing it proxied for moves](#15-a-proxy-stops-being-a-proxy-when-the-thing-it-proxied-for-moves) — ⚠️ **corrects rule 9**
16. [Delegate the wait; never park the goal on it](#16-delegate-the-wait-never-park-the-goal-on-it)

If you read only one of these, read **15**. It is the only rule here that fired because the
*backend* improved, and it is the shape most likely to be hiding in your own client.

---

## 1. Priority-tag control-plane calls

**Rule.** Tag every small, latency-critical *control-plane* call — a goal-loop judge,
context compaction, summarisation, a tool-result post-processor — with high priority,
**regardless of what kind of turn triggered it.** And treat a judge's *transport* failure
as "keep going", not "stop".

**Why tagging matters.** With `--enable-priority-scheduling`, a request that carries no
`priority` field does not land in the middle of the queue. SGLang assigns it the most
extreme value in the losing direction unless `--default-priority-value` is set — below
even an explicit priority of 0 — and logs a warning at boot for exactly that
configuration. (Traced in the scheduler's `_set_or_validate_priority`; the Responses API is
the one exception and defaults to 0.) So an orchestrator that raises its main agent and
leaves its control-plane calls untagged puts them behind *all* bulk work.

**Why "keep going".** `[measured on the reference deployment, 2026-09-29]` An
orchestrator's goal-loop judge — a tiny completion whose only job is to say "continue",
"wait" or "done" — timed out **25 times in one day** while the backend was busy with bulk
subagent work, including **six in a row** between 18:21 and 18:58. Each attempt spent its
full ~30-second timeout before failing.

Hermes Agent, for one, counts consecutive judge transport failures and **auto-pauses the
goal at five** (`DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES = 5`, present in both 0.20.4
and 0.21.5). That rule is designed for a revoked API key, where every future call will fail
too and pausing is right. On a shared self-hosted backend a judge timeout usually means
*busy*, not *broken*, and the same rule can stop a healthy autonomous agent because the box
was loaded. In the run above the session logged "falling through to continue" after each
timeout and kept working; whether a given run of timeouts reaches the pause depends on how
the framework counts them, which is exactly why the policy should be decided deliberately
rather than inherited from a default designed for a different failure.

What to do:

- Send control-plane calls at high priority whatever the turn type — an autonomous turn's
  judge is just as blocking as an interactive one's.
- Set `--default-priority-value` on the server so an untagged call degrades to "ordinary",
  not "last".
- Distinguish *busy* (timeout, 503, queue full) from *broken* (401, DNS, 404). Only the
  second should count toward an auto-pause. If the framework does not distinguish them,
  raise the threshold for a self-hosted backend.

## 2. Prioritise the main agent only when a human is waiting

**Rule.** Raise the main agent's priority for turns a human started and is watching.
Autonomous continuations should run at the same priority as subagents.

**Why.** `[measured, second test card — see the note at the end]` Priority reallocates
throughput; it does not create any. At 12 streams saturated by 14 background streams, giving
a main stream priority cut its time to first token from **5.24 s to 1.11 s** (p90 5.63 →
1.38 s) while its decode rate stayed the same (28.6 vs 28.9 tok/s). The cost landed on
everyone else: background streams completed **113 requests instead of 152 in the same
window — 26 % fewer.**

That trade is worth it when a person is waiting on the main agent's first token. It is not
worth it for an autonomous loop that no one is watching, where it simply starves the
subagents doing the actual work.

## 3. Size subagent context explicitly — never inherit it

**Rule.** State every subagent's context window, output reserve and compaction threshold
explicitly, and check that the geometry is satisfiable. Never let a child inherit its window
from the parent's configuration.

**Why.** `[measured on the reference deployment]` A capacity tool lowered a parent profile's
window to **65,536** to relieve pool pressure. The client framework derived its subagents'
windows from that same value, so every subsequently spawned subagent silently inherited it,
with a 24,576-token output reserve:

```
effective input budget = 65,536 − 24,576             = 40,960
threshold floor        = max(0.75 × 40,960, 64,000)  = 64,000  ≥ 40,960  → degenerate
compaction trigger     = 0.85 × 40,960               = 34,816 tokens
```

At the trigger the child held ~34.9K tokens, of which ~25K (system prompt, tool schemas,
protected head) and ~10.2K (protected tail) were incompressible. Only ~10K could be
summarised — not enough to get back under 34,816. Compaction ran, made no progress, aborted,
and **the agent stalled outright**, with nothing in any log connecting the stall to a config
edit made in a different process.

The check that would have caught it, for window `W`, output reserve `R` and incompressible
floor `F`:

```
0.85 × (W − R)  >  F        with real margin
```

At `W = 65,536`, `R = 24,576`, `F ≈ 35K`: 34,816 > 35,000 is false. Note the second-order
problem too: a 24,576-token reserve is **37.5 %** of a 65,536 window. Keep the output
reservation proportionate to the window.

The Hermes reference implementation resolves and logs the child window explicitly and never
auto-selects a geometry below 81,920 — [`hermes/docs/child-context-window.md`](hermes/docs/child-context-window.md).

## 4. Never advertise a window larger than the pool, or smaller than it grew to

**Rule.** Whatever advertises a context window to clients — a router, a gateway, the
orchestrator's own configuration — must keep it at or below the backend's **live**
`max_total_num_tokens`, re-read it whenever the backend is reconfigured, and raise it when
the pool grows. Reserve pool for what requests actually use, not for every client's
theoretical maximum.

**Too large.** `[measured]` A request that fits the declared context window but not the
pool is refused: a 520,059-token prompt against a 519,040-token pool returned HTTP 400, and
a ~500K prompt at the very edge of that pool came back as an empty completion. This matters
on this backend because the declared window and the pool are separate numbers. The
reference deployment declares `--context-length 540000` and has a **451,264-token pool** — so
the right per-request window to advertise is about 393,216, not 540,000. Note that the pool
moved twice in two days as the server was retuned (643,456 → 522,880 → 451,264), which is the
point of this rule: **re-read it, do not hard-code it.**

**Too small.** `[observed on the reference deployment]` After an engine change grew the pool
to 331,456 tokens, a router kept advertising `max_model_len` 241,728. Every client silently
ran about **20,400 tokens short** of the real 262,144 window until it was corrected. Nothing
errored — requests were simply truncated to a window nobody had chosen.

**Reserve by measured use.** What has to fit the pool is the in-flight set, bounded by
`--max-running-requests`, not the sum of every client's maximum. One long primary session
plus a few small subagents fits; four long primaries do not — see
[Sizing for multiple clients and subagents](../README.md#sizing-for-multiple-clients-and-subagents).

## 5. Grow instead of compact when the pool allows

**Rule.** When a conversation approaches its window and the backend's pool has room, raise
the window rather than compacting. Compact only when the pool genuinely cannot hold the
larger conversation.

**Why.** Compaction rewrites the conversation, so the next request's prefix no longer
matches anything in the prefix cache and the whole history is re-prefilled from scratch.
Growth keeps the prefix byte-identical, so the next turn is a cache hit. `[measured, second
test card]` On this backend losing a cached prefix is expensive: prefill runs at
12.2–12.6K tokens/s up to ~100K and falls to ~8.7K at ~390K, so re-reading a ~290K history
costs about 29.5 s before the first new token. In a multi-turn agent test, turns that lost
their cached prefix took **16.8 s** to first token against **2.5 s** for turns that kept it.

**Caveat — do not grow across a threshold step.** In Hermes Agent, growing a window past
512,000 tokens moves compaction *earlier*, not later (rule 8). Growth only helps if the new
window's compaction trigger is actually above the old one.

No controlled grow-versus-compact benchmark is published here: the evidence above is the
cost of the re-prefill that compaction forces, measured directly.

## 6. Hybrid models: warm conversations need state slots too

**Rule.** On a hybrid linear-attention model like this one, size the backend's
recurrent-state slots for the conversations you want to keep warm, not just for the requests
running at once.

**Why.** A cached prefix is reusable only if the conversation's linear-attention state
snapshot survives alongside its KV. With too few slots an idle conversation's snapshot is
evicted and its next turn re-prefills the full history, even though the KV is still in the
pool. `[measured, second test card]` Six agent conversations at 4 running requests: 12 slots
gave a **5.4 %** prefix-cache hit and 16.8 s to first token; 36 slots gave **95.6 %** and
2.5 s.

Rule of thumb, when live conversations exceed running requests:

```
slots  ≳  3 × max_running_requests  +  3 × (conversations to keep warm)
```

and 3 × running requests when they do not. For the client, this means: **know how many
conversations you keep open between turns, and tell whoever runs the backend.** Full
evidence and the limits of the rule are in
[If you run agents on this model](../README.md#if-you-run-agents-on-this-model-set---max-mamba-cache-size-first).

**If two or more of those conversations are long (hundreds of thousands of tokens), slots are not
enough on their own.** One deep cold prefill donates a checkpoint per 4,096-token chunk and can
evict every other conversation's checkpoint regardless of slot count. The backend needs
`--mamba-max-states-per-path 2`; see
[the README](../README.md#two-or-more-long-agents-on-one-card-add---mamba-max-states-per-path-2).
From the client side the symptom is a long agent whose turn reports `cached_tokens` near zero
right after another lane's cold prefill — measure it, and tell whoever runs the backend.

## 7. Isolate test state from live state

**Rule.** A client framework's own test suite must never be able to write a live allocation,
lease, override or ledger file. Derive every such path from the framework's configurable
home directory, and have the test harness redirect it — automatically, not by convention.

**Why.** It happened twice in this project:

- A context governor read its home directory from a hard-coded default instead of the
  framework's configurable home. **A single test run advanced the live profile's window
  override to expire exactly 4 hours later** (now + 14,400 s), changing the behaviour of the
  real agent the operator was running.
- A shared-pool module hard-coded its run directory in the same way. Any test that ran a
  subagent through the live code path wrote heartbeats into the **production** session
  directory, and could win that directory's lock and rewrite the live pool-allowance file
  every running agent sized itself from.

Both were fixed by deriving the path from the configurable home and having the test
fixtures isolate it. The general rule: if a test *can* reach live state, eventually one will.

## 8. Set the compaction threshold explicitly — beware step functions

**Rule.** Set the compaction threshold explicitly, as a fraction you have checked, rather
than relying on a default that changes with window size.

**Why.** `[reproduced on upstream Hermes Agent 0.20.4 and 0.21.5]` Hermes raises its
compaction threshold to 75 % only for windows **strictly below 512,000 tokens**, and leaves
larger windows at the 50 % default. The rule is a step function, so the trigger in tokens
**drops** when the window crosses 512,000. With a 24,576-token output reserve:

| Window | Effective threshold | Compaction trigger |
|---|---|---|
| 511,999 | 75 % | **365,567** |
| 512,000 | 50 % | 243,712 |
| 524,288 | 50 % | 249,856 |
| 700,000 | 50 % | 337,712 |

**A 524,288-token window compacts about 116K tokens earlier than a 511,999-token one, and even
a 700,000-token window compacts earlier than 511,999.** A user who raises the window from
500,000 to the power-of-two 524,288 expecting more headroom gets much less, and may see
compaction start firing every few turns on a workload that was previously stable.

This bites directly on this backend: the reference deployment declares
`--context-length 540000`. A client that mirrors that declaration as its own window lands
just above the step. Advertising the recommended ~393,216 stays below it.

Reproduce it on a checkout of either version, with an isolated home (rule 7):

```bash
HERMES_HOME=$(mktemp -d) python - <<'PY'
from agent.context_compressor import ContextCompressor as C
for ctx in (511_999, 512_000, 524_288, 700_000):
    pct = C._effective_threshold_percent(ctx, 0.50)
    print(ctx, pct, C._compute_threshold_tokens(ctx, pct, 24576))
PY
```

The fix belongs upstream: the trigger in tokens should be non-decreasing in the window. A
continuous floor — raising the threshold above 512,000 only as far as needed to keep the
trigger at or above the 511,999 case — does that. Until then, set it explicitly.

## 9. Throttle fan-out on VRAM headroom, not only on pool utilisation

> ⚠️ **Read rule 15 with this one.** This rule is correct about the *failure* and wrong as a
> *throttle*, and the reference deployment proved it the hard way about 30 hours after this rule
> was written. Everything below still holds for detecting the OOM. But device-free VRAM stops
> being a capacity proxy the moment the KV pool leaves the device — and a client that had turned
> this rule into a divisor silently stopped throttling at the exact moment the pool grew. Rule 15
> is what replaced it. **Keep VRAM as a floor; do not make it the gate.**

**Rule.** If your client throttles dispatch on backend pressure, read **device-free VRAM** as
well as KV-pool utilisation. A pool-based rule cannot see the failure that actually takes the
lane down.

**Why.** `[measured on the reference deployment, 2026-09-30]` The backend died of a CUDA OOM
while its **KV pool was only 73 % used**. Two agent sessions had fanned out; 14 requests were
queued; the scheduler was chunking 4,096-token prefills; and device-free VRAM was **280.94 MiB**
when prefill asked for 560 MiB. A client reading `token_usage` saw 0.73 and had every reason to
send more.

The asymmetry is structural. KV-pool utilisation describes memory the engine has already
reserved and is managing. What runs out under a deep queue is the **working set for prefill**,
which lives *outside* the pool, in memory no admission knob accounts for. So:

- **`token_usage` is necessary but not sufficient.** It is the right signal for "will this
  request's KV fit". It is silent about "will the engine have room to prefill it".
- **Queue depth is the better proxy** if headroom is not exposed to you. In the incident the
  pool reading rose gently from 0.64 to 0.73 while the queue sat flat at 14 — the queue was
  saying "prefill is the bottleneck" the whole time.
- **Prefer serialising to rejecting**, as in rule 3's reference implementation: a fan-out that
  runs its children one at a time still completes.

**And an alert is not a control.** On the reference deployment a 20-second-cadence monitor had
flagged critical headroom on **every sample that day**, sent a low-VRAM alert **16 h 15 min**
before the crash, and sent a queue-depth alert **53 s** before it. All of it was correct, and
none of it reached anything that could defer work. If your client can read headroom, it should
**act** on it — reduce concurrency, defer a fan-out, wait — not merely log it.

The monitor that produced those readings, why its absolute thresholds were useless (the critical
one fired on **48 of 48** samples at healthy steady state) and what replaced them, are in
[`../ops/observability.md`](../ops/observability.md). That page also carries the other half of this
rule: the sampler it replaced had been **silently dead for 24 h** while reading `active (elapsed)`.

## 10. Bound the ceilings you GRANT, not the reservations you MEASURE

**Rule.** If your client hands out per-session context budgets from a shared pool, the invariant
that checks them must sum the **ceilings it issued**, not the demand it currently observes. An
invariant built on measured demand is mathematically incapable of failing, and will report
healthy while the backend deadlocks.

**Why.** `[measured on the reference deployment, 2026-09-30]` A cross-session broker published a
per-process allowance every 5 s. On a 451,264-token pool it granted **two main agents the entire
pool each**:

```
pid A   main_window 451,264   main_grow_max 451,264   child_grow_max 451,264
pid B   main_window 451,264   main_grow_max 451,264   child_grow_max 451,264
invariant {ok: True, total: 406,137, limit: 406,138}
```

The invariant passed by one token — and it passed **by construction**. It summed each main's
*measured* need, scaled to fit the budget (`reserved = need × f`), so the total could never exceed
the limit no matter what the grants said. It was validating its own arithmetic, not its policy. A
second rail made it worse: a "mains are never shrunk" guard did `mg = max(main_grow_max,
main_window)`, which *raised* any bounded growth ceiling back up to whatever the session had
already grown to.

What that cost, from the broker's own log once the two mains had grown into their grants:

```
pool broker: mains are HOLDING 769,606 of 406,138 usable tokens (pids [A, B] are above their share)
```

with `token_usage ≈ 0.93` and **10 requests queued**. Both mains' prefixes were being evicted and
re-prefilled, which is rule 4's failure mode arrived at from the client side.

**The fix: derive every ceiling from the live pool, and water-fill.** Reserve one child stream per
granted stream out of `usable = pool − cache_reserve`, then distribute the remainder across live
mains: round one gives each `min(equal_share, max(need, guaranteed_lane_window))`, round two
redistributes what the under-users gave up, proportional to unmet want. Clamp on an 8K grid until
the *granted* worst case fits.

`[measured]` On a **451,264-token pool** (`cache_reserve` 10 %, so `usable = 406,138`), that yields:

| Live mains | Granted window each | Child grant | Granted worst case |
|---|---|---|---|
| 1 | **401,408** | 65,536 | 401,817 |
| 2 | **196,608** | 73,728 | 403,045 |
| 3 | **131,072** | 73,728 | 404,275 |
| 4 | **98,304** | 73,728 | 405,503 |
| 5 | 65,536 | 131,072 | 399,767 |

and asymmetrically, with one main demanding 433,782 beside one idle at 40,000: **270,336 / 131,072**,
the difference being a **revocable loan** of 61,386. Revocable matters: at the next 5 s refresh the
loan can be withdrawn, and withdrawing it does **not** shrink the borrower — it only stops granting
it further growth.

**⛔ State the pool with every ceiling you quote.** All of the numbers above are specific to a
451,264-token pool. The same code on the same deployment's current **1,310,720**-token pool grants
two mains **532,480** each. A ceiling table without its pool is meaningless.

**Two constants, and do not confuse them.** The reference implementation has a
`GUARANTEED_LANE_WINDOW` of **131,072** — that is the share a lane is guaranteed *before lending*,
not a floor. The actual floor is a separate and much lower `MAIN_WINDOW_FLOOR` of **64,000**. Proof
that the first is not a floor: at five live mains the grant drops to 65,536.

## 11. Show the share, and degrade honestly when you cannot

**Rule.** If sessions share a pool, the context denominator in your UI is the session's **current
share**, not its configured maximum. And when the share cannot be determined, say so rather than
falling back to a number that looks authoritative.

**Why.** A status bar reading `ctx 175K/393.2K` tells a user they have 218K of room. If the live
share is 196.6K, they have 21K. The configured maximum is the right denominator only on a lane
nobody else is using.

The reference implementation reads a document the broker publishes every ~5 s and renders one
field:

```
⇄ 2 mains · 451.3K pool                      ← normal; the share appears as the ctx denominator
⇄ 2 mains · 451.3K pool · share 196.6K ▲     ← the ▲ marks a session held ABOVE its share
⇄ pool stale 5m ⚠                            ← document present but not being refreshed
no share cap ⚠                               ← document written by a peer that does not bound grants
(field absent)                               ← no document at all: keep the static behaviour exactly
```

Three properties worth copying:

- **Staleness is judged against the publish interval, not guessed.** The document is rewritten
  every ~5 s, so a **90 s** threshold means the publisher is *gone*, not merely busy.
- **A grant above the pool ceiling, or grants that sum above it, is not a grant.** The reader
  treats that document as unbounded and says `no share cap ⚠` rather than rendering a number it
  knows is wrong. That is what makes rolling out rule 10 safe while older peers are still running.
- **The display path is read-only.** It never probes the backend, never requests an allowance, and
  cannot write. A status bar that acquires resources to draw itself is a status bar that changes
  the thing it measures.

## 12. More client concurrency is net-negative here, and the usual instrument hides it

**Rule.** Do not add client-side concurrency to raise throughput on a shared long-context backend.
Measure first — and measure it from **cumulative counters**, never from an occupancy gauge times a
per-stream rate.

**Why the instrument first.** `[measured on the reference deployment, 2026-09-30, 00:00–03:00Z]`
A 20-second-cadence sample of the backend's `running_reqs` gauge averaged **0.26**, which the
obvious formula turns into "the backend is idle 78 % of the time, effective **~22 tok/s**, so add
concurrency". The backend's own **cumulative** decode counter over the same window says **131.4
tok/s sustained** — a factor of **six**. Admission churn is sub-second, so a 20 s point sample of an
occupancy gauge aliases, and it aliases *downward*: the wrong instrument tells you to add exactly
the concurrency that is costing you output.

The right instrument is a difference of monotonic counters:

```
sustained_decode_tok_per_s = (decode_tok_total[t1] − decode_tok_total[t0]) / (t1 − t0)
```

A cumulative counter integrates every request whether or not a sample landed inside it, so it does
not alias at any cadence. If you have no backend access, the client-side equivalent is to parse your
own request log for `in=`/`out=`/`latency=`, reconstruct each request's interval as
`[end − latency, end]`, and then **say which of three different numbers you are quoting**: output
tokens ÷ wall-clock window, output tokens ÷ time-with-a-request-in-flight, or a per-stream rate.
On one 24 h window those were 51.8, 23.5 and ~130 tok/s. They differ by 5×, and quoting the wrong
one is most of how this gets mismeasured.

**Why the conclusion.** `[measured, 24 h window, one profile]` Wasted prefill — the tokens the
server had to recompute beyond genuine prompt growth, i.e. the cached prefix was evicted — rises
sharply with the number of other requests in flight. Pairs are formed **within one session** and only
then bucketed, because consecutive calls from different conversations have unrelated prompt sizes and
mixing them produces nonsense:

| Other requests in flight | Named long-context lanes: wasted tok/call | n | Whole fleet: wasted tok/call | n |
|---|---|---|---|---|
| 0 | 2,692 | 793 | 1,656 | 1,434 |
| 1 | 5,119 | 388 | 3,066 | 835 |
| 2 | 35,421 | 223 | 15,323 | 611 |
| 3 | 71,972 | 131 | 31,366 | 413 |
| 4 | 102,174 | 83 | 36,250 | 378 |
| 5 | **203,547** | 60 | **66,602** | 231 |

A 75× rise for the long-context lanes between zero and five concurrent peers. At ~4,400 prefill
tokens/s that last row is **46 seconds of pure prefill per call that emits no output token.**

And the output does not rise to pay for it. Aggregate output **plateaus from a mean concurrency of
about 1**:

| Mean concurrency | Output tok/s | Prefill tok/s | prefill:decode | Output per unit of concurrency |
|---|---|---|---|---|
| 0.98 | 97.8 | 1,351 | 13.8:1 | 99.7 |
| 2.00 | 102.3 | 3,747 | 36.6:1 | 51.2 |
| 2.55 | 110.6 | 4,067 | 36.8:1 | 43.4 |
| 4.47 | 65.7 | 6,146 | 93.5:1 | 14.7 |
| 5.50 | 54.5 | 6,143 | 112.7:1 | 9.9 |
| 7.28 | 37.2 | 7,197 | 193.4:1 | 5.1 |

Output is flat from ~1.0 to ~2.6 while prefill work triples, then **falls**. Marginal return drops
from 99.7 to 5.1 tok/s per unit of concurrency. Across buckets the prefill-to-decode work ratio runs
from **13.8:1 to 193.4:1** — i.e. 93–99.5 % of the token work on the card produces no output token.

**The control variable is the number of distinct large prefixes resident, not the request count.**
The clearest evidence is a pair of context buckets: requests above 200K ran at mean concurrency 1.81
and produced **37.2 tok/s** at 97.9:1, while 50–100K requests ran at a *higher* mean concurrency of
2.73 and produced **132.0 tok/s** at 20.6:1. More concurrency, one quarter of the output.

## 13. Give background work an explicit priority, and stop paying for calls nobody reads

**Rule.** Every auxiliary call your client makes on its own initiative — title generation,
summarisation, background review, a monitor tick — gets an **explicit** priority below an
interactive turn. And audit what those calls actually send.

**Why priority.** `[measured, 24 h, one profile]` Auxiliary requests were a small minority of
requests on the lane but dominated the mains' wasted prefill: a main's next turn re-prefilled
**143,596 prompt tokens per call** when an aux request had been in flight against **5,028** when
none had. A large aux prefix evicts the mains' resident prefixes, and the mains pay for it on their
next turn. (Those two per-call figures are the measured ones. The shares usually quoted alongside
them — ~11 % of requests, ~84 % of the waste — are derived from two aggregate totals whose
extraction was not preserved; treat them as arithmetic on that run, not as reproducible.)

Note what "explicit" means in the reference implementation: background aux is pinned to the
backend's **normal** priority, **0**, which is strictly below an interactive turn's **100**. It is
not negative. A value below zero would also yield to subagents, which may well be right, but
whether this backend accepts a negative `priority` field was never measured — and a rejected field
fails the whole call. Do not guess at a priority value you have not tested. The *control-plane*
calls of rule 1 stay at **100**; only genuinely background work is demoted.

**Why audit the payload.** `[measured, n = 298 ticks over 24 h]` A monitoring loop issued one
chat completion per tick carrying a **mean of 42,326 prompt tokens** (median 42,079, range
40,695–46,265) — the agent's full system prompt, tool schemas, skills and memory — for a prompt
whose own text ended *"No tools needed — use STATUS_JSON only."* It said it needed no tools while
shipping the tool schemas on every call.

Sending the same request as a bare completion against the same router and the same pinned model
measured **547 prompt tokens** (range 546–550 over 22 ticks, read from the API response's own
`usage.prompt_tokens`). **77× less**, for identical output quality, removing **~12.45 M prompt
tokens per day** from the lane at the measured tick rate. Two caveats worth carrying: that is
prompt tokens *removed from the lane*, not money, and the rate matters — the nominal interval was
240 s but the real cadence was ~306 s because the sleep followed a call whose latency ran 8–187 s,
so 298 ticks/day rather than 360.

The generalisable point is not the 77×. It is that **an agent framework's default call path carries
the agent's whole context**, and most background calls do not need any of it. Measure one.

A smaller instance of the same thing: an LLM call to generate a display title ran **258 times a
day**, once per monitor tick, on one-shot sessions nobody ever named — plus **127 retries** when it
hit an unhealthy replica first. Skipping the model call for one-shot sessions removed all of it.
⛔ Skip the *model call*, not the naming: the first attempt at this left colliding one-shot sessions
nameless, because the cheap instant-title path does not deduplicate.

## 14. Classify an interrupt by provenance, or your autonomous loop will stop on its own

**Rule.** An autonomous loop may be stopped by exactly three things: the operator, completion, or
its budget. A provider stall, a transport cut or a timeout is **not** any of them. Record *who*
interrupted a turn at the seam where the interrupt enters, and never infer it afterwards.

**Why.** `[measured on the reference deployment]` A provider-side stall — the kind that is
**routine on a loaded shared backend, measured at 31–137 s across logged events** — surfaced
internally as an interrupted turn, and the loop labelled it `user-interrupted (Ctrl+C)`. That label
is the single class a resume supervisor must never auto-resume, precisely because it means a human
wants it stopped. So a transient stall became a permanent one: one session paused at **turn 3 of
400**, with 397 turns unspent, and nothing resumed it.

Four defects, each independently sufficient to strand the loop, and all four present at once:

1. **Provenance was inferred, not recorded.** The fix sets a human-interrupt flag only at seams a
   person actually drives — the Ctrl+C binding, the ESC binding, a message typed while the agent is
   running, a voice interjection — and nowhere else. A machine interrupt now keeps the goal active,
   re-arms, consumes no turn and spends no judge call.
2. **The judge prompt instructed the model to stop.** It literally said that a response explaining
   the goal is *"unachievable / blocked / needs user input"* should be treated as **DONE**. That is
   backwards: a goal the agent cannot currently finish is blocked, not complete. The replacement adds
   a distinct `blocked` verdict, states that "broad, multi-part, long-running or simply not finished
   yet" is the **normal mid-flight state of an autonomous loop**, and enumerates the phrases that are
   not done — *blocked, waiting on, pending, as far as I can go, ready to submit*. A deterministic
   regex net (with negation patterns checked first) reclassifies a `done` whose own reason describes
   a block, because the model will still occasionally return "DONE, but X is blocked".
3. **Resume flipped a status without re-arming anything.** `/goal resume` set the state to active
   and cleared the barriers, but only a post-turn hook enqueues continuations and a slash command
   never runs a turn. The goal became "active and idle" — which is the exact stall the resume was
   meant to end, and it made the supervisor a no-op backstop: it correctly found the paused goal,
   correctly waited for an idle pane, correctly sent the resume, and nothing happened.
4. **The supervisor could not see active-but-idle.** It watched for *paused* goals. A goal that is
   active, has turns left, has no wait barrier, and has not taken a turn for 20 minutes is the state
   that actually cost the session, and nothing was looking for it.

**Then bound the re-arm.** Not auto-resuming is a bug; auto-resuming forever is a different one. A
wedged route would spin indefinitely, so after **12 consecutive** machine-interrupted turns the loop
pauses with a *transport-shaped* reason — which the supervisor is allowed to retry — and any
productive turn clears the streak. The asymmetry is the point: a machine interrupt must not pause
like a human one, and must not retry without limit either.

**And distinguish "busy" from "broken" when the judge itself fails.** That is rule 1's second half,
and it is the same error class seen from the other side.

## 15. A proxy stops being a proxy when the thing it proxied for moves

**Rule.** Never build an admission gate by dividing a resource reading by a per-request cost unless
you can say *why that reading responds to load*. When it stops responding, the division does not
fail loudly — it silently returns a large number, and your throttle becomes a multiplier.

**Why.** `[measured on the reference deployment, 2026-09-30]` This is the most portable lesson in
this file, because the client code was *correct* and the deployment changed underneath it.

The client throttled subagent fan-out with `affordable = free_mib // 560`, where 560 MiB was the
largest prefill allocation **measured** to be fatal (rule 9's incident: `Tried to allocate 560.00
MiB`, 280.94 MiB free). That was a sound gate while a prefill's transient VRAM was the scarce
thing. Then two changes landed on the same evening:

- the KV pool moved from VRAM into pinned host RAM — pool 451,264 → 1,310,720 tokens, free device
  VRAM **935–1,533 MiB → 6,463 MiB**;
- a bounded prefill gather flattened prefill transients to **32 MiB at every context length**, down
  from 1,024 MiB at 512K.

Both pushed the same divisor the same way. `6,463 // 560` licensed **eleven** children where it had
licensed about **one**. Nothing errored. Nothing logged a warning. The throttle reported capacity
it had no basis for, and the consequences were measured on the lane within minutes:

| | observed |
|---|---|
| Children the old rule permitted | **~1 → ~11** |
| Main lanes' own turn-average decode | **11–17 tok/s** (against a 158–178 tok/s solo rate) |
| Requests queued | **15** |
| Uncached tokens waiting to be prefilled | **588,268** at the moment the gate was rewritten, and **668,196** read live an hour later — up to **51 %** of the entire pool sitting in the prefill queue |
| KV pool utilisation at the same moment | **14–32 %** — i.e. the pool was fine and said nothing |

**The generalisation.** Free device VRAM was never the quantity that mattered. It was a *proxy* for
prefill working set, and prefill working set left the device. The same trap is waiting in any gate
keyed on a resource whose consumer has been moved, cached, compressed or offloaded — and offloading
is exactly what capacity work does. **Write down what each signal is a proxy for, and re-derive the
gate when that changes.**

### What replaced it

Three conjunctive legs, each read live from the backend's own numbers, none of them a round number:

**(A) Slots, with one reserved per live main *before* any child is counted.** A main that cannot get
a scheduler slot waits however much pool and VRAM are free, and every main needs a slot for its own
control-plane calls (compaction, titling, the goal judge). So children are counted against what is
left after the mains are seated — and the total stream count is capped. The cap is derived from the
measured concurrency curve rather than chosen: on the production geometry a lane keeps **100 % / 74 %
/ 57 %** of its solo decode rate at concurrency 1 / 2 / 3, so **2** is the largest concurrency at
which a lane keeps two thirds of itself. Raising it buys aggregate throughput at a lane's expense,
which is a product decision and therefore a runtime key, not a constant.

> ⛔ **Do not look for an interior aggregate maximum below the backend's own
> `--max-running-requests`.** There isn't one. Aggregate decode rises monotonically through
> concurrency 3 on both pool geometries measured
> ([BENCHMARKS §3.20](../BENCHMARKS.md#320-throughput-versus-concurrency-warm-segmented--and-the-071-retraction)).
> "Cap concurrency where tok/s peaks" has no answer in aggregate terms — the cap has to be keyed on
> a *lane's own* rate, which is the thing a human is actually reading.

**(B) Prefill backlog against a drain budget.** `num_waiting_uncached_tokens` divided by a prefill
rate the client calibrates **from the server's own cumulative counters** —
`total_prefill_uncached_tokens ÷ total_prefill_busy_us` — measured at **8,481 tok/s** and
independently at **8,377 tok/s** an hour later on the same lane. That is checked against a fixed
drain budget (here, the control-plane judge timeout, 30 s). Self-calibrating from the server's own
counters is the point: the divisor then survives a model swap, a quantisation change or a card
change without anyone editing a constant. Log-derived chunk-saturated prefill rates on the same lane
— median **8,958 tok/s**, p25–p75 7,257–10,191 — agree with the counter form to within 6 %.

**(C) Pool residency, with backlog charged before children.** Correctly does not bind at 14–32 %
used. It is in the gate so that it binds when it should.

**And VRAM survives as a floor, not a divisor.** `560 + max_running_requests × 32 = 688 MiB` — the
largest allocation measured fatal, plus one measured transient per slot the scheduler can fill.
Below it, admit nothing and let in-flight work drain. Above it, VRAM contributes **no cap at all**.
The floor separates the three geometries that matter — 287 MiB critical, 935 MiB and 6,463 MiB both
clear — which is the test a threshold has to pass to be worth having. ⛔ It must read **runtime**
free VRAM: on this deployment the boot-time `available_gpu_mem` field reported 3.42 GB while the
card had 287 MiB free while serving.

**Verified effect:** at the live geometry that had yielded eleven children (588,268 uncached waiting
≈ 69 s of backlog, 2 mains, 3 running, 5,651 MiB free), the replacement gate yields **0** child
streams. No main's window, growth ceiling or request priority is touched by any of it.

**Two design notes that generalise past this gate.**

- **Make every threshold resolve at call time** — environment, then a config file, then the module
  default. Retuning a throttle must not mean relaunching the sessions being throttled. The pattern,
  the mtime-keyed parse cache that makes it cheap, and the published "what resolved and from where"
  are in [`../ops/pool-broker.md` §1](../ops/pool-broker.md#1-the-runtime-tunable-config-pattern);
  the annotated config with every default is
  [`../ops/examples/pool-broker.json`](../ops/examples/pool-broker.json).
- **Refuse the *second* concurrent child outright rather than letting six start and self-clamping.**
  A gate that admits and then recovers has already paid the prefill, the eviction and the queue. The
  observed state it was meant to prevent — three children on three of four slots with fifteen queued
  — is reachable by a gate that is merely *eventually* correct.

## 16. Delegate the wait; never park the goal on it

**Rule.** When a long-running job blocks a goal, the goal does not stop. **Something that makes no
model calls** watches the job and reports back; the agent keeps advancing on work it can quote
verbatim from its own goal. And whatever you build to release the wait must not need a turn to run,
because a parked agent fires no turns.

**Why.** `[measured on the reference deployment]` An autonomous session sat at **turn 1 of 400**,
parked 1,131 s on one external process, with **399 turns unspent** — on a turn whose own report
named three other pieces of available work. The operator's pane read:

```
⏳ Goal parked — waiting on session <id>: the goal is not complete because the final
   cut and scoring depend on the running job, which the agent is waiting on.
↻ Loop: next in 30m.
```

Three defects compounded, and the third is the one that generalises:

1. **An un-itemised goal cannot escalate.** With no enumerated sub-criteria, the "are all items
   blocked?" test returns false by design, so the judge is left with a binary: continue, or park
   *everything*. Nothing enumerated the work it was about to abandon.
2. **The whole-goal barrier had no age ceiling** — while per-item barriers had carried one since
   they existed, on the stated reasoning that a process which never exits must not permanently
   remove an item from the work set. The barrier that removed the **whole goal** had none.
3. **⛔ The release was lazy, and lived on a path that itself required a turn.** The barrier was
   re-tested inside the "am I waiting?" check, which ran only from the post-turn hook — and a parked
   goal fires no turns. The stall detector then skipped parked goals *on the grounds that their
   barrier would release them*. So a park had **no backstop at all**: the one component that could
   have noticed deliberately looked away, and the one component that could have released it could
   only run if the thing it was releasing had already been released.

**That third defect is a shape, not a bug.** Any state whose exit condition is evaluated only by
machinery that the state itself suppresses is a deadlock with a progress bar. Check for it wherever
you have a wait: *what runs the predicate, and can it run while the predicate is false?*

### The fix pattern

- **A watcher owns the wait, and makes zero model calls.** It re-tests the barrier's own release
  predicate and reports a bounded outcome. Zero model calls is what makes it free: no context, no
  toolset, **no stream for a concurrency cap or a VRAM floor to throttle**, and nothing to pay the
  measured contention tax with — which on this backend runs from **2,622 wasted prefill tokens**
  with no contention to **114,385** against five in-flight peers (rule 12). Dispatch it on whatever
  rail already delivers background completions, so the report re-enters the session as a real turn
  and **re-judges the goal** — which means a *correctly* parked goal is no longer a dead end either.
- **⛔ Stamp the watcher with a routable owner, or it reports to nobody.** The completion drain
  compares the event's session key against its own and fails closed on an empty one. On the CLI
  surface that key resolved **empty**, so the watcher's report would have queued forever on exactly
  the surface the fix was written for. Fall back to a durable session id, and **refuse to dispatch
  at all when neither resolves** rather than promise a report nobody can claim.
- **Do not give a watcher a liveness probe.** A stale-progress monitor kills a delegation whose
  progress token is frozen, and a watcher's token is frozen by design.
- **Decide "is there other work?" with a code filter, not a prompt instruction.** Asking a model
  "anything else to do?" reliably produces something, and busy-work is worse than the park. So
  require every proposed item to carry a **verbatim quote**, and check mechanically that the quote
  really appears in the goal, its criteria, or the agent's own report. The work has to have already
  been written down — by the user or by the agent itself. Drop any item that merely restates the
  wait. **Every failure direction — no client, transport error, malformed response, an honest
  "nothing", nothing verified — resolves to the park.** A verified probe converts a whole-goal park
  into an *item* barrier, and the continuation names the items and forbids polling.
- **Bound the watchers per goal**, and give the whole-goal barrier the age ceiling the item barriers
  already had.

**Three failures already in rule 14 belong to this same family**, and are not repeated here: a judge
prompt that instructed *"blocked → treat as DONE"*; machine-origin interrupts (provider stalls
measured at **31–137 s**) labelled as human `Ctrl+C`, which is the one class a resume supervisor must
never auto-resume; and a resume command that flipped status without re-arming the continuation,
making the backstop a no-op. **Rule 14's defect 3 and this rule's defect 3 are the same error seen
from two sides** — a state transition that does not arm whatever is supposed to carry it forward.

---

### A note on where the measurements come from

Rules 1, 3, 4, 7, 8, 9 and 10–16 were measured or reproduced on the reference deployment and its
client framework directly. Rules 2, 5 and 6 cite the long-context suite,
which ran on a second card — an RTX PRO 6000 Server Edition at 600 W rather than the 300 W
Max-Q used everywhere else in this repository. Its pass/fail results and ratios transfer;
its absolute tok/s and pool sizes do not. See
[BENCHMARKS §3.8–§3.11](../BENCHMARKS.md#38-long-context-suite--complete-second-card).

**Rules 10–16 carry a caveat of their own.** They come from a single deployment, a single client
framework, and in several cases a single 24-hour window on one agent profile. Where a figure is
derived rather than measured, or where the extraction that produced it was not preserved, that is
said in place — see the parenthetical in rule 13 for the clearest example. The *mechanisms* are
what generalise; the magnitudes are one lane's. Two numbers in particular should be re-measured on
your own lane before you act on them: the wasted-prefill curve in rule 12, which depends entirely on
your prefix sizes, and the monitor-tick payload in rule 13, which depends on how much context your
framework attaches to a background call.

**Pool sizes moved while these rules were being written.** Rule 10's ceiling table is for a
451,264-token pool. The same deployment now runs a **1,310,720**-token host-resident pool
([README §host-resident KV](../README.md#the-pool-is-no-longer-the-binding-constraint-host-resident-kv)),
where the same code grants two mains 532,480 each. None of the rules change; every absolute number
in them does.

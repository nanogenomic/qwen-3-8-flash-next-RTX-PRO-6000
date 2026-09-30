# The client contract

What any agent orchestrator — any client framework, on any engine — has to do to use a
shared, self-hosted backend like this one well.

The engine work in this repository makes the backend faster and larger. It cannot make a
careless client efficient. Every rule below is here because breaking it cost something
measurable on the reference deployment, and each one comes with that evidence. None of it
is specific to one framework; wherever an example names a framework's setting, the rule
underneath applies to every orchestrator.

A **reference implementation** for Hermes Agent lives in [`hermes/`](hermes/). It is MIT
licensed (unlike the rest of this repository, which is Apache-2.0). It currently implements
rule 3 (patch 0002) and own-versus-shared status accounting (patch 0001); the rest of this
contract is pending a rebase — see its README for status.

Contents:

1. [Priority-tag control-plane calls](#1-priority-tag-control-plane-calls)
2. [Prioritise the main agent only when a human is waiting](#2-prioritise-the-main-agent-only-when-a-human-is-waiting)
3. [Size subagent context explicitly — never inherit it](#3-size-subagent-context-explicitly--never-inherit-it)
4. [Never advertise a window larger than the pool, or smaller than it grew to](#4-never-advertise-a-window-larger-than-the-pool-or-smaller-than-it-grew-to)
5. [Grow instead of compact when the pool allows](#5-grow-instead-of-compact-when-the-pool-allows)
6. [Hybrid models: warm conversations need state slots too](#6-hybrid-models-warm-conversations-need-state-slots-too)
7. [Isolate test state from live state](#7-isolate-test-state-from-live-state)
8. [Set the compaction threshold explicitly — beware step functions](#8-set-the-compaction-threshold-explicitly--beware-step-functions)

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
reference deployment declares `--context-length 540000` but has a **522,880-token pool** —
so the right per-request window to advertise is about 393,216, not 540,000.

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

---

### A note on where the measurements come from

Rules 1, 3, 4 (the router incident), 7 and 8 were measured or reproduced on the reference
deployment and its client framework directly. Rules 2, 5 and 6 cite the long-context suite,
which ran on a second card — an RTX PRO 6000 Server Edition at 600 W rather than the 300 W
Max-Q used everywhere else in this repository. Its pass/fail results and ratios transfer;
its absolute tok/s and pool sizes do not. See
[BENCHMARKS §3.8–§3.11](../BENCHMARKS.md#38-long-context-suite--complete-second-card).

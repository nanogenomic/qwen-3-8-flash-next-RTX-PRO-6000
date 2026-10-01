# Hermes Agent modifications for a shared backend

The reference implementation of the [client contract](../README.md) for
[Hermes Agent](https://github.com/NousResearch/hermes-agent): patches that matter once
**several agent sessions share one self-hosted model backend**.

> **Licence: this directory is MIT, unlike the rest of this repository (Apache-2.0).**
> It contains modifications of hermes-agent 0.20.4 by Nous Research, which is MIT
> licensed. Upstream's `LICENSE` ships here unmodified and the Nous Research copyright
> is preserved; the modifications are offered under the same MIT terms. See
> [`NOTICE`](NOTICE) for the attribution and the exact list of changes, and
> [`BASELINE.md`](BASELINE.md) for how the 0.20.4 baseline was identified and verified.

> **Status: twelve patches, plus seven write-ups.** Patches **0001–0002 apply to
> upstream 0.20.4** (tag `v2026.8.18`). Patches **0003–0012 are a series** against a
> stated commit of this fork — the cross-session pool broker with share-bounded grants,
> the live-share status bar, interrupt provenance for autonomous loops,
> background-priority demotion, the overflow tree-view markers and route hotkey, the
> delegated goal wait, and the prefill-capacity gate that replaced a free-VRAM divisor.
> See [the patch series](#the-patch-series) for what each one fixes and
> [`BASELINE.md`](BASELINE.md) for the bases and the verification. **One change is
> documented-only** and says so below. The reasoning behind all of it is in
> [`docs/`](docs/), and the framework-agnostic rules are in [`../README.md`](../README.md).
> A rebase onto upstream **0.21.5** (`v2026.9.24`) is in progress.

> **Read with the rest of this repository.** The concurrency arithmetic in
> [`docs/child-context-window.md`](docs/child-context-window.md) and
> [`docs/design-notes.md`](docs/design-notes.md) was written against a backend running
> 12 recurrent-state slots, where 12 ÷ 3 per request = 4 in flight. That in-flight
> ceiling is still right, but the long-context suite later showed that **slots beyond
> 3 per running request are what keep idle conversations' prefixes cacheable** — see
> [If you run agents on this model](../../README.md#if-you-run-agents-on-this-model-set---max-mamba-cache-size-first).
> Size slots from that section, not from the "three numbers agree on four" argument here.

---

## The thesis

Hermes assumes *a model endpoint*. That assumption is nearly always fine — until
the endpoint stops being yours alone.

Point several Hermes sessions at one self-hosted backend with a finite KV cache,
let each of them spawn subagents, and two things break. Neither announces
itself. Both are invisible right up to the moment they bite.

### 1. A shared backend's global state is not your session's state

The moment your status bar reads anything from the backend — in-flight requests,
KV used, KV total — those numbers describe **the machine**, not **you**. A
brand-new session that has spawned nothing at all will sit there reporting

```
⇉ 3/4 streams      221K/331.5K
```

which any operator reads as *"I have three subagents running and I'm 67% full."*
It means *"the box has three busy slots and 221K of pool in use, and possibly
none of it is mine."* You will make capacity decisions — don't spawn, kill
something, wait — against another session's load, believing it to be your own.

The fix is not a better number. It is **labelling which question each number
answers**:

```
⇉ 0 mine · 3/4 backend · 1 free
```

`0 mine` is the only figure the operator controls. `3/4 backend` is shared with
every other session on the machine. `1 free` is the only honest answer to *"can
I spawn another one right now?"* — and it is the number you actually wanted.

→ [`patches/0001-status-bar-own-vs-backend-accounting.patch`](patches/0001-status-bar-own-vs-backend-accounting.patch)
· write-up in [`docs/status-bar-accounting.md`](docs/status-bar-accounting.md)

**Scope, stated honestly.** Pristine upstream 0.20.4 has no backend-derived
stream readout in its status bar, so upstream does not have this bug — the
backend readout is this fork's own earlier addition, and patch 0001 fixes how
that readout was labelled. The lesson is what generalises: *the instant you
surface a shared backend's occupancy, you have created a number that lies about
ownership unless you label it.* Anyone adding such a readout inherits the bug,
which is why the reasoning is written up rather than just the diff.

One thing in this area *is* upstream-relevant. Upstream's own
`snapshot["active_background_subagents"]` uses
`tools.async_delegation.active_count()` — which counts async-pool **units**, so
a fan-out batch of five children counts as **one**. That is correct for capacity
accounting and wrong for a readout that claims to say how many subagents are
working. `active_task_count()` is the one that expands a batch to its real child
count. Both are upstream; the distinction is documented upstream too, and it is
easy to reach for the wrong one.

### 2. A subagent's context window was inherited, never stated

`_build_child_agent()` constructs the child `AIAgent` without passing a context
window. The child therefore adopts whatever `model.context_length` the parent
profile happens to hold. Edit that value — as any capacity tool relieving pool
pressure will want to — and you have silently reshaped **every subsequently
spawned subagent**, with nothing in the log saying so.

This is not hypothetical. On the deployment these patches run on, a capacity tool
lowered the profile window to **65,536** while `delegation.max_tokens` stayed at
**24,576**. Subagents inherited 65,536. Their compaction trigger became

```
effective input budget = 65,536 − 24,576            = 40,960
threshold floor        = max(0.75 × 40,960, 64,000) = 64,000 ≥ 40,960   → degenerate
compaction trigger     = 0.85 × 40,960              = 34,816 tokens
```

and there they stopped. At the trigger the child held ~34.9K tokens, of which
~25K was system prompt + tool schemas + protected head and ~10.2K was protected
tail. Both incompressible. Roughly 10K was summarisable — not enough to get back
under 34,816 — so compaction ran, failed to make progress, aborted, and the
agent stalled with **"compression is blocked (ineffective)"**. Nothing left to
try, and no line anywhere connecting it to a config edit made in a different
process.

Two separate defects, and the patch fixes both:

- **The window is now stated and logged**, not inherited by accident. Resolution
  order, first hit wins: `HERMES_CHILD_CONTEXT_LENGTH` (env, per-run) →
  `delegation.child_context_length` (profile config) → absent, which inherits
  exactly as before. **Absent config is a no-op**, so this cannot change
  behaviour for anyone who has not opted in.
- **65,536 against a 24,576 output cap is not a viable child geometry at all**,
  so the `auto` ladder never selects it. Its floor is 81,920 — chosen so the
  trigger (48,742) clears the measured ~25K incompressible floor with a real
  summarisable middle above it.

→ [`patches/0002-delegation-explicit-child-context-window.patch`](patches/0002-delegation-explicit-child-context-window.patch)
· write-up in [`docs/child-context-window.md`](docs/child-context-window.md)

---

## Applying these

### Patch 0002 — applies to pristine upstream 0.20.4

```bash
cd /path/to/hermes-agent          # at tag v2026.8.18, or near it
git am /path/to/patches/0002-delegation-explicit-child-context-window.patch
```

Verified: `git am` applies it to pristine 0.20.4 (`tools/delegate_tool.py`
only), the result byte-matches the file in `upstream-modified/`, and it
compiles. On a checkout that has drifted, `git apply --3way` will place the
hunks — they sit in `_build_child_agent()` and in `delegate_task()`'s executor
construction.

The patch is inert until you opt in:

```yaml
# profile config
delegation:
  child_context_length: 98304      # or "auto", or omit to inherit as before
```

```bash
export HERMES_CHILD_CONTEXT_LENGTH=auto     # per-run override
export HERMES_KV_BACKEND=http://127.0.0.1:30000   # where /v1/loads lives
```

### Patch 0001 — has a prerequisite

It modifies a delegation status-bar segment that **pristine upstream does not
have**; it is this fork's own earlier work. It will not `git am` onto a pristine
0.20.4 checkout, and that is expected rather than a packaging mistake.

If your checkout already renders a backend-derived delegation segment, apply it
directly. If it does not, the change is 39 lines and
[`docs/status-bar-accounting.md`](docs/status-bar-accounting.md) gives both
integration points and the code verbatim — it is a short read and a shorter
edit.

### Patches 0003–0012 — a series, in numeric order

```bash
cd /path/to/your-hermes-fork
git am patches/00{03,04,05,06,07,08,09,10,11,12}-*.patch
```

They are **one series with one base**, not ten independent patches: each applies
to the tree the previous one leaves. The base is this fork at commit `5a3c9f5`,
and [`BASELINE.md`](BASELINE.md) records it, the per-patch dependencies, and the
verification — `git apply --check` and `git am` against a clean checkout of that
commit, 27 files compiling, and the 498 tests the series adds or edits passing on
the **published** (scrubbed) tree.

**They will not apply to pristine upstream 0.20.4, and the reason is not
packaging.** Most of what they touch does not exist upstream: `pool_broker.py`,
`pool_policy.py`, `overflow_router.py`, `pool_share.py` and `goal_watchers.py`
are this fork's own modules, and `cli.py`, `goals.py` and
`goal_resume_supervisor.py` carry heavy fork divergence in the exact regions
these hunks sit in. The five new files in the series apply anywhere, because a
new-file hunk has nothing to match; the rest needs hand placement on a different
tree, and `git apply --reject` plus the matching write-up in [`docs/`](docs/) is
the way to do that. `--3way` is not available — its blobs are in the private
tree.

---

## The patch series

Every patch header carries the failure, the measurements and what was changed for
publication. This table is the index.

| # | what it does | the failure it fixes | base | write-up |
|---|---|---|---|---|
| [0001](patches/0001-status-bar-own-vs-backend-accounting.patch) | labels the status bar's own-vs-backend subagent counts | a session that spawned nothing reads `⇉ 3/4 streams` and you throttle yourself against another session's load | fork state; **prerequisite**, see above | [status-bar-accounting](docs/status-bar-accounting.md) |
| [0002](patches/0002-delegation-explicit-child-context-window.patch) | states and logs a subagent's context window instead of inheriting it | a capacity edit to the *parent* profile silently reshaped every future subagent into a geometry where compaction cannot make progress | upstream 0.20.4 (`v2026.8.18`) | [child-context-window](docs/child-context-window.md) |
| [0003](patches/0003-pool-broker-share-bounded-grant-ceilings.patch) | bounds the ceilings the pool broker **grants**, not just the reservations it measures | the invariant summed measured reservations, so it reported `ok` while granting two sessions the whole pool each | fork `5a3c9f5` | [shared-pool-grants](docs/shared-pool-grants.md) |
| [0004](patches/0004-portability-explicit-encodings-pool-overflow.patch) | explicit encodings in the pool and overflow modules | a no-encoding `read_text`/`subprocess` mangles non-ASCII on Windows; 25 findings against a baseline of 1 | 0003 | — |
| [0005](patches/0005-auxiliary-work-yields-to-interactive-lanes.patch) | background auxiliary work takes an explicit low priority; a one-shot run is no longer "interactive" | 83.7% of the interactive lanes' re-prefill waste was overlapped with an auxiliary request at equal priority | 0004 | [autonomous-loop-provenance § Background work](docs/autonomous-loop-provenance.md#background-work-and-what-it-was-really-costing) |
| [0006](patches/0006-overflow-tree-view-markers-and-route-hotkey.patch) | marks overflowed subagents in the tree view; hotkey to toggle the route | a subagent that ran on another provider was indistinguishable from one that ran locally, which makes every after-the-fact capacity question unanswerable | 0005 | — |
| [0007](patches/0007-goal-loop-interrupt-provenance-and-blocked-verdict.patch) | interrupt provenance, a first-class `blocked` verdict, a `/goal resume` that re-arms, and the supervisor's stalled-active class | a machine stall was labelled as the human's `Ctrl+C` — the one class auto-resume must not touch — so a transient became a permanent stall with 397 turns unspent | 0006 | [autonomous-loop-provenance](docs/autonomous-loop-provenance.md) |
| [0008](patches/0008-status-bar-live-shared-pool-share.patch) | the ctx denominator becomes the live granted share | two sessions each showed a percentage against a 393.2K window neither would be allowed to fill, on a pool that had granted them 196.6K each | 0007 | [shared-pool-grants](docs/shared-pool-grants.md) |
| [0009](patches/0009-goal-loop-bounded-machine-interrupt-rearm.patch) | bounds the machine-interrupt re-arm | a provider-cut turn costs no budget, so 0007's re-arm would retry a wedged route forever without reaching `max_turns` | 0008 | [autonomous-loop-provenance](docs/autonomous-loop-provenance.md) |
| [0010](patches/0010-pool-share-read-the-kv-pool-not-the-context-length.patch) | reads the KV pool, not the per-request context length, as the share bound | two plausible fields with the same value: once a KV offload made them diverge, the bar pinned the denominator back to the static window | 0009 | [shared-pool-grants](docs/shared-pool-grants.md) |
| [0011](patches/0011-goal-loop-delegate-the-wait-to-a-watcher.patch) | delegates the wait to a zero-model-call watcher, with a barrier age ceiling and a quote-verified work probe | a goal parked *itself* on one long job at turn 1 of 400, with three other pieces of work named in its own report, and no backstop at all | 0010 | [goal-wait-delegation](docs/goal-wait-delegation.md) |
| [0012](patches/0012-pool-broker-prefill-capacity-child-throttle.patch) | re-keys the child throttle onto slots, prefill backlog and pool residency | a KV offload moved the pool to host RAM, so the free-VRAM divisor licensed ~11 concurrent children where it had licensed ~1 | 0011 | [prefill-capacity-throttle](docs/prefill-capacity-throttle.md) |

### Documented-only, and why

**One** change in this round is not shipped as a patch: fork commit `3272a15`, a
correction to an overflow status message that claimed the runtime route pin was
refusing the route when overflow was merely **disarmed**. The two are different
states with different fixes, and conflating them sends you to look at the pin.

It is held back for a mechanical reason, not a judgement call. Its diff *removes*
a line that is itself an absolute home path, and a patch's removed lines are
quoted from the base it applies to — rewrite one and the patch stops applying,
which is the property [`BASELINE.md`](BASELINE.md) exists to prove. The change,
in full, is: the two-line refusal message

```
"runtime route pin refuses the <provider> route on this lane "
"(see $HERMES_HOME/candidates/<name>.patch)"
```

becomes one line that drops the local path, and the *disarmed* case gets its own
message instead of borrowing the pin's. Ten lines in
`agent/overflow_router.py`, all inside the decision function that returns
`(False, why)`.

Everything else in the round is shipped. The honest summary of this directory is
now: **the mechanisms, the measured numbers, the failure modes, the design
decisions and — for all but that one change — the code.**

---

## What sits between the agent and the engine

A reasonable question when you see a self-hosted backend behind an agent
framework is "where is the adapter?". The honest answer is that **there isn't
one, and there shouldn't be.**

The engine speaks the OpenAI chat-completions API, which is the framework's own
native dialect, so nothing translates between them. Upstream *does* ship adapter
modules — `agent/anthropic_adapter.py`, `agent/bedrock_adapter.py`,
`agent/gemini_native_adapter.py`, `agent/codex_responses_adapter.py`,
`agent/azure_identity_adapter.py` — and they exist precisely because those
providers are **not** OpenAI-shaped. A self-hosted OpenAI-compatible server needs
none of them, and none was written. If you were looking for a module in this
directory named like an adapter, that is why you did not find it.

What *is* there, in the order a request meets it:

1. **The stock client, pointed at a `base_url`.** The main lane builds it in
   `agent/agent_runtime_helpers.py`, auxiliary work in
   `agent/auxiliary_client.py`. No request translation anywhere.
2. **One transport shim** — `PriorityTagTransport` /
   `AsyncPriorityTagTransport` in `agent/pool_broker.py`, mounted in
   `agent/process_bootstrap.py`. It wraps the HTTP transport and, **only** for a
   POST to `/chat/completions` on a known backend host, and **only** when the
   body does not already carry one, adds a single field: `priority`. Everything
   else passes through byte-for-byte, and every failure path returns the request
   unmodified. This is the one thing on the data path that touches a request, and
   it is additive, not translative. It is what patch 0005 uses to demote
   background auxiliary work without editing the broker.
3. **A fail-closed route pin** — `hermes_cli/route_identity.py`. It canonicalises
   a `base_url` (only across proven-equivalent URL components) and checks the
   `(model, endpoint)` pair against the configured pin **before** inference,
   raising `RuntimeRoutePinError` on drift. It accepts the delegation route as a
   second pinned route, so a subagent on a different lane is not drift.
4. **A routing decision, which is not a protocol.**
   `agent/overflow_router.py` decides per dispatch whether a saturated subagent
   goes to the local engine or to the hosted overflow provider, and returns a
   `Decision` carrying the reason. Patch 0006 is what makes that decision visible
   in the tree view.
5. **A control plane, which is not on the data path at all.**
   `agent/pool_broker.py` and `agent/pool_policy.py` read the engine's own
   `/v1/loads` and `/get_server_info` and publish an allowance document that
   every session sizes itself from; `hermes_cli/pool_share.py` only reads it.
   Patches 0003, 0008, 0010 and 0012 all live here. This is the layer the
   [`ops/`](../../ops/) half of this repository describes from the other side.

So the thing people mean by "the adapter" is, concretely: **a route pin and a
priority-tagging transport shim on top of the stock OpenAI client path** — about
twenty lines of wrapper and a `request.content` round-trip. Everything else in
the list is sizing, admission and display: 3,928 lines across
`pool_broker.py` (2,350), `overflow_router.py` (576), `pool_share.py` (450),
`route_identity.py` (277) and `pool_policy.py` (275). None of it translates a
protocol, which is the whole point — when the engine already speaks your
dialect, the work is not adaptation, it is **deciding how much of a shared box
you are allowed to use.**

## What is *not* here, and why

The deployment these patches run on carries a good deal of surrounding machinery: a
lane launcher, a KV budget calculator, an admission guard, a capacity/profile
inspector, and per-session chat rooms for agent-to-operator messaging. **None of
it is published.** It encodes that deployment's capability tiers, model pins,
addresses, and security policy; as source it would be a liability to whoever
read it and useless to whoever tried to run it.

What is worth taking from it is the *shape* of those tools, not their code — so
the patterns are written up generically in
[`docs/design-notes.md`](docs/design-notes.md): the admission guard, the
capacity governor, per-session rooms, and the sizing arithmetic you need before
any of them can make a decision. That document is the reusable half of the work
and is probably more valuable than any single patch here.

Also absent, across every patch and document in this directory: no credentials,
API keys or tokens; no routable address or public host name (the only addresses
that appear are loopback placeholders); no model weights or checkpoints; no
traffic-derived artifacts. [`NOTICE`](NOTICE) lists, exhaustively, the few
deployment-specific strings that *do* survive on lines a patch could not rewrite
without ceasing to apply, and why each one is harmless.

---

## Directory layout

```
LICENSE                       upstream MIT, byte-identical (Nous Research) — governs this directory
NOTICE                        attribution, modification list, publication scrub
BASELINE.md                   how upstream 0.20.4 was pinned; the series' bases and verification
patches/
  # against upstream 0.20.4 / a fork state — see "Applying these"
  0001-status-bar-own-vs-backend-accounting.patch     cli.py  (+37 −2)
  0002-delegation-explicit-child-context-window.patch tools/delegate_tool.py (+339 −1)
  # one series, numeric order, base = fork 5a3c9f5
  0003-pool-broker-share-bounded-grant-ceilings.patch         4 files (+766 −56)
  0004-portability-explicit-encodings-pool-overflow.patch     2 files (+8 −8)
  0005-auxiliary-work-yields-to-interactive-lanes.patch       6 files (+874 −12)
  0006-overflow-tree-view-markers-and-route-hotkey.patch      3 files (+63)
  0007-goal-loop-interrupt-provenance-and-blocked-verdict.patch   8 files (+1219 −37)
  0008-status-bar-live-shared-pool-share.patch                4 files (+1609 −14)
  0009-goal-loop-bounded-machine-interrupt-rearm.patch         4 files (+166 −1)
  0010-pool-share-read-the-kv-pool-not-the-context-length.patch 2 files (+40 −1)
  0011-goal-loop-delegate-the-wait-to-a-watcher.patch          7 files (+2780 −33)
  0012-pool-broker-prefill-capacity-child-throttle.patch       3 files (+1166 −176)
upstream-modified/
  tools/delegate_tool.py      pristine 0.20.4 + patch 0002, for a readable diff
docs/
  status-bar-accounting.md    patch 0001 in full, integration points, the pitfall
  child-context-window.md     patch 0002 — the incident arithmetic, the ladder, the knobs
  design-notes.md             generic patterns for sharing one backend (0005's background)
  shared-pool-grants.md       patches 0003 / 0008 / 0010 — grant ceilings, the live-share bar
  autonomous-loop-provenance.md  patches 0007 / 0009 — why a loop stops unasked
  prefill-capacity-throttle.md   patch 0012 — the free-VRAM divisor that stopped throttling
  goal-wait-delegation.md        patch 0011 — a wait nothing could end, and the watcher that ends it
```

`cli.py` is **not** shipped as a whole file. It is ~1 MB and this fork's copy
carries a great deal of unrelated, deployment-specific divergence from upstream;
shipping it would bury a 39-line change in noise and publish things that have no
business being published. The patch plus the write-up is the legible form.

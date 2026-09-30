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

> **Status: partial, refresh pending.** This directory currently covers **patches
> 0001–0002 against upstream 0.20.4** (tag `v2026.8.18`). A rebase onto upstream
> **0.21.5** (`v2026.9.24`) is in progress and will add the rest of the client contract
> as code: a pool broker, task-sized child windows, grow-instead-of-compact,
> interactive-aware priority, judge priority, an overflow toggle to a hosted provider
> (Cerebras), and a test-isolation guard. Until then, [`../README.md`](../README.md)
> describes those behaviours and the evidence for them; this directory implements two.

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

---

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
and is probably more valuable than either patch.

Also absent: no credentials, no addresses or hostnames, no model weights or
checkpoints, no traffic-derived artifacts.

---

## Directory layout

```
LICENSE                       upstream MIT, byte-identical (Nous Research) — governs this directory
NOTICE                        attribution, modification list, publication scrub
BASELINE.md                   how upstream 0.20.4 was pinned and verified
patches/
  0001-status-bar-own-vs-backend-accounting.patch    cli.py  (+37 −2)
  0002-delegation-explicit-child-context-window.patch tools/delegate_tool.py (+339 −1)
upstream-modified/
  tools/delegate_tool.py      pristine 0.20.4 + patch 0002, for a readable diff
docs/
  status-bar-accounting.md    patch 0001 in full, integration points, the pitfall
  child-context-window.md     the incident arithmetic, the ladder, the knobs
  design-notes.md             generic patterns for sharing one backend
```

`cli.py` is **not** shipped as a whole file. It is ~1 MB and this fork's copy
carries a great deal of unrelated, deployment-specific divergence from upstream;
shipping it would bury a 39-line change in noise and publish things that have no
business being published. The patch plus the write-up is the legible form.

# A wait that nothing could end

The implementation behind [client contract rule 16](../../README.md#16-delegate-the-wait-never-park-the-goal-on-it).
The companion to [autonomous-loop-provenance.md](autonomous-loop-provenance.md), which covers
[rule 14](../../README.md#14-classify-an-interrupt-by-provenance-or-your-autonomous-loop-will-stop-on-its-own)
— same session class, same week, and the two share their deepest defect.
Written up rather than shipped as a diff — see [the README](../README.md#why-this-round-is-write-ups-and-not-patches).

---

## The failure

An autonomous session sat at **turn 1 of 400** and stopped. The operator's pane:

```
⏳ Goal parked — waiting on session <id>: the goal is not complete because the final
   cut and scoring depend on the running job, which the agent is waiting on.
↻ Loop: next in 30m.
```

The live record behind it: active at turn 1 of 400, parked on one external process for **1,131 s**,
with **zero** enumerated subgoals — on a turn whose own report named GPU packing work, queued rows,
and a workflow split it had already agreed with a peer session. **399 turns unspent.** (That figure
is arithmetic on the turn counter, not a logged value.)

The loop was not wrong that it was blocked. It was wrong that being blocked on *one* thing meant
stopping work on *everything*, and it was wrong in a way nothing could recover from.

## Three compounding defects

### 1. An un-itemised goal cannot escalate

The framework has a middle verdict between "continue" and "park" — a defer path, built for exactly
this symptom. It is reached through "are all items blocked?", which **returns false by design when
there are no enumerated items**. A goal with no sub-criteria therefore cannot reach it: the judge is
left with a binary, continue or park everything, and nothing enumerates the work it is about to
abandon.

The defer mechanism was built for this symptom and this is the case it structurally cannot reach.
That is worth stating plainly, because it is a common shape: an escalation path whose precondition
is the very structure the failing case lacks.

### 2. The whole-goal barrier had no age ceiling

Per-item barriers had carried one since they existed, on stated reasoning: a process that never
exits must not permanently remove an item from the work set. The barrier that removes **the whole
goal** had none. The narrower mechanism was the better-guarded one.

### 3. ⛔ The release was lazy, on a path that required a turn

This is the one that generalises.

- The barrier was re-tested inside the "am I waiting?" check.
- That check ran only from the post-turn hook.
- **A parked goal fires no turns.**

So the release could only run if the thing it was releasing had already been released.

And the backstop deliberately looked away: the stall detector **skipped every parked goal**, on the
reasoning that *its barrier will release it*. The release was lazy; the detector assumed it was
eager. Between them, a park had no backstop at all — and the state that actually cost the session
(active, turns remaining, no barrier, no turn taken for twenty minutes) was a state nothing was
watching for.

> **The shape to look for.** Any state whose exit condition is evaluated only by machinery that the
> state itself suppresses is a deadlock with a progress bar. Ask of every wait: *what runs the
> predicate, and can it run while the predicate is false?*
>
> [Rule 14](../../README.md#14-classify-an-interrupt-by-provenance-or-your-autonomous-loop-will-stop-on-its-own)'s
> third defect — a resume that flipped status without re-arming the continuation, so the supervisor
> correctly found the paused goal, correctly waited for an idle pane, correctly sent the resume, and
> nothing happened — is the same error from the other side. A state transition that does not arm
> whatever carries it forward.

## The fix: someone else owns the wait

A **watcher** is dispatched on the rail that already delivers background completions, in the
single-unit form the framework's scheduled-job tool already uses — so it needs no parent agent, and
therefore no edit to the CLI entry point, because all three surfaces already honour the
continue-and-prompt contract.

**A watcher makes zero model calls.** It re-tests the barrier's own release predicate and reports a
bounded outcome. That is what makes it free:

- no context, no toolset;
- **no stream** for a concurrency cap or a VRAM floor to throttle
  ([rule 15](../../README.md#15-a-proxy-stops-being-a-proxy-when-the-thing-it-proxied-for-moves));
- nothing to pay the measured contention tax with — **2,622** wasted prefill tokens with no
  contention, **114,385** against five in-flight peers ([rule 12](../../README.md#12-more-client-concurrency-is-net-negative-here-and-the-usual-instrument-hides-it)).

Its runner enters the delegated-child context, so any incidental call it does make runs at normal
priority; the goal judge keeps the control-plane lane
([rule 1](../../README.md#1-priority-tag-control-plane-calls)).

Three implementation details that each cost a round to find:

- **⛔ A progress probe is deliberately not passed.** The stale-progress monitor kills a delegation
  whose progress token is frozen, and a watcher's token is frozen *by design*.
- **⛔ It lists sessions; it never polls or reads their logs.** Either of those marks the session
  consumed and suppresses the registry's own completion turn — which is the turn the whole fix
  depends on.
- **⛔ It must carry a routable owner, or it reports to nobody.** The completion drain filters on
  *positive* ownership: it compares the event's session key to its own and **fails closed on an
  empty one**. On the CLI surface that key resolves empty — the approval context variable is bound
  only during gateway/TUI turns, and the environment variable is not set there. So the watcher's
  report would have queued forever on exactly the surface the fix was written for. The repair is to
  fall back to the durable session id the goal manager is already constructed with, and to **refuse
  to dispatch at all** when neither resolves, rather than promise a report nobody can claim.

**The report re-enters the session through the completion queue, which forges a fresh turn, which
re-judges the goal.** So a *correctly* parked goal is no longer a dead end either — which matters,
because the fix must not make the loop worse at the cases it was getting right.

## Deciding "is there other work?" honestly

A wait verdict triggers one small auxiliary call, on the same judge task and control-plane lane — no
new model, no fallback provider.

**The guard is code, not a prompt instruction.** Asking a model "anything else to do?" reliably
produces something, and busy-work is worse than the park. So:

1. every proposed item must carry a **verbatim quote**;
2. a verifier checks mechanically that the quote really appears in the goal, its criteria, or the
   agent's own report — the work has to have already been written down, by the user or by the agent
   itself;
3. a second filter drops any item that merely restates the wait;
4. **every failure direction resolves to the park** — no client, transport error, malformed JSON, an
   honest "nothing", or nothing verified.

A verified probe converts the whole-goal park into an **item** barrier, the same mechanism defer
uses, and the continuation names the items and explicitly forbids polling the job.

Plus the ceiling defect 2 was missing: a default maximum age on the whole-goal barrier, enforced in
both the waiting check and the stall detector, so neither can be the only enforcement point.

## And bound it

Not auto-resuming is a bug; auto-resuming forever is a different one. Watchers are bounded per goal.
Rule 14's equivalent — pausing with a *transport-shaped* reason after twelve consecutive
machine-interrupted turns, which the supervisor is then allowed to retry, with any productive turn
clearing the streak — is the same asymmetry: **a machine-caused stop must not behave like a
human-caused one, and must not retry without limit either.**

## What this does not establish

- **One session, one framework, one occurrence.** The 1,131 s park and the 399 unspent turns are a
  single observed event. The mechanism is what generalises.
- **No measurement of how often a verified probe finds real work.** The guard is built to fail
  closed, so the expected failure mode is an unnecessary park, not busy-work — but the rate of
  either is unmeasured.
- **The twenty-minute idle threshold and the twelve-turn streak are defensible defaults**, not
  derived quantities.

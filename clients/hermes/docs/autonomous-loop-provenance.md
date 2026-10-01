# Why an autonomous loop stops when nobody asked it to

The implementation behind [client contract rule 14](../../README.md#14-classify-an-interrupt-by-provenance-or-your-autonomous-loop-will-stop-on-its-own),
plus the background-priority work behind [rule 13](../../README.md#13-give-background-work-an-explicit-priority-and-stop-paying-for-calls-nobody-reads).
Written up rather than shipped as a diff — see [the README](../README.md#why-this-round-is-write-ups-and-not-patches).

---

## The failure

A long-running goal — a loop that takes a turn, asks a judge "continue, wait, or done?", and
re-arms — stopped at **turn 3 of 400** and stayed stopped. Its recorded reason was
`user-interrupted (Ctrl+C)`.

Nobody had pressed Ctrl+C. The backend had stalled.

Provider stalls on a loaded shared backend are routine: **31–137 s** across every logged
occurrence on this deployment, with 33 s and 34 s the most common. Each one surfaced internally as
an interrupted turn, which the loop then labelled as the one class a resume supervisor must never
auto-resume — because that label is supposed to mean *a human wants this stopped*.

So a 30-second transport hiccup became a permanent stall with 397 turns unspent. (That figure is
arithmetic on the turn counter, 400 − 3, not a logged value.)

Four defects were present at once, each independently sufficient:

## 1. Provenance was inferred, never recorded

The pause reason was chosen at the point the loop noticed the turn had been interrupted, by which
time *why* was no longer available. Both a keypress and a dead socket arrive as "the turn did not
finish".

**The fix is a flag set at the seam, not a guess made later.** A single
`_mark_human_interrupt(source)` is called from exactly four places — the Ctrl+C binding, the ESC
binding, a message typed while the agent is running, and a voice interjection — and nowhere else.
Only that flag may produce the `user-interrupted (Ctrl+C)` reason. Everything else is a machine
interrupt: the goal stays active, re-arms, **consumes no turn and spends no judge call**.

The rule generalises past this framework: **an interrupt's provenance is only knowable where it
enters.** If you do not record it there, you cannot recover it, and the default you pick will be
wrong in whichever direction hurts more.

## 2. The judge prompt told the model to stop

The system prompt for the continue/wait/done judge contained, verbatim:

```
- The response explains the goal is unachievable / blocked / needs user input
  (treat this as DONE with reason describing the block).
```

and, in a second variant:

```
- If the response explains the work is blocked / unachievable / needs user input
  (e.g. the stated Stop condition was hit), treat it as DONE with the reason
  describing the block.
```

A goal the agent cannot currently finish is **blocked**. It is not complete. Instructing the judge
to call it DONE converts every transient external dependency into a terminal state.

The replacement does three things:

- **Adds a distinct `blocked` verdict**, widening the set from
  `{done, continue, wait, defer}` to `{done, blocked, continue, wait, defer}`, with its own pause
  marker so a supervisor can tell "needs a human decision" from "finished".
- **States the normal case explicitly.** The strongest line in the new prompt is that a goal which
  is *broad, multi-part, long-running or simply not finished yet* is **the normal mid-flight state
  of an autonomous loop, and never a reason to stop.** It then enumerates the phrases that are not
  done — *blocked*, *waiting on*, *pending*, *as far as I can go*, *ready to submit* — and forbids
  returning DONE in the same sentence as "but X is blocked".
- **Adds a deterministic net behind the model.** Because a model will still occasionally return
  "DONE, but the upload is pending", a regex set reclassifies a `done` whose own reason describes a
  block. Negation patterns are checked **first** and win, and there is a separate branch for the
  contrastive "X is complete BUT Y is pending" shape, which is the form that actually occurred.

A regression test asserts the strings `treat this as done` and `treat it as done` are absent from
every prompt template — the cheapest possible guard against the instruction coming back.

Supporting measurement: across 244 logged judge calls on this deployment the judge returned
**zero** explicit pause verdicts (157 continue, 68 defer, 14 wait, 5 done). The pauses were not
the judge deciding to stop; they were the misclassification in §1 and the false-DONE path above.

## 3. Resume flipped a status without re-arming anything

The resume command set the goal's state to active and cleared its wait barriers. It enqueued
nothing. But only a **post-turn hook** enqueues continuations, and a slash command never runs a
turn — so the goal became *active and idle*, which is the same stall the resume was meant to end.

This made the supervisor a no-op backstop in the most expensive way: it correctly detected the
paused goal, correctly waited for an idle pane, correctly sent the resume, correctly observed the
status change — and nothing worked afterwards. Every component reported success.

The fix enqueues the first continuation as part of resuming, and defers if a human message is
already queued.

## 4. The supervisor could not see "active but idle"

It watched for *paused* goals. The state that actually cost the session was: **active, turns
remaining, no wait barrier, and no goal turn for 20 minutes.** Nothing was looking for that.

Added as an explicit query, with three "uncertainty means not a candidate" guards: a spent turn
budget, a parked barrier (waiting on a pid, a session, or an unexpired timestamp), and a missing
last-turn timestamp. A supervisor that resumes on ambiguous evidence is worse than one that misses
a case.

## Then bound the re-arm

Not auto-resuming a machine interrupt is a bug. Auto-resuming one forever is a different bug: a
wedged route would spin, consuming turns and emitting nothing.

After **12 consecutive** machine-interrupted turns the goal pauses with a *transport-shaped*
reason — a class the supervisor **is** allowed to retry — and any productive turn clears the
streak, as does an explicit resume.

The asymmetry is the whole design: a machine interrupt must not pause like a human one, and must
not retry without limit either.

---

## Background work, and what it was really costing

Same deployment, same week, a different instance of "the client is doing this to itself".

**Auxiliary calls evicted the mains' prefixes.** When an auxiliary request had been in flight, a
main agent's next turn re-prefilled **143,596 prompt tokens**; when none had, **5,028**. The aux
call is cheap; the main pays for it. The fix pins background auxiliary work to the backend's
**normal** priority, `0`, strictly below an interactive turn's `100`, from an explicit allow-list of
five task types. Control-plane calls — the goal judge above among them — stay at `100`.

⛔ **Note what was *not* done.** The priority is 0, not negative. A negative value would also yield
to subagents, which is arguably correct, but whether this backend accepts a negative `priority`
field was never measured — and a rejected field fails the whole call, turning a scheduling
preference into an outage. It is reachable by environment variable and left at 0 by default. Do not
guess at a priority value you have not tested.

**A background call was carrying the whole agent.** A monitoring loop issued one completion per
tick at a **mean of 42,326 prompt tokens** (median 42,079, range 40,695–46,265, n = 298 over 24 h)
— the full system prompt, tool schemas, skills and memory — for a prompt whose own final sentence
was *"No tools needed — use STATUS_JSON only."* The archived request payloads confirm it: that
sentence, and a populated `tools` array, in the same request.

Reissuing it as a bare completion against the same router and the same pinned model measured
**547 prompt tokens** (546–550 over 22 ticks, read from each response's own
`usage.prompt_tokens`). **77× smaller**, same output quality, **~12.45 M prompt tokens/day** off
the lane at the measured 298 ticks/24 h. Two honest qualifiers: that is prompt tokens removed from
the lane, not money; and the nominal 240 s interval produced a real ~306 s cadence, because the
sleep follows a call whose latency ran 8–187 s.

**And a title nobody read.** An LLM call to generate a session display name ran **258 times a
day**, one per monitor tick, on one-shot sessions that are never listed — plus **127 retries** when
it reached an unhealthy replica first. Skipping it for one-shot sessions removed all of it.

⛔ The first attempt at that last one was wrong in an instructive way: it returned early and
skipped the **naming**, not just the model call, leaving colliding one-shot sessions nameless —
because the cheap instant-title path does not deduplicate. Skip the expensive half, keep the
cheap half.

---

## What to take from this

1. **Record provenance at the seam.** You cannot reconstruct who interrupted a turn after the fact.
2. **Read your own judge prompt as an adversary would.** It said "treat this as DONE" and the model
   obeyed.
3. **A status flip is not a state transition.** If a post-turn hook does the real work, a command
   that only sets the status has done nothing — and every component will report success.
4. **Supervise for the state that actually hurt you**, which is rarely the one with a name.
5. **Bound every automatic retry**, including the ones you just added to fix an over-eager stop.
6. **Audit what your framework attaches to a background call.** Most of it is not needed, and the
   prompt may already say so.

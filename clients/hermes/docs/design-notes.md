# Design notes: several agent sessions, one backend

The two patches in this repository are what fell out of running multiple Hermes
sessions — each spawning subagents — against a single self-hosted inference
backend. The surrounding operational tooling that made that arrangement
survivable is **not published**: it encodes one deployment's capability tiers,
model pins, addresses and security policy, so as source it would be a liability
to read and useless to run.

The *shapes* are the reusable part. Four of them, written generically, with the
failure each one exists to prevent.

---

## 0. First, the arithmetic — nothing below can decide anything without it

Every pattern here is a decision, and every decision needs a satisfiability
check rather than a vibe. For a session or child with window `W`, output reserve
`R`, and incompressible floor `F` (system prompt + tool schemas + protected head
+ protected tail):

```
input budget      = W − R
compaction trigger ≈ max(0.75 × (W − R), MINIMUM_CONTEXT_LENGTH)   ... or
                     0.85 × (W − R)  when the floor exceeds the budget
satisfiable       ⟺  trigger > F, with margin
```

If `trigger ≤ F` the configuration is unsatisfiable and the agent will stall
somewhere no retry helps. See
[`child-context-window.md`](child-context-window.md) for a worked case where
this was false by 184 tokens and cost a wedged session.

Resident cost per in-flight request on a hybrid-attention backend is **not**
proportional to the window alone:

```
cost(request) ≈ fixed recurrent state (window-independent)
              + token-linear KV for the full-attention layers only
```

That asymmetry is why **admission is a stronger lever than window sizing** once
windows are small — dropping a request removes a fixed cost that shrinking a
window cannot touch.

---

## 1. Own-vs-shared accounting in every display

**Pattern.** Any figure read from a shared backend is labelled as the backend's.
Any figure about this session is labelled as this session's. Never render one
where a reader will take it for the other.

**Prevents.** An operator making a capacity decision against another session's
load in the belief that it is their own. Full treatment and the specific counter
trap in [`status-bar-accounting.md`](status-bar-accounting.md).

**Generalises to** KV pool used/total, queue depth, tokens/sec, GPU memory,
request rate — every shared number a per-session UI surfaces.

The rule in one line: *the moment you surface a shared resource's occupancy in a
per-session display, you have created a number that lies about ownership unless
you label it.*

---

## 2. Capacity governor

**Pattern.** A process that watches live backend pressure and adjusts
*configuration* — context windows, concurrency ceilings, admission thresholds —
to keep the pool from saturating.

**Prevents.** Thrash: N sessions each sized for an idle backend, collectively
evicting each other's prefix cache.

**⛔ The trap, and it is the reason patch 0002 exists.** A governor that writes a
*main-thread* profile value reshapes everything that **implicitly inherits**
from it. The incident in this repository is exactly that: a governor lowered one
profile window, subagents inherited it because nothing stated theirs, and their
compaction geometry became unsatisfiable. The governor was doing its job
correctly. The inheritance was the defect.

Rules that follow:

- **Every consumer of a governed value states its own.** Implicit inheritance
  across a process or role boundary is the bug, not a convenience.
- **Prefer runtime caps over config writes.** A cap applied to one dispatch is
  reversible, scoped, and cannot outlive the pressure that caused it. A config
  write persists, is global, and is invisible to whoever reads the file later
  expecting their own setting.
- **Log the derivation, not just the value.** `window: 65,536 → 98,304
  (auto/normal, token_usage 0.42)` is debuggable six hours later. `98304` is not.
- **Never let a governed value cross a satisfiability boundary.** Clamp to a
  floor derived from the arithmetic above; refuse rather than ship a geometry
  that cannot compact.

---

## 3. Admission guard

**Pattern.** A gate in front of dispatch that asks *is there room for this right
now?* and, when there is not, **queues** rather than rejecting.

**Prevents.** A fan-out that succeeds at the API level and then wedges, because
all N children were admitted into a pool that can hold four.

**Rules.**

- **Serialise, never reject.** The caller is usually a model. An error teaches it
  that fan-out is unreliable, and it will either stop using a working tool or
  retry the whole batch. Queueing preserves the semantics — every task still
  runs — and costs only latency. Patch 0002 applies its concurrency cap at the
  executor for precisely this reason: the gate above it returns an error, and
  "the pool is busy" must never turn a valid request into a failure.
- **Fail open.** A guard that depends on a monitoring endpoint must treat an
  unreachable endpoint as *permit*, not *deny*. Otherwise a blinking metrics port
  takes the agent down — a strictly worse outcome than the contention it was
  protecting against.
- **Cache the pressure read, under a lock held across the fetch.** A fan-out of N
  resolving its own admission N times must make **one** probe. Holding the lock
  across the HTTP call is the point: the second caller blocks and then reads the
  result, instead of joining a stampede against the backend you are trying not to
  load.
- **Derive the ceiling from the backend, not a constant.** `max_running_requests`,
  recurrent-state slot count and decode batch size are all published; where they
  disagree, the smallest is the truth.

---

## 4. Per-session rooms for agent↔operator messaging

**Pattern.** Each agent session gets its own durable chat room in whatever
messaging system the operator already reads. The agent posts status, questions
and completions there; the operator replies in the same room, and the reply
reaches that session and no other.

**Prevents.** Three problems at once:

- *Attribution loss.* With one shared channel, N sessions' output interleaves
  and "which one asked me this?" becomes unanswerable exactly when it matters.
- *Cross-talk.* An operator's "yes, go ahead" must not be consumable by a
  session that did not ask.
- *Lost questions.* A session blocked on a decision, whose prompt scrolled past
  in a shared channel, waits forever.

**Rules.**

- **Session identity, not agent identity.** Two sessions of the same agent on the
  same host are different rooms. A room keyed to the *agent* re-creates the
  cross-talk it was meant to prevent.
- **Durable, not ephemeral.** The operator is asleep in another timezone. A
  question posted to a stream nobody is watching is a question that was never
  asked; the room must still hold it hours later.
- **Room membership is the authorisation boundary.** Do not put an
  approve/deny-shaped message anywhere a different session can read and act on
  it.
- **The room is for the operator, not for the log.** Verbose progress belongs in
  the agent log. What goes in the room is what a human must read: a decision
  needed, a completion, a failure they will care about.

---

## What did not make it in

Two things worth naming because they are the obvious next steps and are not
solved here.

**Fair sharing across sessions.** Patch 0002's concurrency cap is *cooperative*
— it reads global pressure and voluntarily admits fewer children. Nothing stops
a session that does not run the patch from taking the whole pool, and nothing
arbitrates between two sessions that both back off politely and then both
advance. Real fairness needs a shared lease or a scheduler that all clients
respect, which is a different and larger piece of work.

**Prefix-cache-aware placement.** The ceiling rung exists partly because several
large children evict the parent's radix prefix, but the ladder infers that from
`token_usage` rather than measuring it. `cache_hit_rate` comes back in the same
`/v1/loads` payload and is not yet used. Sizing against measured cache hit rate
rather than raw pool occupancy is, on the evidence here, the most promising
unexplored direction.

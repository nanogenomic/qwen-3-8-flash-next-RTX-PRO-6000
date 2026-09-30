# Own children vs. backend occupancy in the status bar

Patch: [`../patches/0001-status-bar-own-vs-backend-accounting.patch`](../patches/0001-status-bar-own-vs-backend-accounting.patch)
· `cli.py`, +37 −2.

## The number that lies

A status bar segment that reads in-flight requests and KV usage from the model
backend is reporting **the machine**. If the backend is yours alone, that is the
same thing as reporting **you**, and the distinction never comes up. Share the
backend between sessions and the two quietly diverge.

What a session with zero subagents displayed:

```
⇉ 3/4 streams      221K/331.5K
```

Every reading of that is wrong in the same direction: *three of my subagents are
running*, *I am two thirds through my context pool*, *I have one slot left
before I'm saturated*. The truth was that another session was doing all of it.
Act on the misreading and you throttle work that was never competing, or wait
for capacity you already had.

Note that the arithmetic in the display was never wrong. `3/4` was a true fact
about the backend. The defect was entirely in what the number was **implicitly
claiming to be about** — which is the kind of bug that survives review, because
every individual piece of it is correct.

## The change

Count this session's own live children separately and label both figures:

```
⇉ 0 mine · 3/4 backend · 1 free
```

- `0 mine` — per-process, this session's own. The number the operator controls.
- `3/4 backend` — shared with every other client of that backend.
- `1 free` — `slots − backend_streams`, clamped at zero. The answer to the
  question actually being asked: *can I spawn another one right now?*

`mine` comes first because it is the number you are responsible for. Free slots
come last because they are the number you act on.

## ⛔ The pitfall — `_active_children` reads 0 for async delegations

The obvious source for "how many children do I have" is the agent's
`_active_children`, and **it is the wrong one**. That list holds only
synchronous/in-flight children, so for async delegations it is empty. The bar
then cheerfully renders

```
⇉ 0 mine · 3/4 backend · 1 free
```

while this session has four subagents working — the same class of wrong answer
the change set out to fix, merely inverted, and now with a confident label on
it. **This shipped that way first.** It is worth recording because the mistake is
so easy: the attribute is named exactly like the thing you want, it exists, it
returns an `int`, and nothing fails.

Use `tools.async_delegation.active_task_count()`. It is per-process, so it is
genuinely this session's, and it expands a fan-out batch to its real child count.

### Its sibling is also a trap

`tools.async_delegation` exposes two counters, and upstream documents the
difference:

| counter              | counts                        | a 5-child batch is |
|----------------------|-------------------------------|--------------------|
| `active_count()`     | async-pool **units** (slots)  | **1**              |
| `active_task_count()`| child **subagents**           | **5**              |

`active_count()` is right for capacity accounting — a batch does occupy one pool
slot — and wrong for a readout that claims to say how many subagents are
working. Upstream's own `snapshot["active_background_subagents"]` uses
`active_count()`, so it undercounts fan-out. Neither counter is buggy; they
answer different questions, and the status bar wants the second one.

## Integration points

Two sites in `cli.py`. If your checkout has no backend-derived delegation
segment, these are the whole change.

### 1. Status snapshot — add the own-children count

In the delegation block of `_get_status_bar_snapshot()`, alongside where
`delegation_streams` / `delegation_slots` are set:

```python
# ⛔ OWN CHILDREN vs BACKEND OCCUPANCY. delegation_streams is read from the
# SHARED backend and counts every client's in-flight requests -- other
# sessions included. A brand-new session with no children therefore rendered
# "3/4 streams", which reads as "I have 3 subagents" when it means "the box
# has 3 busy slots". Count this session's own live children separately so the
# bar can say which is which, and how many slots are actually left.
# active_task_count() is the module's own "how many subagents are actually
# working right now" figure: it expands a fan-out batch to its child count and
# is per-process, so it is genuinely THIS session's. _active_children is only
# the synchronous/in-flight list and reads 0 for async delegations, which is
# how this first shipped showing "0 mine" next to a busy backend.
_own = None
try:
    from tools.async_delegation import active_task_count as _own_task_count

    _own = int(_own_task_count())
except Exception:
    _own = None
if _own is None:
    _sync = getattr(agent, "_active_children", None)
    _own = len(_sync) if _sync is not None else None
snapshot["delegation_own_children"] = _own
```

The `_active_children` fallback is there only for the case where the import
itself fails; the comment records why it cannot be the primary source.

### 2. Render — label both figures

Where the segment body is built:

```python
slots = snapshot.get("delegation_slots")
own = snapshot.get("delegation_own_children")
# "N mine" first because that is the number the operator controls;
# the backend figure is shared with every other session on the box.
parts = []
if own is not None:
    parts.append(f"{int(own)} mine")
if slots:
    parts.append(f"{stream_count}/{int(slots)} backend")
    parts.append(f"{max(0, int(slots) - stream_count)} free")
else:
    parts.append(
        f"{stream_count} backend stream" + ("" if stream_count == 1 else "s")
    )
body = "⇉ " + " · ".join(parts)
```

replacing:

```python
if slots:
    body = f"⇉ {stream_count}/{int(slots)} streams"
else:
    body = f"⇉ {stream_count} stream" + ("" if stream_count == 1 else "s")
```

`max(0, …)` is load-bearing: the stream count and the slot ceiling come from
different reads, and a transient overshoot must not render as a negative number
of free slots.

## The generalisable rule

> The moment you surface a shared resource's occupancy in a per-session display,
> you have created a number that lies about ownership unless you label it.

It applies to every other shared figure in a multi-session deployment — KV pool
used/total, queue depth, tokens/sec, GPU memory. Each needs the same treatment:
say whose it is, or say that it is everyone's.

# Share-bounded pool grants, and showing the share

The implementation behind [client contract rules 10 and 11](../../README.md#10-bound-the-ceilings-you-grant-not-the-reservations-you-measure).
Written up rather than shipped as a diff — see [the README](../README.md#why-this-round-is-write-ups-and-not-patches).

Everything here is from one deployment of Hermes Agent against one self-hosted SGLang backend.
Names of modules are this fork's; the structure is what matters.

---

## The problem

Several Hermes sessions, in different terminals, each with their own subagents, share one
backend. Nothing in the framework knows that. Each session reads its own profile's
`context_length` and grows into it, and the backend's KV pool is a resource none of them can
see.

A cross-session **broker** was added to publish, every ~5 s, a small JSON document naming each
live session and the window it may use. Sessions read it; one writes it, chosen by a lock.

That part worked. The allocation policy inside it did not.

## The bug: an invariant that could not fail

The broker granted and then self-checked. On a 451,264-token pool it issued:

```
pid A   main_window 451,264   main_grow_max 451,264   child_grow_max 451,264
pid B   main_window 451,264   main_grow_max 451,264   child_grow_max 451,264
invariant {ok: True, total: 406,137, limit: 406,138}
```

Two sessions, each granted the entire pool, and an invariant passing by one token.

It passed **by construction**. The check summed each main's *measured* demand, scaled to fit the
budget:

```python
main_reserved   = sum(r["reserved"] for r in main_rows)   # r["reserved"] = int(r["need"] * f)
children_reserved = child_streams * inflight(child_window_max, child_max_tokens, pct)
reserved_total  = main_reserved + children_reserved
invariant = {"total": reserved_total, "limit": usable, "ok": reserved_total <= usable}
```

Because `f` was chosen to make the mains fit `mains_budget`, `reserved_total` could not exceed
`usable` whatever the grants said. The invariant was validating its own scaling arithmetic, and
was structurally blind to the policy it was supposed to police.

A second rail compounded it. A "mains are never shrunk" guard did
`mg = max(main_grow_max, main_window)`, which *raised* any bounded growth ceiling back to
whatever the session had already grown to — so even a correct grant was undone on the way out.

**What it cost.** Once both mains had grown into their grants, the broker's own log:

```
pool broker: mains are HOLDING 769,606 of 406,138 usable tokens (pids [A, B] are above their share)
```

with pool utilisation around 0.93 and ten requests queued. Both mains' prefixes were being
evicted and re-prefilled every turn — [client contract rule 4](../../README.md#4-never-advertise-a-window-larger-than-the-pool-or-smaller-than-it-grew-to)'s
failure mode, reached from the client side.

> A note on evidence: the two log lines above are preserved. The full instantaneous reading
> often quoted alongside them — a four-decimal utilisation figure, the running/waiting split, a
> zero cache-hit rate — came from a live `/v1/loads` read and from the pre-fix allowance
> document, which the broker rewrites every ~5 s. **Those artifacts are gone.** The geometry is
> reproduced in a regression test as named constants, with the test file stating that it was
> copied from the live stall rather than invented. Treat it as a transcription.

## The fix: derive ceilings from the live pool, then water-fill

`derive_window_ceilings(usable_tokens, mains, child_streams, child_max_tokens,
threshold_percent, ceiling) -> (per_pid_main_ceiling, child_ceiling, detail)`

1. **Start from the live bound, not from config.** `usable = pool − cache_reserve`, where the pool
   is the backend's own reported `max_total_num_tokens` and `cache_reserve` is a fraction (10 %
   here) held back so the prefix cache has somewhere to live.
2. **Reserve one child stream per granted stream** out of `usable`, before the mains are served.
   Subagents that cannot be admitted are not a saving; they are a stall.
3. **Round one:** each main gets `min(equal_share, max(need, GUARANTEED_LANE_WINDOW))`. A main
   wanting less than its share lends the difference.
4. **Round two:** redistribute what the lenders gave up, proportional to unmet want — or evenly
   if nobody wants more.
5. **Clamp on an 8K grid** until the *granted* worst case fits, shrinking child growth first and
   main ceilings only if that is not enough.
6. **Check the grants.** The replacement invariant sums `max(main_window, main_grow_max)` and
   `child_grow_max` over every grant it issued, and reports `window_ceiling_sum`, `held_total`,
   `deadlock_risk` and a `grandfathered` list of sessions already above their new ceiling.

Grandfathering matters: a session that has already grown past its new ceiling is **not** forced
to shrink mid-turn. It stops being granted growth, and the compaction path adopts the smaller
grant only when the resident prompt still fits under the smaller trigger. A ceiling that
retroactively invalidates a live conversation is worse than the oversubscription it fixes.

### Measured ceilings

`[measured by executing the function]` Pool **451,264**, `cache_reserve` 10 %, so
`usable = 406,138`; one child stream at 24,576 max tokens; threshold 0.85:

| Live mains | Granted window each | Child grant | Granted worst case | Feasible |
|---|---|---|---|---|
| 1 | 401,408 | 65,536 | 401,817 | yes |
| 2 | 196,608 | 73,728 | 403,045 | yes |
| 3 | 131,072 | 73,728 | 404,275 | yes |
| 4 | 98,304 | 73,728 | 405,503 | yes |
| 5 | 65,536 | 131,072 | 399,767 | yes |

Asymmetric, one main needing 433,782 beside one idle at 40,000: **270,336 / 131,072**, the
difference a revocable loan of 61,386.

**⛔ Every number above is specific to a 451,264-token pool.** The same code on a
1,310,720-token pool grants two mains 532,480 each. Quote the pool with the ceiling, always.

### Two constants that are easy to confuse

| Constant | Value | What it is |
|---|---|---|
| `GUARANTEED_LANE_WINDOW` | 131,072 | the share a lane is guaranteed **before lending** |
| `MAIN_WINDOW_FLOOR` | 64,000 | the actual floor a grant may not go below |

The first is **not** a floor, and calling it one is wrong in a way the table above disproves: at
five live mains the grant is 65,536, well under 131,072. 131,072 itself is derived, not chosen —
it is the first 8K rung at or above 1.5× the top of the band the deployment's main sessions
actually occupied.

---

## Showing it: the status bar

A context meter reading `ctx 175K/393.2K` on a session whose live share is 196.6K is not
slightly wrong, it is wrong in the direction that makes a user keep going. So the denominator
becomes the share.

The reader is a separate module from the broker, and **read-only by construction**: it parses
the published document and nothing else. It never probes the backend and never requests an
allowance. A status bar that acquires resources in order to draw itself changes what it measures.

Rendered field:

```
⇄ 2 mains · 451.3K pool                       normal — the share is the ctx denominator
⇄ 2 mains · 451.3K pool · share 196.6K ▲      ▲ = this session is holding ABOVE its share
⇄ pool stale 2m ⚠                             document present, not being refreshed
no share cap ⚠                                document written by a peer that does not bound grants
(absent)                                      no document: keep the static behaviour exactly
```

Three decisions worth copying:

- **Staleness is judged against the publish interval.** The document is rewritten every ~5 s, so
  the display threshold is **90 s** — long enough that a busy publisher is never called dead,
  short enough that a *gone* publisher is caught. It is deliberately the same age limit the
  broker applies to its own device-memory sample.
- **An unbounded document is detected, not trusted.** A grant above the pool ceiling, or grants
  summing above it, means the writer predates the fix. The reader says `no share cap ⚠` instead
  of rendering a number it knows is unenforceable. This is what makes a staged rollout safe
  while older peers are still writing.
- **It degrades by width, not by truncation.** Below a width threshold only the markers render;
  below a lower one the field is dropped entirely. A half-drawn capacity figure is worse than none.

---

## What to take from this

1. **An invariant must check the thing you control.** You control grants. Demand is an input.
2. **Derive budgets from the backend's live bound, never from client config.** The pool changes —
   this one went from 451,264 to 1,310,720 — and every derived constant has to move with it.
3. **Grandfather, do not retroactively shrink.** Stop granting growth; let the existing
   conversation finish.
4. **Publish the share to the UI, and make "I don't know" a renderable state.** The failure the
   user needs to see is not an error; it is a quietly wrong denominator.

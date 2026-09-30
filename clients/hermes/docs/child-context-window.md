# Explicit subagent context windows

Patch: [`../patches/0002-delegation-explicit-child-context-window.patch`](../patches/0002-delegation-explicit-child-context-window.patch)
· `tools/delegate_tool.py`, +339 −1. Applies to pristine upstream 0.20.4.
Full modified file: [`../upstream-modified/tools/delegate_tool.py`](../upstream-modified/tools/delegate_tool.py).

## The implicit inheritance

`_build_child_agent()` constructs the child `AIAgent(...)` without passing a
context window. Nothing downstream states one either. So the child adopts
whatever `model.context_length` the parent profile holds at the moment of
dispatch.

That is a reasonable default and an unreasonable *contract*. It means a single
edit to the main thread's profile — the exact edit any capacity tool makes when
it wants to relieve KV pressure — silently reshapes **every subsequently spawned
subagent**, with no log line connecting cause to effect. The parent keeps
working, because the parent is the thing that was measured before the edit. The
children, which nobody measured, are the ones that break.

## The incident

A capacity tool lowered the profile window to **65,536** to relieve pool
pressure. `delegation.max_tokens` stayed at **24,576**. Subagents inherited
65,536:

```
effective input budget = 65,536 − 24,576             = 40,960
threshold floor        = max(0.75 × 40,960, 64,000)  = 64,000  ≥ 40,960  → degenerate
compaction trigger     = 0.85 × 40,960               = 34,816 tokens
```

The floor comes from `MINIMUM_CONTEXT_LENGTH = 64_000` in
`agent/model_metadata.py`, via `_compute_threshold_tokens()` in
`agent/context_compressor.py`. When the floor exceeds the budget the compressor
takes its degenerate branch and the trigger lands at 85% of the budget.

Measured from the compression telemetry at the moment it wedged — two consecutive
attempts, not inferred:

```json
{"main_context_limit": 65536, "effective_threshold": 34816,
 "current_estimated_tokens": 34924, "protected_tail_tokens": 10229,
 "failure_class": "no_progress", "commit_status": "aborted"}
```

Decomposed, at the trigger the child held ≈34.9K tokens:

| region                                          | tokens | compressible |
|-------------------------------------------------|--------|--------------|
| system prompt + tool schemas + protected head   | ~25K   | no           |
| protected tail                                  | ~10.2K | no           |
| summarisable middle                             | ~10K   | yes          |

Compaction needed to get below 34,816 and had ~10K of material to work with
against ~35K of occupancy. It ran, made no progress, aborted, and the agent
stalled with **"compression is blocked (ineffective)"**. There is no retry that
helps: the incompressible floor is larger than the trigger.

**The cause is the geometry, not the compressor.** 65,536 with a 24,576 output
reserve leaves a 40,960 input budget, and a ~25K incompressible floor inside a
40,960 budget cannot be compacted under an 85% trigger. The configuration was
unsatisfiable from the moment it was written, and nothing reported that.

## Two defects, two fixes

### 1. State the window, and log it

Resolved explicitly at dispatch time. First hit wins:

| source                            | accepts                        |
|-----------------------------------|--------------------------------|
| `HERMES_CHILD_CONTEXT_LENGTH`     | int \| `auto` \| `inherit`     |
| `delegation.child_context_length` | int \| `auto`                  |
| *absent*                          | inherit — unchanged behaviour  |

**Absent config is a no-op.** Nobody who has not opted in sees any change.

Every dispatch logs one INFO line, which is the entire point — the child window
becomes visible in the agent log instead of being an invisible consequence of an
edit made in another process:

```
[subagent-0] child context window: 65,536 -> 98,304 tokens
  (delegation.child_context_length=98,304); compaction trigger 64,000,
  output cap 24,576
```

Guard rails: a value below `MINIMUM_CONTEXT_LENGTH` (64,000) is raised to it with
a warning, because `agent_init` refuses to *construct* below that and a
post-construction assignment would otherwise smuggle in a geometry the rest of
the runtime rejects. Booleans, non-integers and non-positive values warn and fall
back to inheriting. Nothing in this path can raise: a child that came up on the
inherited window beats a dispatch that died trying to resize it.

#### Why post-construction rather than a kwarg

`AIAgent.__init__` has no `context_length` parameter. Adding one means editing
`run_agent.py` and `agent/agent_init.py`, where the window is entangled with LM
Studio runtime loading, ollama `num_ctx` capping, plugin engine selection and the
`MINIMUM_CONTEXT_LENGTH` construction gate. The compressor's `context_length`
setter already re-derives `threshold_percent`, `threshold_tokens`,
`tail_token_budget` and `max_summary_tokens` coherently — which is the whole
contract needed — so assigning through it is the surgical form.
`_config_context_length` is set as well, so later resolution paths (display,
model switching) read the same number.

### 2. Never let `auto` choose an unsatisfiable geometry

The ladder's floor is **above** 65,536, deliberately:

| rung      | window  | compaction trigger | selected when                          |
|-----------|---------|--------------------|----------------------------------------|
| ceiling   | 131,072 | —                  | `token_usage < 0.35`, nothing queued   |
| nominal   | 98,304  | 64,000             | `token_usage < 0.70`, nothing queued   |
| floor     | 81,920  | 48,742             | anything queued, or `usage ≥ 0.70`     |
| *unknown* | 98,304  | 64,000             | pressure probe unavailable             |

- **81,920** → trigger `0.85 × (81,920 − 24,576) = 48,742`, which clears the
  measured ~25K incompressible floor with a real summarisable middle above it.
- **98,304** → trigger `max(0.75 × 73,728, 64,000) = 64,000`, non-degenerate,
  leaving ~27K of compressible middle above the ~25K floor. This is the value
  recommended as an explicit default.
- The ceiling exists because subagent work here is bounded — one goal, a fresh
  conversation, `skip_context_files` + `skip_memory` — so a child does not need
  the main thread's 262,144, and handing it one is how three children evict the
  parent's radix prefix.

Three rungs rather than a formula, on purpose: a clever curve nobody can reason
about is worse than boundaries an operator can check against `/v1/loads` by eye.
An explicit integer may exceed the ceiling; `auto` never does.

#### The pressure probe fails open, always

`auto` reads `/v1/loads` from the backend — richer than `/get_server_info`, and
the right signal because it carries `num_running_reqs`, `num_waiting_reqs`,
`token_usage` and `cache_hit_rate` together. Configure with
`HERMES_KV_BACKEND` (default `http://127.0.0.1:30000`); note that on a
router/backend split only the backend port publishes `/v1/loads`.

Cached for 5s under a lock held **across** the fetch, so a fan-out of N children
makes one HTTP call and the rest read the result — the alternative is a stampede
of probes against the very backend you are trying not to load. Timeout 1.5s. Any
error, any timeout → `None` → the configured value is used unchanged. **This
probe sizes a child; it must never be able to delay or fail a dispatch.**

## Concurrency is the stronger lever

The patch also caps child concurrency at runtime from the same signal, and the
reason is architectural rather than taste.

On a hybrid-attention backend, most layers are linear/recurrent rather than full
attention. On the model measured here, 36 of 48 layers are gated-delta-net, so
each in-flight request pins a **fixed** recurrent state that does **not** shrink
with the context window, plus a token-linear KV cost for the 12 full-attention
layers. Three independent numbers agreed on the same in-flight ceiling:

```
max_mamba_cache_size 12 / 3 slots per request  = 4 requests
max_running_requests                             4
cuda_graph_max_bs_decode                         4
```

Marginal arithmetic at the small end of the ladder:

| action                          | saves                        |
|---------------------------------|------------------------------|
| drop one child at 81,920        | 48,742 resident + 24,576 decode |
| shrink a child 98,304 → 81,920  | 15,258                       |
| shrink a child 81,920 → 64,000  | 15,232 — and re-enters the wedge |

Once the windows are already small, admitting fewer children beats shaving
windows. Shrinking past the floor buys about the same as the previous rung *and*
walks back into the stall.

The cap therefore **only ever lowers the worker count for one dispatch**:

| backend state                | workers          |
|------------------------------|------------------|
| `num_waiting_reqs > 0`       | 1                |
| `token_usage ≥ 0.70`         | `min(free, 2)`   |
| `token_usage ≥ 0.45`         | `min(free, 3)`   |
| otherwise                    | as configured    |

Every requested task still runs — they are **serialised, not rejected** — and
`delegation.max_concurrent_children` on disk is untouched. This is applied at the
executor rather than at the `len(tasks) > max_children` gate above it, because
that gate returns an *error* to the model, and "the pool is busy" must never turn
a valid fan-out into a failure.

## Configuring it

```yaml
delegation:
  child_context_length: 98304      # int, or "auto", or omit to inherit
  max_tokens: 24576                # the output reserve the arithmetic uses
```

```bash
export HERMES_CHILD_CONTEXT_LENGTH=auto
export HERMES_KV_BACKEND=http://127.0.0.1:30000
```

If you take nothing else from this: **check that your child geometry is
satisfiable before you trust it.** Given a window `W`, an output reserve `R` and
an incompressible floor `F` (system prompt + tool schemas + protected head +
protected tail), you need

```
0.85 × (W − R)  >  F        with meaningful margin
```

or compaction has nothing to work with and the agent will stall where no retry
helps. At `W = 65,536`, `R = 24,576`, `F ≈ 35K`: `34,816 > 35,000` is false, and
that is the whole incident in one line.

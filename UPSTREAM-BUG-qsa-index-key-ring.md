# Upstream bug: the Qwen sparse-attention pending index-key ring aliases at the shipped verify width

**This is a correctness bug in upstream SGLang, not an optimization, and it should be reported
to [`sgl-project/sglang`](https://github.com/sgl-project/sglang/issues).** This document is
written so it can be filed more or less as-is.

- **Affects:** `QwenSparseAttnBackend` — the full-attention backend for Qwen3.8-Flash-Next and
  any model using Qwen sparse attention (QSA) — whenever speculative decoding is enabled with
  a verify window `W > 1`.
- **Present in:** upstream `6fa3fe69e2e5e19b75cadd9fc285b72634551992`, the base commit of this
  fork, and in the older PR #36497 tree (`73a255206f`).
- **Symptom:** none. No crash, no warning, no NaN. Attention selects the wrong blocks.
- **Severity:** silent wrong results on the default configuration of a shipped model.
- **Fix in this fork:** commit `389ad4f`, `feat(qsa): multi-group pending ring for verify
  windows > compress ratio`.

---

## The bug

The QSA indexer keeps, per request, a ring of *pending* compressed index keys — the keys for
token groups that are still being accumulated. Upstream sizes that ring at exactly
`compress_ratio` rows per request and addresses it:

```
req_pool_idx * compress_ratio + (position % compress_ratio)
```

That addressing is only correct if a single forward touches at most `compress_ratio`
consecutive positions' worth of pending state.

It does not. **A forward that writes `W` consecutive positions and compresses a group that
ends inside that window touches up to `W + compress_ratio - 1` consecutive positions.** So the
ring aliases as soon as `W > 1`: two distinct positions within one forward map to the same row,
and the later write clobbers the earlier one.

## Why this is not a corner case

For Qwen3.8-Flash-Next the shipped configuration has `indexer_compress_ratio: 4` in
`config.json` and `speculative_num_draft_tokens: 4`, so **`W == compress_ratio == 4`**.

A verify window of 4 covers **every** residue class mod 4. Therefore the group that completes
part-way through the window is compressed from the **next** group's tokens whenever the
committed length is not a multiple of the ratio — which is **three verify cycles out of four**.

The engine does not notice, because the compressed pending key is consumed only by the
block-selection top-k. A wrong key means the indexer scores the wrong blocks, so attention
runs over a wrong sparse block set. The output is plausible and the logs are clean.

Note also that upstream *already knows* `W` can exceed the ratio — the same file rejects it
explicitly:

```python
draft_tokens = int(getattr(spec_info, "draft_token_num", 0) or 0)
if draft_tokens > self.compress_ratio:
    # The pending-group ring keys state by position % ratio; a verify
    # window wider than the ratio would collide within one forward.
    raise NotImplementedError(
        "Qwen QSA requires speculative_num_draft_tokens <= the QSA "
        f"compress ratio ({self.compress_ratio}): the pending "
        f"index-key ring holds one group; got {draft_tokens}")
```

The guard is off by `compress_ratio - 1`. The comment identifies the right mechanism; the
threshold lets the aliasing case through, and `W == ratio` is the default.

## The fix

Size the ring to cover the widest reachable span and address modulo that size:

```
R = compress_ratio * ceil((W + compress_ratio - 1) / compress_ratio)
row = req_pool_idx * R + (position % R)
```

and relax the guard to `W <= R - compress_ratio + 1`.

Four places index this ring and **all four must change together**, which is the part most
likely to be missed in a partial fix:

1. the eager metadata builders,
2. the CUDA-graph Triton kernel,
3. the graph-replay host fallback,
4. the **prefill/decode-disaggregation pending-state transfer infos** — whose page is one
   request's *whole ring*, not one group.

`W == 1` keeps the historical ratio-sized layout **bit for bit**, so a non-speculative
deployment is unaffected by the change.

In this fork the ring size is derived in a new `python/sglang/srt/layers/attention/qsa/config.py`
(`qsa_pending_ring_size`, `qsa_max_draft_tokens`) so the four call sites cannot drift apart.
`SGLANG_QSA_PENDING_RING_SLOTS=<n>` pins the old ring, purely so an A/B against an older build
can reproduce its addressing.

`speculative_eagle_topk > 1` remains rejected, for an unrelated reason: tree verify multiplies
the routed-expert union.

## Reproducing it

Two tests ship with the fix.

**`test/registered/unit/attention/test_qsa_pending_ring.py`** — **117 CPU test cases**, no GPU
needed. They drive the production slot builders over a range of prefill lengths, verify widths
1–8, exhaustive accept patterns, and rejection/rollback streams. Crucially, they **assert that
a ratio-sized ring aliases**, so the sizing cannot be quietly reverted later:

```bash
pytest test/registered/unit/attention/test_qsa_pending_ring.py
```

**`test/registered/kernels/ops/attention/qsa/test_qsa_wide_verify_ring.py`** — the GPU
correctness test: it compares the wide-ring result against a narrow reference on device and
reports `max|delta|` per location.

```bash
pytest test/registered/kernels/ops/attention/qsa/test_qsa_wide_verify_ring.py
```

Either test file demonstrates the bug against unpatched upstream.

## What has *not* been measured

Stated plainly, because it matters for how the report should be framed.

**There is no end-to-end quality A/B isolating this fix.** The fix landed as the first commit
of this fork and every subsequent measurement was taken with it in. A paired run — the same
build with `SGLANG_QSA_PENDING_RING_SLOTS` pinned to the old aliasing layout versus derived —
across the quality-gate suites would quantify how much output quality the aliasing actually
costs. **That run was never done.**

So the claim here is: **the addressing is provably wrong, and it is wrong on the default
configuration.** The claim is *not* a measured benchmark delta. Given that the consumer is a
top-k over block scores, the effect is plausibly partial — a wrong block set still contains
many relevant blocks — which would explain why it has gone unnoticed. That is a hypothesis,
not a measurement.

## Suggested issue text

> **Title:** QSA pending index-key ring aliases at `speculative_num_draft_tokens == indexer_compress_ratio` (silent wrong block selection)
>
> The pending index-key ring in `QwenSparseAttnBackend` holds `compress_ratio` rows per request
> and is addressed `position % compress_ratio`. A forward writing `W` consecutive positions and
> compressing a group ending inside that window touches up to `W + compress_ratio - 1`
> consecutive positions, so the ring aliases for any `W > 1`.
>
> Qwen3.8-Flash-Next ships `indexer_compress_ratio: 4` and
> `speculative_num_draft_tokens: 4`, so `W == ratio == 4`, and a verify window covers every
> residue class. The group completing mid-window is then compressed from the next group's
> tokens on three verify cycles out of four. This is silent: the compressed key is consumed
> only by the block-selection top-k, so the engine attends over a wrong sparse block set with
> no error.
>
> The existing guard in the same file (`if draft_tokens > self.compress_ratio: raise
> NotImplementedError`) names the correct mechanism but is off by `compress_ratio - 1`, and the
> default configuration sits inside the gap.
>
> Suggested fix: size the ring `R = ratio * ceil((W + ratio - 1) / ratio)`, address
> `position % R`, and relax the guard to `W <= R - ratio + 1`. Four sites index the ring and all
> must change: the eager builders, the CUDA-graph Triton kernel, the graph-replay host fallback,
> and the PD pending-state transfer infos (whose page is one request's whole ring, not one
> group). `W == 1` keeps the old layout bit for bit.
>
> A patch with 117 CPU tests (which assert that the ratio-sized ring aliases) and a GPU
> correctness test is available at
> https://github.com/nanogenomic/qwen-3-8-flash-next-RTX-PRO-6000 — see
> `UPSTREAM-BUG-qsa-index-key-ring.md` and commit `389ad4f`.

---

## A second, latent hazard in the same area, for the same report

Not the same bug, and **not fixed here**, but it belongs in any upstream conversation about
this code path.

`[measured on an SM120 card]` The CUTLASS MoE finalize path produces **wrong output** if the
router ever emits a **duplicate expert id with a non-zero weight**. The prefix sum `break`s on
the first matching slot, so the duplicate slot's `unpermuted_row_to_permuted_row` entry is
never written and finalize reads a stale row — **max absolute error 15.5** against a
merged-weight reference, with **23 of 24 test cases wrong**.

This is latent today because the stock router cannot emit duplicates. It becomes live the
moment anything writes expert ids directly — a batch-aware or piggybacking routing scheme, for
example. Two representations *are* safe, and were verified bitwise-equal to a narrow `[T, k]`
reference (12/12 cases across both finalize paths): dropping a slot as `-1`, or as an id
`>= num_experts`. Masking by setting weight 0 on a live id is also numerically exact but
**saves no time** (267 vs 265 µs at T=4), so it is not a useful alternative.

Either the kernel should handle duplicates, or the constraint should be asserted rather than
assumed.

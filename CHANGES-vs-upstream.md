# CHANGES vs upstream — full specification

Base: upstream SGLang `6fa3fe69e2e5e19b75cadd9fc285b72634551992`.
This fork: 32 commits, 43 files, +6,933 / −77.

**How numbers are labelled.** `[measured]` means a number produced by a run on the
hardware named. `[GPU2-proxy]` means it was measured on a **second, smaller SM120 card
(RTX PRO 4000, 70 SMs, ~546 GB/s)** with tensor shapes scaled, and is a *relative* signal
only — a GPU2-proxy microsecond figure is never a claim about throughput on the RTX PRO 6000.
`[arithmetic]` means it was computed from measured inputs (byte counts from checkpoint
headers × measured bandwidth). `[modelled]` means it comes from a calibrated model, not a
run. `[not measured]` is stated where it applies. Anything without a measurement is said to
have none.

**Read the noise band first.** Repeat cells of the same configuration on the same harness
gave 149.81 vs 140.42 tok/s — **6.3 %**. On a single sweep, single-stream differences below
about 6 % are not resolvable. Several 4-concurrent figures are n=1. [BENCHMARKS.md](BENCHMARKS.md)
has the detail.

---

## Contents

- [1. Correctness fixes](#1-correctness-fixes)
- [2. Kernel levers that shipped on](#2-kernel-levers-that-shipped-on)
- [3. Capacity levers](#3-capacity-levers)
- [4. Levers that are present and off](#4-levers-that-are-present-and-off)
- [5. Incomplete work](#5-incomplete-work)
- [6. Instrumentation](#6-instrumentation)
- [7. Why not 2x single-stream](#7-why-not-2x-single-stream)
- [8. Things that are upstream's, not this fork's](#8-things-that-are-upstreams-not-this-forks)
- [9. Commit list](#9-commit-list)

---

## 1. Correctness fixes

### 1.1 QSA pending index-key ring aliasing — a real upstream bug

| | |
|---|---|
| **Enabled by** | nothing — it is always on. `SGLANG_QSA_PENDING_RING_SLOTS=<n>` pins the *old*, aliasing layout, for an A/B against an older build. |
| **Files** | `python/sglang/srt/layers/attention/qsa/config.py` (new), `qsa/metadata.py`, `qsa/graph_metadata.py`, `qsa/qsa_indexer.py`, `qwen_sparse_attn_backend.py`, `python/sglang/srt/environ.py` |
| **Tests** | `test/registered/unit/attention/test_qsa_pending_ring.py` (117 CPU cases), `test/registered/kernels/ops/attention/qsa/test_qsa_wide_verify_ring.py` (GPU) |
| **Quality verdict** | This produces *different* attention output from upstream, because upstream's output was wrong. |

The pending index-key ring held exactly `compress_ratio` rows per request, addressed
`req_pool_idx * ratio + position % ratio`. A forward that writes `W` consecutive positions and
compresses a group ending inside that window touches up to `W + ratio - 1` consecutive
positions, so the ring aliases as soon as `W > 1`.

**At the shipped `W == ratio == 4` this is already wrong on three verify cycles out of four**:
a verify window covers every residue class, so the group completing mid-window gets
compressed from the *next* group's tokens whenever the committed length is not a multiple of
the ratio. It is silent, because the compressed key is consumed only by the block-selection
top-k — the result is attention over the wrong blocks, not a crash.

The fix sizes the ring `R = ratio * ceil((W + ratio - 1) / ratio)` and addresses
`position % R`, in all four places that index it: the eager builders, the CUDA-graph Triton
kernel, the graph-replay host fallback, and the PD pending-state transfer infos (whose page
is one request's whole ring, not one group). The target-verify guard relaxes to
`W <= R - ratio + 1`. **`W == 1` keeps the historical ratio-sized layout bit for bit.**
`speculative_eagle_topk > 1` is still rejected.

Full write-up, including why we believe it is upstream's and not a property of this fork:
**[UPSTREAM-BUG-qsa-index-key-ring.md](UPSTREAM-BUG-qsa-index-key-ring.md)**. **It should be
reported to `sgl-project/sglang`.**

### 1.2 Shared-scratch reservation: boot-order failures

Two follow-up fixes to §2.4's shared workspace. Both were found by real boot failures, and
without them the merged tree **failed to boot**.

**(a) A reservation covered by an existing buffer must be idempotent**
(patch `0030`, development SHA `486b099529`).
FlashInfer autotune hands out the shared packed-KV scratch *before*
`init_cuda_graph_state` reserves it, so a boot-time reservation of the same size raised
`already allocated at 33792 rows when a backend reserved 33792` — equal values, and it still
raised. A reservation `<=` the allocated size now succeeds; only a larger one raises.
Tests cover allocate-then-reserve equal / smaller / larger, plus a boot-order smoke
(autotune handout → every tier reserves → per-tier capture handout).

**(b) Regrow if autotune allocated undersized before capture**
(patch `0031`, development SHA `436fa5cf69`).
Autotune may hand out at the *active* tier width, below the widest reservation made later in
`init_cuda_graph_state`. Until a handout happens under CUDA-graph capture, no graph holds the
pointer, so a larger reservation now drops the undersized buffer and the next handout
allocates at the reserved size. After the first captured handout, growth still raises.

**(c) Two further reservation fixes** in the same area: patch `0023`
`fix(qsa): normalize the device key so cuda and cuda:0 share one buffer` (two spellings of
the same device were allocating two buffers) and patch `0028`
`fix(qsa): bound the shared reservation by the widest reachable batch`.

### 1.3 Adaptive-state sizing fixes

Three fixes for buffers sized at the wrong adaptive-speculation width. These matter whenever
more than one speculation tier exists, and are inert at a single static width.

| Fix | What was wrong |
|---|---|
| `fix(mamba): reserve verify intermediate state at the adaptive maximum` | The mamba verify intermediate state was reserved at the active width, not the deepest reachable one. |
| `fix(qsa): one max-width MTP selection buffer serves every adaptive state` | Each adaptive state was sizing its own MTP selection buffer. |
| `fix(qsa): require num_steps + 1 >= num_draft_tokens for MTP index sharing` | MTP index sharing was permitted in configurations where it is not valid. |

### 1.4 A duplicate-expert-id kernel hazard, documented not fixed

`[measured, GPU2-proxy]` The CUTLASS MoE finalize path is **wrong** if the router ever emits
a duplicate expert id with a non-zero weight: the prefix sum `break`s on the first matching
slot, the duplicate slot's `unpermuted_row_to_permuted_row` entry is never written, and
finalize reads a stale row — max abs error 15.5 against the merged-weight reference, **23 of
24 test cases wrong**. Dropped expert slots are therefore represented **only** as `-1` (or an
id `>= num_experts`), which is bitwise-equal to a narrow `[T, k]` reference (12/12 across two
finalize paths). Masking by setting weight 0 on a live id also gives exact output but **saves
nothing** (267 vs 265 µs at T=4).

This is latent upstream — the stock router cannot emit duplicates — and is **not fixed here**.
It is recorded because any future batch-aware routing scheme must dedupe per token before the
kernel. It is also a candidate upstream report.

---

## 2. Kernel levers that shipped on

These three are what the headline `+19.9 %` / `+17.9 %` consists of, together with the
capacity work in §3 and a speculative token map that is **not shipped** (see README).

### 2.1 `SGLANG_QWENOPT_FUSE_SBMOE` — 4-launch small-batch NVFP4 MoE

| | |
|---|---|
| **Enabled by** | `SGLANG_QWENOPT_FUSE_SBMOE=1` (off by default). `SGLANG_QWENOPT_FUSE_SBMOE_MAX_T` (default 16) sets the fallback threshold. |
| **Files** | `python/sglang/kernels/ops/moe/qwenopt_small_batch_nvfp4_moe.py` (new, 349 lines), `srt/layers/moe/moe_runner/flashinfer_cutlass.py`, `srt/models/qwen2_moe.py` |
| **Quality verdict** | **BITWISE equal** to flashinfer through sglang's own `_run_flashinfer_cutlass`, on layers 20 and 5, real router on real-token rows, `T = 1..16` with ~42 windows each. `frac_differ 0.0` at every T. Graph replay equals eager. Deterministic. |
| **Scope** | `T <= 16` and NVFP4 MoE only. Falls back to stock for prefill (`T > 16`) and for the BF16 MTP experts. |

Each MoE call drops from **9 launches** (3 prefix sums, expand, strides, GEMM1, act, GEMM2,
finalize) to **4**: input quant, fc1+SwiGLU+fp4 quant, fc2, and a top-k-ordered fp32 finalize.
Each expert's weights are read once.

**Measured effect:**
- `[GPU2-proxy]` µs per call at T=4, off → on: layer 20 **137 → 113**, layer 5 **196 → 173**.
  In a 188-SM emulation, **90.7 → 63.8**.
- `[arithmetic, from the GPU2-proxy deltas and a measured 16.2 ms cycle]` priced at
  **0.8–1.5 ms/cycle** (central 1.1–1.3), the largest single lever in the fusion set.
- `[measured]` Decode-path equivalence, SBMOE alone: **PASS at conc 1 and conc 4**;
  degeneracy 0/192.

It is bitwise-safe with dynamic-k because it drops invalid expert ids (`-1` or `>= E`) the
same way the CUTLASS path does — see §1.4 for why that representation and not weight-0
masking.

### 2.2 `SGLANG_QWENOPT_FUSE_HC` — fused hyper-connection chain

| | |
|---|---|
| **Enabled by** | `SGLANG_QWENOPT_FUSE_HC=1` (off by default). `SGLANG_QWENOPT_FUSE_HC_SHARED=0` disables the deferred shared-expert add. |
| **Files** | `python/sglang/kernels/ops/elementwise/hc_fused_tail.cuh` (new), `hc_fused_tail.py` (new), `kernels/ops/gemm/hc_mix.py`, `srt/layers/hyperconnection.py`, `srt/models/qwen4_exp.py`, `srt/models/qwen2_moe.py` |
| **Tests** | `test/registered/kernels/ops/elementwise/test_hc_fused_tail.py` (11/11 pass) |
| **Quality verdict** | The tail kernel alone is **bitwise equal** to apply + norm, `T = 1..16`, on real weights. **The full sublayer is within ≤ 1 bf16 ulp**: at most **37 of 163,840 elements flip**, from the gate-dot summation reorder and the pre-existing `hc_mix` atomics — which are already run-to-run nondeterministic at the same level. Graph replay is correct. |
| **Requires** | `CUDA_HOME` set and `ninja` on `PATH`, for the first-use JIT during graph warmup. TP=1 for the deferred shared add. |

Two things fuse: the inject gate folds into `hc_mix`'s down-projection so `hc_combine_gate`
stops running, and a new JIT CUDA kernel `hc_fused_tail` combines combine-apply, the next
sublayer's grouped RMSNorm, and (on the MLP side) `routed + σ(g)·shared`. The shared add is
deferred through `Qwen2MoeSparseMoeBlock.forward(defer_shared_add=)`.

**Measured effect:**
- **Launches: 9 → 4 per layer** (attention 4 → 2, MLP 5 → 2) ≈ **240 fewer launches per
  verify cycle** plus ~20 in the draft graphs.
- `[GPU2-proxy]` HC µs per layer, fused vs original: T=1 **0.965** (slower — the T=1 gate
  cost on 70 SMs), T=4 **0.925**, T=16 **0.897**; i.e. **−3.2 µs per layer at T=4** on a
  48-deep chain in one CUDA graph replayed 200×, 12 → 7 kernels.
- `[arithmetic]` priced **0.15 ms/cycle**, upper bound 0.54.
- `[measured]` With both fuse levers on (FUSEALL): decode-path **PASS at conc 1 and conc 4**,
  degeneracy 0/192.

`SGLANG_QWENOPT_FUSE_HC=1` **does not run in prefill.** It gates on
`_FUSED_MIX_MAX_ROWS = 16`, so at `T > 16` the original code path runs and is
bitwise-original — which is why the prefill-only NLL suite cannot see this lever at all
(BENCHMARKS.md §Quality instruments).

### 2.3 `--bf16-gemm-backend sm120gemv` — SM120 skinny BF16 GEMM

| | |
|---|---|
| **Enabled by** | `--bf16-gemm-backend sm120gemv`. A new choice on an existing flag; `auto` is unchanged. |
| **Files** | `python/sglang/kernels/ops/gemm/sm120_bf16_skinny_gemm.py` (new), `srt/layers/quantization/unquant.py`, `srt/layers/quantization/fp8.py`, `srt/arg_groups/fields/exec_.py` |
| **Tests** | `test/registered/unit/kernels/test_sm120_gemm_dispatch.py` |
| **Quality verdict** | Lossless. |
| **Measured effect** | **+3.2 % single-stream** `[measured]`. `[GPU2-proxy]` 1.00–1.25× vs cuBLAS on cold weights under graph replay; the M>16 tiling and the M=1 small-weight fallback landed as a follow-up commit. |

Triton skinny BF16 GEMM for `M <= 32` on SM12x with in-kernel split-K, cuBLAS otherwise.
The motivation is measured: an M2 profile of the stock engine showed `cutlass_80_wmma_*` —
**Ampere-generation** kernels — taking **6.61 ms of a 19.63 ms decode cycle (33.7 %)** on a
Blackwell card, while the NVFP4 MoE path already used a native SM120 kernel. So it is
specifically the BF16 dense path that was falling back. `sm120gemv` also removes
`splitKreduce` (0.40 ms/cycle) and most small-GEMM launch latency.

**The best gate-PASSING single-lever configuration measured was GEMV-only**: 155.05 tok/s
single-stream, 408.55 aggregate at conc 4, pool 279,680, lossless NLL gate **PASS** at mean
dNLL 0.00511.

### 2.4 Shared QSA trtllm workspace — `SGLANG_QSA_SHARE_SCRATCH`

| | |
|---|---|
| **Enabled by** | on by default. Auto-off under two-batch overlap / PDMux (the only paths with a second forward stream). |
| **Files** | `python/sglang/srt/layers/attention/qsa/shared_scratch.py` (new, 376 lines), `qwen_sparse_attn_backend.py`, `srt/environ.py` |
| **Tests** | `test/registered/unit/speculative/test_adaptive_shared_ws.py` (24 passed, 0 failed, 1 skipped) |
| **Quality verdict** | Allocation-site change only; no numerics change. |

Adaptive speculation builds one QSA backend set per tier, and each instance otherwise owns a
private **128 MiB** trtllm workspace plus a private packed-KV gather buffer. This shares one
max-size set process-wide. Its measured value is **pool headroom**: adaptive speculation with
private workspaces cost **−67 % of the KV pool**. See §1.2 for the three boot-order fixes this
needed.

### 2.5 `SGLANG_HC_MIX_PREFETCH` — prefetch the up-weight

| | |
|---|---|
| **Enabled by** | `SGLANG_HC_MIX_PREFETCH=1` (off by default). |
| **Files** | `python/sglang/kernels/ops/gemm/hc_mix.py` |
| **Quality verdict** | Lossless; the configuration carrying it passed the lossless NLL gate at mean dNLL **0.00511**. |
| **Measured effect** | **Neutral on its own.** It prefetches the up-weight before the phase barrier and drops a zeroing barrier. Published because it is harmless, tested, and part of the gate-PASSING configuration — not because it was shown to pay. |

### 2.6 SM120 skinny MXFP8 GEMM — `SGLANG_SM120_MXFP8_SKINNY`

| | |
|---|---|
| **Enabled by** | `SGLANG_SM120_MXFP8_SKINNY=1` (off by default). Only relevant to MXFP8 checkpoints. |
| **Files** | `python/sglang/kernels/ops/gemm/sm120_mxfp8_skinny_gemm.py` (new), `srt/layers/quantization/fp8.py` |
| **Quality verdict** | `[measured, GPU2-proxy]` **PASS on all 9 QAD shapes × M 1..32. Worst relative error 3.4 %**, against the fp32 dequant reference — which is the W8A8 activation-quant floor (~2.7 %). Every shape routed to the skinny kernel, including N=48 and the fused `in_proj_ba` N=96. |
| **Known gap** | **The prefill path (`M > 32`) goes through FlashInfer CUTLASS `mm_mxfp8` and was not GPU-tested.** That is exactly where the QAD blocker in §5.1 landed. Do not read "N=48/96 resolved" as a statement about prefill. |

---

## 3. Capacity levers

### 3.1 KV pool — what the +15.7 % is made of

`[measured]` Stock **241,728** tokens → **279,680** on the test card → **331,456** in
production. The production figure is larger only because that card carried less foreign VRAM.
Contributions, all measured:

- `--mem-fraction-static 0.98` instead of 0.958, with **3.39 GB of runtime headroom measured
  on the production card** at that setting. ⚠ **That headroom is enough for steady-state
  serving and not always enough for a lazy allocation made after serving starts** — the
  reference deployment was killed once by a constrained-decoding request taking a lazy Triton
  kernel-load path with 0.54 GiB free, and its recorded history has the same failure class
  killing the lane **twice in one day at 0.99** (2.20 GB headroom), once after 18 h 52 m of
  uptime. The mitigation (a grammar-constrained warmup request at startup, which costs no KV
  pool) and the full failure chain are in
  [README.md](README.md#operational-warning---mem-fraction-static-098-is-tight). Anyone copying
  this configuration should read that before treating 0.98 as free — in particular, **a memory
  fraction that survives a soak can still kill the lane hours later.**
- `--max-mamba-cache-size 12` with `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`, instead of 16.
  Mamba slots trade against KV at **~0.093 GB ≈ 3,750 pool tokens per slot**, and **each
  running request costs 3 mamba state slots** (`mamba num: 45` observed at
  `#running-req: 15`), i.e. **~11,250 pool tokens per running request**.
- The shared QSA workspace of §2.4, which removes the per-tier duplication.
- An upstream MTP weight-residency improvement — **not this fork's**, see §8.

`[measured]` A related admission finding worth stating because it looks like a hardware
ceiling and is not: **the 4-concurrent aggregate "ceiling" was an admission artefact.**
`--max-mamba-cache-size 16` admits only ~5 requests. Unclamping it reached **723.4 tok/s at
12 concurrent** — at the cost of giving up long context entirely (pool **63,680**). That
configuration is not what ships; it is recorded because it bounds what this card can do when
context is not needed.

### 3.2 `--kv-cache-dtype fp8_e4m3` — opt-in, lossy

Not a change in this fork (the flag is upstream's), documented here because this fork's
measurements are what qualify it.

| | Value |
|---|---|
| KV pool | **542,912 tokens** on the test card (2.25× stock). On the production card, **643,456** — the largest pool measured on this hardware. The multiplier is **1.941× on both**, which is why it is a property of the dtype and not of one card's spare VRAM. |
| Single-stream, short context | 159.8 tok/s, **−4.0 %** vs bf16 KV on this tree |
| Single-stream @250k | **145.8 tok/s** |
| 4-concurrent aggregate | 411.4 tok/s |
| GSM8K | **0.953 → 0.953**, identical |
| Needle retrieval | **1.0 → 1.0**, identical |
| Decode-path equivalence | **PASS at conc 1, WARN at conc 4 (p = 0.033)** |

The conc-4 WARN is a small real divergence, so this is labelled **lossy and opt-in**. Note
the indexer key cache does **not** shrink: `index_state_dtype` is hardcoded `torch.bfloat16`
in `qsa_kv_pool.py`.

Note also that **declaring a larger `--context-length` costs pool**: the same fp8 build and card
gives 542,912 tokens at a 262k declaration but **519,040** at 540k, about **−4.4 %**. The maximum
pool is only available at a modest declared window.

There is one blind spot worth stating: fp8 KV is **bitwise equal to bf16 KV at a
single-chunk prefill (ctx 512)** `[measured]`, because a single-chunk prefill attends over
fresh K/V rather than the quantized cache. The single-chunk NLL check therefore cannot see
KV dtype at all. Coverage for it comes from the multi-chunk NLL buckets, the decode-path
test (decode reads the cache) and needle retrieval.

### 3.3 Native NVFP4 PLE table

| | |
|---|---|
| **Enabled by** | `text_config.ple_embedding_dtype="nvfp4"` on a checkpoint that carries a native NVFP4 PLE table. Requires `--ple-offload-embedding` (the loader raises without it). |
| **Files** | `python/sglang/srt/models/qwen4_exp.py` |
| **Tests** | `test/manual/qwen4_exp_ple_nvfp4_gather_gpu.py` |
| **Quality verdict** | `[measured, GPU2-proxy, 200k real rows]` **PASS, bit-identical** to `dequantize_nvfp4` rounded to bf16. Out-of-range ids write zeros. |
| **Measured effect** | Pinned host table **26.8 GiB instead of 47.7 GiB** (uint8 `[320,001,536, 90]`), i.e. **−20.9 GiB of host RAM**. It also removes the FP8 re-code error, which was **1.38 % rel-RMS** on the same rows. |

The table is host-resident and dequantized inside the gather. PLE per-token traffic is
~20,480 B over PCIe gen4 ×16 — negligible bandwidth, non-negligible latency, and not
removable: `--no-ple-offload-embedding` would need 47.7 GiB more VRAM on a card already
holding 82.45 GB of weights.

### 3.4 1M context on one card — measured, and not the default

`[measured]` Full-quality 1M context works on a single card, using a bf16 KV cache spilled
to host RAM (**1,310,720-token host pool, 30.00 GiB pinned**) plus **static YaRN factor 4**.

**Needle retrieval: 5 of 5 pass, 4 of them beyond the native 262,144 limit.**

| Prompt | Needle distance | Prefill / TTFT |
|---|---|---|
| 300k | 299k | 45 s |
| 520k | 518k | 108 s |
| 760k | mid | 230 s |
| 990k | 987k | 414 s |
| 990k | 90k | 414 s |

**Decode, primary stream alone:** 250k **157.0 tok/s** · 500k **154.4** · 900k **141.0**.
TTFT 34 / 69 / 235 s (prefill ~7.3k tok/s at 250k and 500k, ~3.8k at 900k).

**Primary plus short agent streams in parallel:** at 250k the primary decodes ~92.9 tok/s
while three 8k short streams average 109.4 each (~421 aggregate); at 500k, 92.8 + 104.8
(~407); **at 900k the mixed run failed** — a GPU OOM in the chunked-prefill prefix gather
(`forward_extend` materializes the whole prefix's K and V, ~1.8 GB per layer call at 900k,
against a 4.6 GB margin), which killed the scheduler. The host-KV kernels themselves did not
fail.

Host RAM floor during the run: **63 GiB MemAvailable**. Host↔GPU transfer measured at
26.5 GB/s memcpy and ~19–22 GB/s for the decode gather.

**Caveats, stated plainly.** This is not the default lane and should not be:
- YaRN f=4 is **static and process-wide** — every request in that process gets it. It shifts
  short-context logits: mean |Δlogprob| **0.066–0.283 nats/token** against native on real
  text. Short-context benchmark scores did not move measurably (GSM8K 0.9533 = 0.9533,
  HumanEval 0.875 = 0.875, NIAH 4k–64k 39/40 vs 40/40, NLL within ±0.005 nats at ≤4k), but
  IFEval and the determinism suite on f=4 were **not measured**.
- **The automated quality gate for this configuration did not complete** (rc=2, request
  errors after the 900k mixed failure). There is **no decode-path verdict for the 1M
  configuration**.
- The long primary never got a radix-cache hit (`#cached-token: 0` on every primary
  prefill, while the 8k shorts did hit). **Cause not established.** Consequence: every turn
  of a long primary re-prefills its whole context — 234 s at 900k.
- The vendor's own MRCR-8needle numbers for this model fall to **40.5 at 512k and 26.4 at
  1M** (from 93.0 at 256k) `[vendor-published]`. Needle retrieval passing is **not** a
  statement about multi-needle reasoning at these depths.

### 3.5 Native RoPE past the trained window — no YaRN needed to 400k

`[measured]` Separately from the YaRN work above, and **more useful than it**: with fp8 KV and
**no rope override at all**, needle retrieval passes at ~200k (in-range control), ~300k (1.14×
the trained 262,144) and **~400k with the needle 398k tokens back** (1.53×, 57.1 s), and fails
at ~500k (1.91×, empty completion, 81.7 s). The 520,059-token case was an **HTTP 400 capacity
refusal** — pool 519,040 < prompt — and is **not** a position failure. Booting above the
derived window needs `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`.

**Native extrapolation therefore holds to at least 400k and has broken by 500k.** This is worth
more than the YaRN path for an agent lane, because YaRN f=4 is static and process-wide and
measurably shifts short-context logits, whereas native extrapolation costs nothing at short
context.

Three caveats, and they are load-bearing:
- **No quality gate was run at 400k.** Four needles at one offset each is a capability probe,
  not a quality verdict.
- **The 400k–500k cliff is unbracketed** — nothing in between was tested.
- fp8 KV is required to get a pool that admits these prompts, and fp8 KV is itself opt-in lossy
  (§3.2).

*Hypothesis, not a finding:* only **12 of 48 layers are full-attention** and
`partial_rotary_factor` is **0.25**, so RoPE acts on a quarter of the dimensions in a quarter of
the layers — a small positional surface, which would be consistent with graceful extrapolation.
No experiment here isolates that, and it should not be cited as the mechanism.

---

## 4. Levers that are present and off

### 4.1 Dynamic expert count (dynamic-k) — measured not to pay for single-stream

| | |
|---|---|
| **Enabled by** | `SGLANG_QWENOPT_DYNK` (inline JSON) or `SGLANG_QWENOPT_DYNK_FILE`; `SGLANG_QWENOPT_DYNK_CONTROL` (JSON polled every 2 s, live change with no graph recapture); `SGLANG_QWENOPT_DYNK_STATS_FILE` (in-kernel counters, written every 5 s); `SGLANG_QWENOPT_DYNK_CHECK=1` (duplicate assert). Modes `fixed`, `cumprob`, `learned`. **All off by default.** |
| **Files** | `python/sglang/srt/layers/moe/dynk.py` (new, 345 lines), `kernels/ops/moe/moe_fused_gate.py`, `srt/layers/moe/topk.py`, `srt/models/qwen2_moe.py` |
| **Quality verdict** | Off is **bitwise identical to the original kernel** `[measured, GPU2-proxy]`. `cumprob` with τ=1 and `k_min=k_max=k` is bitwise equal to fixed-k; with k=10, bitwise equal to off. Every byte-saving setting is **lossy at the per-layer level**. |
| **GPU end-to-end A/B** | **Not run.** Deferred; CPU and GPU2-proxy verified only. |

**Why it is off — the router is flat.** `[measured, 12,261 tokens/layer × 48 layers, real
traffic]` The top-10 experts hold a median of only **0.159 (~16 %) of the 512-way softmax
mass**. Renormalized top-10 weights: 0.207 / 0.146 / 0.118 / 0.101 / 0.089 / 0.080 / 0.073 /
0.067 / 0.062 / 0.059. Only **0.5 %** of (token, layer) pairs reach 80 % of the mass with
≤ 4 experts.

**The engine's own FP4 activation noise floor is ~11 %** `[measured: 11.3 % median relative
error between the engine's MoE output and an fp32 reference]`. Any truncation whose layer
error is above that is doing visible damage. Measured layer error (‖dropped tail‖/‖y‖, full
MoE-block output including the shared expert):

| Rule | mean k | mean err | p90 | p99 |
|---|---|---|---|---|
| dynamic cum ≥ 0.95 (renorm) | 9.73 | **2.0 %** | 7.7 % | 12.1 % |
| dynamic cum ≥ 0.90 (renorm) | 8.76 | 13.4 % | 17.7 % | 23.0 % |
| dynamic cum ≥ 0.80 (no renorm) | 7.26 | 19.8 % | 25.6 % | 32.0 % |
| static k=9 (renorm) | 9 | 12.3 % | 17.1 % | 22.8 % |
| static k=8 (renorm) | 8 | 19.9 % | 26.5 % | 33.1 % |
| static k=4 (renorm) | 4 | 63.1 % | 77.9 % | 88.2 % |

**And the payoff is small, because routed experts are only 17.9 % of a bs=1 cycle.**
`[arithmetic on measured byte counts and measured bandwidth]`

| Setting | Δ bytes/cycle | Speed |
|---|---|---|
| τ 0.95 (≈ lossless per layer) | −0.09 GB | **×1.004** |
| τ 0.90 (12–13 % layer error) | −0.48 GB | ×1.023 |
| τ 0.80 (20–25 % layer error) | −0.98 GB | **×1.048** |
| ceiling: zero routed bytes | −3.79 GB | ×1.217 |

**Verdict: dropped for single-stream.** The lossless version is worth 0.4 %; ~5 % costs
~20 % per-layer error against an ~11 % noise floor.

**One variant does look worth finishing, and its GPU test is deferred.** A *batch-aware*
fill — keep each token's own top-m, then fill the remaining slots only from experts already
in the group's union **and** in that token's own top-16, then renormalize — exploits the fact
that routed experts are **37 % of a bs=4 cycle and 44 % of a bs=8 cycle**.
`[measured, 6 layers, 120 real groups]` at 8 streams × 4 verify tokens, fill m=8 cuts
distinct experts **158.4 → 134.7** at **11.0 % mean layer error** (vs 19.9 % for a plain drop
to top-8 at the same expert count). `[modelled]` that is **×1.067 at bs=4 and ×1.071 at
bs=8**. At 4×4 the measured error is 13.6 %, not 11.0 %. It is built and CPU-tested; **the
GPU test was deferred and never run.** It needs a lossy NLL gate before anyone believes it.

Also measured and **not worth building**: dynamic-k on the MTP draft (draft routed experts
are 0.295 GB/cycle, so even k=0 caps at ~1.4 %); expert prefetch (every expert is already
VRAM-resident, and PCIe is ~25 GB/s against 1.29 TB/s HBM); expert pruning or hot-expert
pinning (no expert exceeds 0.448 per-token inclusion; the 122 never-selected (layer, expert)
pairs are worth 0.34 GB of VRAM and pruning them is lossy out-of-distribution); grouping
requests by routing similarity (across streams the union equals the popularity-only null:
37.0 vs 36.8 at 4×1, 67.3 vs 66.9 at 8×1). And if dynamic-k ever ships, **do not renormalize
the kept weights** — it adds 3–8 points of layer error at every threshold.

### 4.2 Learned dynamic-k gate — trained, evaluated offline, GPU test deferred

| | |
|---|---|
| **Enabled by** | dynamic-k mode `learned` (off). |
| **Files** | `python/sglang/srt/layers/moe/dynk.py`, `kernels/ops/moe/moe_fused_gate.py` |
| **Trained on** | **CPU**, on a 128-thread host (4 processes × 4 threads, capped at 1600 % CPU / 64 GB RAM). No GPU was used for training. |
| **Model** | Per-layer MLP 16 → 32 → 8, 48 × 808 = **38,784 parameters**, predicting `log10` of the truncation error from the sorted top-10 probabilities and 6 derived features. |
| **Weights** | **Not published.** |

`[measured, held-out 25 % of ~80k token-layers per layer, split by sequence]` At matched
mean k it beats the probability cutoff everywhere:

| mean k | Learned gate | Cumprob cutoff | Learned p95 | Cutoff p95 | Per-layer oracle |
|---|---|---|---|---|---|
| 5.0 | 31.7 % | 32.1 % | 44.2 % | 45.7 % | — |
| 6.0 | 26.0 % | 26.7 % | 36.7 % | 38.7 % | — |
| 7.0 | 20.7 % | 21.6 % | 30.0 % | 32.3 % | — |
| 8.0 | 15.4 % | 16.6 % | 23.5 % | 26.0 % | 14.5 % |
| **9.0** | **8.1 %** | 11.0 % | 16.9 % | 19.0 % | 7.4 % |

**Only the k≈9 setting is under the engine's own ~11 % FP4 noise floor**, and it is within
0.7 points of the per-layer oracle there. `[arithmetic]` it is worth about **+1.5–2 %
single-stream** — **below the 3 % bar that was set for shipping it**, so its value is as a
stacking component at concurrency, not as a stand-alone win. Runtime cost `[measured,
GPU2-proxy]`: **+0.9 µs per router call** (down from +2.8 µs before the features were rebuilt
from the top-10 rounds only), ≈ 0.2 ms/cycle ≈ 1 % of an ~18 ms cycle.

**Its GPU paired test was deferred and never run.**

### 4.3 Adaptive speculation — measured to cost more than it returns

| | |
|---|---|
| **Enabled by** | the adaptive tier configs in `python/sglang/srt/speculative/configs/`. Off in every shipping configuration. |
| **Files** | `python/sglang/srt/speculative/adaptive_spec_params.py` (new), `srt/speculative/eagle_worker_v2.py`, `srt/managers/scheduler_components/metrics_reporter.py`, `batch_result_processor.py`, plus the two tier configs |
| **Tests** | `test/registered/unit/speculative/test_adaptive_wide.py`, `test_adaptive_shared_ws.py` |
| **Quality verdict** | Lossless by construction — it changes speculation width, not numerics. |

`[measured]` It **does** raise accept length above 4.0 — it is the only thing that ever moved
accept length, and one decode line hit **4.10**, breaking the 4.00 cap. But it costs
**−3.8 % throughput and −67 % of the KV pool** (duplicated per-tier QSA scratch, which §2.4
then addresses). It is the reason the shared-scratch work exists, and it is still off.

Two tier configs ship in-tree: `adaptive_flashnext.json` (4 tiers, bs1 candidates [3,5,7])
and `adaptive_flashnext_lean.json` (VRAM-lean, bs1 [3,7], 9 batch-size captures instead of
13). Hysteresis in both is **derived from a tok/s break-even model, not tuned**.

### 4.4 Killed by measurement — no code was written

`[measured]` These were priced, tested, and dropped. They are listed so nobody spends the
time again.

| Candidate | Why |
|---|---|
| **PDL** (programmatic dependent launch) | Saves 0.21–0.28 µs per boundary **only when both sides are our own kernels**; −0.06 to +0.02 µs (nothing) next to cuBLAS, flashinfer or torch, which never trigger `launch_dependents`. Only 72 boundaries per cycle qualify → **0.017 ms/cycle**. Even all ~1650 small kernels on PDL ceilings at ~0.4 ms. |
| **`torch.compile` / inductor** | **3.2× SLOWER** on the HC layer chain with the kernel count unchanged: inductor cannot analyze `hc_mix`'s `tt.elementwise_inline_asm`, assumes every input is mutated, and `hc_mix` goes 12.4 → 49.1 µs. It also fails on `fused_sigmoid_gating_delta_rule_update_kernel` (loop-carried state promoted fp32 → fp64). Draft graphs are never compiled. |
| **MTP draft-chain merge** | ≤ 0.15 ms/cycle. `copy_result_to_cpu` blocks merging verify. |
| **GDN gated-RMSNorm rewrite** | Its 10 µs trace duration was PDL early-launch overlap; **exclusive** time is 2.0 µs per call, so ≤ 0.07 ms/cycle. |
| **Glue removal** | Nothing removable in the MoE and GDN paths. |
| **Tensor / pipeline parallel across two cards** | **No NVLink** (PCIe host bridge, gen4 ×16); a recorded TP=2 attempt on this box measured **20.2 tok/s** against 141 single-card; all-reduce costs ~5.06 ms/token on SM120; SM120 has no custom all-reduce. This model fits one card, so TP is not a capacity mechanism here. |
| **Disabling PLE host offload** | Needs 47.7 GiB more VRAM on a card already holding 82.45 GB of weights. |

---

## 5. Incomplete work

### 5.1 QAD / MIXED_PRECISION checkpoint loader — **boots, does not serve**

| | |
|---|---|
| **Enabled by** | `--quantization modelopt_mixed` on a ModelOpt MIXED_PRECISION checkpoint. |
| **Files** | `python/sglang/srt/models/qwen4_exp.py`, `srt/models/qwen4_exp_mtp.py`, `srt/layers/quantization/fp8.py`, `srt/mem_cache/kv_cache_configurator.py`, `srt/mem_cache/qsa_kv_pool.py` |
| **Tests** | `test/manual/qwen4_exp_qad_load_dryrun.py` |
| **Status** | **Incomplete. Do not use.** |

**What works** `[measured]`: the loader fix boots a pristine QAD checkpoint with
`quant_algo=MIXED_PRECISION`. A CPU dry-run passes cleanly — 301,730 tensors, 0 unconsumed,
0 double-consumed, 0 params unwritten, 0 dtype-changing writes, exact PLE row coverage — and
the same dry-run **fails on unpatched upstream**, reproducing the original crash on CPU. It
uses 26.8 GiB of pinned host memory for PLE instead of 47.7, and gives a **371,456-token KV
pool** (+33 % vs the lossless build here, +54 % vs stock).

It also fixes a **silent** wrongness: without the patch, the vision tower's FP8 values would
be copied into bf16 parameters with their scales dropped — a wrong model that errors nowhere.
And the MTP experts are W4A16_NVFP4 with no `input_scale`, which `flashinfer_cutlass` would
run as W4A4 with a neutral 1.0 activation scale, silently changing the draft's numerics
(acceptance only, not output correctness); `--speculative-moe-runner-backend marlin` is exact
W4A16 and is the fix.

**What fails** `[measured]`: **every inference request**. CUTLASS MXFP8 requires
**n ≥ 128 and k ≥ 128**; one projection has **n = 96**:

```
ValueError: MXFP8 requires n >= 128 and k >= 128 for CUTLASS MXFP8. got m=122, n=96, k=2560
  flashinfer/gemm/gemm_base.py:4868 _check_mm_mxfp8_problem_size
```

24 of 24 evaluation requests errored, 0 tokens produced. `SGLANG_SM120_MXFP8_SKINNY=1` did
**not** route these small-N GEMMs away from CUTLASS (see §2.6 — the skinny kernel covers the
decode path, `M <= 32`; this failure is on the prefill path).

**The attempted fallback refused to run, correctly.** A small-N fallback guard reported:
`needs the plain [N, K//32] uint8 scale, got shape=(10240,) (expected (96,80))`. So the MXFP8
`weight_scale` is **pre-swizzled and padded at load time**: `10240 = 128 × 80`, i.e. N padded
96 → 128 and flattened. The guard computed nothing rather than producing wrong output.

**What it needs:** a verified de-swizzle — reshape to `128 × 80`, slice to `96 × 80` — with a
proof that the layout is row-major rather than interleaved; or a loader that keeps a plain
copy of the scale for small-N layers. That was not guessed at. It remains **opt-in and lossy
even when fixed**, because the QAD checkpoint is itself lossy relative to BF16 (this engine
path is lossless *with respect to the checkpoint*).

### 5.2 MTP draft-head retraining — no in-engine gain, and a train/serve trap

**Outcome: two retrained heads, neither deployed. Published as a negative result, because the
offline-to-engine non-transfer is the interesting part.**

#### The final head measured flat in-engine

`[measured]` Offline on held-out traffic, the final head (v3) beat stock everywhere, including
at full context with the engine's own sparse attention:

| Offline metric (accepted tokens per step) | Stock | v3 |
|---|---|---|
| Windowed, sampled / greedy | 2.474 / 2.783 | 2.575 / 2.950 |
| Out-of-distribution, sampled / greedy | 2.222 / 2.629 | 2.285 / 2.764 |
| Full ~16k context, sparse attention, sampled / greedy | 2.556 / 2.881 | 2.596 / 2.973 |

`[measured]` **On a B200 running plain upstream SGLang it also won in the real engine, in all
four cells** (stock is 4 reps; the in-distribution T=1 cell is n=240 at concurrency 4, 2 reps):

| Cell | Stock mean [range] | v3 (reps) | Δ |
|---|---|---|---|
| OOD, T=1 | 2.055 [2.027–2.082] | 2.139, 2.125 | **+3.7 %** |
| OOD, T=0 | 2.505 [2.431–2.541] | 2.562, 2.554 | **+2.1 %** |
| In-distribution, T=1, n=240 | 2.498 [2.491–2.506] | 2.555, 2.530 | **+1.8 %** |
| In-distribution, T=0 | 2.904 [2.890–2.921] | 2.971, 2.989 | **+2.6 %** |

`[measured]` **On this tree's fused / SM120 path it was flat.** Same eval-split probe, same
production configuration, only the checkpoint changed:

| | Stock | v3 |
|---|---|---|
| Accept length (counter method) | 2.1079 | **2.1076** |
| Single-stream | 180.5 | 180.0 / 185.8 |
| 4-concurrent aggregate | 413.9 | 428.0 / 431.0 (n=1, inside noise) |

**It was not deployed. The cause of the non-transfer is unknown.** The two candidate
differences are the hardware (SM100 vs SM120) and this tree's fused decode path; neither was
isolated. What can be said is narrow and worth saying: **a draft head that wins on a different
GPU generation and a different kernel path is not evidence it will win on yours.**

#### The trap: the first retrain regressed hard, from a train/serve mismatch

This is the part worth reading before retraining an MTP head on this architecture.

`[measured]` The first retrained head **regressed badly in-engine** — accepted tokens per step
**2.108 → 1.677**, single-stream **180.5 → 120–131 tok/s**, 4-concurrent **414 → 309–327** —
while its offline numbers looked good (+4.4 % sampled, +5.3 % greedy).

**Root cause: the training reimplementation attended densely where the engine attends
sparsely.** The MTP layer's QSA indexer (`index_qk_proj`) takes the attention *input* and
selects the top-512 of 4-token blocks (budget 2,048); decode steps then reuse that selection.
The PyTorch training model attended **densely** over its windows. The retrain moved exactly the
weights that feed the indexer — `fc_embedding` by 113 %, `fc_hidden` by 57 %, `attn_hc` by
8–26 % — so at serving time the indexer was selecting a *different* context subset than the one
the head had been trained against.

**And the equivalence gate did not catch it, for a structural reason: it only covered ≤ 2,048
tokens, where dense and sparse selection are identical.** The gate was measuring a regime in
which the bug cannot exist.

The fix that made offline and in-engine agree was twofold:
1. **Freeze everything upstream of the attention input** — `fc_embedding`, `fc_hidden`,
   `pre_fc_norm_*`, `attn_hc`, and the MoE gate router — so the indexer sees exactly stock
   inputs and selects exactly what the engine selects.
2. **Train with the engine's own sparse selection**, and extend the gate past the ratio.
   `[measured]` A sparse-attention gate at 8k (40 held-out sequences, 26,336 generated
   positions, the engine's own MTP as reference) gives sparse-torch 0.6514 vs engine 0.6511
   (Δ +0.0003, argmax agreement 0.9778), against dense-torch Δ −0.0008 / 0.9745.

`[measured]` That reframing also deflated the apparent win: the windowed-dense evaluation had
shown **+4.4 % / +5.3 %**; at real context with sparse attention the same head is **+0.5 % /
+1.4 %**. The offline gain was substantially an artefact of the dense approximation.

**Generalisable lesson:** when a model's attention is sparse and *input-dependent*, any
training reimplementation must reproduce the selection, and the equivalence gate must run
**past the compression ratio** — otherwise it certifies a regime where the two implementations
are trivially equal. This is the same class of error as the QSA ring aliasing in §1.1: a check
whose window is narrower than the mechanism it is supposed to test.

**Neither head is published**; both were trained on private traffic. The hooks that produced
their training data are in §6.

Ceiling worth knowing: `[arithmetic]` **even a perfect draft head tops out at 2.72 accepted
tokens per step** under production sampling.

## 6. Instrumentation

### 6.1 MTP hidden-state dump

| | |
|---|---|
| **Enabled by** | `SGLANG_MTP_HIDDEN_DUMP_DIR`. **Completely inert without it.** Tuning: `SGLANG_MTP_HIDDEN_DUMP_GATE`, `..._MAX_TOKENS`, `..._QUEUE`, `..._SHARD_GIB`. |
| **Files** | `python/sglang/srt/debug_utils/mtp_hidden_dump.py` (new, 324 lines), `srt/speculative/eagle_worker_v2.py`, `srt/models/qwen4_exp_mtp.py` |

Writes, from inside the scheduler process, the exact tensors the MTP head consumes, for
draft-head training. It also writes a `SERVER.json` so a dump client can refuse to talk to a
non-dump instance, and supports a `KEEP_FROM` tail-window control so only the last N tokens
of a stream are kept.

### 6.2 Per-tier adaptive acceptance counters

`srt/speculative/eagle_worker_v2.py`, `srt/managers/scheduler_components/metrics_reporter.py`,
`batch_result_processor.py`. Reports a **width-correct** accept rate per adaptive tier. The
naive accept-length counter is not comparable across tiers.

### 6.3 Dynamic-k kept-slot counters

`SGLANG_QWENOPT_DYNK_STATS_FILE` writes in-kernel per-slot kept-expert counters every 5 s.

---

## 7. Why not 2x single-stream

Stated plainly because it is the most useful negative result here.

`[measured]` Single-stream decode on this card is bandwidth-bound at **9.885 GB of weight
traffic per target forward pass**, of which **86.6 % is BF16 that the NVFP4 quantization
never touched** — the 36 gated-delta-net layers alone are 4.173 GB (42.2 %), plus
hyper-connections 13.0 %, `lm_head` 12.9 %, full attention 12.5 %. Only the routed experts
(13.4 %) are NVFP4. The stock engine already ran at **88 % of the memory bandwidth measured
on the card**.

`[source-verified]` And speculation is hard-capped. Two independent limits, read out of the
installed tree, not inferred:

1. `speculative_eagle_topk` must be 1 — `assert self.speculative_eagle_topk in (None, 1)`,
   and the QSA backend raises `NotImplementedError` otherwise. **No tree drafting; the draft
   is a linear chain.**
2. `speculative_num_draft_tokens <= indexer_compress_ratio = 4`. So **accept length ≤ 4.00
   per forward, permanently, on this engine.** Independently corroborated: across 738 logged
   decode samples the observed accept length maxes at exactly 4.00, never higher.

`[modelled, fitted to two measured context points and validated out-of-sample within 5 % on
three others]` 300 tok/s single-stream would need accept length **4.03 at zero context, 4.38
at 32k, 5.41 at 128k** against that cap of 4.00. Even a hypothetical perfect draft falls 1 %
short at zero context and 26 % short at 32k.

`[arithmetic]` The draft depth is also already optimal: at the measured per-step accept rate
of 0.667, `nstep/ndraft` 3/4 maximizes throughput (165.1 tok/s at 32k in the model, against
162.0 at 2/3 and 161.7 at 4/5). **Do not sweep it.**

**So 2× is not reachable by any flag, kernel, backend or CUDA-graph change.** It requires
removing bytes — requantizing the dense BF16 path — which is a checkpoint change behind a
quality gate, not a serving change. `[modelled]` dense BF16 → FP8 gives ~244 tok/s at 32k and
→ NVFP4 ~289. **The lossless kernel work in this tree reached 1.2×.** Aggregate throughput is
a different story: it is bounded by KV-pool VRAM, not by kernels, and ~800 tok/s aggregate is
reachable at moderate per-stream context.

---

## 8. Things that are upstream's, not this fork's

Stated explicitly so this fork does not take credit for them. All three were part of the gain
from *moving to* base `6fa3fe69e2`, and none is a commit here.

- **SM120 routing for QSA decode.** The long-context decode slowdown — a decode cycle whose
  cost grew with context because the trtllm-gen paged decode path was gated behind
  `is_sm100_supported()` and an SM120 card fell through to an FA2 varlen prefill-shaped
  kernel — **is already fixed at this base commit**: `_resolve_trtllm_sparse_decode()` gates
  on `is_sm100_supported() or is_sm120()`. The measured effect is real and large (per-token
  latency 14.7 → 15.4 ms from 325 to 250k context, versus growing with depth on the older
  engine), and it is upstream's.
- **MTP weight residency.** The older engine kept the MTP head's own 512-expert MoE resident
  at 4.37 GB, which held the KV pool down; this base loads it at ~0.24 GB.
- **`--kv-cache-dtype fp8_e4m3` for QSA.** The flag and the QSA support are upstream's; only
  the measurements in §3.2 are this fork's.

One further finding is a **configuration** recommendation, not a fix here: NVFP4 plus
speculative verification without `--fp4-gemm-backend flashinfer_cudnn` is on record hitting a
CUTLASS FP4 GEMM race that propagated NaN through speculative verification into repetition
loops. The safe-on-record setting is to pin `flashinfer_cudnn` **and wipe the FlashInfer JIT
cache when changing it** (stale kernels keep the bug). This fork does not fix that race; it
pins the flag. Related upstream issues referenced during this work: `#38290`, `#36811`,
`#37111` (NaN router bias → repeated-token collapse under NEXTN), `#38851` (FP8-KV NEXTN
accept length collapse), `#38319` / `#38355` (poisoned prefix cache), `#37326` (accept-length
decay over uptime).

---

## 9. Commit list

31 functional commits plus one commit that ships the adaptive-speculation tier configs
in-tree. The original development history had 39 commits: 8 were merge commits from
integrating parallel feature branches, and those are branch bookkeeping that was dropped when
linearizing onto the upstream base.

**The published tree was verified byte-identical to the original 39-commit merge-based head**
before publication hygiene was applied. That hygiene — making the optional build and
checkpoint paths environment-configurable, replacing development-host names in comments with
their roles, adding SPDX lines, and removing internal tracker ids from source comments — was
folded into **every** revision of **every** affected blob rather than added as a commit on the
end, so no revision in `patches/` contains a development-machine path. It changes no
executable line: outside the four files whose hardcoded paths became environment variables,
the only differences from the development tree are comments and docstrings.

**The patch number is the stable identifier.** `git am` assigns new commit SHAs on every
application, so this table maps patch file to the SHA in the development tree, for anyone
correlating against the engineering notes. `COMMIT-MAP.tsv` has the same mapping in
machine-readable form.

| Patch | Development SHA | Subject |
|---|---|---|
| `0001` | `5d5bbd7a51` | feat(qsa): multi-group pending ring for verify windows > compress ratio |
| `0002` | `1fa1337b1e` | test(qsa): GPU correctness test for the widened pending ring |
| `0003` | `53c85dc0aa` | feat(spec): scheduler-side hidden-state dump for MTP draft-head training |
| `0004` | `195682cce2` | feat(spec): dump writes SERVER.json so the dump client can refuse a non-dump instance |
| `0005` | `6bbc07ec1e` | feat(spec): KEEP_FROM tail-window control for the MTP hidden dump |
| `0006` | `d76fed293a` | fix(qsa): require num_steps + 1 >= num_draft_tokens for MTP index sharing |
| `0007` | `91dcb90c1c` | feat(gemm): sm120gemv BF16 backend, Triton skinny GEMM for M<=32 on SM12x |
| `0008` | `0eec3a1293` | perf(hc_mix): prefetch up-weight before the phase barrier, drop zeroing barrier |
| `0009` | `a06f56e8b9` | fix(mamba): reserve verify intermediate state at the adaptive maximum |
| `0010` | `00897e9e66` | fix(qsa): one max-width MTP selection buffer serves every adaptive state |
| `0011` | `52391633d3` | feat(spec): per-tier adaptive acceptance counters, width-correct accept rate |
| `0012` | `6395eefe36` | perf(gemm): sm120gemv M>16 tiling and M=1 small-weight fallback |
| `0013` | `a98e221c94` | perf(mxfp8): SM120 skinny MXFP8 GEMM on native block-scaled MMA, opt-in |
| `0014` | `a4aa369ee7` | fix(qwen4_exp): load ModelOpt MIXED_PRECISION QAD checkpoints |
| `0015` | `cc1ba2b840` | feat(moe): dynamic expert count (fixed-k / cumprob / learned) fused into the Triton router, off by default |
| `0016` | `63d92e2f47` | feat(moe): dynamic-k per-slot kept-slot counters + stats writer (SGLANG_QWENOPT_DYNK_STATS_FILE) |
| `0017` | `4309772a86` | test(gemm): CPU dispatch + CLI coverage for the SM120 skinny BF16/MXFP8 GEMM |
| `0018` | `9d08f8ffa5` | feat(moe): dynamic-k live control file (per-slot tau/k_min/k_max on device) for one-boot arm sweeps |
| `0019` | `d7e1d0a51b` | feat(qwen4_exp): native NVFP4 PLE table (host-resident, dequant in gather) |
| `0020` | `7621d4a796` | perf(moe): learned dynamic-k gate features from the top-10 rounds only (F=16), +0.9 us/call on GPU2 |
| `0021` | `27c30a8f3d` | perf(qsa): one shared trtllm workspace + packed-KV scratch across QSA backends |
| `0022` | `bcf45ab741` | test(qsa): pin the shared-scratch allocation plan and its byte budget |
| `0023` | `6a5f2da329` | fix(qsa): normalize the device key so cuda and cuda:0 share one buffer |
| `0024` | `705ca532c9` | perf(moe): SGLANG_QWENOPT_FUSE_MOE_FINALIZE alias for flashinfer fused finalize |
| `0025` | `f8fc8b5a35` | perf(hc): fused hyper-connection chain behind SGLANG_QWENOPT_FUSE_HC=1 |
| `0026` | `22fc5c8ffd` | test(qsa): correct the selection width -- qsa_token_topk IS the budget |
| `0027` | `d0ecae43d8` | perf(moe): lossless ILP finalizeMoeRouting via private flashinfer build (SGLANG_QWENOPT_FUSE_MOE_FINALIZE_ILP) |
| `0028` | `837e52ff80` | fix(qsa): bound the shared reservation by the widest reachable batch |
| `0029` | `12d6193745` | perf(moe): 4-launch expert-centric small-batch NVFP4 MoE for T<=16 (SGLANG_QWENOPT_FUSE_SBMOE) |
| `0030` | `486b099529` | fix(qsa): reservation covered by an existing shared buffer is idempotent |
| `0031` | `436fa5cf69` | fix(qsa): regrow the shared scratch if autotune allocated it undersized before capture |
| `0032` | — | feat(spec): ship the adaptive-speculation tier configs in-tree |

The development head these were verified against was `be24acf0a5`, a merge commit whose tree
the series reproduces. Note that four files differ from the development tree by design (the
environment-variable change); see §2 of [BENCHMARKS.md](BENCHMARKS.md) for what was and was
not re-verified after it.

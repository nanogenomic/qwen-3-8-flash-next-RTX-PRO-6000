# qwen-opt — SGLang optimizations for Qwen3.8-Flash-Next on one RTX PRO 6000 Blackwell

A fork of [SGLang](https://github.com/sgl-project/sglang) carrying decode-path kernel
work, a QSA correctness fix, and memory-capacity work for serving **Qwen3.8-Flash-Next
(NVFP4)** on a **single NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 96 GB, 300 W)**.

**Base commit:** upstream SGLang `6fa3fe69e2e5e19b75cadd9fc285b72634551992`
(*"[PD] Keep the sampling mask of a replayed rebootstrap token (#41235)"*).
32 commits on top, 43 files, +6,933 / −77.

> **Every performance lever in this tree is env-gated and OFF by default.** With no
> `SGLANG_QWENOPT_*` variable set and no `--bf16-gemm-backend sm120gemv`, the engine takes
> the original upstream code paths. The two exceptions are the **QSA index-key ring fix**
> (a correctness fix — it is always on, and `SGLANG_QSA_PENDING_RING_SLOTS` exists only to
> pin the old, aliasing layout for an A/B) and `SGLANG_QSA_SHARE_SCRATCH`, which defaults
> to on and only changes *where* a buffer is allocated.

---

## Headline measured results

Against **the same model on the stock engine** (SGLang PR #36497 @`73a255206f`) on an
identical card. Temperature 0. The stock figures are the production lane's own **logged**
rates; this tree's are single measurement sweeps on a dedicated card. See the noise band
before reading any single-stream difference under ~6 % as real, and BENCHMARKS.md §1.2 for
why this is a cross-engine comparison rather than a single-variable A/B.

| Metric | Stock engine | This tree (all lossless levers on) | Δ |
|---|---|---|---|
| Single-stream, short context | 138.8 tok/s | **166.4 tok/s** | **+19.9 %** |
| Single-stream peak (@325 tokens) | — | 180.5 tok/s | — |
| 4-concurrent aggregate | 351 tok/s | **413.9 tok/s** | **+17.9 %** |
| KV pool (test card) | 241,728 tokens | **279,680 tokens** | +15.7 % |
| KV pool (production card, less foreign VRAM resident) | 241,728 tokens | **331,456 tokens** | +37 % |
| Per-token latency, 325 → 250k context | grows with depth | 14.7 → 15.4 ms | +4.5 % over 770× context |

### Single-stream throughput across the full context range

`[measured]` Decode speed on this card is **near-flat in context depth** — the single largest
qualitative change versus the older engine, where per-token cost grew with depth. Per-token
latency moves only **14.7 → 15.4 ms from 325 tokens to 250k**, i.e. **+4.5 % across a 770×
increase in context**.

No single configuration spans the whole range, because what limits depth is the KV pool, not
speed. So the columns below are three different arms; each is labelled, and none of the numbers
is interpolated.

| Context depth | bf16 KV (the shipping config) | + fp8 KV (opt-in lossy) | host-KV + YaRN f=4 (1M arm) |
|---|---|---|---|
| 325 tokens | **180.5** | 167.3 | — |
| 48.6k | **175.6** | 164.2 | — |
| ~95–98k | **155.5** (@98k) | 154.4 (@95k) | — |
| 250k | *not swept* | **145.8** | **157.0** |
| 500k | *not swept* | — | **154.4** |
| 900k | *not swept* | — | **141.0** |

All tok/s, temperature 0, single stream.

**Read the "not swept" cells correctly — they are a gap in the measurement, not a limit of the
build.** The shipping bf16-KV configuration holds a **279,680-token pool** (331,456 in
production), so it can serve well past 98k; its sweep simply used harness cells at 4k / 48.6k /
95.4k and never went deeper. **A deep single-stream sweep on the exact shipping configuration
was not run** — it is listed as an open item in [BENCHMARKS.md §6](BENCHMARKS.md#6-open-measurements).
What the deeper rows do establish is that the *engine* sustains 141–157 tok/s out to 900k.

Caveats on the two right-hand columns: fp8 KV is opt-in lossy (decode-path WARN at concurrency
4, p = 0.033). The 1M arm additionally uses a host-RAM-resident KV pool and **static YaRN f=4**,
which is process-wide and shifts short-context logits by 0.066–0.283 nats/token against native —
so it is a separate lane, not a drop-in. Its quality gate returned INVALID; see
[CHANGES §3.4](CHANGES-vs-upstream.md#34-1m-context-on-one-card--measured-and-not-the-default).
The native-RoPE 400k result below is a **retrieval** probe — no throughput was measured on it.

**Quality:** the decode-path equivalence test — the instrument that actually exercises the
decode kernels — is **PASS at concurrency 1 and at concurrency 4**, with **0 / 192
degenerate outputs**. Neither that test nor the NLL suite certifies *bitwise* equality,
because decode on this stack is nondeterministic even for the unmodified baseline (4–7 of
96 identical output pairs within the baseline itself).

**How "lossless" was decided.** The obvious instrument, teacher-forced NLL, is the wrong
one here. It runs prefill in 4,096-token chunks, so it never executes the decode kernels
these levers change (they gate at `T <= 16`). Beyond one chunk it is also nondeterministic
run to run, even with no lever on. The gate is therefore built around three rules:

- **Decode-path equivalence is primary.** The test uses 48 real prompts that fit in one
  prefill chunk. Each is decoded greedily for 384 tokens at concurrency 1 and 4, using
  output logprobs only. Agreement length and \|Δlogprob\| are compared against a measured
  baseline-vs-baseline spread.
- **ctx-512 NLL is a hard check.** It is bit-deterministic on the baseline, so a lossless
  claim fails on any token moving by more than 1e-4. This check is blind to KV-cache dtype.
- **Multi-chunk NLL is judged only against a measured baseline repeat.** It is never judged
  against a fixed budget.

Every verdict is PASS / WARN / FAIL / INVALID / NOT_MEASURABLE. A suite with more than 5 %
errors is INVALID, never PASS. [BENCHMARKS §4](BENCHMARKS.md#4-quality-instruments--what-each-one-can-and-cannot-see)
covers what each instrument can and cannot see, the statistical power at each sample size,
and the open gaps. No strict-mode (≤ 2-point) verdict was issued. The harness is published in
[`tools/qgate/`](tools/qgate/README.md), so you can gate your own changes the same way.

**Optional, lossy, off by default —** `--kv-cache-dtype fp8_e4m3`:

| | Value |
|---|---|
| KV pool | **542,912 tokens** on the test card (2.25× stock); **643,456** on the production card — the largest measured on this hardware |
| Single-stream, short context | 159.8 tok/s (−4.0 % vs bf16 KV here) |
| Single-stream @250k context | **145.8 tok/s** |
| 4-concurrent aggregate | 411.4 tok/s |
| GSM8K | 0.953 → 0.953 (identical) |
| Needle retrieval | 1.0 → 1.0 (identical) |
| Decode-path equivalence | PASS at conc 1, **WARN at conc 4** (p = 0.033) |

The conc-4 WARN is a small but real divergence, which is why fp8 KV ships **labelled lossy
and opt-in** rather than as a default.

### Context beyond the trained window, with no YaRN

`[measured]` With fp8 KV for the pool and **no rope override of any kind**, this model
retrieves past its trained 262,144-token window on native RoPE alone. Needle retrieval, one
needle per request in a real-text haystack:

| Prompt | × trained window | Result | Wall |
|---|---|---|---|
| ~200k (in-range control) | 0.76× | **PASS** | — |
| ~300k | 1.14× | **PASS** | — |
| ~400k, needle 398k tokens back | 1.53× | **PASS** | 57.1 s |
| ~500k | 1.91× | **FAIL** — empty completion | 81.7 s |
| 520,059 | 1.98× | **HTTP 400 — capacity refusal**, pool was 519,040 < prompt. Not a position failure. | — |

So **native extrapolation holds to at least 400k and has broken by 500k.** Booting above the
derived window needs `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` (SGLang otherwise refuses),
and reaching a pool that can admit these prompts needs `--kv-cache-dtype fp8_e4m3` — which is
itself opt-in lossy (decode-path WARN at concurrency 4).

**Read this as a capability probe, not a quality result.** Four needles are not a quality
verdict: **no quality gate was run at 400k**, and the 400k–500k cliff is **unbracketed** —
nothing between 400k and 500k was tested. If you want the window, gate it first.

*Hypothesis, not a finding:* only **12 of 48 layers** are full-attention (the other 36 are
gated-delta-net linear attention), and `partial_rotary_factor` is **0.25**, so RoPE covers a
quarter of the dimensions in a quarter of the layers. That is a small positional surface to
extrapolate, which would be consistent with graceful degradation rather than a hard edge — but
it was not tested, and it is not an explanation anyone should rely on.

This matters practically because it is **cheaper than YaRN**: static YaRN f=4 is process-wide,
shifts short-context logits by 0.066–0.283 nats/token against native, and needs a separate
process. Native extrapolation to 400k costs nothing at short context.

Full methodology, harness, rep counts and noise bands: **[BENCHMARKS.md](BENCHMARKS.md)**.
Per-lever specification: **[CHANGES-vs-upstream.md](CHANGES-vs-upstream.md)**.

---

## The correctness fix, separately

`qwen_sparse_attn_backend.py` sized the QSA pending index-key ring at exactly
`compress_ratio` rows per request and addressed it `position % ratio`. At the shipped
verify width `W == ratio == 4` that **aliases on three verify cycles out of four**, so the
compressed index key for a group is built from the *next* group's tokens and block
selection attends to the wrong blocks. It is silent — the compressed key is only consumed
by the top-k.

This is an upstream bug, not a property of this fork, and it should be reported to
`sgl-project/sglang`. Write-up, reproduction and the reasoning:
**[UPSTREAM-BUG-qsa-index-key-ring.md](UPSTREAM-BUG-qsa-index-key-ring.md)**.

---

## Quickstart

### Repository layout

This repository holds **only what this fork changes**, plus the documentation and the commit
series — not a second copy of all of SGLang. Concretely:

```
README.md  CHANGES-vs-upstream.md  BENCHMARKS.md  UPSTREAM-BUG-qsa-index-key-ring.md
LICENSE                  upstream SGLang's Apache-2.0 licence, unmodified
make-fork.sh             builds the full, rebasable fork locally (below)
patches/                 the 32 commits as a git-am-able series against the base commit
COMMIT-MAP.tsv           patch file -> development SHA, for the engineering notes
python/  test/           the 43 changed files, at their exact upstream paths
tools/qgate/             the quality-gate harness behind BENCHMARKS §4 (not part of the engine)
```

The source files sit at their real upstream paths, so you can read them here, diff them
against an upstream checkout, or `rsync` them over one.

### Build the rebasable fork

```bash
git clone https://github.com/nanogenomic/qwen-3-8-flash-next-RTX-PRO-6000
cd qwen-3-8-flash-next-RTX-PRO-6000
./make-fork.sh /path/to/sglang-qwenopt
```

That shallow-clones upstream at the pinned base commit and applies the series, giving a real
fork where the normal git operations work:

```bash
cd /path/to/sglang-qwenopt
git diff upstream-base..qwenopt                          # exactly this fork's change set
git log  --oneline upstream-base..qwenopt                 # the 32 commits
git rebase --onto sglang/main upstream-base qwenopt       # rebase onto newer upstream
```

To apply the change set onto an upstream checkout you already have:

```bash
cd /path/to/your/sglang
git checkout 6fa3fe69e2e5e19b75cadd9fc285b72634551992
git am /path/to/this/patches/0*.patch
```

### Run it

This is the configuration the measured results were taken on, transcribed from the reference
deployment's own service unit rather than reconstructed:

```bash
export SGLANG_QWENOPT_FUSE_SBMOE=1     # 4-launch small-batch NVFP4 MoE, bitwise-identical
export SGLANG_QWENOPT_FUSE_HC=1        # fused hyper-connection chain, within 1 bf16 ulp
export SGLANG_HC_MIX_PREFETCH=1
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
export CUDA_HOME=/usr/local/cuda       # SGLANG_QWENOPT_FUSE_HC JIT-compiles one CUDA kernel
export FLASHINFER_CUDA_ARCH_LIST=12.0 MAX_JOBS=8
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
export TRTLLM_ENABLE_PDL=1

python -m sglang.launch_server \
  --model-path <your Qwen3.8-Flash-Next NVFP4 checkpoint> \
  --quantization modelopt_fp4 --trust-remote-code \
  --context-length 262144 --mem-fraction-static 0.98 \
  --attention-backend flashinfer --sampling-backend flashinfer \
  --moe-runner-backend flashinfer_cutlass \
  --bf16-gemm-backend sm120gemv \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size 12 --page-size 64 \
  --max-running-requests 4 --cuda-graph-max-bs-decode 4 \
  --cuda-graph-backend-prefill=disabled \
  --chunked-prefill-size 4096 --max-prefill-tokens 4096 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --ple-offload-embedding --enable-metrics --enable-cache-report
```

Two notes on flags that are easy to get wrong:

- **`--speculative-algorithm NEXTN`, not `EAGLE`.** NEXTN is what selects this model's MTP
  head. The server then *reports* `EAGLE` in `/get_server_info`, because NEXTN resolves onto
  the EAGLE worker internally — so the launch flag and the reported value legitimately differ.
- **`--fp4-gemm-backend` is left unset here**, which resolves to `auto`, because that is what
  the measurements ran on. Separately, `flashinfer_cudnn` is the one setting on record as
  known-safe against a CUTLASS FP4 GEMM race that propagated NaN through speculative
  verification into repetition loops. Whether `auto` resolves to a safe kernel on SM120 was
  **not traced**. If you pin `flashinfer_cudnn`, wipe the FlashInfer JIT cache at the same
  time — stale kernels keep the bug.

`SGLANG_QWENOPT_FUSE_HC=1` needs `CUDA_HOME` set and `ninja` on `PATH` at first use, for
the `hc_fused_tail` JIT during CUDA-graph warmup.

### Operational warning: `--mem-fraction-static 0.98` is tight

0.98 buys the KV pool, and it leaves only about **3.39 GB of device headroom** on a 96 GB card
`[measured]`. That is enough for steady-state serving and **not** always enough for a *lazy*
allocation made after serving starts.

**This failure class is recurring, not a one-off.** The reference deployment's own recorded
operational history has it happening twice in a single day at `--mem-fraction-static 0.99`, on
2026-09-11 — **once after 18 h 52 m of continuous uptime**, and once earlier for want of 48 MiB.
Both were **runtime allocation failures during serving**, not CUDA-graph capture failures.
Measured at 0.99 on that lane:

| | |
|---|---|
| KV pool | 270,208 tokens |
| Static headroom reported at boot | **2.20 GB** |
| At the OOM | process held **93.44 GiB**; card had **190.06 MiB** free; the allocation needed **312.00 MiB** |
| Co-resident on the same card | one process holding 550 MiB **and growing**, plus a launcher holding a further 698 MiB |

The note that replaced the 0.99 configuration puts it better than a paragraph can:

> 2.20 GiB of static headroom is not a margin, it is a countdown.

The lesson that generalises: **a memory fraction that survives a soak can still kill the lane
hours later.** The 18 h 52 m death is the important one — nothing in a start-up check or a
five-minute soak would have caught it. Anything else resident on the card, or anything that
grows, eats the same budget.

At the shipped 0.98 the headroom is 3.39 GB rather than 2.20 GB, which is more margin — and
**the same class of failure still occurred.** At 16:45:24 on 2026-09-29 the deployment died
with:

```
Triton kernel 'apply_token_bitmask_inplace_kernel' device-loaded after serving started
(free device mem: 0.54 GiB). Pre-load it during engine init to avoid CUDA OOM.
```

→ scheduler exception → `SIGQUIT` → `kill_process_tree`. A constrained-decoding request
(tool-call / JSON, i.e. `response_format`, a grammar, or a tool schema) took a **lazy Triton
kernel-load path with no VRAM left to load into** — the same shape as the 0.99 deaths above: an
allocation that arrives *after* the static pool is already committed. Nothing in this fork's
levers caused it; it is what a high static memory fraction costs.

The mitigation that was applied — and the one to prefer, because it **costs no KV pool** —
is to issue one grammar-constrained request at startup, so the kernel loads while memory is
still plentiful:

```bash
curl -sf "127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"<served-model-name>",
  "messages":[{"role":"user","content":"Reply with a JSON object containing one key \"ok\" set to true."}],
  "max_tokens":16, "temperature":0, "response_format":{"type":"json_object"}}'
```

Run it after `/health` returns, and make it non-fatal — a warmup hiccup must never keep the
lane down. If you would rather not depend on a warmup step, lower `--mem-fraction-static`
and accept the smaller pool.

> **⛔ If you wire this into a service manager, detach it.** A post-start hook generally runs
> *inside* the unit's start timeout. On the reference deployment that timeout is **90 s**, while
> the warmup polls for health — which takes **~9 minutes on a 206-shard checkpoint load**. Run
> inline, the hook does not return, the service manager concludes the unit failed to start,
> kills it, and restarts it. **It did that on every boot until the hook was detached** (four
> restarts before it was caught). Either launch the warmup as its own transient unit so the
> hook returns immediately, or raise the start timeout well past the load time. Do not leave it
> inline: the mitigation for one outage becomes a restart loop, which is worse than the problem
> it fixes.

### Sizing for multiple clients and subagents

The benchmarks above are single-stream and 4-concurrent. Neither tells you how to size this
when several agent sessions — each spawning subagents — share one backend. That is a different
problem, and it has an architecture-specific answer.

**The KV pool is shared by every client, and demand oversubscribes trivially.** A live reading
from the reference deployment at 18:11Z, on the 331,456-token bf16 pool:

| Field | Value |
|---|---|
| `num_used_tokens` | 254,144 (**76.7 %** of pool) |
| `num_running_reqs` | 3 |
| `num_waiting_reqs` | 6 |
| `num_waiting_uncached_tokens` | **335,949** — more than the entire pool |
| `num_total_tokens` | 590,093 (**1.78×** oversubscribed) |
| `cache_hit_rate` | 0.0 |

That is one instantaneous reading, not a benchmark. It is here because of what it shows.

**The failure mode is not an OOM — the scheduler queues. It is prefix-cache collapse.** Queued
requests have their cached prefixes evicted and then re-prefill from scratch. At ~250k context
that is tens of seconds per turn, every turn. So *"it queues, so it is fine"* is the wrong
conclusion: nothing errors, and the lane just becomes slow in a way that looks like the model
got worse. For contrast, the same deployment normally serves **64 % of prefill tokens from
cache** (218 prefill batches with >50k cached, max 203,904) `[measured]` — a `cache_hit_rate`
of 0.0 under load is the signal that this is happening.

#### The cost model is affine, not linear — and that is architectural

This model has **36 gated-delta-net (linear attention) layers and only 12 full-attention
layers**, which splits the memory cost in two:

- The **12 full-attention layers** carry KV that is **linear in tokens**: 24,576 B/token at
  bf16 for K+V `[arithmetic from the checkpoint config]`, or **27,392 B/token (26.75 KiB)**
  once the compressed indexer keys and the MTP draft's own KV are included. The measured pool
  line worked out at 26.0 KiB/token.
- The **36 GDN layers** carry a **recurrent state that is constant per sequence** — it does not
  grow with context. Each running request costs **3 mamba state slots** `[measured:
  "mamba num: 45" observed at 15 running requests]`.

So the pool cost is roughly `N × fixed + total_tokens × rate`. Two consequences matter more
than the formula:

1. **Concurrency is capped by `--max-mamba-cache-size`, not by the token budget.** At 12 slots
   and 3 slots per request that is **4 concurrent requests**, which is why the shipping config
   pairs `--max-mamba-cache-size 12` with `--max-running-requests 4`. Raise one without the
   other and nothing happens.
2. **Every extra mamba slot comes out of the KV pool 1:1** — `[measured]` **~0.093 GB ≈ 3,750
   pool tokens per slot**, so **~11,250 pool tokens per additional running request**.
   Concurrency and context trade directly against each other.

The practical inversion: **for many small requests the fixed term dominates, so admitting fewer
clients can free more pool than shrinking each client's context.** That is the opposite of the
usual intuition, and it follows directly from 36 of 48 layers being context-length-independent.

#### Sizing principle

**What must fit the pool is the in-flight set — bounded by `--max-running-requests` — not every
client's theoretical maximum.** That makes the arithmetic tractable:

- One long-context primary session at ~190k resident plus three small subagents at ~35k each is
  ~295k, which fits a 331k pool.
- Four long-context sessions at ~190k each is ~762k, which thrashes.

So: **give the primary session the full window, and size subagents explicitly and small.**
Subagent work is bounded — a subagent that searches, reads and reports does not need the
primary's window — and sizing it small is what buys the primary its context. Treat the numbers
above as an illustration of the method, not as your budget; substitute your own pool and your
own session shapes.

#### The trap: subagents that inherit their context window

If your orchestrator derives a subagent's context window from the **parent's** configured
`context_length`, then editing that one value silently reshapes every subagent spawned
afterwards. This is worth stating because of how it fails.

On the reference deployment a capacity-tuning change lowered a profile's `context_length` to
65,536. Subagents inherited it, and their compaction trigger became:

```
(65,536 window − 24,576 output reserve) × 0.85 = 34,816 tokens
```

At that point **compaction could not free enough to get back under its own threshold, and
agents stalled outright** — not slowly, not with a warning. Note the second-order problem: a
24,576-token output reservation is **37.5 % of a 65,536-token window**. A reservation sized for
a large window becomes pathological in a small one.

Two rules, and they hold for any orchestrator on any engine:

1. **Size subagent context explicitly. Never inherit it from the parent.**
2. **Keep the output reservation proportionate to the window**, and check that the compaction
   threshold leaves enough room for compaction to actually succeed.

#### fp8 KV is the capacity lever, and it is the honest trade

`[measured]` The KV pool goes up **1.94×**, and that multiplier now has **two independent
confirmations on cards with different amounts of foreign VRAM resident**:

| Card | bf16 pool | fp8 pool | Ratio |
|---|---|---|---|
| Test card | 279,680 | 542,912 | **1.941×** |
| Production card (read from `/get_server_info` at 18:31Z) | 331,456 | **643,456** | **1.941×** |

Agreeing to three decimal places across different occupancy is worth stating plainly: **the
1.94× is a property of the dtype change, not of one card's spare memory.** **643,456 tokens is
the largest pool measured on this hardware**, at `--context-length 262144`,
`--mem-fraction-static 0.98` and `--max-running-requests 4`.

That is the single biggest capacity change available, and it buys roughly one extra long-context
session or a handful of subagents.

What it costs, stated as precisely as the measurements allow:

| | |
|---|---|
| GSM8K (n=150) | 0.9533 → 0.9533 — **identical** |
| Needle retrieval | 1.0 → 1.0 — **identical** |
| Degeneracy | **PASS** |
| Decode-path equivalence, 1 stream | **PASS** |
| Decode-path equivalence, 4 streams | **WARN**, p = 0.033 |
| Single-stream throughput | **−4.0 %** |

The WARN is a real divergence and is why this is labelled lossy. But be precise about what the
instrument measures: **divergence from the bf16 reference, not degradation.** No task metric
moved. It is labelled lossy because it changes numerics by design, and because a small real
divergence at concurrency 4 is exactly what you would expect from quantizing the cache the
decode path reads. Decide accordingly: for an agent lane that is pool-starved, 1.94× the pool
for no measured task-metric change is usually the right trade; for anything where bit-level
agreement with a bf16 baseline matters, it is not.

**An independent check that fp8 does not cost much single-stream throughput in practice.**
`[measured, live, 18:31Z]` Three sequential 512-token greedy completions of the same prompt
through the serving router on the production card after warm-up: **175.3 / 183.3 / 173.1 tok/s**
(mean ~177). These are **router-path chat completions, not harness cells** — different prompt,
different path, different cell definition — so they are **not** directly comparable to the sweep
numbers in [BENCHMARKS.md](BENCHMARKS.md). They are a sanity check, and they are consistent with
the harness result on the other card, where fp8 measured 159.75 against bf16's 166.42 (−4.0 %).

> **⛔ Benchmarking caveat: the first request after boot is garbage.** In the same run the very
> first completion measured **39.6 tok/s** and the next three measured 173–183 — **about 4.5×
> too low.** Anything that benchmarks immediately after a boot, or any monitoring that samples
> early, will record that number and it means nothing. Warm up first, and discard the first
> completion.

One non-obvious interaction: **declaring a very large `--context-length` itself costs pool.**
`[measured]` The same fp8 build and card gives **542,912** tokens at a 262k declaration but
**519,040** at 540k — about **−4.4 %** for the larger declared window. Raising
`--context-length` past the checkpoint's trained window additionally requires
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`. So **the window you advertise and the pool you
get trade against each other, and the maximum pool is only available at a modest declared
context.** Do not declare a window you do not intend to serve.

### Deployment gotcha: a router that pins the context window

If a router or gateway fronts the backend and advertises its own `max_model_len`, **raise it
when the KV pool grows.** The reference deployment's router kept advertising 241,728 after
the pool became 331,456, so every client silently ran about 20,400 tokens short of the real
262,144 window until it was corrected. The backend does not complain, and neither does the
client — requests simply get truncated to a window nobody asked for.

**A speculative token map is supported and helps materially, but no map ships here.** The
maps used in the measurements were built from private serving traffic. `--speculative-token-map`
takes a file listing the hot token ids to restrict the draft head's output vocabulary to;
build one by tokenizing a corpus representative of *your* traffic and taking the most
frequent ids (a 24k-id map was used here against a 248,320-token vocabulary). Without a map,
this tree is still faster than stock, but the single-stream figures above are not reproducible
— see BENCHMARKS.md, which states which rows depend on a map.

**No model weights, checkpoints, or trained draft heads are in this repository.**

---

## Environment variables and flags

Everything is off unless listed otherwise. Details and measured effects per lever in
[CHANGES-vs-upstream.md](CHANGES-vs-upstream.md).

### Performance levers

| Variable / flag | Default | Effect |
|---|---|---|
| `SGLANG_QWENOPT_FUSE_SBMOE=1` | off | 4-launch expert-centric small-batch NVFP4 MoE for `T <= 16`. **Bitwise-identical** output. Largest single lever. |
| `SGLANG_QWENOPT_FUSE_SBMOE_MAX_T` | 16 | Token count above which SBMOE falls back to the stock path. |
| `SGLANG_QWENOPT_FUSE_HC=1` | off | Fused hyper-connection chain: 9 → 4 launches per layer. Within **1 bf16 ulp**. |
| `SGLANG_QWENOPT_FUSE_HC_SHARED=0` | on with `FUSE_HC` | Disables the deferred shared-expert add, removing one of the two 1-ulp sources. |
| `--bf16-gemm-backend sm120gemv` | `auto` | Triton skinny BF16 GEMM for `M <= 32` on SM12x, cuBLAS otherwise. **+3.2 % single-stream, lossless.** |
| `SGLANG_HC_MIX_PREFETCH=1` | off | Prefetch the hc_mix up-weight before the phase barrier. Measured **neutral** on its own. |
| `SGLANG_SM120_MXFP8_SKINNY=1` | off | Native block-scaled-MMA skinny MXFP8 GEMM on SM120. Only relevant to MXFP8 checkpoints. |
| `SGLANG_QWENOPT_FUSE_MOE_FINALIZE_ILP=1` | off | Loads a locally built flashinfer `fused_moe_120` with an ILP finalize (bitwise). Requires `SGLANG_QWENOPT_FI_MOE_SO`. Redundant when SBMOE is on. |
| `SGLANG_QWENOPT_FI_MOE_SO` | unset | Path to that `.so`. No default; the ILP lever raises without it. |
| `SGLANG_QWENOPT_FUSE_MOE_FINALIZE=1` | off | flashinfer's own fused finalize. **LOSSY and non-deterministic** (bf16 atomic adds). Not in the lossless stack. |
| `SGLANG_QSA_SHARE_SCRATCH` | **on** | One shared 128 MiB trtllm workspace + packed-KV scratch across QSA backends, instead of one per adaptive tier. Auto-off under two-batch overlap / PDMux. |

### Measured not to pay — present, documented, off

| Variable | Why it is off |
|---|---|
| `SGLANG_QWENOPT_DYNK`, `..._FILE`, `..._CONTROL`, `..._STATS_FILE`, `..._CHECK` | Dynamic expert count. The router is flat (top-10 hold ~16 % of the 512-way mass), so the single-stream ceiling is ~5 % and costs ~20 % per-layer error. See CHANGES §Dynamic-k. |

### Instrumentation — inert unless set

| Variable | Effect |
|---|---|
| `SGLANG_MTP_HIDDEN_DUMP_DIR` | Enables the scheduler-side MTP hidden-state dump. Everything else in that feature is inert without it. |
| `SGLANG_MTP_HIDDEN_DUMP_GATE`, `..._MAX_TOKENS`, `..._QUEUE`, `..._SHARD_GIB` | Dump window, budget, queue depth and shard size. |
| `SGLANG_QSA_PENDING_RING_SLOTS` | Pins the pending index-key ring to a fixed row count. `0` derives it correctly. **Set it only to reproduce an older build's (aliasing) addressing in an A/B.** |

### Upstream variables this fork's results depend on

Not ours, listed because a result above needs them.

| Variable / flag | Why it appears here |
|---|---|
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` | Required to boot with `--context-length` above the window SGLang derives from the checkpoint. Needed for the native-extrapolation result and for any YaRN configuration. |
| `--kv-cache-dtype fp8_e4m3` | Opt-in lossy. **1.94× the KV pool** (279,680 → 542,912), and required to reach a pool that admits ~400k-token prompts. |
| `--max-mamba-cache-size` **and** `--max-running-requests` | These two must be set **together**: 3 mamba slots per request means the cache size divided by 3 is the real concurrency cap, whatever `--max-running-requests` says. See [Sizing for multiple clients and subagents](#sizing-for-multiple-clients-and-subagents). |
| `--context-length` | Declaring a larger window **costs pool** (measured: 542,912 tokens at 262k vs 519,040 at 540k), so do not advertise a window you do not intend to serve. |

### Build helpers (optional levers only)

| Variable | Used by |
|---|---|
| `QWENOPT_FI_BUILD_DIR`, `QWENOPT_FI_CACHED_OPS`, `QWENOPT_VENV`, `QWENOPT_FI_CUTLASS_BACKEND` | `qwenopt_fi_finalize/mk_build.py`, for the optional ILP finalize build. |
| `QWENOPT_QAD_CKPT`, `QWENOPT_QAD_FP8_OVERLAY` | `test/manual/qwen4_exp_ple_nvfp4_gather_gpu.py`. |
| `QWENOPT_ADAPTIVE_CONFIG_DIR` | Overrides the in-tree adaptive-speculation tier configs. |

---

## Hardware and software baseline

Everything measured on one card:

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| Compute capability | **12.0 (SM120)**, 188 SMs, 300 W |
| VRAM | 97,887 MiB |
| Measured bandwidth | 1,271.9 GB/s read-only reduction · 1,203.1 GB/s copy · 985.8 GB/s 10-of-512 row gather |
| Interconnect | PCIe gen 4 ×16, **no NVLink** |
| Driver | 610.43.02 |
| CUDA toolkit | 13.3 (`nvcc` V13.3.73) |
| Python | 3.12.3 |
| torch | 2.13.0+cu130 (CUDA 13.0 build) |
| triton | 3.7.1 |
| flashinfer-python | **0.6.18** |
| sglang-kernel | **0.4.7** |
| transformers | 5.12.1 |
| cuDNN / CUTLASS DSL | 9.20.0.48 / 4.6.2 |

The **stock comparison baseline** ran a different, older venv: SGLang PR #36497
@`73a255206f`, **flashinfer 0.6.17**, **sglang-kernel 0.4.6.post1**, same driver, torch and
triton. Both venvs are named where it matters in BENCHMARKS.md, because the stock-vs-this
comparison is a cross-engine comparison, not a single-variable A/B.

### The model

Qwen3.8-Flash-Next, NVFP4-quantized: 48 layers (**36 gated-delta-net** + **12 full-attention**
with Qwen sparse attention), **512 routed experts with 10 active**, per-layer-embedding
(PLE) n-gram table host-resident in pinned memory, MTP speculative draft head
(EAGLE, 3 steps / topk 1 / 4 draft tokens). Total parameter count is **176 B**; the
active-parameter count is not independently verified here. Only the routed experts are
NVFP4 — **86.6 % of decode weight traffic is BF16 the quantization never touched**, which
is why the wins in this tree are kernel-launch and capacity wins rather than bandwidth wins,
and why single-stream throughput on this engine is hard-capped well below 2× (see
CHANGES §7).

---

## What is *not* here

- **No model weights, checkpoints, or quantized tables.**
- **No speculative token map.** Maps are traffic-derived. Build your own (above).
- **No trained MTP draft head.** The retrained head that was evaluated is traffic-derived
  and is not published. It also **regressed in-engine** and was not deployed — see
  CHANGES §Adaptive speculation and draft head.
- **No benchmark corpora and no run data.** The quality-gate harness **is** included
  ([`tools/qgate/`](tools/qgate/README.md)), with a script that downloads the public datasets from
  their publishers. It does not include the baseline and candidate result files behind the
  published verdicts, or the private code corpus used for some NLL windows. That corpus is
  optional; the harness falls back to public-domain text without it.

## License

Upstream SGLang is **Apache-2.0**, and `LICENSE` in this repository is upstream's, unmodified.
The files added by this fork carry `SPDX-License-Identifier: Apache-2.0` alongside their
copyright line and are offered under the same terms.

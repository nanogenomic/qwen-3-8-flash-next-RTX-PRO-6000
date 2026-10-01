# CONTEXT-PROFILES — how to run high context, and what it costs in speed

A decision guide for serving Qwen3.8-Flash-Next on **one 96 GB RTX PRO 6000** at long context,
written as copyable flag profiles. Pick a profile from the first table, copy its launch block,
then read the speed section before you decide whether it is "slower".

Every figure is labelled the way the rest of this repository labels them:

- `[measured]` — read from an engine log, `/get_server_info`, `/metrics`, `nvidia-smi` or a
  benchmark artifact, with n and window given.
- `[estimated]` — derived from measured inputs by a stated method (for example, tokens from
  characters ÷ 4). The method is given; the result is not a measurement.
- `[arithmetic]` — computed from configuration or architecture constants, not observed.

The evidence behind this page lives in [BENCHMARKS §3.21](BENCHMARKS.md#321-the-mamba-checkpoint-path-cap--the-flag-that-makes-multi-lane-high-context-work)
(the path cap) and [§3.22](BENCHMARKS.md#322-what-high-context-costs-in-speed--per-step-versus-delivered)
(speed). This page does not repeat what is already established elsewhere; it links to it.

---

## 1. Which one do I need?

| What you need to serve | Profile | Per-step cost vs the fastest config | Pool |
|---|---|---|---|
| One agent up to ~400K, **or** two up to ~200K each, and nothing else long | **A — on-GPU KV** | baseline | **451,264** tokens `[measured]` |
| Two or more long agents (250K–400K each) plus subagents, on one card | **B — host-RAM KV** | **+2 %** at <50K context, **+11–15 %** at 50K–450K, single stream `[measured, depth-matched]` | **1,310,720** tokens `[measured]` |
| One request near 1M tokens | not a profile here — see [CHANGES §3.4](CHANGES-vs-upstream.md#34-1m-context-on-one-card--measured-and-not-the-default) | — | lossy YaRN lane, quality gate INVALID |

**Whichever you pick, add `--mamba-max-states-per-path 2` if more than one long conversation
shares the card.** It is an upstream SGLang flag, it costs no VRAM, and without it two long
agents on one card evict each other's prefix-cache checkpoints and re-prefill their whole
history most turns. That is section 2, and on the reference deployment it mattered more than the
choice between A and B.

---

## 2. The flag every multi-lane profile needs: `--mamba-max-states-per-path 2`

**This is the most transferable finding in the repository.** It is not specific to this fork,
nor to this model: it follows from how SGLang caches prefixes for **any hybrid model whose
linear-attention or SSM layers carry a recurrent state** and are served through SGLang's mamba
radix cache. It was measured only on this model; the mechanism is general.

**Mechanism.** Qwen3.8-Flash-Next has 36 gated-delta-net (linear-attention) layers and 12
full-attention layers. A cached prefix is reusable only up to the deepest radix-tree node that
**also holds a saved recurrent-state checkpoint**; having the KV is not enough. Under
`--mamba-radix-cache-strategy extra_buffer_lazy`, **every 4,096-token prefill chunk donates one
checkpoint** to the tree. Checkpoints live in the `--max-mamba-cache-size` pool on their own LRU
list — `--radix-eviction-policy` governs only the full-attention KV.

So a single cold 350K prefill inserts ~86 checkpoints into a 36-slot pool `[arithmetic:
350,000 ÷ 4,096]`. Every other conversation's turn-boundary checkpoint is pushed out. Their
KV is still resident, but it is unusable without the checkpoint, so their next turn reports
`cached=0` and re-prefills everything, and *that* prefill evicts the first conversation's
checkpoint. Two long agents settle into a re-prefill ping-pong. Any cold prefill longer than
~147K tokens (36 slots × 4,096) can flush every idle checkpoint in the pool `[arithmetic]`.

`--mamba-max-states-per-path N` keeps only the `N` deepest checkpoints on each root-to-tail
path and frees the shallower interior ones; tail, fork and locked nodes are always kept.

**Evidence** `[measured]`:

| | production flags (cap unlimited) | `--mamba-max-states-per-path 2` |
|---|---|---|
| Two mains at 300K and 330K, +1.5K tokens per turn, a cold 45K subagent every round (Modal, production replica) | **0 / 10** turns hit the cache; **31.4–37.3 s** per main turn | **10 / 10** hit; **0.70–0.85 s** per main turn |
| Same, plus 3 warm subagent lanes × 4 turns per round | — | mains **12 / 12**, subagents **72 / 72** |
| Same, with `--max-mamba-cache-size 64` | — | 12 / 12 and 72 / 72 — **no additional gain** |
| Production card, before → after the flag (full eras, see §5) | cold deep-prefill arrivals **131 / h** | **15 / h** |

**Do not "fix" this by lowering `--max-mamba-cache-size` to reclaim VRAM.** The pool looks
over-provisioned because the logged `mamba usage` counts only states held by running requests,
not checkpoints held by the cache. On the production card, before the flag, `/metrics` showed
**≤ 3 of 36 slots free in 99.3 % of 3,368 one-second samples**, median 28 held by the cache
`[measured]`. After the flag the pool is *still* full most of the time (≤ 3 free in 69.9 % of
1,255 samples) — the cap does not empty the pool; it changes which checkpoints are in it.

**Do not use cap 1.** An agent turn reuses the checkpoint at the *previous* prompt's end, which
becomes an interior node as soon as that turn finishes. Cap 2 keeps the tail plus that one.

Where this sits next to the earlier slot-count finding: [§3.10](BENCHMARKS.md#310-multi-turn-agents-state-slots-decide-whether-the-prefix-cache-works-second-card)
showed that 60–80K conversations need enough slots (`--max-mamba-cache-size 36` at 4 running
requests). That is still true. The path cap fixes a different case — a deep cold prefill
flushing the pool — and the two compose. Full evidence:
[BENCHMARKS §3.21](BENCHMARKS.md#321-the-mamba-checkpoint-path-cap--the-flag-that-makes-multi-lane-high-context-work).

---

## 3. Profile A — on-GPU KV: maximum per-step speed, bounded pool

Use when one long lane (or two medium ones) is the whole workload.

Both profiles share one base command. It is README "Run it" with the
[recommended configuration](README.md#recommended-configuration) applied and the path cap added:

```bash
export SGLANG_QWENOPT_FUSE_SBMOE=1 SGLANG_QWENOPT_FUSE_HC=1 SGLANG_HC_MIX_PREFETCH=1
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 TRTLLM_ENABLE_PDL=1
export CUDA_HOME=/usr/local/cuda FLASHINFER_CUDA_ARCH_LIST=12.0 MAX_JOBS=8
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1      # needed for --context-length above 262,144

COMMON=(
  --model-path "<your Qwen3.8-Flash-Next NVFP4 checkpoint>"
  --quantization modelopt_fp4 --trust-remote-code
  --context-length 540000
  --kv-cache-dtype fp8_e4m3                              # opt-in lossy; see README
  --attention-backend flashinfer --sampling-backend flashinfer
  --moe-runner-backend flashinfer_cutlass --bf16-gemm-backend sm120gemv
  --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer_lazy
  --max-mamba-cache-size 36                              # do not lower it -- section 2
  --mamba-max-states-per-path 2                          # section 2
  --page-size 64
  --max-running-requests 4 --cuda-graph-max-bs-decode 4
  --cuda-graph-backend-prefill=disabled
  --chunked-prefill-size 4096 --max-prefill-tokens 4096
  --speculative-algorithm NEXTN --speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  --ple-offload-embedding --enable-metrics --enable-cache-report
  --max-queued-requests 64
  --crash-dump-folder /path/to/crashdumps
)
```

Profile A:

```bash
python -m sglang.launch_server "${COMMON[@]}" --mem-fraction-static 0.97
```

| | Value |
|---|---|
| Pool at `--mem-fraction-static 0.97` | **451,264 tokens** `[measured, production card, /get_server_info]` |
| Pool at 0.98 | 522,880 `[measured]` — **and the lane died of a runtime CUDA OOM** with 287 MiB of device memory free while serving ([§3.14](BENCHMARKS.md#314-runtime-cuda-oom-under-a-deep-queue-2026-09-30--why-098-is-not-usable)). Do not use it. |
| What bounds it | After weights and the MTP head the card has **12.03 GiB** left, and **3.10 GiB** of that is the 36-slot recurrent-state pool, which is fixed `[measured, boot log]`. The rest is KV at ~14 KB/token (12,288 target K/V + 1,024 draft K/V + 768 compressed index, fp8 `[arithmetic]`) plus graph capture and activations |
| Practical use | **one lane up to ~400K, or two lanes up to ~200K each.** Above that, a subagent's admission evicts a main's prefix ([§3.15b](BENCHMARKS.md#315b-two-long-context-mains-on-one-card--the-result-the-pool-was-for) reproduces it) |
| Per-step decode, warm | **13.7 / 17.6 / 20.6 ms** at 1 / 2 / 3 streams `[measured, §5]` |

⚠️ An earlier internal estimate put this profile's ceiling at "~485K tokens". That arithmetic
treated SGLang's logged "GB" as 10⁹ bytes; SGLang logs GiB. Redone in GiB the same deductions
give ~520K, which is what 0.98 measured — and 0.98 is not survivable. **Quote the measured
451,264.**

---

## 4. Profile B — host-RAM KV: a large pool, for a per-step tax

Use when two or more long agents must share the card. The 12 full-attention layers' K/V (and
the MTP draft layer's) move to pinned host RAM; only the compressed index and a hot cache stay
on the GPU ([§3.15](BENCHMARKS.md#315-host-resident-qsa-kv-pool--1310720-tokens-on-one-96-gb-card)).

```bash
# the exports and COMMON array from Profile A, then:
export SGLANG_QSA_HOST_KV=all
export SGLANG_QSA_HOST_KV_MAX_GB=24          # fail fast at startup instead of meeting the OOM killer
export SGLANG_QSA_HOST_KV_CACHE_SETS=4096    # what production runs; see the sizing table below
export SGLANG_QSA_HOST_KV_STATS=0            # keep 0 on this tree -- see below
export SGLANG_OOM_MAX_REQ_RETRACTIONS=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m sglang.launch_server "${COMMON[@]}" \
  --max-total-tokens 1310720 \
  --mem-fraction-static 0.97
#   --max-total-tokens is MANDATORY with host KV; --mem-fraction-static has no effect here
```

The flags and environment that matter here are the reference deployment's live configuration,
read back from the running process rather than from a unit file.

| Setting | Why |
|---|---|
| `--max-total-tokens 1310720` | **Mandatory.** Without it the pool sizes itself off the static-memory budget and demands **~89 GiB** of pinned host RAM `[arithmetic, §3.15a]`. |
| Host RAM cost | **16.25 GiB pinned** at fp8 for 1,310,720 tokens (15.00 target + 1.25 draft) `[measured, boot log]`. Pinned pages are neither swappable nor reclaimable — budget them like VRAM ([§3.18a](BENCHMARKS.md#318a-host-ram-is-a-boot-time-budget--and-a-correction-to-this-section)). |
| `SGLANG_QSA_HOST_KV_MAX_GB=24` | Caps the host allocation at startup with a named error; 24 GiB admits up to 2,097,152 fp8 tokens. |
| `--mem-fraction-static` | **No effect in this mode** `[measured, Modal]`: 0.94 and 0.97 booted identically — same pool, same 8.88 GiB free at pool end, same 5,948 MiB free after boot. With `--max-total-tokens` set, the pool is fixed by the flag and the fraction does not change what is left. |
| `SGLANG_QSA_HOST_KV_STATS=0` | ⛔ **On this tree, `=1` can kill the boot.** The stats reporter was a background thread doing a device-to-host copy plus `synchronize()` every 30 s; if a tick lands inside CUDA-graph capture, PyTorch's default `capture_error_mode="global"` invalidates the capture. That is the cause of the one boot failure in [§3.19](BENCHMARKS.md#319-nondeterministic-cuda-graph-capture-failure-on-the-host-kv-build). A fix exists on a development branch and is not yet in this patch series; until it is, leave stats off. |
| `SGLANG_QSA_HOST_KV=all`, not `target` | `target` keeps the draft layer's K/V on the GPU: same hit rate, same step time, ~0.94 GiB less free VRAM `[measured, Modal]`. Not worth it. |
| A restart policy | Still recommended (`Restart=always`, `RestartSec=30`). It is what made the one capture failure cost ~6 minutes instead of a dark card. |
| Per-step decode, warm | **15.6 / 19.9 / 24.5 ms** at 1 / 2 / 3 streams `[measured, §5]` |

### Hot-cache size: 16,384 sets is the better trade if you have the VRAM

`[measured, Modal RTX PRO 6000 Server Edition, PCIe Gen5, production replica]` One 250K stream
× 3 repetitions, then a 4-stream mixed cell (two mains at 250K and 200K plus two subagents).
Hit rate is the 12 target layers' hit fraction of cacheable selections in the 250K cell; step
time is `1000 × accept_len × running_req / gen_throughput` from the engine's own decode lines.

| Hot cache | GPU cost (target + draft) | Target-layer hit | Step, 1 stream @250K | Step, 4 streams | Free VRAM after boot |
|---|---|---|---|---|---|
| **4,096 sets** (production) | 1.02 GiB | **0.776** | 16.51 ms (n = 17) | 25.06 ms (n = 26) | 9,072 MiB |
| 2,048 sets × 4-way | 2.04 GiB | 0.839 | 16.20 ms (n = 18) | 24.00 ms (n = 29) | 8,028 MiB |
| **16,384 sets** | **4.06 GiB** | **0.908** | **15.49 ms** (n = 15) | **23.31 ms** (n = 27) | 5,948 MiB |
| 4,096 sets × 4-way (same bytes as 16,384) | 4.06 GiB | 0.919 | 15.49 ms (n = 15) | 23.36 ms (n = 28) | 5,948 MiB |

- **16,384 sets: hit 0.776 → 0.908, step −6.2 % (1 stream) and −7.0 % (4 streams), for +3.05 GiB
  of VRAM.** On this tree that is `SGLANG_QSA_HOST_KV_CACHE_SETS=16384` — the same direct-mapped
  layout, 12 layers × 5 slots × sets × 4 tokens × 1,024 B. The cell itself ran on a development
  tree; the 16,384-set geometry on *this* tree was not separately booted.
- **Associativity buys ~1 point of hit at equal bytes and nothing measurable in step time.** It
  is not a substitute for capacity: 4-way at half the bytes reaches 0.839, not 0.91. Set-
  associative lookup is not in this patch series.
- These cells ran on a **Gen5 Server Edition at 600 W**; on the Gen4 Max-Q the miss cost is
  higher, so the saving there is likely larger. That is reasoning, not a measurement.
- Production still runs 4,096 sets. The step-time saving is a Modal measurement; it has not
  been re-measured on the production card.

### Check your PCIe link before you budget the tax

The per-step tax in this mode is the cost of hot-cache misses read over PCIe. **The reference
card runs at Gen4 because of the host, not the GPU** `[measured]`:

```bash
nvidia-smi --query-gpu=name,pcie.link.gen.max,pcie.link.gen.gpumax,pcie.link.gen.hostmax,pcie.link.width.max --format=csv
# reference deployment:  pcie.link.gen.max=4, gpumax=5, hostmax=4, width 16
```

| Platform | Zero-copy sequential read, host → GPU |
|---|---|
| Reference card, Max-Q Workstation, host-limited to Gen4 ×16 | **20.27 GB/s** `[measured, 2026-09-29]` |
| Modal Server Edition, Gen5 ×16 | **42.52** and **45.40 GB/s**, two runs `[measured, different SKU]` |

A Gen5 host should therefore **roughly halve the host-KV per-step tax** `[estimated]`. That
estimate assumes the tax is miss traffic, which the depth-matched data supports (+2 % below
50K, where the hot cache covers almost everything) but does not prove, and no same-card
Gen4/Gen5 A/B exists. [§3.20](BENCHMARKS.md#320-throughput-versus-concurrency-warm-segmented--and-the-071-retraction)'s
statement that no Gen4/Gen5 *decode factor* has been measured still stands.

---

## 5. Speed: what high context costs, and which number to look at

Measure per-step cost as **step time**, not tok/s:

```
step_ms = 1000 × accept_len × running_req / gen_throughput      (all three from one "Decode batch" line)
```

`accept_len` (speculative tokens accepted per step) depends on what is being generated; on the
production lane its 10th–90th percentile spans **~1.9–3.2** within one era `[measured]`. A tok/s
comparison silently absorbs that, and several tok/s comparisons made while writing this page
had to be withdrawn because of it. Step time divides it out.

### 5.1 Per step: host KV is 13–19 % slower than on-GPU KV at the eras' own depths

`[measured, production card, engine log frozen at 2026-10-01 03:07:37 UTC]` Warm filter: drop
any decode line within 10 log lines of a prefill batch with ≥ 50,000 pending tokens. Each decode
line is a 40-step average. Depth is the median of `#full token ÷ #running-req`.

| Streams | On-GPU KV (6.4 h) | Host KV, before the path cap (2.6 h) | **Host KV + path cap (0.8 h)** | Δ, cap era vs on-GPU |
|---|---|---|---|---|
| 1 | 13.73 ms · n = 1,406 · depth 68K | 15.66 · n = 711 · 247K | **15.59** · n = 1,440 · 91K | **+14 %** |
| 2 | 17.57 · n = 1,078 · 89K | 20.34 · n = 599 · 185K | **19.87** · n = 516 · 200K | **+13 %** |
| 3 | 20.56 · n = 460 · 98K | 24.41 · n = 247 · 200K | **24.50** · n = 143 · 162K | **+19 %** |

⛔ **This table is not an A/B.** The eras differ in context depth (see the depth column), in
client mix, and in build: the on-GPU era ran this series through patch 0032 (development
`be24acf0a5`); both host-KV eras ran through 0037 (`0a6519021b`), a fast-forward that adds host
KV (0033–0034), the bounded prefill gather (0035) and non-fatal OOM (0036–0037). The path cap
changes nothing per step (compare the two host-KV columns), as expected — it is a cache policy.

Depth-matched, which removes the depth confound but not the others `[measured, same log]`:

| Per-stream depth | 1 stream: on-GPU → host KV | 2 streams: on-GPU → host KV |
|---|---|---|
| 0–50K | 13.61 (n = 282) → 13.89 (n = 460) · **+2 %** | 17.21 (234) → 18.75 (232) · +9 % |
| 50–100K | 13.70 (719) → 15.27 (688) · **+11 %** | — |
| 100–200K | — | 17.68 (407) → 20.18 (421) · **+14 %** |
| 200–300K | 14.16 (210) → 16.30 (65) · **+15 %** | 18.12 (106) → 20.52 (370) · **+13 %** |
| 350–450K | 14.49 (185) → 16.64 (683) · **+15 %** | — |

Within each mode, the cost of depth itself, single stream: **on-GPU +6.5 %** from 0–50K to
350–450K; **host KV +20 %** over the same span `[measured]`. Host KV makes depth more expensive,
because a deeper context selects more blocks the hot cache does not hold.

### 5.2 Delivered: the system does ~60 % more work per minute anyway

Per step is not what an agent operator feels. With the path cap the lane stops spending most of
its time re-prefilling, and that outweighs a 13–19 % slower step by a wide margin.

| Era (same windows as 5.1) | Main agents' turns / min | Main agents' output, tok / min | Engine-wide generated tok / min | New prefill tok / min | Est. share of wall time prefilling | Cold deep-prefill arrivals / h | Deep prefix hits / min |
|---|---|---|---|---|---|---|---|
| On-GPU KV | 2.21 | ~1,080 | ~3,730 | 426K | ~75 % | 158 | 0.46 |
| Host KV, no cap | 2.80 | ~1,250 | ~3,690 | 347K | ~65 % | 131 | 2.18 |
| **Host KV + path cap** | **4.06** | **~2,100** | **~5,950** | **70K** | **~13 %** | **15** | **4.33** |

How each column was produced:

- **Turns and main-agent output** `[measured turns / estimated tokens]`: the two long-running
  main agents' assistant messages in the agent framework's own message store. Tokens are
  characters ÷ 4 of content + reasoning + tool calls, **counting reasoning once** (the store
  keeps it in two columns). Output volume depends on what the agents were doing.
- **Engine-wide generated tokens** `[estimated]`: Σ over decode log lines of `40 × accept_len ×
  running_req` — every client on the card, including subagents and scheduled jobs.
- **New prefill tokens** `[measured]`: Σ `#new-token` over prefill batch lines.
- **Share of wall time prefilling** `[estimated]`: new prefill tokens ÷ the measured
  chunk-saturated prefill rate (9,477 tok/s on-GPU, 8,958 host KV —
  [§3.20](BENCHMARKS.md#prefill-rate-for-anyone-sizing-a-backlog-gate)).
- **Cold deep-prefill arrivals** `[measured]`: prefill batches with no cached prefix that raise
  the pending queue to ≥ 150,000 tokens. **Deep prefix hits**: prefill batches with ≥ 100,000
  cached tokens.

**The point to take away: per step, host KV is 13–19 % slower; delivered output is ~60 % higher
engine-wide and ~70–95 % higher for the main agents, because prefill time fell from roughly
two-thirds of the wall clock to about an eighth.** "Is it slower?" depends on which number you
mean, so measure both on your own workload.

Limits, stated: the path-cap era is **49 minutes**, and its rates move a lot between 10-minute
windows (new-prefill tokens per minute ranged 9K–173K). It overlaps the hours before it in
workload but is not a controlled comparison. The direction is corroborated three independent
ways — prefill tokens, cold arrivals and deep hits — and by the Modal A/B in §2, which is
controlled. The magnitude is provisional.

---

## 6. Measure it on your own lane

| Question | Where to read it |
|---|---|
| Per-step cost | `step_ms` above, from `Decode batch` lines; bucket by `#running-req`; drop lines near a large prefill |
| Are lanes losing their prefixes? | `#cached-token` on `Prefill batch` lines; a long agent's turn showing `0` after it previously hit is the symptom |
| Is the checkpoint pool thrashing? | `/metrics`: `sglang:mamba_available_tokens`, `sglang:mamba_evictable_tokens`, `sglang:mamba_used_tokens` (they count states, despite the name). **Not** the logged `mamba usage`, which excludes cached checkpoints |
| How much time goes to prefill | Σ `#new-token` per minute ÷ your prefill rate; or `/v1/loads` `total_prefill_uncached_tokens ÷ total_prefill_busy_us` ([§3.20](BENCHMARKS.md#prefill-rate-for-anyone-sizing-a-backlog-gate)) |
| Host-KV hot-cache hit | Read it at steady state, never the first sample ([§3.15](BENCHMARKS.md#315-host-resident-qsa-kv-pool--1310720-tokens-on-one-96-gb-card)); and see the `STATS` warning above |
| Device memory in host-KV mode | `nvidia-smi`, **not** `/v1/loads` `kv_cache_gb`, which reports the host pool ([§3.20](BENCHMARKS.md#-operational-trap-v1loads-reports-the-host-pool-as-memorykv_cache_gb)) |

## 7. Corrections made while writing this page

Figures that circulated in internal drafts and were wrong. None of them was published here
before; they are listed so they are not reintroduced.

| Draft figure | What the source says |
|---|---|
| Main-agent output 1,671 → 1,825 → 2,968 tok/min (+78 %) | Reasoning text was counted twice. Counted once, over the full eras: **~1,080 → ~1,250 → ~2,100** (+68 % against the no-cap era, +94 % against on-GPU). The direction holds. |
| "Prefill share of engine batches 89 % → 45–52 %" | That was a ratio of *log lines*, and decode is logged once per 40 steps, so it was not a share of batches. Replaced by new-prefill tokens and an estimated share of wall time (**~65 % → ~13 %**). |
| Profile A ceiling "~485K" | GiB/GB mix-up; see §3. Use the measured 451,264. |
| "Host-KV lanes run 370–410K, on-GPU ~200K" as the depth confound | The single-stream host-KV samples have a median depth of 91K. Replaced by the depth-matched table. |
| Depth term "+12 % from 0–50K to 300–350K" within host KV | Measured: **+18.6 %** to 300–350K (16.48 ms, n = 253) and +20 % to 350–450K. |
| Deep prefix hits per minute "1.0 → 4.1–5.3" | Over the full eras: **2.18 → 4.33**. |
| "≤ 3 of 36 slots free in 91 % of 4,623 samples" | Those samples straddle the restart. Before the cap: **99.3 % of 3,368**; after: 69.9 % of 1,255. |

Open questions that remain open are in [BENCHMARKS §6](BENCHMARKS.md#6-open-measurements).

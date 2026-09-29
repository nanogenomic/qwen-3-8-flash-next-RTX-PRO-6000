# BENCHMARKS — methodology, results, and what the numbers cannot tell you

Everything here was measured on one physical machine with two RTX PRO 6000 Blackwell Max-Q
cards. The **stock baseline** ran on one card as a live serving lane; the **candidate** ran on
the other. Read §1 before reading any table.

---

## 1. Read this first: the noise band, and what is and is not an A/B

### 1.1 Measurement noise

`[measured]` **Repeat cells of the identical configuration on the identical harness gave
149.81 and 140.42 tok/s — a 6.3 % spread.** On a single sweep, single-stream differences below
about 6 % are not resolvable. Several 4-concurrent figures are single runs (n=1) and are
annotated where that matters.

Other measured noise floors, all from same-configuration repeats:

| Quantity | Noise floor |
|---|---|
| Greedy determinism (same config, 16 prompts × 2 reps) | **0 / 16 identical.** Median first divergent token 13, minimum 4. Exact-match "lossless" tests are meaningless on this stack. |
| Shared-prefix mean \|Δlogprob\|, cross-run, same config | 0.027 |
| Decode-path identical output pairs, baseline vs itself | **4–7 of 96** |
| Single-chunk prefill (ctx 512) logprobs, same config | **bit-deterministic** — max token \|Δ\| **0.0** |
| Multi-chunk prefill NLL, same config, per-window \|dNLL\| | 4k **0.043** · 16k **0.045** · 64k **0.053** (SD 0.062–0.069), max single-token \|Δ\| up to 4.62 |
| GSM8K, same config A vs B (n=150) | 1 discordant pair; +0.7 points |
| HumanEval, same config A vs B (n=80) | 6 discordant; +2.5 points |
| IFEval, same config A vs B (n=120) | 11 discordant; +2.5 points |
| Degenerate-output rate, baseline | **2 / 492 (0.41 %)** under greedy decoding |
| Accept-length, boot to boot, identical config | 0.03 |

**The 6.3 % single-stream spread and the 0/16 greedy determinism are the two facts that
constrain every claim in this document.**

### 1.2 The headline comparison is cross-engine, not a single-variable A/B

The `138.8 → 166.4` figure compares:

- **Stock:** SGLang PR #36497 @`73a255206f`, venv with **flashinfer 0.6.17** and
  **sglang-kernel 0.4.6.post1**, `--mem-fraction-static 0.958`, `--max-mamba-cache-size 16`,
  no token map, BF16 KV, on the card that was also serving live traffic. Its figures are
  **mined from that lane's own scheduler log** — 738 decode and 471 prefill samples from one
  boot segment, p50 per `#running-req` — not from a dedicated sweep. Its accept length of
  2.52 is therefore measured on its own traffic mix and **is not comparable** row-to-row with
  this tree's accept lengths, which come from a held-out eval split.
- **This tree:** base `6fa3fe69e2` + these 32 commits, venv with **flashinfer 0.6.18** and
  **sglang-kernel 0.4.7**, `--mem-fraction-static 0.98`, `--max-mamba-cache-size 12` with
  `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`, graph-bs 4, **a 24k hot-token speculative map**, BF16
  KV, both fuse levers on, `--bf16-gemm-backend sm120gemv`, `SGLANG_HC_MIX_PREFETCH=1`, on a
  dedicated card.

**Many variables move at once**, including the engine version, two library versions, the
scheduler profile, and a speculative token map that is **not shipped in this repository**.
Treat `+19.9 %` as *"what this configuration achieved against what was previously in
production"*, which is the question it was measured to answer — not as the effect of the
kernels alone.

For the kernel effect on its own, see the clean same-engine A/B in §3.

### 1.3 "GPU2 proxy" figures

Several per-kernel microsecond figures are labelled `[GPU2-proxy]`. Those were measured on a
**second, smaller SM120 card — an RTX PRO 4000, 70 SMs, ~546 GB/s** — with tensor `N`
dimensions scaled to the RTX PRO 6000's per-SM work, because the 188-SM card was not
available for microbenchmarks. **A GPU2-proxy microsecond number is a relative signal about a
kernel, never a claim about throughput on the RTX PRO 6000.** Where a GPU2-proxy figure was
used to price an end-to-end effect, the result is labelled `[arithmetic]` and given as a
range.

---

## 2. Harness

| | |
|---|---|
| **Temperature** | **0** for every throughput and equivalence measurement. The degeneracy suite deliberately runs production sampling: T=1.0, top_k 20, top_p 0.95. |
| **Throughput harness** | A fixed code/math prompt through the chat endpoint, `max_tokens >= 512`, warmed up before measuring. Prose through `/v1/completions` measures ~2.3× lower under speculative decode and was not used. |
| **Source of truth for rates** | `accept_len` and `gen_throughput` were read from the **scheduler log**, not from client wall time. |
| **First request after a boot is discarded** | `[measured]` The first completion after a boot ran at **39.6 tok/s** where the next three ran at 173–183 — **~4.5× too low**. Any sweep or monitor that samples early records a meaningless number. Warm up, then discard the first completion. |
| **Router-path spot checks are labelled as such** | A handful of figures in this document are chat completions through the serving router rather than harness cells (different prompt, path and cell definition). They are sanity checks and are **not** comparable row-to-row with the sweep. Every one is marked where it appears. |
| **Config verification** | Every instance's configuration was read back from `/get_server_info`, and for env-gated levers from `/proc/<pid>/environ`, rather than trusted from the launch line. |
| **Concurrency** | 1 and 4. **8 and 16 were dropped** because the shipping profile caps at `--max-running-requests 4`. |
| **Context sweep** | prefix lengths 4k / 16k / 48k / 96k / 192k at bs=1, fitting ms-per-verify-cycle against context. |
| **Quality gate** | A separate harness, published in [`tools/qgate/`](tools/qgate/README.md), with the suites in §4, a measured noise floor from same-config repeats, and explicit PASS / WARN / FAIL / INVALID / NOT_MEASURABLE verdicts. A suite with >5 % errors or unplanned skips is **INVALID** and is dropped from verdicts rather than reported as PASS or FAIL. |

**One difference between the measured tree and the published tree.** Publication made four
files' hardcoded paths environment-configurable (see CHANGES §9). Three of the four are an
optional build script, a manual numerics probe, and a unit test — none of them runs in the
serving path. The fourth, `moe_runner/flashinfer_cutlass.py`, changed a **default value**:
`SGLANG_QWENOPT_FI_MOE_SO` no longer has one, so the optional ILP-finalize lever now raises
instead of silently looking in a fixed directory. **That lever is off in every measurement
here, and is redundant when SBMOE is on.** No measurement was re-run after that change, and
nothing in the measured path was altered by it.

**Benchmark data is not in this repository, and neither are the run results.** The NLL and
reference-agreement suites used a private code corpus. The needle-in-a-haystack haystacks are
public-domain Gutenberg texts: War and Peace, Moby Dick, Pride and Prejudice, A Tale of Two
Cities and Great Expectations. GSM8K, IFEval, HumanEval, HumanEval+, MBPP+ and MMLU-Pro are the
standard public sets.

`tools/qgate/fetch_data.sh` downloads every public file from its publisher, and checks it
against content checksums of the exact files used here. All 11 files matched when the script
was tested. Without the private corpus, the harness scores public-domain text in its place.

---

## 3. Results

### 3.1 Single-stream and aggregate, stock vs this tree

`[measured]` Temperature 0. Stock column is the previously-in-production configuration; see
§1.2 for why this is not a single-variable comparison.

| Metric | Stock | This tree, lossless levers | This tree + fp8 KV (opt-in) |
|---|---|---|---|
| Single-stream, short context | 138.8 | **166.4** (+19.9 %) | 159.8 (+15.1 %) |
| Single-stream peak (@325 tokens) | — | 180.5 | 167.3 |
| 4-concurrent aggregate | 351 | **413.9** (+17.9 %) | 411.4 (+17.1 %) |
| KV pool, test card | 241,728 | **279,680** (+15.7 %) | **542,912** (2.25×) |
| KV pool, production card | 241,728 | **331,456** (+37 %) | — |
| Per-token time, 325 → 250k context | grows with depth | — | 14.7 → 15.4 ms (+4.5 % over 770× context) |

Single-stream by context depth is in **§3.1b**, which is the one place those figures live.

The fp8-KV column is a **2-rep mean**. The lossless column comes from a single run of the
final build with both fuse levers on.

A separate live check on the production lane after deployment measured **176.4 tok/s** on a
768-token reply through the serving router.

### 3.1b Single-stream across the full context range, and where it was not swept

`[measured]` Consolidated from the three arms that between them cover the range. No cell is
interpolated, and each column is a different configuration.

| Depth | bf16 KV (shipping) | + fp8 KV (opt-in lossy) | host-KV + YaRN f=4 (1M arm) |
|---|---|---|---|
| 325 tokens | **180.5** | 167.3 | — |
| 48.6k | **175.6** | 164.2 | — |
| ~95–98k | **155.5** (@98k) | 154.4 (@95k) | — |
| 250k | *not swept* | **145.8** | **157.0** |
| 500k | *not swept* | — | **154.4** |
| 900k | *not swept* | — | **141.0** |

tok/s, temperature 0, single stream. Per-token latency over the same span:
**14.7 → 15.4 ms from 325 tokens to 250k, +4.5 % across a 770× context increase.**

**The "not swept" cells are a measurement gap, not a capability limit.** The shipping bf16-KV
arm holds a 279,680-token pool (331,456 in production) and can serve far past 98k; its context
sweep used harness cells at 4k / 48.6k / 95.4k and stopped there. **A deep single-stream sweep
on the exact shipping configuration was never run** (§6). The right-hand columns show the
engine sustaining 141–157 tok/s to 900k, but each carries its own cost: fp8 KV is opt-in lossy,
and the 1M arm adds a host-resident KV pool plus static process-wide YaRN f=4 whose quality gate
returned INVALID.

Note that the 1M arm is *faster at 250k* than the fp8 arm (157.0 vs 145.8) — it uses bf16 KV
spilled to host RAM rather than quantized on-device KV, and its hot cache ran at a 0.89–0.95 hit
rate in steady decode. Those are different mechanisms, not a contradiction.

### 3.2 Same-engine A/B: the fusion levers on their own

`[measured]` Same tree, same boot profile, same prompts; the only difference is whether
`SGLANG_QWENOPT_FUSE_SBMOE` and `SGLANG_QWENOPT_FUSE_HC` are set. **This is the clean
measurement of what the kernels are worth.**

| Leg | Single-stream | 4-concurrent aggregate | Accept length |
|---|---|---|---|
| Levers off (same tree) | 143.61 | 401.4 | 2.1042 |
| Both levers on | **169.20** (+17.8 %) | **416.18** (+3.7 %) | 2.0426 |

By context on the same A/B: 170.6 @4k · 174.9 @48.6k · 162.2 @95.4k.

Note the accept-length drop, 2.104 → 2.043. Boot-to-boot accept spread on identical
configurations has been measured at 0.03, so a 0.06 drop is **at the edge of resolvable** and
was not attributed. Since the fused kernels are bitwise (SBMOE) and ≤1 bf16 ulp (HC) on the
decode path, a real distribution change is not the expected mechanism.

### 3.3 The full arm table

`[measured]` Every configuration that was booted and measured, so the shipped one can be seen
in context. "Gate" is the quality-gate verdict where one was run.

| Arm | Single-stream | conc-4 agg | Peak agg | Accept | KV pool | Gate |
|---|---|---|---|---|---|---|
| Stock reference | 138.8 | 351.2 | — | 2.52 (own traffic) | 241,728 | reference |
| Stock engine twin, for an A/B | **failed to boot ×2** | — | — | — | — | — |
| New engine, **no token map** | 130.3 | 327.6 | 347.2 @8 | 2.1578 | 269,376 | WARN |
| New engine + 32k token map | 147.2 | 386.9 | 388.2 @8 | 2.1125 | 269,376 | cache PASS |
| + fp8 KV | 140.4 | 387.2 | 387.2 @4 | 2.0885 | **523,008** | **PASS** (lossy) |
| mamba 64 / mrr 16 | 152.3 | 388.5 | **723.4 @12** | 2.0932 | 63,680 | not run |
| mamba 32 + skip-lock | 149.1 | 347.9 | 684.9 @10 | 2.1192 | 183,680 | not run |
| mrr 4 / mamba 12 | 150.3 | 394.3 | 394.3 @4 | 2.1165 | **277,952** | not run |
| **+ sm120gemv only** | **155.05** | **408.55** | — | 2.1096 | 277,952 | — |
| + sm120gemv + HC prefetch | 152.25 | 408.45 | — | 2.1651 | 279,680 | **PASS**, mean dNLL 0.00511 |
| + adaptive-wide speculation | 152.6 (2 cells) | 396.6 | — | **2.2109** | 92,800 | not run |
| + both fuse levers | **169.20** | **416.18** | — | 2.0426 | 277,952 | see §4.4 |
| QAD checkpoint arm | **failed at first request** | — | — | — | 371,456 | — |

**Two things in this table are worth stating explicitly.**

*The aggregate ceiling was an admission artefact, not the card.* `--max-mamba-cache-size 16`
admits only ~5 requests, because each running request costs **3 mamba state slots**
(`mamba num: 45` observed at `#running-req: 15`). Unclamping it reached **723.4 tok/s at 12
concurrent** — but with a KV pool of 63,680, i.e. no long context. Mamba slots trade against
KV at **~0.093 GB ≈ 3,750 pool tokens per slot**, so ~11,250 pool tokens per running request.
The shipped profile deliberately takes the pool.

*Adaptive speculation is the only thing that ever raised accept length* (2.2109, and a decode
line hit 4.10, breaking the 4.00 cap) and it is still off, because it cost −3.8 % throughput
and −67 % of the pool.

### 3.3b Pool-capacity constants, for multi-client sizing

`[measured / arithmetic from the checkpoint config]` The constants a deployment needs in order
to size a shared backend. The guidance built on them is in
[README.md](README.md#sizing-for-multiple-clients-and-subagents).

| Quantity | Value | Basis |
|---|---|---|
| Full-attention KV, bf16 | **24,576 B/token** | 12 full-attn layers × 2 KV heads × 256 head_dim × (K+V) × 2 B `[arithmetic]` |
| Compressed indexer keys | 768 B/token | 12 layers × 1 head × 128 dim × 2 B ÷ 4 (compress ratio) `[arithmetic]` |
| MTP draft KV | 2,048 B/token | 1 layer × 2 heads × 256 × (K+V) × 2 B `[arithmetic]` |
| **Total per context token** | **27,392 B (26.75 KiB)** | sum of the above; a measured pool line worked out at 26.0 KiB/token |
| Mamba state slots per running request | **3** | `[measured]` "mamba num: 45" observed at 15 running requests |
| Pool cost of one mamba slot | **~0.093 GB ≈ 3,750 pool tokens** | `[measured]` |
| Pool cost of one extra running request | **~11,250 pool tokens** | 3 × the above |
| Concurrency cap | `--max-mamba-cache-size` ÷ 3 | 12 slots → 4 concurrent, which is why the shipping config pairs 12 with `--max-running-requests 4` |
| KV pool, bf16 → fp8, test card | 279,680 → **542,912** | **1.941×** `[measured]` |
| KV pool, bf16 → fp8, production card | 331,456 → **643,456** | **1.941×** `[measured, read from `/get_server_info` at 18:31Z]`. **Largest pool measured on this hardware.** |
| Cost of declaring a larger window | 542,912 tokens at a 262k declaration vs **519,040** at 540k | `[measured]`, same fp8 build |

**Why the cost is affine rather than linear.** Only 12 of 48 layers are full attention; the other
36 are gated-delta-net linear attention, whose recurrent state is **constant per sequence** and
does not grow with context. Pool cost is therefore ≈ `N × fixed + total_tokens × rate`. The
operationally useful consequence is that **for many small requests the fixed term dominates**, so
reducing the number of admitted clients can free more pool than reducing each client's context.

One measurement quality note: two sources give different per-slot state sizes — **~0.093 GB**
`[measured on this engine, and the figure used above because it is the pool trade-off that
matters]` and **~56 MiB** `[measured on the older engine]`. The ~0.093 GB figure covers more than
the raw GDN state, so they are not directly comparable and are not averaged here.

**Prefix-cache behaviour, which is what actually degrades under oversubscription.** `[measured]`
In normal operation the deployment serves **64 % of prefill tokens from cache** — 218 prefill
batches with more than 50k cached tokens each, maximum 203,904. A live reading at 18:11Z on the
331,456-token pool showed the oversubscribed state instead: 254,144 tokens used (76.7 %), 3
running and 6 waiting requests, **335,949 waiting uncached tokens — more than the whole pool** —
for 590,093 total (**1.78×** oversubscribed), at `cache_hit_rate` 0.0. That reading is a single
instantaneous sample from the live lane, not a benchmark cell, and is reported as such. It is
included because it identifies the failure mode: **the scheduler queues rather than erroring, and
the cost lands as prefix eviction and full re-prefill**, which at ~250k context is tens of seconds
per turn.

### 3.3c fp8 KV on the production card — live readings, 18:31Z

`[measured, live]` fp8 KV went live on the reference deployment's production card. Configuration
read back from `/get_server_info` rather than trusted from the launch line:

| | |
|---|---|
| `kv_cache_dtype` | `fp8_e4m3` |
| `max_total_num_tokens` | **643,456** (that card's bf16 figure was 331,456 → **1.941×**) |
| `context_length` | 262,144 |
| `max_running_requests` | 4 |
| `max_mamba_cache_size` | 12 |
| `mem_fraction_static` | 0.98 |
| `bf16_gemm_backend` | `sm120gemv` |
| `speculative_algorithm` | reported as `EAGLE` — the launch flag is `NEXTN`; see [README](README.md#run-it) |

**Single-stream, router path, three sequential 512-token greedy completions of the same prompt
after warm-up: 175.3 / 183.3 / 173.1 tok/s** (mean ~177).

**These are not harness cells.** They are chat completions through the serving router, on a
different prompt and a different code path from the sweep in §3.1b, so they must not be compared
row-to-row with it. What they are good for is one thing: confirming that fp8 does not cost
meaningful single-stream throughput in practice, which is consistent with the harness A/B on the
other card (fp8 159.75 vs bf16 166.42, **−4.0 %**).

The first completion in that same run measured **39.6 tok/s** — see the first-request caveat in
§2. It is excluded from the three figures above and is the reason the caveat is there.

### 3.4 Acceptance rule used for shipping

A single-stream gain counted only if, on the same boot: 4-concurrent aggregate ≥ 386.9,
8-concurrent ≥ 388.2, and 4-concurrent TTFT p50 did not regress past 0.331 s.

Under that rule: **sm120gemv-only passed and was the conservative recommendation**; the fuse
levers passed on speed and were shipped on the decode-path verdict (§4.4); fp8 KV **failed**
the aggregate leg (8-concurrent 377.0 < 388.2) and is therefore documented as a **pool lever**
rather than a throughput lever; adaptive-wide failed.

### 3.5 Where the decode cycle goes

`[measured, from a 30-cycle decode trace of the stock configuration at bs=1]` This is the
measurement that directed the whole effort, so it is worth publishing.

| Quantity | ms per cycle |
|---|---|
| Cycle wall (median, verify-start to verify-start) | **16.74** |
| Device busy (interval union, all streams) | 16.18 |
| **Device idle inside cycles — all bubbles** | **0.25 (1.5 %)** |
| — inter-kernel gaps < 1 µs (681 per cycle) | 0.107 |
| — gaps 1–20 µs (23 per cycle) | 0.051 |
| — gaps ≥ 100 µs (one host stall every ~15 cycles) | 0.090 |

**There is essentially no host overhead to remove.** CUDA-graph replay already hides launch
cost; bubbles are 1.5 % of the cycle. The recoverable "fixed overhead" is **on-GPU time in
~1,650 small latency-bound kernels per cycle** (2,135 launches per cycle in total: 1,857
verify + 135 draft + 81 draft-extend + ~60 eager). That is why the levers in this fork are
**kernel-count** levers — fusion — and not scheduling levers, and it is why PDL and
`torch.compile` were killed by measurement (see CHANGES §4.4).

Weight-streaming kernels (dense GEMM + NVFP4 grouped GEMM + `hc_mix`) account for 13.21 ms of
kernel-sum across 487 launches; everything else is **6.19 ms across 1,649 launches**.

One correction worth recording, because it changed the priorities: **trace durations
double-count PDL-overlapped kernels.** Exclusive times are much smaller than trace durations
— GDN gated norm 2.0 µs, not 12.3; shared-expert gate-mul-add 1.15 µs, not 6.1; MoE prefix
sums 0.44 µs, not 2.15. Two candidate levers died on that correction alone.

### 3.6 Measured device bandwidth

`[measured]` On the RTX PRO 6000:

| Probe | Result |
|---|---|
| Read-only reduction, 3.0 GB buffer | **1,271.9 GB/s** |
| Copy (1 read + 1 write) | 1,203.1 GB/s |
| 10-of-512 row gather (MoE-shaped) | **985.8 GB/s** |

Vendor spec for this SKU is 1,792 GB/s. The measured figure is what the arithmetic in
CHANGES.md uses. The stock engine was already running at **88 % of the measured read
bandwidth**.

### 3.7 Host↔GPU transfer, for the host-KV path

`[measured, GPU2-proxy, PCIe gen4 ×16 — the same generation and width as the target card]`

| Path | Result |
|---|---|
| `cudaMemcpy` pinned H2D / D2H | 28.2 / 28.6 GB/s |
| Zero-copy sequential kernel read | 23.1 GB/s |
| Zero-copy scatter write, 8192-token chunk | 3.4 GB/s |
| Decode gather from host, 1 / 4 / 8 / 16 rows per layer | 0.180 / 0.725 / 1.58 / 3.14 ms (21–23 GB/s) |
| Decode gather from HBM, same kernel | 0.035 / 0.049 / 0.111 / 0.270 ms |

`[measured, on the target card during the 1M run]` 26.5 GB/s memcpy and ~19–22 GB/s for the
decode gather.

Host-KV hot cache hit rate in steady decode: **0.89–0.95** (cumulative 0.92), cutting host
bytes read per selected token from 2,048 to **92–223 B**, i.e. a **9–22× reduction in PCIe
traffic**. During the needle phase the hit rate was only **0.35–0.40**, because each needle
decodes at most 48 tokens right after its prefill.

### 3.8 Native context beyond the trained window, with no YaRN — suite stage 2 of N

**Status: in progress.** Retrieval is measured to 530K. Reasoning at length, normal-length
equivalence under a larger declared window, and any quality verdict past 262,144 are **not yet
established**. No production deployment has been changed on this basis.

#### 3.8a Second card: stage 1 grid, stage 2 bracket, over-limit refusal

**Hardware differs from the rest of this document.** NVIDIA RTX PRO 6000 Blackwell **Server
Edition, 600 W**, 97,887 MiB, SM120, driver 580.95.05 — against the 300 W Max-Q, driver 610.43.02,
used everywhere else. **Nothing in this subsection may be compared against a throughput figure
elsewhere in this document.**

Engine: a replica of the production configuration — this tree's final build, `--kv-cache-dtype
fp8_e4m3`, `SGLANG_QWENOPT_FUSE_SBMOE=1`, `SGLANG_QWENOPT_FUSE_HC=1`, `SGLANG_HC_MIX_PREFETCH=1`,
`--bf16-gemm-backend sm120gemv`, `--speculative-algorithm NEXTN` with the stock MTP draft head,
`--mem-fraction-static 0.98`, `--max-running-requests 4`, `--max-mamba-cache-size 12` — booted with
`--context-length 540000` and `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`, **no rope override**.
Configuration read from the suite's own source, not from a description of it. KV pool at that
declaration: **689,728 tokens** (the reference card gives 519,040 at the same declaration; the
difference is free VRAM between the cards).

Harness: one synthetic passphrase needle per request in a real-text public-domain haystack, a
distinct book window and passphrase per trial, temperature 0, thinking off, scored by exact match.

| Stage | Length | × 262,144 | Depths | Trials | Pass | Mean wall (this card) |
|---|---|---|---|---|---|---|
| 1 | 300K | 1.14× | 0.02 / 0.50 / 0.95 | 3 | **9 / 9** | 30.7 s |
| 1 | 400K | 1.53× | 0.02 / 0.50 / 0.95 | 3 | **9 / 9** | 46.1 s |
| 1 | 440K | 1.68× | 0.02 / 0.50 / 0.95 | 3 | **9 / 9** | 54.3 s |
| 1 | 470K | 1.79× | 0.02 / 0.50 / 0.95 | 3 | **9 / 9** | 61.2 s |
| 2 | 480K | 1.83× | 0.02 / 0.95 | 1 | **2 / 2** | 63.6 s |
| 2 | 500K | 1.91× | 0.02 / 0.95 | 1 | **2 / 2** | 68.5 s |
| 2 | 510K | 1.95× | 0.02 / 0.95 | 1 | **2 / 2** | 70.5 s |
| 2 | 530K | **2.02×** | 0.02 / 0.95 | 1 | **2 / 2** | 76.0 s |
| — | 691,728 | — | 0.02 / 0.95 | 1 | HTTP 400 × 2 | refused in ~3 s |

**Grid: 36 / 36. Bracket: 8 / 8. 44 of 44 retrievals.** Every completion was the exact passphrase
with `finish_reason: stop`; zero errors and zero empty completions among the 44. Prompt tokens ran
within +61 of the nominal length (e.g. 470,056–470,061), from the needle and question text.

The two over-limit requests were refused immediately — *"input (691,793 tokens) is longer than
the model's context length (540000)"* — which is correct behaviour and **not** a retrieval result.

**How much weight each half carries.** The grid is 9 retrievals per length across start, middle
and end. The bracket is 2 per length at start and end only, one trial, one seed. The bracket
establishes that **no retrieval ceiling appears below 530K on this card**; it does not
characterise 480K–530K as tightly as the grid characterises 300K–470K.

#### 3.8b Reference card: the original four-needle probe, re-read

The earlier probe on the 300 W reference card (fp8 KV, `--context-length 540000`, pool 519,040):
~200K control **PASS**; ~300K **PASS**; ~400K with the needle 398K back **PASS** (57.1 s); ~500K
**empty completion** (81.7 s); 520,059 **HTTP 400 capacity refusal** (pool < prompt).

That was previously concluded as *"holds to 400k and has broken by 500k."* **Superseded.** The
500K position passes 2/2 on the second card at the same declaration with no YaRN, so position
alone cannot account for the reference card's empty completion. The **probable** cause is capacity:
a ~500K prompt, plus the draft head's KV and the state-slot reservations, sits at the edge of a
519,040-token pool. Note the contrast with 520,059, which was *refused* rather than admitted —
the 500K case was admitted and then produced nothing, consistent with running out of room after
admission rather than before it. **Probable, not confirmed: 500K has not been re-run on the
reference card.**

#### 3.8c Stage 1b: long-document aggregation

**Design.** Five synthetic ledger records inserted at depths **0.1, 0.3, 0.5, 0.7 and 0.9** of a
real-text document. One question needing all five: the record with the largest value (argmax) and
the total (sum), in a fixed answer format. Direct answer, **thinking off**, temperature 0, scored
exactly. Two trials at each of 100K, 300K, 400K, 440K, on the second card.

| Length | argmax | sum | Both |
|---|---|---|---|
| 100K (control, inside the native window) | 2 / 2 | **1 / 2** | 1 / 2 |
| 300K | 2 / 2 | 1 / 2 | 1 / 2 |
| 400K | 2 / 2 | 0 / 2 | 0 / 2 |
| 440K | **1 / 2** | 0 / 2 | 0 / 2 |
| **Total** | **7 / 8** | **2 / 8** | **2 / 8** |

**The sum column is confounded, and is not evidence about context length.** It fails at the 100K
control, which sits inside the native window where no extrapolation is involved. With thinking off,
the model is doing five-term addition in a single forward answer, so the cell measures unaided
arithmetic. The six wrong sums, as (returned vs correct): 231 vs 302 (−23.5 %) at 100K; 211 vs 254
(−16.9 %) at 300K; 150 vs 164 (−8.5 %) and 259 vs 352 (−26.4 %) at 400K; 183 vs 196 (−6.6 %) and
224 vs 219 (+2.3 %) at 440K. Five of the six are **under**-counts, which would fit a record being
missed as readily as a slip in addition — this run cannot tell those apart (below), so no mechanism
is claimed.

argmax — closer to a lookup — holds at 7/8, with its one miss at 440K. With n=2 per length that
single miss is not interpretable as a trend.

**No conclusion about reasoning at length is drawn from 3.8c.** This run predates the suite's
per-record recall instrumentation, so it cannot distinguish "did not find a record" from "found all
five and added them wrong". The follow-up that can — asking the model to list every record before
answering, plus the same task with thinking on — is running.

#### 3.8d Why this is operationally interesting, and what it is not

It is cheaper than YaRN: static YaRN f=4 is process-wide and shifts short-context logits by
0.066–0.283 nats/token against native. Native extrapolation costs nothing at short context —
**if** the normal-length equivalence test (262K boot vs 540K boot) comes back clean, which has not
happened yet.

It is not a quality verdict. Needle retrieval saturates and catches only catastrophic failure
(§4.7). No accuracy suite, NLL or decode-path verdict has been issued at any length past 262,144.

*Hypothesis only:* 12 of 48 layers are full-attention and `partial_rotary_factor` is 0.25, so RoPE
acts on a quarter of the dimensions in a quarter of the layers — a small positional surface,
consistent with graceful extrapolation to 2.02×. Not isolated by any experiment here.

Reproducibility: both cards' runs used a private, traffic-derived speculative token map. A map
changes only which draft tokens are proposed and the target verifies them, so it cannot change a
greedy completion; the pass/fail results above do not depend on it.

### 3.9 1M context

See [CHANGES §3.4](CHANGES-vs-upstream.md#34-1m-context-on-one-card--measured-and-not-the-default)
for the full table — needles 5/5 with 4 beyond native, decode 157.0 / 154.4 / 141.0 tok/s at
250k / 500k / 900k, the 900k mixed-stream failure, and the caveats. **Its automated quality
gate returned INVALID (rc=2) and there is no decode-path verdict for that configuration.**

---

## 4. Quality instruments — what each one can and cannot see

This matters more than usual here, because the obvious instrument is the wrong one. The
harness is in [`tools/qgate/`](tools/qgate/README.md); its README gives the commands and the
verdict labels.

### 4.1 The suites

| Suite | Data | n | Metric |
|---|---|---|---|
| gsm8k | GSM8K test | 150 (fixed seeded subset; 1,319 in strict mode) | exact numeric match, greedy, 4,096 max tokens |
| ifeval | Google IFEval, official checkers | 120 (541 strict) | prompt-level strict |
| humaneval | HumanEval | 80 (164 strict) | pass@1 greedy, tests run in a subprocess under rlimits |
| niah | Gutenberg haystack, 1 target + 3 distractor keyed needles | 5 lengths × 5 depths × 2 | exact 7-digit match; `confused` = returned a distractor |
| nll | Gutenberg + a private code corpus | 6 corpora × 5 depths | teacher-forced NLL of a 64/128-token window at depth C |
| **decpath** | 48 real prompts (16 GSM8K, 16 HumanEval, 16 IFEval), all < 500 tokens | 96 per concurrency | greedy 384 tokens, 2 reps, at conc 1 and 4: identical outputs, agreement length, mean \|Δlogprob\| over the agreeing prefix |
| degen | GSM8K + HumanEval + IFEval | 60 (200 strict) | production sampling T=1.0, 6,144 max tokens, at the lane's concurrency |
| prefix | code corpus, 8k / 24k / 49k prefixes | 6 | cold / warm / late drift, impossible token ids |
| mmlupro / heplus / mbppplus | MMLU-Pro, EvalPlus HumanEval+ / MBPP+ | 1,200 / 164 / 378 | strict mode only |

### 4.2 Why NLL is the wrong instrument for these levers

`[measured]` **The NLL suite is teacher-forced prefill with 4096-token chunking, so `T >> 16`.
Both fuse levers gate at `T <= 16`.** They therefore never execute during the NLL suite.
Proved directly: at ctx 512 (a single prefill chunk), the fused build's window logprobs are
**bitwise identical** to four other builds — `diff_frac 0.00`, `mad 0.0000`.

`[measured]` Worse, **multi-chunk prefill logprobs are nondeterministic run-to-run in this
engine, independently of any lever.** Two configurations whose prefill mathematics are
identical by construction differed on **89 % of tokens at 4k**, with mean |Δlogprob|
0.095 / 0.147 / 0.163 at 4k / 16k / 64k. A fixed dNLL budget of 0.02 nats/token was
approximately **2 standard errors of the gate's own noise** at that sample size.

Consequence: the NLL rules were rebuilt. Single-chunk windows (ctx ≤ 3,968) are a **hard**
check — the baseline is bit-deterministic there, so a lossless claim FAILs on any token
|Δlogprob| > 1e-4. Multi-chunk buckets are tested **only** against a measured base-vs-base
spread, never against a fixed budget, and the verdict is `UNCALIBRATED` rather than FAIL when
no repeated baseline exists.

**Blind spot to know about:** the ctx-512 hard check **cannot see KV-cache dtype**. fp8 KV is
bitwise equal to bf16 at a single-chunk prefill, because that prefill attends over fresh K/V
rather than the quantized cache. KV dtype is covered by the multi-chunk buckets, by decpath,
and by needle retrieval.

### 4.3 The decode-path test is the primary instrument

Built because of §4.2. It uses prompts under 500 tokens so prefill is a single deterministic
chunk, then measures **decode**: greedy, 384 tokens, output logprobs only, 2 repeats, at
concurrency 1 (bs1 verify shapes) and 4 (bs4 graph shapes).

Verdict rule: **BITWISE** if every output is identical; **FAIL** if mean |Δlogprob| exceeds
`max(1.5 × noise, noise + 0.002, 0.003)` **and** a one-sided Mann-Whitney per-prompt test
gives p < 0.01, **or** if agreement length is significantly shorter than noise at p < 0.01
**and** its median is below 0.75× the noise median. **WARN** at p < 0.05, otherwise **PASS**.
Noise comes from baseline rep0-vs-rep1 within a run plus baseline-run-1-vs-run-2 across boots.

### 4.4 Decode-path results

`[measured]`

| Run | Concurrency | Verdict | Identical (candidate / noise) | Agreement-length median (cand vs noise) | Mean \|Δlogprob\| (cand vs noise) |
|---|---|---|---|---|---|
| SBMOE only | 1 | **PASS** | 6 / 96 | 49 vs 52 | 0.0150 vs 0.0143 |
| SBMOE only | 4 | **PASS** | 4 / 96 | 46 vs 54 | 0.0146 vs 0.0147 |
| Both fuse levers | 1 | **PASS** | 4 / 96 | 50 vs 52 | 0.0156 vs 0.0143 |
| Both fuse levers | 4 | **PASS** | 7 / 96 | 47 vs 54 | 0.0151 vs 0.0147 |

**Degeneracy: 0 / 192 for SBMOE and 0 / 192 for both levers**, against a baseline reference
bad-rate of 0.41 %.

**Stated plainly: neither decpath nor NLL certifies bitwise equality end to end**, because
decode on this stack is nondeterministic even for the unmodified baseline — the noise column
itself only produces 4–7 identical pairs out of 96. What the table supports is that the fused
levers are **indistinguishable from the baseline at the resolution available**, which together
with the kernel-level bitwise and 1-ulp results (CHANGES §2.1, §2.2) is the lossless claim
being made. It is not a claim of bit-exactness in the served engine.

For **fp8 KV**, the same test gives **PASS at concurrency 1 and WARN at concurrency 4
(p = 0.033)** — a small but real divergence, alongside identical GSM8K (0.953 → 0.953) and
identical needle retrieval (1.0 → 1.0). That combination is why fp8 KV is shipped labelled
lossy and opt-in.

### 4.5 NLL result for the fuse levers, and why it is a WARN not a FAIL

`[measured]` On a fixed 0.02 budget the fused build initially **FAILED** at mean dNLL 0.02251
(0.04943 at the 16k bucket). Re-scored against a measured base-vs-base spread it is **WARN**:
mean dNLL +0.0114 over 24 windows against noise −0.0029 over 48 windows, excess +0.0144,
Welch p = 0.037. Against the final build's own base-vs-base seed (a wider spread, SD
0.06–0.07) the same comparison is **PASS** (excess +0.015, p 0.12). The two noise sources
disagree; neither is a FAIL.

Leave-one-out calibration confirms the rule is not simply lenient: two lossless configurations
scored as candidates against the others give PASS at p 0.95 and 0.65.

**Since the fused kernels provably do not execute during the NLL suite (§4.2), this is a
statement about the gate, not about the kernels.** The decisive instrument is §4.4.

Half of the NLL-only re-gate was run. The **baseline leg ran twice on one boot of the final
build, at 96 paired windows** (`--nll-ctx 512,4096,16384,65536 --nll-starts 4 --nll-window 64`),
and it is the source of the base-vs-base spread in §1.1. It confirms the premise of §4.2
directly: all 24 ctx-512 windows are bit-identical between the two runs (max token |Δ| 0.0),
and every multi-chunk bucket differs (per-window |dNLL| 0.043 / 0.045 / 0.053 at 4k / 16k /
64k, SD 0.062–0.069). **The lever leg on that same boot was not run**, so FUSE's NLL verdict
still rests on the cross-boot comparisons above.

### 4.6 Statistical power — what these sample sizes can actually resolve

`[arithmetic: one-sided exact binomial sign test on discordant pairs, α = 0.05]` Publishing
this because "benchmark unchanged" is a much weaker statement than it looks.

| Suite | n | Measured discordance | Resolvable drop |
|---|---|---|---|
| GSM8K | 150 | 1 / 150 | ~4 points |
| HumanEval | 80 | 6 / 80 | ~8.8 points (the 10-point ceiling is the practical limit) |
| IFEval | 120 | 11 / 120 | ~7 points |
| NIAH | 10 per bucket | saturates at 100 % | only ≥ 3/10 in a bucket — a catastrophe detector |

`[arithmetic, 80 % power]` In strict mode: MMLU-Pro at n=1,200 resolves **1.86 points**;
GSM8K full at n=1,319 resolves **0.56 points**; all binary suites pooled at n=3,602 resolve
**≈ 1.0 point**. **Code alone does not reach 2 points** — every public HumanEval+ and MBPP+
item together gives 3.0 points, and getting code to 2 points needs ~1,200 code items.

**No strict-mode verdict was issued for any configuration in this fork.** The strict
references were never built: they need a baseline-configuration instance on the second card,
and that instance failed to boot twice. So the accuracy statements here rest on the
lower-powered suites, at the resolutions above.

### 4.7 Known weaknesses of the instruments, stated

- **NIAH is a lexical-match proxy.** It saturates at 100 % up to 98k on the baseline, so it
  catches catastrophic retrieval failure, not subtle degradation. The vendor's own
  MRCR-8needle numbers for this model fall from 93.0 at 256k to **40.5 at 512k and 26.4 at
  1M**; this gate would not see an MRCR-style collapse.
- **Gutenberg texts are memorized** (NLL 0.04–0.10 nats/token at ctx 512), which is why a
  private code corpus was used alongside them.
- **7.7 % of short-context items hit the 4,096-token budget while still reasoning** (5 GSM8K,
  13 IFEval, 9 HumanEval). These count as failures on both sides of every comparison.
- **The baseline itself emits degenerate loops at 0.41 %** (2 of 492 under greedy) — one
  HumanEval case collapsed into a single token repeated 1,519 times, one IFEval case into a
  period-3 loop. Four greedy re-runs did not reproduce it. It is nondeterministic and **has
  not been attributed**, and it is the reference rate the degeneracy gate is measured against.
- **A bug that fires only at a batch size the harness never forms is not caught.** The degen
  suite is run at the lane's own `max_running_requests` for this reason.
- **Retraction pressure is never deliberately induced**, so the prefix-cache probe detects
  poisoning if it happens during a run but does not provoke it.
- **GPQA-Diamond is not included** — the dataset is gated.

### 4.8 Radix-cache integrity (the poisoned-prefix race)

Upstream has an unfixed chunked-prefill / radix-insert race (sglang #38319, fix PR #38355 still
open) that can leave a cached prefix corrupted until the cache is flushed. The same prefix then
emits an impossible token id on every hit. This fork does not change that code, so the gate
checks for it rather than assuming it away.

**The first version of the probe never touched the cache.** It scored prompt logprobs of a
window after each prefix. In this engine, any request that asks for prompt logprobs caps its
own prefix match (`schedule_batch.py:1414`). Every warm and late request therefore reported
`cached_tokens = 0`, on the stock engine, on the old engine on the test card and on the new
engine. What it measured was prefill recompute drift, not cache integrity. That produced one
false FAIL, which was withdrawn.

**Version 2 is generation-only.** It sends raw input ids, asks for a greedy 32-token
continuation, and returns output-token logprobs only. There is no prompt-logprob request, so
the prefix match is not capped. Each of three code-corpus prefixes (8k, 24k and 49k tokens)
runs three times: cold (first sight), warm (an immediate repeat) and late (after the other
suites have churned the cache).

The verdict rule:

- **FAIL** on any impossible token id.
- **FAIL** on a degenerate continuation beyond the reference + 1.
- **FAIL** if the mean |Δlogprob| over the shared greedy continuation exceeds max(0.1, 3 × the
  reference).
- **NOT_COMPARABLE** if no warm or late request actually hit the cache.

`[measured]`

| Run | Cache hits (warm/late pairs) | Largest cached prefix | Max mean \|Δlogprob\| on shared continuation | Impossible ids | Verdict |
|---|---|---|---|---|---|
| Stock engine, production lane (reference) | 4 / 6 | 49,088 tokens | 0.111 | 0 | reference |
| This tree, token-map configuration | 4 / 6 | 49,088 tokens | 0.101 (limit 0.334) | 0 | **PASS** |

The two misses in each row are warm repeats that landed before lazy insertion into the cache.
Only 3 prefixes were used, and the race needs retraction under memory pressure, which the probe
does not provoke. So this row says no poisoning occurred during the run. It does not say the
race cannot happen.

---

## 5. Reproducing this

You will need: an RTX PRO 6000 Blackwell (or another SM120 card, with the caveat that pool
sizes scale with VRAM), a Qwen3.8-Flash-Next NVFP4 checkpoint, and **a speculative token map
built from your own traffic** — the map used here is not shipped and several single-stream
rows depend on one (§1.2 and the `no token map` row in §3.3, which is 130.3 tok/s against
147.2 with a map).

1. Check out this repository, or apply `patches/*.patch` onto upstream at the base commit.
2. Launch with the command in [README.md](README.md#run-it).
3. Confirm from the boot log that the levers are actually on: exactly one
   `QSA shared packed-KV scratch allocated` line with 0 regrown and no `ValueError` at graph
   capture; no JIT error for `hc_fused_tail`; and the attested line showing
   `bf16_gemm_backend=sm120gemv`. Verify the environment from `/proc/<pid>/environ`, not from
   your launch line.
4. Measure the A/B in §3.2 rather than the cross-engine comparison in §3.1 — it is one
   variable, and it is the number the kernels are responsible for.
5. Run at least two repeats of every cell. **The spread is 6.3 %.**

---

## 6. Open measurements

Recorded so this is not mistaken for a finished evaluation.

| Item | Status |
|---|---|
| Dynamic-k paired GPU A/B (fixed-k, cumprob, learned gate, batch-aware fill) | **never run.** CPU- and GPU2-proxy-verified only. |
| Learned-gate paired GPU test | **never run.** |
| NLL-only re-gate of the fuse levers (§4.5) | **half run.** Baseline twice on one boot, 96 windows: done, and it is the §1.1 noise seed. The lever leg on that boot: **not run.** |
| Strict-mode references and any strict verdict | **never built** — the baseline-config twin failed to boot twice. |
| Same-instance NLL repeat (the true NLL noise floor rather than a cross-instance one) | **done** on the final build (same boot, 96 paired windows; §1.1, §4.5). It was never done for the stock-engine baseline configuration, whose twin failed to boot. |
| IFEval, decpath and degen on the YaRN f=4 / 1M configuration | **not measured** — the run was stopped mid-suite, and the later gate returned INVALID. |
| Cache-integrity (v2, cache-exercising) prefix probe on this engine | **measured once, PASS** (§4.8) — on the token-map configuration only, 3 prefixes. Not repeated on the final lossless stack. |
| A multi-client / multi-subagent capacity benchmark (aggregate throughput and per-turn latency at a realistic mix of one long-context primary plus N small subagents) | **not run.** §3.3b gives the constants to size with and one live oversubscription reading; there is no swept measurement of the mixed workload, so the sizing guidance is arithmetic plus a single observation, not a benchmark. |
| Prefill re-cost under deliberate prefix eviction (how many seconds a turn actually costs once a ~250k prefix has been evicted) | **not measured directly.** Inferred from the measured prefill rate, not timed under induced eviction. |
| A deep single-stream context sweep on the exact shipping bf16-KV configuration (beyond 95.4k, up to its 279,680-token pool) | **not run.** The harness cells stopped at 95.4k, so §3.1b's deeper rows come from the fp8-KV and host-KV arms instead. This is the single most useful missing measurement for anyone sizing a long-context agent lane on the shipping config. |
| The native-RoPE retrieval ceiling | **not found.** §3.8a passes through 530K (2.02×) on the second card; above that the declared window and the pool bind before position does. The 480K–530K bracket is thin (2 per length, start and end only). |
| Whether the reference card's 500K empty completion was capacity | **probable, not confirmed** — 500K has not been re-run on the reference card (§3.8b). |
| Recall vs arithmetic in the long-document aggregation task, and the same task with thinking on | **running.** §3.8c cannot separate them: the run predates per-record recall instrumentation. |
| Decode-path equivalence at normal lengths, 262K boot vs 540K boot | **running.** Until it lands, native extrapolation is not shown to be free at short context. |
| Single-stream decode throughput at 400K+ on native RoPE | **not measured.** Both long-context runs were retrieval probes; §3.8a's wall times are per-request and prefill-dominated on a 600 W card, so they are not decode throughput and are not comparable to §3.1b. |
| A quality gate at any length past 262,144 | **not run.** Retrieval passes are a capability probe, not a certified operating point. |
| Wall times for the ~200k and ~300k native-RoPE needle cases | **not recorded** in the run ledger; omitted rather than estimated. |
| Why the retrained MTP head won on a B200 under plain upstream SGLang (+1.8 % to +3.7 % in all four cells) and was flat on this tree's SM120 fused path (2.1076 vs 2.1079) | **cause not established.** GPU generation and the fused decode path were never isolated from each other. |
| Downtime duration of the 2026-09-29 16:45:24 constrained-decoding OOM | **not recorded** in the sources consulted; the failure chain and the mitigation are, so no duration is claimed. |
| The 900k mixed-stream failure, after the OOM fix | **not re-run.** |
| Why the long primary never gets a radix-cache hit | **cause not established.** |
| Accept-length decay over uptime | snapshots recorded per suite; **no trend analysis done.** |
| Whether `fp4_gemm_runner_backend='auto'` resolves to a safe kernel on SM120 | **not traced.** Flagged, not asserted — which is why the launch command pins `flashinfer_cudnn`. |

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
- **This tree:** base `6fa3fe69e2` + these 32 commits (the 2026-09-30 cutover in §3.15–§3.18 adds six more; the headline table predates them), venv with **flashinfer 0.6.18** and
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
| Pool cost of one extra running request | **29,120 pool tokens** | `[measured, second card]` 710,336 / 652,096 / 593,856 / 477,376 at 4 / 6 / 8 / 12 streams. Its 3 slots are ~12,480 of that (~4,160 per slot there); the rest is other per-stream reservations. An earlier revision estimated ~11,250 from the slots alone, which undercounts. |
| Concurrency cap | `--max-mamba-cache-size` ÷ 3 | 12 slots → 4 concurrent. Slots beyond 3 per running request hold idle conversations' state and are what keeps multi-turn prefixes cacheable (§3.10) |
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

### 3.8 Long-context suite — complete (second card)

**Hardware differs from the rest of this document.** NVIDIA RTX PRO 6000 Blackwell **Server
Edition, 600 W**, 97,887 MiB, SM120, driver 580.95.05 — against the 300 W Max-Q, driver 610.43.02,
used everywhere else. Engine: a replica of the production configuration (this tree's final build;
fp8 KV; `SGLANG_QWENOPT_FUSE_SBMOE=1`, `SGLANG_QWENOPT_FUSE_HC=1`, `SGLANG_HC_MIX_PREFETCH=1`,
`SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`; `sm120gemv`; NEXTN 3/1/4 with the stock draft head;
`--mem-fraction-static 0.98`; `--max-running-requests 4`; `--max-mamba-cache-size 12` unless stated),
read from the suite's own source and each boot's `/get_server_info`. **No rope override.** Boots above
262,144 set `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`.

**What transfers:** pass/fail, same-boot ratios and the shape of every curve. **What does not:**
absolute tok/s, TTFT and pool sizes. Nothing from this section appears in any other table.

KV pool on this card, fp8, by declaration (mrr 4, 12 slots): 262,144 → **710,336**; 540,000 →
**689,728** (−2.9 %); 680,000 → **686,336** (−3.4 %). The reference card gave 519,040 at 540,000.

#### 3.8a Retrieval — 60 / 60

One synthetic passphrase needle per request in a real-text haystack of five public-domain books;
distinct book window and passphrase per trial; temperature 0; thinking off; 48 max tokens; exact match.

| Length | × 262,144 | Declared | Depths | Trials | Pass | Wall (this card) |
|---|---|---|---|---|---|---|
| 300K | 1.14× | 540,000 | 0.02 / 0.50 / 0.95 | 3 | **9 / 9** | 30.7 s |
| 400K | 1.53× | 540,000 | 0.02 / 0.50 / 0.95 | 3 | **9 / 9** | 46.1 s |
| 440K | 1.68× | 540,000 | 0.02 / 0.50 / 0.95 | 3 | **9 / 9** | 54.3 s |
| 470K | 1.79× | 540,000 | 0.02 / 0.50 / 0.95 | 3 | **9 / 9** | 61.2 s |
| 480K | 1.83× | 540,000 | 0.02 / 0.95 | 1 | **2 / 2** | 63.6 s |
| 500K | 1.91× | 540,000 | 0.02 / 0.95 | 1 | **2 / 2** | 68.5 s |
| 510K | 1.95× | 540,000 | 0.02 / 0.95 | 1 | **2 / 2** | 70.5 s |
| 530K | 2.02× | 540,000 | 0.02 / 0.95 | 1 | **2 / 2** | 76.0 s |
| 560K | 2.14× | 680,000 | 0.02 / 0.95 | 2 | **4 / 4** | 84–89 s |
| 600K | 2.29× | 680,000 | 0.02 / 0.95 | 2 | **4 / 4** | 96 s |
| 640K | 2.44× | 680,000 | 0.02 / 0.95 | 2 | **4 / 4** | 110–111 s |
| 672K | **2.56×** | 680,000 | 0.02 / 0.95 | 2 | **4 / 4** | 121–123 s |
| 691,728 | — | 540,000 | 0.02 / 0.95 | 1 | HTTP 400 × 2 | refused in ~3 s |

Every one of the 60 completions was the exact passphrase with `finish_reason: stop`. Pool headroom at
the 680,000 declaration was 126,228 / 86,228 / 46,228 / **14,228** tokens at 560K / 600K / 640K /
672K. A scheduler-log search of those runs for retraction, pool-full, abort, eviction or OOM events
matched nothing but the server-arguments line. The over-limit request was refused up front ("input
(691,793 tokens) is longer than the model's context length (540000)") — correct behaviour, not a
retrieval result.

**No position ceiling was found.** The binding limit at the top of the range is the pool.

**Reference card, re-read.** The original probe on the 300 W card — ~200K control, ~300K and ~400K
PASS; ~500K empty completion; 520,059 HTTP 400 capacity refusal — was first concluded as "holds to
400k and has broken by 500k". **Superseded.** 500K passes here at the same declaration and build, so
position cannot explain it; a ~500K prompt plus draft-head and state-slot reservations sits at the
edge of that card's 519,040 pool. **Probable, not confirmed** — 500K was not re-run there. The
contrast with 520,059 (refused outright) fits: the 500K request was admitted and then ran out of room.

#### 3.8b Short-context equivalence under a larger declaration — decode-path PASS

`qgate.py compare` with a 262,144-declared boot as baseline, a 540,000-declared boot as candidate,
and a **re-run of the 262,144 boot as noise**. 48 real prompts (GSM8K / HumanEval / IFEval, each under
500 tokens), greedy, 384 tokens, 2 reps, at concurrency 1 and 4.

| Conc | Candidate mean \|Δlogprob\| | Noise | Tolerance | p (Δ larger) | Agreement-length median, cand / noise | Identical, cand / noise | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 0.017117 | 0.015714 | 0.02357 | 0.121 | 50 / 54 | 1 / 96 · 8 / 144 | **PASS** |
| 4 | 0.016336 | 0.016268 | 0.0244 | 0.634 | 52 / 49 | 3 / 96 · 8 / 144 | **PASS** |

Degeneracy PASS: 0 bad of 192 for candidate, baseline and noise alike, no impossible token ids.

**Lossless relative to run-to-run noise, not bitwise.** The candidate diverges from the baseline by
the same amount a repeat of the baseline does.

**The gate's overall verdict on this comparison is FAIL, and here is exactly why.** Its
`long_context_claim` rule fails any candidate whose declared context exceeds 262,144 unless that same
run contains needle cells past 262,144 with ≥ 0.8 accuracy. This run was decode-path only, so it
carries none. The rule is doing its job — it stops anyone certifying a long window from a
short-context test — and the evidence it wants is §3.8a's 60 retrievals, gathered by a separate
harness rather than inside this gate run. Decode-path and degeneracy, which are what this comparison
measures, both PASS. Anyone re-running the published gate on this configuration will see the same
FAIL for the same reason.

#### 3.8c Reasoning at length — n = 2 per cell

Five synthetic ledger records at depths 0.1 / 0.3 / 0.5 / 0.7 / 0.9 of a real-text document; one
question needing all five (the largest record, and the total); temperature 0; exact scoring.

| Length | 1b direct answer, thinking off: argmax / sum / both | 1c list every record, then answer: recall / both | 1d thinking on, 3,000-token budget: both |
|---|---|---|---|
| 100K (in-window control) | 2/2 · 1/2 · 1/2 | 5/5 · 2/2 | 2/2 |
| 300K | 2/2 · 1/2 · 1/2 | 5/5 · 2/2 | 2/2 |
| 400K | 2/2 · 0/2 · 0/2 | 5/5 · 2/2 | 1/2 |
| 440K | 1/2 · 0/2 · 0/2 | 5/5 · 2/2 | not run |
| 470K | not run | 5/5 · 2/2 | 1/2 |
| **Total** | **7/8 · 2/8 · 2/8** | **5/5 every row · 10/10** | **6/8** |

- **1c resolves what 1b could not.** When asked to list the records before answering, the model
  recalled all five exactly in every case to 470K and got both answers right 10 of 10 times. So the
  records are retrievable at every length, and the direct-answer errors in 1b are most likely the
  one-shot addition of five numbers with thinking off. That fits 1b failing at the 100K control, and
  five of its six wrong sums being under-counts. An earlier revision of this section could not
  distinguish recall from arithmetic; this is the stage that separated them.
- **1d's two misses both stopped on the token budget** (`finish_reason: length`, empty answer) —
  neither is a wrong answer. 1d has no recall figure: its output is reasoning text rather than a
  record list, so the list-parsing recall metric does not apply and reads 0 on every row.
- **n = 2 per cell.** Each cell is a signal, not a rate.

#### 3.8d Speed — relative only

Single stream, 540,000 declaration, 2 reps, first request discarded. Per-stream decode is the median
decode-phase rate.

| Prompt | TTFT p50 | Prefill rate | Per-stream decode, rep 1 / rep 2 |
|---|---|---|---|
| 4,044 | 0.32 s | 12.5K tok/s | 173.6 / 157.0 |
| 10,586 | 0.86 s | 12.3K tok/s | 173.4 / 168.0 |
| 48,575 | 3.87 s | 12.6K tok/s | 159.5 / 158.8 |
| 95,437 | 7.85 s | 12.2K tok/s | 166.6 / 161.4 |
| 194,603 | 17.4 s | 11.2K tok/s | 163.8 / 160.4 |
| 291,818 | 29.5 s | 9.9K tok/s | 152.3 / 157.6 |
| 387,773 | 44.8 s | 8.7K tok/s | 157.8 / 170.7 |

- **Decode is flat in context**, 152–174 tok/s from 4K to ~390K. Qwen sparse attention reads at most
  ~2,048 selected tokens per step, so context barely touches decode.
- **Prefill slows with length** — 12.2–12.6K tok/s to ~100K, 8.7K by ~390K — so time to first
  token grows a little faster than linearly. That is where long context costs you.
- Draft acceptance, by counters over each full curve: 2.404 and 2.403.
- **Short replies decode faster per token.** On one boot, 128-token generations ran 1.20–1.43× faster
  per token than 512-token ones (cell means), with accept length 3.0–3.41 against 2.44–2.56. Early
  reply tokens are easier to draft.

### 3.9 1M context

See [CHANGES §3.4](CHANGES-vs-upstream.md#34-1m-context-on-one-card--measured-and-not-the-default)
for the full table — needles 5/5 with 4 beyond native, decode 157.0 / 154.4 / 141.0 tok/s at
250k / 500k / 900k, the 900k mixed-stream failure, and the caveats. **Its automated quality
gate returned INVALID (rc=2) and there is no decode-path verdict for that configuration.**

---

### 3.10 Multi-turn agents: state slots decide whether the prefix cache works (second card)

A multi-turn agent simulation at `--max-running-requests 4`: **6 conversations**, each re-sending its
full, growing history every turn — starting at 60,000 tokens, adding 2–4K of real text plus its own
256-token reply per turn, for 7 turns. Hit fraction is the engine's `cached_tokens / prompt_tokens`,
averaged per request over turns 1–6 (36 requests).

| Slots (`--max-mamba-cache-size`) | Pool | Mean hit | Requests with >50 % hit | Mean TTFT |
|---|---|---|---|---|
| 12 | 710,336 | **5.4 %** | 2 / 36 | 16.8 s |
| 24 | 660,416 (−7.0 %) | 2.7 % | 1 / 36 | 16.8 s |
| 36 | 610,496 (−14.1 %) | **95.6 %** | 36 / 36 | **2.5 s** |

**Controls**, 12 slots, 5 turns each: 1 conversation 95.1 % hit / 0.29 s TTFT; 2 conversations
95.5 % / 0.30 s; 4 conversations 89.4 % / 0.62 s.

**Reading.** Prefix caching itself works — the controls hit 89–96 %. What fails at 6 conversations is
contention for state slots. On this hybrid model a cached prefix is reusable only if the conversation's
linear-attention state snapshot survives alongside its KV; with 4 running requests using 3 slots each,
12 slots leave nothing to hold an idle conversation's snapshot, it is evicted, and the next turn
re-prefills the full history although its KV is still resident. **It is a threshold:** 24 slots was
*worse* than 12, and 36 fixed it outright.

**The rule of thumb, and its limits.** When live conversations exceed running requests, slots ≳
3 × running + 3 × conversations kept warm — 30 here, consistent with 24 failing and 36 working. When
they do not, 3 × running sufficed (the controls). The six observations bracket this workload's
threshold in (24, 36] and do not pin it. A "+1 slot per conversation" rule is contradicted by the
24-slot result (it predicts 18 would suffice). Treat the formula as a conservative estimate and
measure the hit rate on your own workload.

**Cost.** ~49,920 pool tokens per 12 slots on this card (~4,160 per slot).

### 3.11 Streams and priority scheduling (second card, same-boot ratios)

**Pool per stream.** 710,336 / 652,096 / 593,856 / 477,376 at 4 / 6 / 8 / 12 running requests (slots
3 × running, graph batch = running): exactly **−29,120 per extra stream**. Its 3 slots account for
~12,480; the rest is other per-stream reservations.

**Uniform 8K prompts, 256-token replies, one fresh 8-stream boot:**

| Concurrent | Aggregate tok/s | Per-stream decode, median (mean) | TTFT p50 | Accept (counters) |
|---|---|---|---|---|
| 1 | 111.1 | 178.6 (178.6) | 0.88 s | 2.631 |
| 2 | 141.6 | 124.3 (124.3) | 1.25 s | 2.457 |
| 4 | 188.2 | 87.3 (90.9) | 2.75 s | 2.482 |
| 6 | 205.0 | 62.7 (65.4) | 2.74 s | 2.544 |
| 8 | 191.2 | 49.3 (64.5) | 4.26 s | 2.496 |

A separate 12-stream boot measured 210.0 / 212.8 aggregate at 33.8 / 35.2 per stream (median). Same
boot, 4 → 6 streams buys **+9 %** aggregate; 12 streams is about +12–13 % over the 4-stream cell,
across boots. Per-stream decode falls ~2.5× over the same range. Medians are used because the
per-stream distribution has a long fast tail at higher concurrency (8-stream max 148.9).

**Accept length is not a function of concurrency.** Counter-bracketed on the fresh boot it is flat at
2.46–2.63 from 1 to 8 streams. An earlier reading of ~1.85 at higher stream counts came from a gauge
averaging a synthetic workload of forced continuations from random mid-document windows; that workload
drafts at 1.85 / 1.84 / 1.87 / 1.84 at **4** / 6 / 8 / 12 streams — flat, including at 4. The content
set the number, not the batch size.

**Mixed load** (2 main streams at 180K + subagents at 20–40K, 256 out, fresh windows so no prefix
reuse, 240 s window) is prefill-bound by construction — aggregate 27.7 / 36.3 / 42.7 / 53.3 tok/s at
4 / 6 / 8 / 12 streams, with main-stream TTFT 18.4 / 17.2 / 17.5 / **36.0 s**. More streams admit more
work but make the long main stream wait twice as long at 12.

**Priority scheduling**, 12 streams, 14 background streams at priority 0 (so there is a queue), main
stream at priority 100, 12 sequential main requests of 2–6K prompts:

| | FCFS | Priority on |
|---|---|---|
| Main alone: TTFT / decode | 0.48 s / 127.9 | 0.47 s / 125.0 |
| Main saturated: TTFT median / p90 | **5.24 / 5.63 s** | **1.11 / 1.38 s** |
| Main saturated: decode | 28.6 | 28.9 |
| Background decode | 29.7 | 27.7 |
| Background requests completed in window | 152 | **113 (−26 %)** |

Priority buys **admission** (4.7× faster first token) and not decode, which the batch sets. Its cost
falls on background throughput. At FCFS, saturated main-stream TTFT at 4 / 6 / 8 streams was 7.1 /
5.9 / 6.4 s.

**Untagged requests go last.** With priority scheduling on and `--default-priority-value` unset, a
chat or completions request without a `priority` field is assigned the most extreme value in the
losing direction by the scheduler — below an explicit 0 — and SGLang logs a boot warning for that
configuration. The Responses API is the exception (default 0). Set `--default-priority-value`, or
tag every request, or a client's untagged auxiliary calls will queue behind all background work.

### 3.12 The recommended configuration on the production card

`[measured, read from /get_server_info]` The reference deployment's production card (300 W Max-Q)
was switched to the configuration recommended by §3.8–§3.11, first at `mem_fraction_static` 0.98
and then — after the OOM in §3.14 — at 0.97, which is what it runs now:

| Setting | 2026-09-29 (0.98) | **2026-09-30 (0.97), current** |
|---|---|---|
| `context_length` | 540,000 (no YaRN) | 540,000 (no YaRN) |
| `max_mamba_cache_size` | 36 | 36 |
| `max_running_requests` | 4 | 4 |
| `kv_cache_dtype` | fp8_e4m3 | fp8_e4m3 |
| `enable_priority_scheduling` | off | off |
| `mem_fraction_static` | 0.98 | **0.97** |
| **`max_total_num_tokens`** | 522,880 | **451,264** |

**The 0.98 row is what died** (§3.14); it is kept because the pool projection below was made
against it. This closes the one measurement the suite left for the reference hardware. A projection of ~420K,
made beforehand by subtracting the 36-slot cost from 519,040, was wrong because 519,040 was the
*test* card's pool at 540,000; the production card carries less foreign VRAM. From its own measured
262,144 baseline of 643,456, the second card's ratios (−2.9 % for the declaration, ≈ −99,840 for 24
extra slots) predict ≈ 524,900 — within 0.4 % of the measured 522,880.

Only the resulting pool is measured here. Prefix-cache hit rate and time to first token under the
production workload at 36 slots have not yet been measured on this card, at either fraction.

### 3.13 Static memory fraction: the headroom/pool trade, measured

Four production configurations, each figure read from that boot's own log rather than derived.
"Boot `available_gpu_mem`" is what SGLang prints at the end of start-up; "device-free while
serving" is nvidia-smi free on the same card in steady state, sampled every 20 s.

| `--mem-fraction-static` | Slots | Declared ctx | KV pool | Boot `available_gpu_mem` | Device-free while serving |
|---|---|---|---|---|---|
| 0.98 | 12 | 262,144 | 643,456 | 3.39 GB | — |
| 0.98 | 36 | 540,000 | 522,880 | 3.42 GB | **287 MiB** |
| 0.97 | 36 | 540,000 | **451,264** | **4.64 GB** | **1,737–1,739 MiB** |

**The two columns on the right are not the same measurement, and the difference is large enough
to be dangerous.** At 0.98 the boot figure read 3.42 GB while the card actually had 287 MiB free
during serving — flat across **2,406 consecutive samples, 13 h 22 min**. The gap is PyTorch's
caching allocator continuing to reserve after the boot figure is printed. Earlier revisions of
this repository quoted the boot figure as "device headroom"; that was wrong.

**Slots cost pool, not headroom.** 12 → 36 slots moved boot-reported headroom 3.39 → 3.42 GB —
unchanged — while the pool fell 643,456 → 522,880. Decomposed, the ~3 % declared-context cost
(§3.3b) accounts for ~19K of that and the 24 extra slots for ~102K, matching the ~100K estimated
in §3.10. So the state-slot recommendation is not implicated in the OOM below, and does not need
to be revised.

**The pool cost of 0.01 of memory fraction is ~1.9× the naive prediction.** 1 % of a 96 GiB card
is ~983 MiB; at ~27 KB per context token that implies ~38K tokens. Measured: **71,616**. Lowering
the fraction shrinks the whole reservation, and the non-pool overheads inside it do not shrink in
proportion. Measure this per deployment rather than deriving it.

### 3.14 Runtime CUDA OOM under a deep queue (2026-09-30) — why 0.98 is not usable

`[measured, production card]` At 16:44:17, at `--mem-fraction-static 0.98` with a 522,880-token
pool:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 560.00 MiB.
GPU 0 has a total capacity of 95.01 GiB of which 280.94 MiB is free.
Process <other> has 676.00 MiB memory in use.
Including non-PyTorch memory, this process has 93.92 GiB memory in use.
```

→ scheduler exception → `SIGQUIT` → `kill_process_tree`. Down **12 min 54 s** (serving again at
16:57:11). The traceback ends in `_forward_prefill_batch` → `model_runner.forward` → the **eager**
runner: prefill, not decode, and not graph capture.

Scheduler state through the final nine seconds, from its own log:

| Time | `full token usage` | `#queue-req` | `#pending-token` | `mamba usage` | `#running-req` |
|---|---|---|---|---|---|
| 16:44:09 | 0.64 | 14 | 1,530,905 | 0.06 | 1 |
| 16:44:12 | 0.67 | 14 | 1,514,521 | 0.06 | 1 |
| 16:44:15 | 0.71 | 14 | 1,498,137 | 0.06 | 1 |
| 16:44:17 | **0.73** | 14 | 1,485,849 | 0.06 | 1 |

**The KV pool had 27 % free and the state slots were 6 % used.** What was exhausted was
driver-level VRAM outside the pool — the working set for chunked 4,096-token prefill against a
14-deep queue, with another process holding 676 MiB of the card. **No KV-pool admission signal
can see this**: a guard reading `token_usage` sees 0.73 and admits more. Peak queue depth that
day measured 16.

This is the third instance of one failure class, all on this lane: **an allocation arriving after
the static pool is committed.** The other two were two runtime OOMs in one day at 0.99 (one after
18 h 52 m of uptime) and a lazy Triton kernel load at 0.98 with 0.54 GiB free. The fix for the
third was 0.97; the fix for the second was a warm-up request; the first is what set 0.98.

#### The monitoring signal existed and was not consumed

| Signal | Sent / first seen | Relative to crash |
|---|---|---|
| Free VRAM flat at 287 MiB | 03:22:24Z | −13 h 22 min |
| Critical-headroom flag, every sample | 00:00:04Z | all day |
| Low-VRAM alert | 00:28:44Z | −16 h 15 min |
| Queue-depth alert | 16:43:24Z | **−53 s** |
| Unit-down alert | 16:44:24Z | +8 s |

Correct, early, and inert: none of it fed admission or dispatch. **Monitoring headroom is not the
same as controlling for it.** The client-side obligation is
[client contract rule 9](clients/README.md#9-throttle-fan-out-on-vram-headroom-not-only-on-pool-utilisation).

### 3.15 Host-resident QSA KV pool — 1,310,720 tokens on one 96 GB card

`[measured, production card, 2026-09-30]` This is the largest capacity change in the repository
and it does not come out of VRAM. The 12 full-attention layers' K/V — and, at
`SGLANG_QSA_HOST_KV=all`, the MTP draft layer's as well — are allocated in **pinned, UVA-mapped
host RAM** and read zero-copy. What stays on the card is only the QSA *compressed index*, which
is what decode's top-k actually scans.

The arithmetic that makes it work is the same 36/12 split as everywhere else in this repository.
At fp8 the host-side cost is 12,288 B/token for the target layers plus 1,024 B/token for the
draft; the GPU-side compressed index is **832 B/token** (768 for 12 layers at 1 index KV head ×
128 dim / ratio 4 × 2 B, plus 64 for the draft layer). A token costs ~16× less VRAM in the host-KV
layout than in the on-GPU one, which is where the 2.9× comes from.

**The production boot, line for line from the engine's own log** (delta, `--mem-fraction-static
0.97`, `--kv-cache-dtype fp8_e4m3`, 36 slots, declared context 540,000):

| | On-GPU KV (16:56:52Z) | **Host-resident KV (23:34:00Z)** |
|---|---|---|
| `max_total_num_tokens` | 451,264 | **1,310,720** (**2.905×**) |
| Host RAM, pinned + UVA-mapped | — | **16.25 GiB** (15.00 target + 1.25 draft) |
| GPU hot cache | — | **1.02 GiB** (0.94 target + 0.08 draft) |
| Boot `available_gpu_mem` | 4.64 GB | **8.23 GB** |

Both right-hand figures are read from `/get_server_info` and the boot log of the running lane, not
derived. The host-side allocation is logged explicitly as
`12 layers x (K (1310784, 2, 256) + V (1310784, 2, 256)) torch.uint8 = 15.00 GiB pinned+mapped
host memory (UVA zero-copy)`, and again at `1 layers ... = 1.25 GiB` for the draft.

**⛔ Read the last row with §3.13 in mind.** `available_gpu_mem` is a *boot* figure, and §3.13 is
the section that exists because an earlier revision of this repository mistook one for steady
state. 4.64 → 8.23 GB is a boot-to-boot comparison and nothing more. The steady-state number on
the host-KV configuration is **not yet soaked**: `nvidia-smi` read **5,863 MiB free** on the card
about six minutes after the lane came up, against **1,737–1,739 MiB** for the on-GPU
configuration in §3.13 — but §3.13's figure is the flat floor of 2,406 samples over 13 h 22 min,
and this one is a single early reading on a lane that has not yet run a full day. Treat the
direction as established and the magnitude as provisional.

**The GPU hot cache is what keeps PCIe out of the decode path.** `SGLANG_QSA_HOST_KV_CACHE_SETS`
sets a direct-mapped cache per (layer, request slot), 4 tokens per set; at 4,096 sets that is
16,384 tokens per slot and, as logged, `12 layers x 5 request slots x 4096 sets x 4 tokens = 0.94
GiB GPU`.

**⛔ Read the hit rate at steady state, not at the first sample.** The cache's counters are
cumulative from boot, so the first 30-second window reports itself as the cumulative figure and
looks far better than the lane sustains. Both readings, from the same production lane:

| Layer group | First 30 s after ready (23:34:21Z) | **Steady state (23:45:51Z)** | Selected tokens behind the steady figure |
|---|---|---|---|
| 12 target layers | 0.9734 hit · 27 host B/token | **0.8121 hit · 198 host B/token** | 1.34 × 10⁹ |
| MTP draft layer | 0.8991 hit · 103 host B/token | **0.8774 hit · 118 host B/token** | 1.68 × 10⁸ |

**Publish the right-hand column.** 198 host bytes per selected token against 12,288 bytes per
token of resident K/V is still a **62× reduction** in PCIe traffic, and that is the real result;
the left-hand column would have claimed 455×. An independent run on the second card measured
0.89–0.95 in steady decode, cumulative 0.92, at 92–223 bytes/token — consistent with the right-hand
column and not with the left. This is the same boot-versus-steady-state error §3.13 exists to warn
about, applied to a different counter.

#### 3.15a `--max-total-tokens` is mandatory, not tuning

With host KV the pool is no longer bounded by VRAM, so if you do not bound it explicitly SGLang
sizes it to consume the whole `--mem-fraction-static` budget. At 832 B/token on the GPU that is a
**~7M-token pool demanding ~89 GiB of pinned host RAM** `[arithmetic, from the 832 B/token
measured GPU cost and the budget at 0.97]`. The flag is what converts freed VRAM into headroom
instead of into an unbounded host allocation. `SGLANG_QSA_HOST_KV_MAX_GB` is the fail-fast behind
it: at 24 GiB it caps the pool at 2,097,152 fp8 tokens and dies at startup with a named error
rather than walking into the host OOM killer.

#### 3.15b Two long-context mains on one card — the result the pool was for

`[measured on Modal, RTX PRO 6000 Blackwell **Server Edition**, 600 W, PCIe Gen5 ×16, driver
580.95.05 — see the caveat below]` Two main agents at priority 100 holding 250K and 200K tokens,
each re-sending its full history, with two subagents added on the third turn. The control is the
**same card, same engine, same script**, with `SGLANG_QSA_HOST_KV` off.

**Be precise about the control's pool.** It was **not** pinned to the production lane's size. It
ran with `max_total_tokens: None` and sized itself off the static budget, landing at **451,840** —
within 0.13 % of delta's 451,264 by coincidence, not by construction. A second control boot of the
same configuration self-sized to 451,648. That the three agree so closely is a property of the
model and the memory fraction; it is not a control variable anyone set.

| Third turn, 2 mains + 2 subagents | Host KV, pool 1,310,720 | On-GPU control, pool 451,840 |
|---|---|---|
| main A (250K) — cached prefix | **249,984** | **none** |
| main A — TTFT | **1.2 s** | **22.9 s** |
| main A — decode | **106.3 tok/s** | 25.6 tok/s |
| main B (200K) — cached prefix | **200,000** | **none** |
| main B — TTFT | **1.0 s** | **41.1 s** |
| main B — decode | **117.4 tok/s** | 118.2 tok/s |
| `cache_hit_rate` after the turn | **0.9949** | **0.0** |
| Wall time for the turn | **32.1 s** | 71.4 s |

**Both mains lose their prefix entirely in the control and neither does on host KV.** That is the
finding. 450K of resident main-agent context fits a 451,840-token pool with nothing left over, so
the moment two subagents ask for admission the mains' prefixes are evicted and re-prefilled from
scratch. It is the prefix-cache collapse of §3.3b, reproduced deliberately rather than observed
once. Note also that it is *not* visible as an error: the control completed every request.

Earlier turns of the same script, for context — at turn 2, with no subagents, 450K still fits and
**the control keeps its cache too** (`cache_hit_rate` 0.9998 in both arms, TTFT ~2 s in both). The
divergence appears exactly when demand exceeds the pool, which is the point.

**Retrieval and multi-turn on the host-KV arm**, same card:

| Probe | Result |
|---|---|
| Needles, 2 concurrent at 250K | **2 / 2** |
| Needles, 300K / 470K / 520K × depths 0.02 / 0.5 / 0.95 | **9 / 9** |
| Multi-turn, 6 conversations × 7 turns from 60K, +2–4K per turn | **95.56 %** mean cached fraction from turn 1 on; 36 / 36 turns above threshold; **0 errors**; mean TTFT 2.9 s |
| Two mains at 500K each (1.0M tokens live) | runs; see the limits below |
| Two mains at 520K each (1,040,080 tokens live) | runs; the on-GPU control **cannot** — that cell did not execute |

The multi-turn figure is `hit_frac_turn_ge1_mean = 0.9556` from the run's own summary. Turn 0 is
0.0 by construction (cold); turns 1–6 measured 0.9502, 0.9513, 0.9565, 0.9578, 0.9567, 0.9614 —
flat, not decaying. **⛔ Scope: those conversations start at 60,000 tokens and grow to ~80,000 —
they are not the 250–500K main agents.** Do not read 95.6 % as a statement about long mains.

**Headroom during a cold 2 × 250K prefill**, which is a during-load measurement rather than a boot
one: minimum free VRAM **3,414 MiB on the control against 6,962 MiB on host KV** (3.33 → 6.80 GiB).
Non-pool peak over idle was 1,090 MiB on the control and 1,248 MiB on host KV.

#### 3.15c Quality gate: decode-path lossless, composite verdict not assembled

`[measured, Modal, same card]` The host-KV arm was gated against the on-GPU arm as baseline, with
a second baseline run as the noise leg — the §4.3 protocol, 48 prompts × 384 greedy tokens at
concurrency 1 and 4, `claim: lossless`.

| Leg | Verdict |
|---|---|
| Decode-path equivalence, concurrency 1 | **PASS** (agree-median ratio 0.944; mean \|Δlogprob\| 0.0154 against a 0.0155 noise leg, tolerance 0.0232) |
| Decode-path equivalence, concurrency 4 | **PASS** (ratio 1.069; 0.0150 against 0.0149, tolerance 0.0224) |
| Degeneracy | **PASS** — 0 bad of 192 scanned, 0 loops, 0 impossible ids |
| Prefix-cache integrity (gen mode, 12 pairs, 6 cache hits) | **PASS** |
| Long-context claim | **FAIL** |
| **Overall** | **FAIL** |

**Be precise about that FAIL, because it is not a quality result.** The long-context leg fails on
a *scope* rule: declaring `--context-length 540000` obliges the gate to find NIAH cells beyond
262,144 scoring ≥ 0.8 **inside the same bundle**, and this bundle contained only the decode-path
suites — `niah_cells_past_native: 0`. The needles that satisfy the rule were run, passed 9 / 9 at
300K / 470K / 520K, and are in §3.15b; they were simply a separate invocation. So:

- **What is established:** on the decode path, moving the KV pool to host RAM is indistinguishable
  from keeping it on the GPU, at both concurrencies, with no degeneracy and no prefix-cache
  divergence. Accept length is 2.90 candidate against 2.92 baseline at concurrency 4.
- **What is not:** a single assembled verdict covering both the decode path and retrieval past the
  trained window. That is a harness-composition gap, and it is listed in §6.

Do not cite "the host-KV configuration failed its quality gate". Do not cite "it passed", either.

#### 3.15d What the host-KV result does not establish

- **⛔ The validation card is not the production card, and the difference is bigger than PCIe.**
  Modal's is an RTX PRO 6000 Blackwell **Server Edition at 600 W** on **PCIe Gen5**; the reference
  deployment's is a **Max-Q Workstation Edition at 300 W** on **PCIe Gen4** (§ Hardware baseline).
  Double the power limit is a larger confound than the link generation, and **no cross-SKU scaling
  factor was measured** — so treat every absolute tok/s above as belonging to the Server Edition.
  Expect the production card to be slower; by how much is not established here.
- **A cold prefill starves its peer's decode, and host KV is not the cause.** In the two-mains
  cold turn, the main holding a cached prefix decoded at **24.3 tok/s** while the other cold-
  prefilled 200K. The on-GPU control does the same thing at **27.3 tok/s**. So this is chunked-
  prefill scheduling, not the host pool — and it is not fixed by anything in this section.
- **At 2 × 500K one main re-prefills every turn, and the cause is not established.** On turn 2
  main A arrived with no cached prefix and spent 75.1 s on TTFT; on turn 3 it was main B's turn to
  lose it. The pool holds 1,310,720 and the two mains need ~1.0M, so this is not capacity. Decode
  for the main that *kept* its cache fell to **7.1–7.2 tok/s** while its peer cold-prefilled 500K
  — far worse than the 24–27 tok/s at 250K. **2 × 500K is demonstrated to run, not recommended.**
- **The host-KV steady-state VRAM floor is not soaked** (above).
- **Host RAM is now a boot-time failure mode.** See §3.18a.

### 3.16 Bounded QSA prefill gather — the transient that scaled with context

`[measured]` This is the root-cause fix behind the OOM history in §3.14, and it is the single
most important correctness-of-sizing change in this repository.

**The bug.** QSA's prefill path materialised **four full-context transients per full-attention
layer, per chunk** (`index_select` over the request's whole `req_to_token` row, then `torch.cat`).
So peak non-pool VRAM scaled with the **total context of the request**, not with
`--chunked-prefill-size` — which defeats the entire purpose of chunked prefill. A deployment can
lower its chunk size to 4,096, watch the pool sit at 27 % free, and still die in prefill, because
the thing that grew was never the chunk.

The allocation trace makes the "four" literal — at a 382,293-token context in fp8, where one
full-context K or V is 382,293 × 512 B = 186.67 MiB, the old path's allocations ≥ 16 MiB are
`[186.67, 186.67, 186.67, 186.67, 48.0]` against the new path's `[32.0, 48.0]`. **In the host-KV
tree it is two, not four**, and that tree's old peaks are correspondingly half the table below.

Upstream fixed the same bug class in the **DSA** backend in `63d320c723`. QSA had no equivalent.

**Micro-benchmark, isolated kernel, two independent runs** (`m1`, `m2`; same card, same script).
`peak_transient_MiB` is the gather's own allocation, excluding the output:

| Context | KV dtype | Old peak | **New peak** | Old kernel, median ms | **New kernel, median ms** |
|---|---|---|---|---|---|
| 65,536 | fp8 | 128.0 MiB | **32.0 MiB** | 2.642 / 2.609 | **2.507 / 2.481** |
| 131,072 | fp8 | 256.0 MiB | **32.0 MiB** | 2.894 / 2.876 | **2.535 / 2.527** |
| 262,144 | fp8 | 512.0 MiB | **32.0 MiB** | 3.359 / 3.310 | **2.554 / 2.538** |
| 382,293 | fp8 | 746.67 MiB | **32.0 MiB** | 3.698 / 3.683 | **2.575 / 2.564** |
| 524,288 | fp8 | **1,024.0 MiB** | **32.0 MiB** | 4.100 / 4.125 | **2.579 / 2.558** |
| 382,293 | bf16 | **1,495.33 MiB** | **32.0 MiB** | 5.368 / 5.337 | **3.121 / 3.113** |

Two values per cell are the two runs. The old column is a straight line in context length; the new
one is **flat at 32 MiB at every length and both dtypes**. At 524,288 in fp8 the kernel is also
**1.59× faster** (4.100 → 2.579 ms), because it stops writing and re-reading a gigabyte.

**Bit-exactness.** All **12 cases in both runs** report `bit_exact: true`, `max_abs_diff: 0.0`,
`n_diff: 0`, and no NaNs on either side. This is an unusually strong result for a memory
optimisation and it is why the change is unconditional rather than env-gated.

**In-engine A/B, same pool (689,728), same idle VRAM (94,661 MiB), needle-in-a-haystack at three
depths:**

| Prompt | Old peak device used | **New peak device used** | Old TTFT | **New TTFT** | Old prefill | **New prefill** |
|---|---|---|---|---|---|---|
| 100,000 | 96,067 MiB | 96,067 MiB | 7.85–8.23 s | 7.72–8.64 s | 12.2–12.7K tok/s | 11.6–13.0K tok/s |
| 300,000 | **97,149 MiB** | **96,093 MiB** | 29.6–29.7 s | **29.0 s** | ~10.1K tok/s | ~10.4K tok/s |
| 500,000 | **97,237 MiB** | **96,093 MiB** | 66.6–67.2 s | **65.1–65.7 s** | 7.44–7.50K tok/s | **7.61–7.68K tok/s** |

9 / 9 needles pass in both arms and the engine survives both. The number that matters is the
second column: on a 97,887 MiB card the old tree peaked at **97,237 MiB — 650 MiB of headroom
left** at a 500K prompt, while the new tree peaked at 96,093 MiB, leaving **1,794 MiB**. That
**1,144 MiB** is the margin the 2026-09-30 death did not have.

**⛔ And the honest headline: at the production cell it made no measurable difference.** A
2 × 250K cold-prefill peak-VRAM A/B on the host-KV configuration measured the **same non-pool peak
on both trees — +1,248 MiB over idle either way.** The benefit above is real, reproducible and
bit-exact, and it did not show up in the one cell that most resembles the production workload. The
most likely reading is that at 250K the old transient (512 MiB at 262K) is not what sets the peak
on that configuration — something else is — but **that was not established**, and the cell is the
reason this section does not claim an end-to-end win.

**Honest counters.** Two cells get slightly worse and both are small: at a 1,512-token context the
fixed bound costs **4.0 MiB against 2.95 MiB** (the bound is a block, so it has a floor), and the
`wide_int64_slots` case peaks at **64 MiB rather than 32** and measures 3.188 ms against 3.125 ms
in run 1 — within the run-to-run spread, but not an improvement.

**One result did not reproduce and is excluded from the claim.** On the host-UVA cases, run 1
reported the new path at 341–376 ms against the old path's 6.2–12.5 ms — a 30–55× *regression* —
while runs 2 and 3 agree with each other in the opposite direction (4.99–5.68 ms new against
5.70–15.15 ms old) and report a new peak of 159–189 MiB rather than 32 MiB. Runs 2 and 3 agreeing
against run 1 is suggestive of first-touch page-fault cost on the UVA mapping in run 1, but **that
is a hypothesis and the discrepancy is unexplained.** No host-UVA kernel-time claim is made here.

### 3.17 Non-fatal OOM — retract the batch instead of killing the lane

`[measured on Modal, Server Edition card, production flags]` Every incident in §3.14 has the same
shape: one allocation fails, the scheduler raises, `SIGQUIT` follows, and the whole lane — every
healthy in-flight request included — dies for ~9–13 minutes. This makes an EXTEND-batch OOM
retract the offending requests and continue.

An eight-test injection battery, run twice:

| Run | Forced OOMs injected | Real OOMs | Recoveries | Engine alive after every test |
|---|---|---|---|---|
| 1 | 39 | **6** | **45** | **yes** |
| 2 | 39 | **3** | **42** | **yes** |

"Real" means an OOM produced by genuine memory pressure rather than injection; the difference
between the runs is how many of those the same battery happened to provoke. The upstream registered
scheduler suite is **100 passed** on this tree, so the retraction path does not regress admission.
(A first attempt at that suite collected 9 errors from its own harness defect — an `IndexError` in
the test runner — and was re-run; 100/100 is the re-run.)

**⛔ Scope, and it is narrower than "EXTEND only".** The recovery declines unless **nine**
conditions hold: overlap scheduling enabled, `forward_mode == EXTEND`, `tp_size == 1`,
`pp_size == 1`, no MLP sync required, no disaggregation, HiSparse off, PDMux off, and no diffusion-LM
config. It also declines if the device-wide drain it attempts first does not complete.

**Decode OOMs and driver-level errors stay fatal**, deliberately. The handler catches
`torch.OutOfMemoryError` and nothing else, because that is raised by the caching allocator *before*
the failing kernel launches, so the CUDA context is intact. A driver-level failure — the 2026-09-29
`ncclUnhandledCudaError: Failed to CUDA calloc` — leaves the communicator undefined, and surviving
it would serve garbage. `SGLANG_OOM_MAX_REQ_RETRACTIONS` (default 3) bounds how many times one
request may be retracted before it is failed: without a cap, a request whose context can never fit
re-admits and re-OOMs forever, which is no forward progress, no crash and no alert — a worse outage
than the crash it replaced. Measured behaviour at the cap: three recoveries, then `503`, then the
next request gets `200` and the lane is alive.

**Peers across the OOM complete, and they are NOT bit-identical — neither are OOM-free reruns.**
This engine is not run-to-run deterministic in this configuration (`--enable-deterministic-inference`
does not fix it), so the test is two-sided: over 12 runs, reference-versus-reference pairs first
diverged at a mean of **6.0 tokens** with mean \|Δlogprob\| **0.094** on the shared prefix, and
OOM-involved pairs at **5.3 tokens** / **0.084**. Indistinguishable from the engine's own noise.
**Never cite this as "identical".**

**⛔ What was and was not validated on the merged ship tree.** The 39-injection battery above ran
on the `oom-retract` branch **with host KV off** (pool 451,712–451,840), not on the ship tree. The
first ship-tree injection run was **invalidated by a harness defect** — every request failed with
`TypeError: Object of type BatchEncoding is not JSON serializable`, giving `all_injects: 0`,
`all_recoveries: 0`, `verdict_pass: false`. It was **re-run, and passed: 2 injections, 2
recoveries, every needle correct, engine alive at the end.** So the ship tree at the 1.31M host-KV
pool has a passing but **much smaller** sample — two events, not forty-five — plus a clean idle
audit under `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE` (`pool_check_leak: false`, 1,310,656 of
1,310,720 available, `req_pool_ok`, no byte violations).

### 3.18 Model weights on NVMe — boot time, measured

`[measured, production card, both boots from the same log on the same day]` The reference
deployment's 126 GiB of weights moved off a RAID6 spindle array onto NVMe.

| | RAID6 (16:47:48 → 16:57:11) | **NVMe (23:29:34 → 23:34:19)** |
|---|---|---|
| Target model weight load | 288.58 s | **130.67 s** |
| MTP draft weight load | 160.44 s | **20.75 s** |
| Weight load, total | 449.02 s | **151.42 s** (**2.96×**) |
| Wall clock, restart → "fired up and ready" | **563 s** | **285 s** (**1.98×**) |

The weights are **126.0 GiB** (measured on disk) across 206 shards, read twice per boot.

**Do not quote the disk bandwidth ratio as a boot speedup.** An O_DIRECT 4 GiB sequential read of
the same shard is recorded at **5.8 GB/s on NVMe against 293 MB/s on the array — 19.8×** — but the
boot phase it governs improved **2.96×**, and total boot **1.98×**. Checkpoint loading is not a
sequential read: it is 206 shards through a quantizing loader, and the array was never the only
cost. The 19.8× is a property of the filesystems; 2.96× is what it bought.

⚠️ **Provenance of that pair is weaker than everything else in this section.** The method is
recorded but **no result artifact was retained** — it is a self-asserted measurement, and a second
figure for the same array, **290.6 MB/s**, is on record from a different method (seek-bound,
149–159 KiB requests against a 512 KiB RAID chunk, 9.5–13.6 ms waits), which does carry its
supporting detail. Treat 293 MB/s and 5.8 GB/s as approximate and unreproduced. The boot-time
figures either side of them are read from the lane's own log and are not affected.

An earlier estimate of 3.6–4.1 minutes for the post-move boot was **optimistic**: the measured
figure is **4 min 45 s**, because the estimate did not include the ~110 s the host-KV pinning and
CUDA-graph capture phase takes after the last weight lands. The measured number supersedes it.

#### 3.18a Host RAM is a new boot-time failure mode

`[measured, 2026-09-30]` The cutover to host-resident KV did not come up first time. Two boots
were killed with `status=9/KILL` — at 23:23:09 and 23:28:58 — before the 23:29:34 boot succeeded.
The unit recorded a **202.1 GiB memory peak** on a **251 GiB** host, and the failed boot at
23:23:44 had already finished loading weights (`Load weight end. elapsed=184.70 s`) and died
afterwards, in the phase that pins the host KV.

The lane is configured `MemoryMax=infinity`, so nothing bounded it but the host. The mechanism to
take away: **pinned pages are neither swappable nor reclaimable**, so a host-resident KV pool
converts a VRAM budget into a host-RAM budget, and the host-RAM budget has to be reasoned about
with the same care §3.13 applies to VRAM — including whatever page cache the previous instance
left behind. `SGLANG_QSA_HOST_KV_MAX_GB` fails fast on a *mis-sized pool*; it does not protect
against the host simply being full at the moment of the pin.

Weight-load time is also not stable under that pressure: the boot that died took **184.70 s** to
load the target weights where the boot that succeeded took **130.67 s**, on the same NVMe.

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
| The native-RoPE retrieval ceiling | **not found.** §3.8a passes through 672K (2.56×) with 14,228 tokens of pool to spare; the pool binds before position does. |
| Whether the reference card's 500K empty completion was capacity | **probable, not confirmed** — 500K has not been re-run on the reference card (§3.8a). |
| A quality gate or accuracy benchmark at any length past 262,144 | **not run.** §3.8 establishes retrieval, recall and short-context equivalence, not accuracy at length. |
| Reasoning at length beyond n = 2 per cell | **not run.** §3.8c is a signal per cell, not a rate. |
| Prefix-cache hit rate and TTFT on the production card at 36 slots, under its real workload | **not yet measured.** §3.12 measures only the resulting pool. |
| The lowest safe `--mem-fraction-static` for this workload | **still not bracketed.** 0.99 and 0.98 both died (§3.14); 0.97 has ~1.74 GB device-free while serving and has not. Nothing between 0.97 and 0.98 was tried, and no headroom figure is known to be *sufficient* — only that 287 MiB is not. The host-KV cutover (§3.15) and the bounded gather (§3.16) both enlarge the margin at 0.97, which makes the question **less urgent and no more answered**: the bracket was not re-attempted after either change. |
| How much prefill working set a queue of depth N actually needs | **not measured.** §3.14 shows a 14-deep queue exhausting ~287 MiB of headroom; the relationship to queue depth, chunk size and prompt length was not characterised, so 0.97 is a measured-safe point rather than a derived margin. |
| Whether the production OOM recurs at 0.97 under a deeper queue | **open.** Peak queue depth on the incident day measured 16; 0.97 has not yet been exercised at that depth. |
| A quality gate at any length past 262,144 | **not run.** Retrieval passes are a capability probe, not a certified operating point. |
| Wall times for the ~200k and ~300k native-RoPE needle cases | **not recorded** in the run ledger; omitted rather than estimated. |
| Why the retrained MTP head won on a B200 under plain upstream SGLang (+1.8 % to +3.7 % in all four cells) and was flat on this tree's SM120 fused path (2.1076 vs 2.1079) | **cause not established.** GPU generation and the fused decode path were never isolated from each other. |
| Downtime duration of the 2026-09-29 16:45:24 constrained-decoding OOM | **not recorded** in the sources consulted; the failure chain and the mitigation are, so no duration is claimed. |
| The 900k mixed-stream failure, after the OOM fix | **not re-run.** |
| Why the long primary never gets a radix-cache hit | **cause not established.** |
| Accept-length decay over uptime | snapshots recorded per suite; **no trend analysis done.** |
| Whether `fp4_gemm_runner_backend='auto'` resolves to a safe kernel on SM120 | **not traced.** Flagged, not asserted. The published launch command leaves `--fp4-gemm-backend` unset — which resolves to `auto`, because that is what every measurement here ran on. `flashinfer_cudnn` is the one setting on record as known-safe against the CUTLASS FP4 GEMM race; pinning it is a deliberate departure from the measured configuration, not the default. |

Added by the 2026-09-30 cutover (§3.15–§3.18):

| Item | Status |
|---|---|
| Steady-state device-free VRAM on the host-KV configuration | **not soaked.** One reading of 5,863 MiB about six minutes after boot (§3.15), against §3.13's 13 h 22 min flat floor for the on-GPU configuration. The comparison is boot-fresh against soaked and should not be read as a like-for-like margin. |
| Why one main re-prefills every turn at 2 × 500K | **cause not established** (§3.15d). Not capacity: the pool holds 1,310,720 and the two mains need ~1.0M. It alternates between the two mains across turns. |
| Cross-SKU scaling from the validation card to the production card | **not measured.** Modal is a Server Edition at 600 W on PCIe Gen5; production is a Max-Q at 300 W on Gen4 (§3.15d). Every absolute tok/s in §3.15b belongs to the Server Edition. No factor is published because none was measured. |
| Non-fatal OOM on the **merged ship tree** at any useful sample size | **2 events.** The 39-injection battery ran on the `oom-retract` branch with host KV off; the ship-tree re-run passed with only 2 injections / 2 recoveries (§3.17). Not a comparable sample. |
| Why the bounded gather showed no peak-VRAM benefit at the production 2 × 250K cell | **cause not established** (§3.16). Both trees measured +1,248 MiB over idle. The micro and needle A/Bs both show the benefit; this cell does not. |
| Steady-state hot-cache hit rate as a function of workload | **one steady reading per card.** 0.8121 on the production lane at 11 minutes, 0.92 cumulative on the second card. No sweep over context mix, and the first-30-s figure (0.9734) is an artifact of cumulative counters (§3.15). |
| NVMe-versus-array sequential bandwidth | **artifact not retained** (§3.18). Method recorded, result self-asserted, and a second figure for the array (290.6 MB/s) disagrees slightly from a different method. |
| Host-UVA kernel time under the bounded gather | **did not reproduce.** Run 1 disagrees with runs 2 and 3 by 30–55× in the opposite direction, and reports a different peak (§3.16). Unexplained; no claim made. |
| A single assembled quality verdict covering both the decode path and retrieval past 262,144 | **not built.** Both legs pass in separate bundles; the composite verdict returns FAIL on a scope rule, not on a measurement (§3.15c). |
| Whether the host-KV lane survives a deep queue at 0.97 | **not exercised.** The bounded gather removes the transient that killed §3.14's boot, and non-fatal OOM is meant to absorb what is left, but neither has been tested against a 14–26-deep queue on the production card. |
| Prefix-cache hit rate and TTFT on the production card under its real workload | **still not measured**, at any memory fraction or pool size. §3.15's hot-cache hit rates are host-KV cache hits, which is a different quantity. |
| How much host RAM the pinning phase needs at boot, and under what contention | **not characterised.** Two boots were OOM-killed before one succeeded (§3.18a); the relationship to page cache left by the previous instance was not measured. |
| Whether `SGLANG_QSA_HOST_KV_CACHE_SETS` 2048 or 4096 is the better trade | **not A/B'd on one boot.** 4096 measured 0.9734 cumulative hit for 0.94 GiB of GPU; 2048 was reported at ~0.92 for half the VRAM, on a different run. |

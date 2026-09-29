# qgate — the quality gate behind BENCHMARKS §4

This is the harness that produced every quality verdict in
[BENCHMARKS.md §4](../../BENCHMARKS.md#4-quality-instruments--what-each-one-can-and-cannot-see).
It talks to any OpenAI-compatible **SGLang** server over HTTP. It uses `/v1/chat/completions`,
plus `/generate` for raw-token-id requests, and `/get_server_info` and `/v1/loads` for the
server fingerprint and load back-off. It never starts, stops or reconfigures a server.

**Included:** the harness, its wrapper scripts, a data fetcher, and the IFEval checkers
(Apache-2.0, see `ifeval_lib/NOTICE`).

**Not included:**
- the run data behind the published verdicts (baseline and candidate result files);
- any benchmark corpus;
- a tokenizer copy.

`fetch_data.sh` downloads the public datasets from their publishers and checks them against
the content checksums of the files we used. The private code corpus used for some NLL and
reference-agreement windows is **not** published. It is optional (see below).

---

## The one methodological point that matters

**Teacher-forced NLL cannot see decode-time levers, so it cannot certify them.**

- NLL is scored in prefill: a whole window is pushed through the model in 4,096-token chunks,
  so every kernel runs at `T >> 16` tokens per step.
- The levers that make decode fast run only at `T <= 16`: fused MoE and hyper-connection
  tails, skinny decode GEMV, and the speculative-verify graph shapes. They **never execute**
  during an NLL measurement.
- A fused build is therefore bitwise identical to the stock build on single-chunk NLL.
  That tells you nothing about decode.

Two further measurements make a fixed NLL budget worse than useless:

1. **Beyond one prefill chunk, prompt logprobs are nondeterministic run to run**, even for
   the unmodified engine. About 88 % of tokens differ at ≥ 4k between two runs of identical
   prefill math.
2. **A single chunk (ctx 512) is bit-deterministic** (max token |Δ| exactly 0.0 across runs).

The gate is built accordingly:

| Instrument | Role |
|---|---|
| **decpath**, the decode-path test | **Primary for anything that touches decode.** 48 real prompts, all under 500 tokens, so prefill is one deterministic chunk. Greedy decode of 384 tokens with **output** logprobs only, 2 repeats, at concurrency 1 and 4. Agreement length and \|Δlogprob\| over the agreeing prefix are compared with a measured baseline-vs-baseline spread. |
| **nll at ctx 512** | Hard, near-exact check for prefill and weight changes. A lossless claim FAILs if any token moves by more than 1e-4. **Blind to KV-cache dtype**: a single-chunk prefill attends over fresh K/V, not the cache. |
| **nll at ≥ 4k** | Judged **only** against a measured spread from a repeated baseline run: z-test plus a practical floor. With no repeated baseline the verdict is `UNCALIBRATED`, never FAIL. |
| accuracy suites, niah, degen, prefix | Task accuracy, retrieval, degeneracy and radix-cache integrity (below). |

---

## Setup

```bash
pip install tokenizers pandas pyarrow scipy langdetect immutabledict nltk
./fetch_data.sh                              # public datasets -> ./data, then content checksums
cp /path/to/checkpoint/tokenizer.json tokenizer/tokenizer.json   # or: export QGATE_TOKENIZER=...
```

- The IFEval checkers download the NLTK `punkt_tab` tokenizer on first import.
- The needle suite draws key words from `/usr/share/dict/words`; set `QGATE_WORDS` to use
  another word list.
- **The tokenizer must be the served model's.** The raw-id suites (`nll`, `decpath`, `prefix`,
  `refagree`) build token ids locally, so every server scores identical bytes.
- **Impossible token ids.** `QGATE_VOCAB_SIZE` (default 248077, the Qwen3.8 tokenizer) sets the
  first id no healthy model can emit. Set it for another model.

Everything is plain Python standard library plus the packages above. There is no GPU code in
the harness.

---

## Commands

### 1. Baseline (once per hardware class)

```bash
python3 qgate.py run --url http://127.0.0.1:30000 --model <served-name> --label base_A --out runs/base_A.json
python3 qgate.py run --url http://127.0.0.1:30000 --model <served-name> --label base_B --out runs/base_B.json
```

Run it twice on the **unmodified** configuration. `base_B` versus `base_A` is the noise floor,
and every threshold below is relative to it. As a sanity check, `compare` of A against B must
itself give PASS.

### 2. Candidate verdict

```bash
QGATE_BASELINE=runs/base_A.json QGATE_NOISE=runs/base_B.json \
  ./qgate_verdict.sh http://127.0.0.1:30000 <served-name> my_change lossless quick   # or: lossy, full
```

- `quick` takes about 12–15 minutes on a dedicated card. It runs gsm8k, humaneval, niah to 64k,
  nll, degen and the prefix probe.
- `full` takes about 45–60 minutes.
- Exit codes: `0` for PASS/WARN, `1` for FAIL, `2` for INVALID.

### 3. Decode-path test (about 5 minutes per run)

```bash
./qgate_decpath.sh run http://127.0.0.1:30000 <served-name> BASE    # unmodified
./qgate_decpath.sh run http://127.0.0.1:30000 <served-name> BASE2   # unmodified again (same boot is fine)
./qgate_decpath.sh run http://127.0.0.1:30000 <served-name> CAND    # candidate
./qgate_decpath.sh verdict BASE CAND BASE2
```

### 4. Strict mode (certifying lossy changes to within about 2 points)

```bash
./run_strict_ref.sh http://127.0.0.1:30000 <served-name>            # two reference runs, ~3 h each
QGATE_BASELINE=runs/strict_ref_R1.json QGATE_NOISE=runs/strict_ref_R2.json \
  ./qgate_verdict.sh http://127.0.0.1:30000 <served-name> my_change lossy strict
```

Strict mode adds the following. The reference-agreement suite teacher-forces the candidate on the
reference's own greedy continuations, so both servers score identical tokens.

| Suite | Size |
|---|---|
| MMLU-Pro | 1,200 items, stratified by category |
| HumanEval+ | all 164 |
| MBPP+ | all 378 |
| GSM8K | all 1,319 |
| IFEval | all 541 |
| reference agreement | 4k, 32k and 128k context |
| degenerate generations | 200, at production sampling |

Strict mode prints one of:
- `LOSSLESS-EQUIVALENT`: everything is within the measured noise;
- `LOSSY-ACCEPTABLE`: no drop of more than 2 points can be established;
- `NOT-ACCEPTABLE`: a hard failure;
- `NOT-CERTIFIABLE`: no failure, but the sample is too small to exclude a 2-point drop.

---

## Verdict labels

| Label | Meaning |
|---|---|
| `PASS` | Within the measured noise, or no significant regression. |
| `WARN` | Borderline: one condition of a FAIL met but not both (e.g. p < 0.05 without the practical-size condition), or a count above the reference that is still within its allowance. Report it; do not treat it as a pass. |
| `FAIL` | A regression that is both statistically significant and larger than the practical floor. Also any impossible token id. |
| `INVALID` | **More than 5 % of a suite's items errored or were skipped unexpectedly.** That suite is dropped from every verdict and the run exits 2. A broken tunnel or a dead server is not a model result, so it is never scored PASS or FAIL. |
| `BITWISE` | decpath only: every output identical to the baseline. |
| `UNCALIBRATED` | No repeated-baseline run to measure noise against (multi-chunk NLL, or decpath without divergence data). No PASS or FAIL is issued. |
| `NOT_MEASURABLE` | The server cannot admit the test. For example, a context-length claim past native whose KV pool cannot hold such a request. |
| `NOT_COMPARABLE` | Baseline and candidate share no measured items for that suite. |

"Designed" skips do not count toward the 5 % rule: a context longer than the server's, prompt
logprobs refused on a protected server, or no private corpus configured. A suite made entirely
of designed skips is marked `not_measured`.

---

## Suites

| Suite | Data | Default n | Metric |
|---|---|---|---|
| gsm8k | GSM8K test | 150 (1,319 in strict mode) | exact numeric match, greedy |
| ifeval | Google IFEval, vendored official checkers | 120 (541) | prompt-level strict |
| humaneval | HumanEval | 80 | pass@1 greedy, tests run in a subprocess under rlimits |
| heplus, mbppplus | EvalPlus HumanEval+ v0.1.10, MBPP+ v0.2.0 | 164, 378 | pass on base plus augmented inputs, judged against the canonical outputs |
| mmlupro | TIGER-Lab MMLU-Pro | 1,200 | letter match |
| niah | Gutenberg haystack; 1 target and 3 distractor keyed needles | 5 lengths × 5 depths × 2 | exact 7-digit match; `confused` means it returned a distractor |
| nll | Gutenberg, plus the optional private corpus | per corpus × depth × `--nll-starts` | teacher-forced NLL of a 64/128-token window |
| decpath | GSM8K, HumanEval and IFEval prompts under 500 tokens | 48 × 2 reps × 2 concurrencies | agreement length, \|Δlogprob\| over the agreeing prefix |
| refagree | documents plus a question, at 4k, 32k and 128k | 8 per length | teacher-forced \|Δlogprob\| on the reference continuation |
| degen | GSM8K, HumanEval and IFEval prompts | 60 (200) | production sampling (T 1.0, top-k 20, top-p 0.95). An output is degenerate if it has an impossible id, a run of 8+ token 0, any token repeated 32+ times, a periodic tail loop, or garbled text. |
| prefix | 8k, 24k and 49k prefixes | 3–6 | generation-only radix-cache integrity probe: cold, warm and late repeats |

Every output from every generative suite is also checked for degeneracy.

The harness measures its own limits (BENCHMARKS §4.6). The smallest detectable drop at 80 % power
comes from the discordance between two baseline runs:

| Suite | n | Smallest detectable drop |
|---|---|---|
| MMLU-Pro | 1,200 | 1.9 points |
| GSM8K | 1,319 | 0.6 points |
| All binary suites pooled | ~3,600 | about 1 point |
| HumanEval+ and MBPP+ together | 542 | 3.0 points |

Code alone does not reach 2 points with the public items.

---

## Safety on a live server

- **Load back-off.** Every request first reads `/v1/loads`. It waits while the lane has queued
  requests, or while the KV pool could not hold the request under `--pool-frac`. Long needle
  cells run one at a time.
- **Prompt-logprob requests are dangerous on a memory-tight server.** They build a
  `scored_tokens × vocab` logits tensor outside the KV pool.
  - List such servers in `QGATE_NO_PROMPT_LOGPROBS` as comma-separated URL roots. Prompt-logprob
    suites (`nll`, `refagree` teacher forcing) then record a designed skip for them.
  - `decpath`, `prefix` and every accuracy suite use output logprobs only, and are safe anywhere.
  - NLL scores 64–128-token windows only, never the whole prompt.
- `--degen-conc` should equal the server's `max_running_requests` on a dedicated instance, so
  every batch shape the server can form actually occurs during the test.

## Optional private corpus

Public texts are memorised (Gutenberg NLL is 0.04–0.10 nats/token at ctx 512), so they say
little about how well a model uses context. Set `QGATE_CODE_CORPUS=/path/to/text` to add windows
from a document you know was never trained on; our published NLL figures used a private code
corpus.

- Without it, `nll` scores Gutenberg only.
- The prefix probe and `refagree` fall back to Gutenberg text.
- Numbers are only comparable between runs that used the same corpus.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `QGATE_DATA` | `./data` | where `fetch_data.sh` put the datasets |
| `QGATE_TOKENIZER` | `./tokenizer/tokenizer.json` | the served model's tokenizer |
| `QGATE_MODEL` | `default` | served model name when `--model` is omitted |
| `QGATE_VOCAB_SIZE` | `248077` | first impossible token id |
| `QGATE_CODE_CORPUS` | unset | optional private text for nll, prefix and refagree |
| `QGATE_NO_PROMPT_LOGPROBS` | unset | URL roots that must never receive prompt-logprob requests |
| `QGATE_WORDS` | `/usr/share/dict/words` | needle key words |
| `QGATE_BASELINE`, `QGATE_NOISE`, `QGATE_NLL_NOISE`, `QGATE_PREFIX_REF`, `QGATE_RUNS`, `QGATE_CONC`, `QGATE_N_MMLUPRO`, `QGATE_NIAH_LENGTHS` | — | wrapper inputs (see each script's header) |

## Known weaknesses

These are the harness's limits, stated before you rely on it (BENCHMARKS §4.7):
- The needle test is a lexical-match proxy and saturates at 100 %.
- The prefix probe detects cache poisoning if it happens during a run, but does not provoke the
  memory pressure that causes it.
- A kernel bug at a batch shape the harness never forms is not caught.
- GPQA-Diamond is not included, because the dataset is gated.

# Idle-gap background filler

Fill the quiet periods between interactive turns with a low-priority background
batch job, so idle GPU time produces useful output without slowing the
interactive sessions. This note records a full A/B measured on one RTX PRO 6000
Blackwell Server Edition (600 W) on Modal, with the standard S8 + FP8-dense
build and flags (chunked prefill 4096, NEXTN 3/1/4, host QSA KV pool 1,310,720
tokens, priority scheduling on, mamba cache 48, 8 running requests), with and
without `--prefill-decode-interval 8`. Temperature 0.

## Design

- Interactive sessions are the protected workload: realistic agent-like sessions
  at 1/2/4/8 concurrent streams, contexts of 32K / 131K / 500K, think and
  tool-call gaps drawn from a 1-60 s log-uniform distribution. Submitted at a
  high request priority (10).
- The filler is a low-priority (-10) batch worker (long generations from a fixed
  prompt set). It is submitted only while no interactive turn is active, and any
  in-flight filler request is aborted (`/abort_request`) the moment an
  interactive turn arrives.
- Because the filler priority is below the server default (0), both interactive
  traffic and ordinary default-priority traffic preempt it.

## Results (4 interactive streams, 3 repeats, mean +/- 95% CI)

| Arm | Interactive first-token p90 (s) | Interactive decode tok/s | Total tok/s | Filler tok/s | Cache-hit rate |
|---|---|---|---|---|---|
| Off (no filler) | 61.4 +/- 2.0 | 143.7 +/- 20 | 23.1 +/- 6.7 | 0 | 0.10 |
| Idle-gated filler | 61.5 +/- 2.2 (+0.2%) | 146.5 +/- 16 | 42.1 +/- 6.4 (+82%) | 17.7 +/- 4.2 | 0.04 |
| Always-on filler | 62.6 | 97.1 (-32%) | - | 39.7 | - |

First-token p90 is dominated by the cold 500K/131K prefills that both arms pay
equally, so it is the control for the filler's effect; the longest single-stream
stall was identical across off and idle (65.7 s). The always-on row shows why
gating matters: with the filler co-occupying decode batch slots it cuts
interactive decode by a third even though it is lower priority. Priority alone
orders the queue; it does not stop an already-running request from sharing each
decode step, so the filler must be submitted only in idle gaps.

## Filler auto-throttles with interactive load

| Interactive load | Interactive decode tok/s | Filler tok/s | Filler degenerate |
|---|---|---|---|
| 1 stream | 283 | 145 | 0 |
| 2 streams | 204 | 149 | 0 |
| 4 streams | 146 | 18 | 0 |
| 8 streams | 21 | ~0 | 0 |
| Saturation (12 small sessions, tight gaps) | - | 12.1 | 0 |

At 8 streams the server is interactive-saturated, so the filler gets essentially
no idle gap and yields entirely (2 jobs, both preempted). Under a near-saturated
load it still adds about 12 tok/s (about 6% more output) at under 10%
first-token cost (p90 9.5 -> 10.25 s).

## Preemption, KV and quality

- Preemption latency (abort to stream close): p50 0.68 s, p90 2.24 s, max
  2.42 s. A filler prefill chunk already in flight is not interrupted, which
  sets the ~2.4 s tail, but it lands inside interactive think-gaps so
  first-token p90 is unchanged. Resume is a fresh low-priority job.
- 500K follow-up: filler running in the think-gaps of a 500K session does not
  evict the 500K prefix. Follow-up first token 2.5-2.9 s (off: 2.7-3.2 s), about
  500K tokens still served from cache, while the filler completed 22,301 tokens
  in those gaps. Peak pool use unchanged (0.53), no memory pressure.
- The one measurable cost is a lower overall cache-hit rate in mixed 4-stream
  traffic (0.10 -> 0.04); it did not cause a first-token regression here.
- Quality under filler load: interactive needles 9/9 (131K/262K/500K at depths
  0.1/0.5/0.9), GSM8K 96/100. Filler output never degenerate (0 flagged across
  ~33K filler tokens: solo, idle, saturation and quality runs).

## Verdict and how to implement

SHIP, idle-gated. Criteria met: interactive first-token p90 within 10% (idle
+0.2% at 4 streams, +7.9% under saturation), decode unchanged, no quality loss,
filler not degenerate. Condition: it must be idle-gated, not always-on.

Implementation: use the SGLang priority scheduler
(`--enable-priority-scheduling --default-priority-value 0
--radix-eviction-policy priority`). Run the filler at a priority below the
server default so interactive and default-priority traffic preempt it; add an
idle gate at the scheduler or job-queue level that submits filler work only when
no interactive turn is active; abort in-flight filler with `/abort_request` when
an interactive turn arrives. Do not co-run the filler during active interactive
decode.

## With `--prefill-decode-interval 8`

Repeating the 4-stream A/B with `--prefill-decode-interval 8` (3 reps) leaves the
tradeoff unchanged: idle-gated filler adds +82% total output (24.2 -> 44.0
tok/s), interactive first-token p90 is within noise (63.0 -> 62.7 s, -0.5%),
peak pool use is unchanged (0.52 -> 0.48), and the filler stays non-degenerate.

Follow-up: all numbers here are on the 1,310,720-token host pool. For a larger
pool (with or without `--prefill-decode-interval 8`), re-run the same A/B on a
matching configuration before relying on it there; pool size is the one
variable this study did not vary.

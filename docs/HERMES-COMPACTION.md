# Hermes context compaction: blocking vs background (design note)

Status: proposal only. Hermes was not modified. Measured 2026-10-28 (ET) on Modal, 1× RTX PRO 6000,
S8-FP8 profile (max-running 8, chunk 4096, mamba 48, NEXTN 3/1/4, host KV pool 1,310,720 tokens).
Raw data: `flashnext-modal-bench/results_trough/r20261028tr/hm_*`. Benchmark detail: BENCHMARKS.md §3.24.

## Scenario

There are 8 sessions on one server, matching 4 panes × 2 workers. One Hermes session (H) sits at about 250K tokens and
hits its compaction threshold. The other 7 run agent loops at 131K, 65K and 5× 32K contexts, with think gaps of 1–15 s.

- **Blocking (today's behaviour):** H stops, the summarizer runs, then H resumes on summary + last turn.
- **Background:** the summarizer runs on a snapshot of H's history while H keeps working on its full context.
  When the summary arrives, H swaps to summary + everything it produced since the snapshot, verbatim.
- **Warm summarizer:** the summary request reuses H's exact prompt prefix, so the server's prefix cache already holds it.
  **Cold summarizer:** a different prompt, which is what Hermes does today: `_serialize_for_summary` sends a re-serialized
  transcript to the auxiliary model, so all ~250K tokens are prefilled again.

## Results (250K session, 8 workers)

| Server flags | Mode | Summarizer | H stall (longest gap, s) | Summary time (s) | Swap first token (s) | Others' longest gap (s) | Peak KV (tokens, % of pool) |
|---|---|---|---|---|---|---|---|
| base | blocking | warm | 7.1 | 6.7 | 0.35 | 0.88 | 721,920 (55%) |
| base | background | warm | **2.7** | 5.7 | 0.69 | 0.41 | 718,400 (55%) |
| base | blocking | cold | 31.4 | 31.1 | 0.36 | 24.4 | 723,072 (55%) |
| base | background | cold | 26.3 | 35.2 | 1.31 | 25.3 | 940,416 (72%) |
| interval 8 | blocking | warm | 8.3 | 7.9 | 0.35 | 0.80 | 685,952 (52%) |
| interval 8 | background | warm | **2.9** | 9.8 | 0.86 | 0.38 | 687,360 (52%) |
| interval 8 | blocking | cold | 38.2 | 37.5 | 0.67 | 0.79 | 754,624 (58%) |
| interval 8 | background | cold | 26.8 | 36.2 | 1.47 | 0.83 | 916,672 (70%) |
| interval 8 | background | cold, summarizer priority 0 / H priority 10 | 21.8 | 37.5 | 1.41 | 1.93 | 926,464 (71%) |

"interval 8" means `--prefill-decode-interval 8`. After compaction, H's turns run at 0.7–1.0 s TTFT on a 9–25K prompt.

What this shows:

1. **Background compaction only pays off when the summarizer shares H's prefix.** Warm, H's stall falls from 7–8 s to about 2.8 s (−60%).
   Cold, H's next turn needs about 7K new tokens of prefill. That prefill queues behind the summarizer's 250K prefill, because chunked prefill
   serves one request at a time, so H stalls 26 s, almost as long as blocking (31–38 s).
2. **Priority helps only partly.** With the summarizer at low priority, H's first turn went ahead of it (TTFT 2.8 s). H's next turn
   still waited 19.7 s, because a summarizer prefill already in progress is not pre-empted. H's stall was 21.8 s, against 26.8 s at equal priority.
3. **Cold background uses about 200K more peak KV** (70–72% of the pool vs 52–58%), because the full context and the summarizer's copy are both live.
4. **The server flag matters more for everyone else.** With interval 8, the other 7 sessions never stalled more than 2 s during a cold compaction.
   On base flags they froze for 24–25 s.

## Proposal

```yaml
compression:
  mode: auto            # blocking | background | auto
  background_start_ratio: 0.8   # start the background summary early so it lands before threshold (0.9)
  summarizer_priority: 0        # sent as request priority; foreground turns use 10
  auto_fallback_queue_depth: 4  # auto -> blocking when server num_queue_reqs >= this
```

- **Prefix-sharing summary request (the most important change).** In background or auto mode, build the summary request as
  H's live message list plus a final "summarize the above" user message, instead of a re-serialized transcript. That turns the 25–30 s cold prefill into a cache hit
  of about 258K tokens and a 1.5 s first token. This assumes the summarizer runs on the same server/model as the session. If it uses a different model, no prefix can be shared.
- **auto:** use background only when (a) the summarizer can share the prefix and (b) the server's `sglang:num_queue_reqs` is below the threshold and KV usage
  leaves room for the extra copy (`token_usage < 0.6`). Otherwise fall back to blocking, which is no worse on stall when cold and uses less KV.
- **Swap:** the replacement context is summary + every message since the snapshot, verbatim. If the summary lands mid-turn, swap at the next turn boundary.
- **Hotkey:** `Alt+K` (escape, k) cycles blocking → background → auto. The key is currently unbound in `cli.py`. Also add `/compress bg` as a one-shot background compaction next to the existing `/compress here [N]`.

Pros: about 2.5× shorter stall when warm, and the session keeps producing output (about 380 tokens during a 5.7 s summary). Combined with
`background_start_ratio`, the user may see no stall at all.

Cons: it needs prefix sharing to help. It uses more KV while both copies are live. The swap prompt is a bit larger (7–9K vs about 1K tokens), so the first
turn after the swap costs 0.7–1.5 s instead of 0.35 s. The summary also misses whatever happened after the snapshot, which is why that tail is kept verbatim.

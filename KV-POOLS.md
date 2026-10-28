# KV-POOLS — bigger host-RAM KV pools with the GPU hot cache

How to raise the KV pool of the S8-FP8 profile ([CONTEXT-PROFILES §4b](CONTEXT-PROFILES.md#4b-profile-s8-fp8-8-bit-dense-cap-8--host-ram-kv-eight-streams))
past its benchmarked 1,310,720 tokens, what it costs in host RAM and GPU memory, and what it buys.
All numbers are measured on one RTX PRO 6000 Blackwell Server Edition (600 W) on Modal, temperature 0,
S8-FP8 flags with `--context-length 540000`, `--max-mamba-cache-size 48`, `--max-running-requests 8`,
host QSA KV and 4,096 hot-cache sets (runs `r20261028kv`, 2026-10-28), plus boot checks on a 300 W
Max-Q card serving other traffic at 2,097,152 and 4,194,304 tokens. No code change is needed: this is flags and one environment variable.

## 1. Short version

| Pool `--max-total-tokens` | `SGLANG_QSA_HOST_KV_MAX_GB` | Host RAM pinned (target + draft) | Free GPU after boot | 500K sessions kept cached | Use it when |
|---|---|---|---|---|---|
| 1,310,720 (S8-FP8 default) | 24 (default 100 is fine too) | 15.00 + 1.25 GiB | 9.61 GB | 2 | baseline |
| 1,572,864 | 24 | 18.00 + 1.50 GiB | 9.37 GB | 3 | cautious middle step |
| **2,097,152 (recommended)** | **28** | **24.00 + 2.00 GiB** | **8.97 GB** | **4** | several long agents at once |
| 4,194,304 | 50 | 48.00 + 4.00 GiB | 7.14 GB | 8 | eight long agents at once (boot on a 300 W Max-Q card matched Modal) |

The pool buys **room, not speed**. Decode speed is the same at every size measured, 1.31M to 4M (section 3). What
changes is how many long sessions stay resident: at 1.31M a third 500K session evicts the others and
their next turn re-reads 500K tokens (52–107 s to first token); at 2M four 500K sessions stay cached
and follow-ups start in 2.5–3.5 s; at 4M eight do (2.6–7.0 s with all eight active).

## 2. Flags and environment

The pool lives in pinned host memory (UVA): **12,288 bytes per token** for the 12 full-attention
layers' FP8 K/V, plus **1,024 bytes per token** for the NEXTN draft layer's own pool. The GPU keeps only
the QSA index keys and the hot cache, so the GPU cost of a bigger pool is small (0.24 GB at 1.5M,
0.64 GB at 2M).

```bash
# S8-FP8 exports from CONTEXT-PROFILES §4b, then for the 2M pool:
export SGLANG_QSA_HOST_KV_MAX_GB=28     # guard, checked PER POOL; the 12-layer pool is 24.00 GiB at 2M
# ...and in the launch line, replace --max-total-tokens 1310720 with:
  --max-total-tokens 2097152
```

`SGLANG_QSA_HOST_KV_MAX_GB` is a safety guard, not a reservation: boot refuses a host pool larger
than this many GiB. It applies to each pool separately, so size it to the 12-layer target pool:
`tokens × 12,288 / 2^30`, rounded up with a little slack (1.31M → 15.0, 1.5M → 18.0, 2M → 24.0 GiB).
Keep `--context-length 540000` (section 5).

Boot log lines to check:

```
QSA host KV: ... 24.00 GiB ...            # target pool (2M)
QSA host KV: ... 2.00 GiB ...             # draft pool
max_total_num_tokens=2097152
available_gpu_mem=8.97 GB                 # after CUDA-graph capture
```

## 3. Measured (Modal, one GPU per run)

| Pool | Decode, 1 / 2 / 4 / 8 short streams (tok/s, aggregate) | 131K: decode, first token | 500K: decode, first token | Needles 131–500K | GSM8K n=100 | Peak host RSS (container) |
|---|---|---|---|---|---|---|
| 1,310,720 | 274.9 / 402.8 / 609.3 / 811.2 | — | — | 9/9 (earlier run) | 0.97 (earlier run) | 93.0 GiB |
| 1,572,864 | 273.4 / 402.1 / 596.3 / 802.7 | 243.6 tok/s, 14.2 s | 227.2 tok/s, 53.0 s | 9/9 | 0.97 | 96.5 GiB |
| 2,097,152 | 273.9 / 403.2 / 604.5 / 804.9 | 241.0 tok/s, 12.2 s | 230.9 tok/s, 53.2 s | 9/9 | 0.98 | 103.1 GiB |
| 4,194,304 (same-container A/B, run `r20261028ab`) | 273.0 / 405.3 / 599.4 / 807.7 (2M in the same container: 278.7 / 403.0 / 603.7 / 807.5) | 230.6 tok/s, 12.4 s | 225.5 tok/s, 53.0 s | 9/9 | 0.97 | — |

Several 500K sessions decoding at once (total decode tok/s from the server's own decode log; the
benchmark client counts streamed chunks, about 3 tokens each with the draft model, so client-side
per-stream rates under concurrency read low):

| Concurrent 500K sessions | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| 1.31M pool | 224.6 | 342.3 | (third session evicts: next first token 52.5 s and 106.9 s instead of 2.5 s) | — | — | — |
| 1.5M pool | 229.8 | 317.7 | 396.7 | — | — | — |
| 2M pool | 237.3 | 335.6 | 385.3 | 429.3 | — | — |
| 4M pool | 233.3 | 350.6 | — | 456.3 | 444.9 (slow container, one sample) | all 8 cached; total rate not logged |

Follow-up first token for a session that stayed cached: 2.45–3.53 s at 2M with 4 × 500K resident. At 4M with 8 × 500K resident: 2.6–7.0 s, all eight fully cached (run `r20261028p40`). The 4M pool costs 1.83 GB of free GPU against 2M and no measurable speed; it needs `SGLANG_QSA_HOST_KV_MAX_GB=50`. Full numbers: [BENCHMARKS §3.25.4](BENCHMARKS.md#3254-the-4m-pool-and-eight-500k-sessions).

### Hot-cache hit rate

73–85 % of attended tokens were served from the GPU hot cache in single-stream runs at 131K–900K,
and 55–64 % with two 900K sessions at once (150–280 host bytes read per attended token, versus 1,024
uncached). The hot cache is per request slot, so **pool size does not change the hit rate**; keep
4,096 sets (16,384 does not boot with this profile, see §4b).

## 4. Check on a 300 W Max-Q card serving other traffic

At 2,097,152 tokens, applied as a systemd drop-in on an RTX PRO 6000 Max-Q (300 W) card that was serving other traffic:

- Restart to `/health` 200: **3 min 59 s**. No restarts since (`NRestarts=0`).
- Boot log: `max_total_num_tokens=2097152`, host KV **24.00 + 2.00 GiB**, **8.97 GB** free GPU after
  boot (Modal: 8.97 GB; at 1.31M the same card showed 9.61 GB, again identical to Modal), weights 70.99 GB.
- `nvidia-smi` 90,688 / 97,887 MiB; 377 GiB host RAM still available.
- Sanity: short chat OK; a 500K-token needle 1/1 in 67.4 s.

### 4.1 Boot at 4,194,304 tokens with `--prefill-decode-interval 8`

Armed as `60-kvpool.conf` (4M, replacing the 2M file, which is kept as `60-kvpool.conf.bak-2m`) plus
`70-pdi8.conf`, same card. Boot log: host KV **48.00 + 4.00 GiB**, `max_total_num_tokens=4194304`,
**7.14 GB** free GPU after boot (Modal predicted 7.14 GB). `/health` 200, `NRestarts=0`. Quality and
long-session checks on this card at 4M have not been re-run yet.

## 5. 1M context is lossy — keep 540K

With `--context-length 1048576` on the 2M pool: 900K decode 212.0 tok/s, first token 107.8 s, two
900K sessions 277.5 tok/s total. But needles at 900K passed **1 of 3** (depth 0.1 and 0.9 came back
empty). The model is trained to 262,144 positions; 540K passes 9/9, 900K does not. A bigger pool is
for more sessions, not longer ones.

## 6. systemd drop-in example and rollback

systemd reads only `*.conf` in the `.d` directory, so stage the file under another suffix
(`.staged`) and rename it to arm it — an unrelated `daemon-reload` or a `Restart=always` restart then
cannot pick it up early. A drop-in that changes the command must clear it with a bare `ExecStart=`
and restate the whole line.

```ini
# /etc/systemd/system/<your-unit>.service.d/60-kvpool.conf
[Service]
Environment=SGLANG_QSA_HOST_KV_MAX_GB=28
ExecStart=
ExecStart=/path/to/venv/bin/python -m sglang.launch_server \
  --model-path /path/to/qwen38-flash-next-nvfp4 \
  --served-model-name qwen38-flash-next --host 0.0.0.0 --port 8011 \
  --quantization modelopt_fp4 --trust-remote-code \
  --context-length 540000 --mem-fraction-static 0.97 \
  --max-total-tokens 2097152 \
  --attention-backend flashinfer --sampling-backend flashinfer \
  --moe-runner-backend flashinfer_cutlass \
  --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer_lazy \
  --page-size 64 --chunked-prefill-size 4096 --max-prefill-tokens 4096 \
  --cuda-graph-backend-prefill=disabled --cuda-graph-max-bs-decode 8 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --ple-offload-embedding --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder --enable-metrics --enable-cache-report --log-level info \
  --max-running-requests 8 --max-mamba-cache-size 48 --mamba-max-states-per-path 2 \
  --speculative-token-map /path/to/hot_24k.pt \
  --bf16-gemm-backend sm120gemv --kv-cache-dtype fp8_e4m3 \
  --enable-priority-scheduling --default-priority-value 0 --radix-eviction-policy priority \
  --max-queued-requests 64 --crash-dump-folder /path/to/crashdumps
```

For 1.5M use `--max-total-tokens 1572864` and `SGLANG_QSA_HOST_KV_MAX_GB=24`. The S8-FP8 exports
(`SGLANG_SM120_ONLINE_MXFP8`, `SGLANG_SM120_ONLINE_FP8_DENSE`, …) stay in the base drop-in.

Apply (one restart, about 4 minutes down on a single replica):

```bash
D=/etc/systemd/system/<your-unit>.service.d
sudo mv $D/60-kvpool.conf.staged $D/60-kvpool.conf
sudo systemd-analyze verify /etc/systemd/system/<your-unit>.service
sudo systemctl daemon-reload
systemctl show <your-unit> -p ExecStart | tr ' ' '\n' | grep -A1 max-total-tokens   # expect 2097152
sudo systemctl restart <your-unit>
curl -s http://<host>:8011/health
grep -E "QSA host KV:|max_total_num_tokens|available_gpu_mem" <your server log> | tail -5
```

If the unit logs to a file rather than journald, grep that file, not `journalctl`.

Rollback (back to the base profile):

```bash
sudo rm -f $D/60-kvpool.conf
sudo systemctl daemon-reload && sudo systemctl restart <your-unit>
```

Roll back at once if boot OOMs or `available_gpu_mem` after boot is far below the figures above.

## 7. Optional: expert-usage census (off by default)

[`patches/optional/0043-opt-in-expert-census.diff`](patches/optional/0043-opt-in-expert-census.diff)
counts, on the GPU and CUDA-graph safe, every routed-expert pick per MoE layer (48 layers × 512
experts, top-10, plus the draft layer), every vocabulary id fed into a forward, and the per-layer
embedding (PLE) rows touched, split by a per-request workload tag. A daemon thread writes the delta
to a compressed `.npz` every interval. It is **not** part of the `make-fork.sh` series; apply it on
top of the fork if you want it:

```bash
./make-fork.sh /tmp/sglang-qwenopt
git -C /tmp/sglang-qwenopt apply "$PWD"/patches/optional/0043-opt-in-expert-census.diff
```

| Variable | Default | Meaning |
|---|---|---|
| `SGLANG_EXPERT_CENSUS` | `0` | `1` turns it on. Off: a boolean check at each hook, no kernels, no buffers |
| `SGLANG_EXPERT_CENSUS_DIR` | — | where the `.npz` deltas and `ple_bitset.npy` go |
| `SGLANG_EXPERT_CENSUS_INTERVAL_S` | `600` | flush interval (60 was used in testing: 99–264 KB per flush) |
| `SGLANG_EXPERT_CENSUS_TAGS` | `default,hermes,sdk,bio,chat` | allowed workload tags; others count as `default` |
| `SGLANG_EXPERT_CENSUS_PLE_BLOCK` | `4096` | PLE rows per counting block |

Tag a request with `{"custom_params": {"census_tag": "hermes"}}` in the OpenAI body or the
`x-custom-labels: {"census_tag": "hermes"}` header.

Overhead, same build and session, census off vs on (Modal, `r20261028cen`): short-prompt decode
−0.9 %; 1 / 2 / 4 / 8 streams −1.8 / −0.5 / +1.6 / −0.5 % client-side and −0.8 / +2.4 / −1.0 / −1.5 %
from the server log. Round-to-round spread is about ±2 %, so no overhead is measurable.

What a first census showed (about 64,000 tokens, all tags): 258 of 24,576 experts never picked, but
covering 99 % of picks still needs a median 406 of 512 experts per layer, and a benchmark-only window
left 577 unused, most of which chat, bio, SDK and tool-calling prompts then used. A short window is
not a safe trim list; mask before deleting, and gate on needles, GSM8K and held-out tasks.

## 8. Options: 2M, 4M, and the interval flag

Three opt-in choices on top of the S8-FP8 profile. None is the default. Copyable files for all of
them are in [`ops/examples/kv-pools/`](ops/examples/kv-pools/): systemd drop-ins (`*.conf.staged`)
and a plain launch script for non-systemd setups.

| Option | Pool `--max-total-tokens` | `SGLANG_QSA_HOST_KV_MAX_GB` | 500K sessions kept cached | Free GPU after boot | File |
|---|---|---|---|---|---|
| 2M (4 × 500K) | 2,097,152 | 28 | 4 | 8.97 GB | `60-kvpool-2m.conf.staged` |
| 4M (8 × 500K) | 4,194,304 | 50 | 8 | 7.14 GB | `60-kvpool-4m.conf.staged` |
| 4M + interval 8 | 4,194,304 | 50 | 8 | 7.14 GB | `70-pdi8.conf.staged` (self-contained) |

Without systemd:

```bash
cd ops/examples/kv-pools
MODEL=/path/to/model TOKEN_MAP=/path/to/hot_24k.pt PY=/path/to/venv/bin/python \
  POOL=4m PDI8=1 ./launch-s8fp8-kvpool.sh        # POOL=1310k|1p5m|2m|4m, PDI8=0|1
```

Or by hand: take the S8-FP8 launch line, set `--max-total-tokens 4194304`,
`export SGLANG_QSA_HOST_KV_MAX_GB=50`, and optionally append `--prefill-decode-interval 8`.

### 8.1 4M vs 2M (measured, Modal, one RTX PRO 6000 at 600 W)

| | 4M pool | 2M pool |
|---|---|---|
| Aggregate decode, 1 / 2 / 4 / 8 short streams, same container (run `r20261028ab`) | 273.0 / 405.3 / 599.4 / 807.7 tok/s | 278.7 / 403.0 / 603.7 / 807.5 tok/s |
| Free GPU after boot | 7.14 GB | 8.97 GB |
| Host RAM pinned (target + draft) | 48.00 + 4.00 GiB | 24.00 + 2.00 GiB |
| 500K sessions that stay cached | 8 of 8 (run `r20261028p40`) | 4 |
| Follow-up first token, sessions cached | 2.6–7.0 s with all eight active | 2.45–3.53 s with four resident |
| Needles 131–500K / GSM8K n=100 | 9/9, 0.97 | 9/9, 0.98 |

The speed differences (−2.0 / +0.6 / −0.7 / 0.0 %) are within noise. The 4M pool costs 1.83 GB of
free GPU and 26 GiB more pinned host RAM, and buys room for four more 500K sessions.

### 8.2 The optional `--prefill-decode-interval 8` flag

Makes the scheduler run decode steps between prefill chunks, so a long prefill in one session no
longer freezes the others. Measured on Modal (runs `r20261028tr3`/`tr4`/`tr5`, [BENCHMARKS §3.24](BENCHMARKS.md#324-troughs-compaction-and-between-turn-2026-10-28)):

| | Without | With interval 8 |
|---|---|---|
| Worst per-stream stall, 4 streams | 7.79 s | 0.76 s |
| Other streams frozen during a cold 300K prefill | 28.2 s | at most 1.27 s |
| Cold 300K prefill time | 29.5 s | 37.5 s (+27 %) |
| Per-stream speed, 2 busy streams | — | −11 % (−2 % at 4 streams; 1 stream unchanged) |

With the 4M pool (run `r20261028c40p8`, [BENCHMARKS §3.25.5](BENCHMARKS.md#3255-the-4m-pool-together-with---prefill-decode-interval-8)):
boots with the same 7.14 GB free GPU, all eight 500K sessions cached, follow-up first token
6.6–9.0 s with all eight active, GSM8K 0.98. Short-stream speed was 261.9 / 360.9 / 548.2 / 744.8 tok/s
at 1 / 2 / 4 / 8 streams, 4–11 % below 4M alone, but that run was in a different container and
containers read 5–10 % apart on their own, so treat 4–11 % as an upper bound on the flag's cost.

Use it when several agents share the card and one of them regularly sends a long cold prompt (for
example a compaction). Skip it if you care most about raw prefill speed or two-stream throughput.

### 8.3 Known not to work with this profile

- `--enable-mixed-chunk`: crashed (device-side assert) about 1 minute into 2-stream load.
- Overlap scheduler off (`--disable-overlap-schedule`): will not start; the lazy mamba buffer needs it.
- `--max-mamba-cache-size 112`: no gain at 4 streams (−3.8 %), about 3.5 GB more memory, 500K follow-ups 22–26 % slower.
- `SGLANG_QSA_HOST_KV_STATS=1`: the stats thread can break CUDA-graph capture at boot. Keep it `0`.
- `--context-length` above 540,000: lossy (section 5).

### 8.4 Apply and roll back (systemd)

Only one `60-kvpool.conf` may be armed. systemd keeps the last `ExecStart` in name order, and
`60-kvpool-4m.conf` sorts **before** `60-kvpool.conf`, so arming the 4M file under its own name next
to the 2M file leaves the 2M pool in force without any error. The 4M file must replace the 2M one.

```bash
D=/etc/systemd/system/<your-unit>.service.d
# 2M -> 4M
sudo mv $D/60-kvpool.conf $D/60-kvpool.conf.bak-2m
sudo cp 60-kvpool-4m.conf.staged $D/60-kvpool.conf
# optional: add the interval flag (self-contained, wins over 60-kvpool.conf)
sudo cp 70-pdi8.conf.staged $D/70-pdi8.conf
sudo systemd-analyze verify /etc/systemd/system/<your-unit>.service
sudo systemctl daemon-reload
systemctl show <your-unit> -p ExecStart | tr ' ' '\n' | grep -E -A1 'max-total-tokens|prefill-decode-interval'
sudo systemctl restart <your-unit>
curl -s http://<host>:8011/health
grep -E "QSA host KV:|max_total_num_tokens|available_gpu_mem" <your server log> | tail -5
# expect 48.00 GiB and 4.00 GiB host KV, max_total_num_tokens=4194304, available_gpu_mem=7.14 GB
```

Roll back the interval flag only (falls back to whatever `60-kvpool.conf` holds):

```bash
sudo rm -f $D/70-pdi8.conf && sudo systemctl daemon-reload && sudo systemctl restart <your-unit>
```

Roll back 4M to 2M:

```bash
sudo rm -f $D/70-pdi8.conf
sudo mv $D/60-kvpool.conf.bak-2m $D/60-kvpool.conf
sudo systemctl daemon-reload && sudo systemctl restart <your-unit>
```

Roll back at once if boot fails on the host-KV guard (raise `SGLANG_QSA_HOST_KV_MAX_GB` only if
the host has the RAM), if it OOMs, or if `available_gpu_mem` after boot is far below 7.14 GB.

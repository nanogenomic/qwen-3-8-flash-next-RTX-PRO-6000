# KV-POOLS — bigger host-RAM KV pools with the GPU hot cache

How to raise the KV pool of the S8-FP8 profile ([CONTEXT-PROFILES §4b](CONTEXT-PROFILES.md#4b-profile-s8-fp8-8-bit-dense-cap-8--host-ram-kv-eight-streams))
past its benchmarked 1,310,720 tokens, what it costs in host RAM and GPU memory, and what it buys.
All numbers are measured on one RTX PRO 6000 Blackwell Server Edition (600 W) on Modal, temperature 0,
S8-FP8 flags with `--context-length 540000`, `--max-mamba-cache-size 48`, `--max-running-requests 8`,
host QSA KV and 4,096 hot-cache sets (runs `r20261028kv`, 2026-10-28), plus one live production
restart at 2,097,152 tokens. No code change is needed: this is flags and one environment variable.

## 1. Short version

| Pool `--max-total-tokens` | `SGLANG_QSA_HOST_KV_MAX_GB` | Host RAM pinned (target + draft) | Free GPU after boot | 500K sessions kept cached | Use it when |
|---|---|---|---|---|---|
| 1,310,720 (S8-FP8 default) | 24 (default 100 is fine too) | 15.00 + 1.25 GiB | 9.61 GB | 2 | baseline |
| 1,572,864 | 24 | 18.00 + 1.50 GiB | 9.37 GB | 3 | cautious middle step |
| **2,097,152 (recommended)** | **28** | **24.00 + 2.00 GiB** | **8.97 GB** | **4** | several long agents at once |

The pool buys **room, not speed**. Decode speed is the same at all three sizes (section 3). What
changes is how many long sessions stay resident: at 1.31M a third 500K session evicts the others and
their next turn re-reads 500K tokens (52–107 s to first token); at 2M four 500K sessions stay cached
and follow-ups start in 2.5–3.5 s.

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

Several 500K sessions decoding at once (total decode tok/s from the server's own decode log; the
benchmark client counts streamed chunks, about 3 tokens each with the draft model, so client-side
per-stream rates under concurrency read low):

| Concurrent 500K sessions | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1.31M pool | 224.6 | 342.3 | (third session evicts: next first token 52.5 s and 106.9 s instead of 2.5 s) | — |
| 1.5M pool | 229.8 | 317.7 | 396.7 | — |
| 2M pool | 237.3 | 335.6 | 385.3 | 429.3 |

Follow-up first token for a session that stayed cached: 2.45–3.53 s at 2M with 4 × 500K resident.

### Hot-cache hit rate

73–85 % of attended tokens were served from the GPU hot cache in single-stream runs at 131K–900K,
and 55–64 % with two 900K sessions at once (150–280 host bytes read per attended token, versus 1,024
uncached). The hot cache is per request slot, so **pool size does not change the hit rate**; keep
4,096 sets (16,384 does not boot with this profile, see §4b).

## 4. Production verification (2,097,152 tokens, 2026-10-28)

Applied as a systemd drop-in on the production RTX PRO 6000 host:

- Restart to `/health` 200: **3 min 59 s**. No restarts since (`NRestarts=0`).
- Boot log: `max_total_num_tokens=2097152`, host KV **24.00 + 2.00 GiB**, **8.97 GB** free GPU after
  boot (Modal: 8.97 GB; at 1.31M the same host showed 9.61 GB, again identical to Modal), weights 70.99 GB.
- `nvidia-smi` 90,688 / 97,887 MiB; 377 GiB host RAM still available.
- Sanity: short chat OK; a 500K-token needle 1/1 in 67.4 s.

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

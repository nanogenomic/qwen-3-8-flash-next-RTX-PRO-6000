# KV pool options: copyable files

Opt-in configs for the S8-FP8 profile. Nothing here is a default; the repo's default stays the
1,310,720-token pool of [CONTEXT-PROFILES §4b](../../../CONTEXT-PROFILES.md#4b-profile-s8-fp8-8-bit-dense-cap-8--host-ram-kv-eight-streams).
Measured trade-offs and rollback: [KV-POOLS.md §8](../../../KV-POOLS.md#8-options-2m-4m-and-the-interval-flag).

| File | What it does |
|---|---|
| `60-kvpool-1p5m.conf.staged` | systemd drop-in: 1,572,864-token pool, guard 24 GiB |
| `60-kvpool-2m.conf.staged` | systemd drop-in: 2,097,152-token pool, guard 28 GiB (four 500K sessions cached) |
| `60-kvpool-4m.conf.staged` | systemd drop-in: 4,194,304-token pool, guard 50 GiB (eight 500K sessions cached). Arm it **as `60-kvpool.conf`, replacing the 2M file** |
| `70-pdi8.conf.staged` | systemd drop-in: self-contained 4M pool + `--prefill-decode-interval 8` |
| `launch-s8fp8-kvpool.sh` | the same choices without systemd: `POOL=4m PDI8=1 ./launch-s8fp8-kvpool.sh` |

The drop-ins assume your base unit (or an earlier drop-in) already sets the S8-FP8 and host-KV
environment (`SGLANG_QSA_HOST_KV=all`, `SGLANG_SM120_ONLINE_MXFP8=true`, …); they only override the
guard and the command. Replace every `/path/to/...` and the host/port with yours.

Files stay inert while named `*.staged`: systemd reads only `*.conf` in the `.d` directory.

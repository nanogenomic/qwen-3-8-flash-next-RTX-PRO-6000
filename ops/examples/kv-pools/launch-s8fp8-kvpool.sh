#!/usr/bin/env bash
# Plain launch (no systemd) of the S8-FP8 profile with a choice of host-RAM KV pool and the
# optional --prefill-decode-interval 8 flag. See KV-POOLS.md section 8 for measured trade-offs.
#
#   POOL=2m ./launch-s8fp8-kvpool.sh            # 2,097,152 tokens: four 500K sessions cached
#   POOL=4m ./launch-s8fp8-kvpool.sh            # 4,194,304 tokens: eight 500K sessions cached
#   POOL=4m PDI8=1 ./launch-s8fp8-kvpool.sh     # 4M + shorter freezes while others prefill
#
# Set MODEL, TOKEN_MAP, PY (python in the venv with this fork installed), HOST, PORT as needed.
set -euo pipefail

POOL=${POOL:-2m}          # 1310k | 1p5m | 2m | 4m
PDI8=${PDI8:-0}           # 1 adds --prefill-decode-interval 8
MODEL=${MODEL:-/path/to/qwen38-flash-next-nvfp4}
TOKEN_MAP=${TOKEN_MAP:-/path/to/hot_24k.pt}
PY=${PY:-python}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8011}
CRASH_DIR=${CRASH_DIR:-/tmp/sglang-crashdumps}

case "$POOL" in
  1310k) TOKENS=1310720; GUARD=24 ;;
  1p5m)  TOKENS=1572864; GUARD=24 ;;
  2m)    TOKENS=2097152; GUARD=28 ;;
  4m)    TOKENS=4194304; GUARD=50 ;;
  *) echo "POOL must be 1310k, 1p5m, 2m or 4m" >&2; exit 2 ;;
esac

# Profile A exports (CONTEXT-PROFILES.md)
export SGLANG_QWENOPT_FUSE_SBMOE=1 SGLANG_QWENOPT_FUSE_HC=1 SGLANG_HC_MIX_PREFETCH=1
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 TRTLLM_ENABLE_PDL=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
# Profile B host-RAM KV exports
export SGLANG_QSA_HOST_KV=all
export SGLANG_QSA_HOST_KV_MAX_GB=$GUARD     # guard per pool; target pool = tokens x 12,288 bytes
export SGLANG_QSA_HOST_KV_CACHE_SETS=4096
export SGLANG_QSA_HOST_KV_STATS=0           # keep 0: =1 can break CUDA-graph capture at boot
export SGLANG_OOM_MAX_REQ_RETRACTIONS=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# S8-FP8 exports (CONTEXT-PROFILES.md section 4b)
export SGLANG_SM120_ONLINE_MXFP8=true SGLANG_SM120_ONLINE_FP8_DENSE=rowwise
export SGLANG_QSA_SORT_TOPK=0 SGLANG_TOOL_STRICT_LEVEL=2 SGLANG_QSA_MQA_SPLITK=auto

EXTRA=()
[ "$PDI8" = "1" ] && EXTRA+=(--prefill-decode-interval 8)

exec "$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name qwen38-flash-next --host "$HOST" --port "$PORT" \
  --quantization modelopt_fp4 --trust-remote-code \
  --context-length 540000 --mem-fraction-static 0.97 \
  --max-total-tokens "$TOKENS" \
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
  --speculative-token-map "$TOKEN_MAP" \
  --bf16-gemm-backend sm120gemv --kv-cache-dtype fp8_e4m3 \
  --enable-priority-scheduling --default-priority-value 0 --radix-eviction-policy priority \
  --max-queued-requests 64 \
  --crash-dump-folder "$CRASH_DIR" \
  ${EXTRA[@]+"${EXTRA[@]}"}

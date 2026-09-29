#!/usr/bin/env bash
# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# qgate_verdict.sh -- one command from a candidate server to a PASS / WARN / FAIL / INVALID verdict.
#
#   qgate_verdict.sh URL MODEL LABEL CLAIM [PROFILE] [extra `qgate.py run` args...]
#     URL      server root, e.g. http://127.0.0.1:30000
#     MODEL    served_model_name
#     LABEL    short name for the output files
#     CLAIM    lossless (spec decode / MTP, CUDA graphs, torch.compile, kernels, scheduler flags)
#              lossy    (fp8 KV, weight requant, YaRN -- anything that changes numerics by design)
#     PROFILE  quick | full (default) | strict
#
# Required:  QGATE_BASELINE=path/to/baseline.json   (a `qgate.py run` of the unmodified config)
# Optional:  QGATE_NOISE=second_baseline.json       (same config again: the noise floor)
#            QGATE_NLL_NOISE=a.json,b.json          (more repeated-base runs for the NLL spread)
#            QGATE_PREFIX_REF=ref.json              (prefix-cache reference, if the baseline has none)
#            QGATE_RUNS=dir (default ./runs)   QGATE_CONC (default 4)
# strict:    QGATE_BASELINE / QGATE_NOISE must be the two reference runs from run_strict_ref.sh
#
# Exit: 0 PASS/WARN, 1 FAIL, 2 INVALID (errors/unplanned skips > 5 % in a suite -- fix and re-run).
set -u
cd "$(dirname "$0")"
URL=$1; MODEL=$2; LABEL=$3; CLAIM=$4; PROFILE=${5:-full}; shift $(( $# < 5 ? $# : 5 ))
RUNS=${QGATE_RUNS:-runs}; mkdir -p "$RUNS"
PY=${PYTHON:-python3}
: "${QGATE_BASELINE:?set QGATE_BASELINE to a baseline qgate.py run}"
NOISE=""; [ -n "${QGATE_NOISE:-}" ] && NOISE="--noise $QGATE_NOISE"
CONC=${QGATE_CONC:-4}
STRICT=""
case "$PROFILE" in
  quick)  SIZES="--suites nll,gsm8k,niah,humaneval,degen --prefix-n 3 --niah-lengths 4096,16384,32768,65536 --nll-ctx 512,4096,16384,65536 --degen-n 30 --degen-max-tokens 4096" ;;
  full)   SIZES="--niah-lengths 4096,16384,32768,65536,98304 --nll-ctx 512,4096,16384,65536,98304" ;;
  strict) SIZES="--suites refagree,mmlupro,gsm8k,heplus,mbppplus,ifeval,niah,nll,degen --max-tokens 16384 --n-gsm8k 1319 --n-ifeval 541 \
            --ref-file $QGATE_BASELINE --prefix-n 3 --n-mmlupro ${QGATE_N_MMLUPRO:-1200} --degen-n 200 --degen-max-tokens 16384 \
            --nll-window 64 --niah-lengths ${QGATE_NIAH_LENGTHS:-4096,32768,131072} --niah-trials 4 --nll-ctx 512,4096,32768,131072"
          STRICT="--strict" ;;
  *) echo "PROFILE must be quick | full | strict"; exit 2 ;;
esac
$PY qgate.py run --url "$URL" --model "$MODEL" --label "$LABEL" --out "$RUNS/cand_$LABEL.json" \
  --conc "$CONC" --degen-conc "$CONC" --pool-frac 0.95 $SIZES "$@" > "$RUNS/cand_$LABEL.log" 2>&1
RC=$?
if [ $RC -ne 0 ] && [ $RC -ne 2 ]; then echo "qgate run crashed (rc=$RC), see $RUNS/cand_$LABEL.log"; exit 2; fi
$PY qgate.py compare --baseline "$QGATE_BASELINE" --candidate "$RUNS/cand_$LABEL.json" $NOISE --claim "$CLAIM" $STRICT \
  ${QGATE_NLL_NOISE:+--nll-noise $QGATE_NLL_NOISE} ${QGATE_PREFIX_REF:+--prefix-ref $QGATE_PREFIX_REF} \
  --out "$RUNS/verdict_$LABEL.json" > /dev/null
VRC=$?
$PY - "$RUNS/verdict_$LABEL.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("OVERALL:", d["overall"], "| invalid:", d.get("invalid"), "| unverified:", d.get("claims_unverified"))
for k, v in d["metrics"].items():
    if isinstance(v, dict) and "verdict" in v:
        extra = {x: v[x] for x in ("base_acc", "cand_acc", "lost", "gained", "p_worse") if x in v}
        print(f"  {k:20s} {v['verdict']:15s} {extra}")
if d.get("strict_verdict"):
    st = d["strict"]
    print("STRICT VERDICT:", d["strict_verdict"])
    for k, e in st["suites"].items():
        print(f"  {k:10s} {e['verdict']:13s} delta {e['delta']:+.4f} lower95 {e['delta_lower95']:+.4f} mde80 {e['mde_80pct']} n {e['n_paired']}")
    if st.get("pooled"):
        print("  pooled", {k: st["pooled"].get(k) for k in ("n", "delta", "delta_lower95", "p_worse", "mde_80pct")})
    for L, x in st["refagree"]["cand_vs_ref"].items():
        print(f"  refagree {L}: {x}")
    for tag in ("reasons_not_equivalent", "reasons_not_acceptable", "not_measured"):
        for r in st[tag]:
            print(f"  {tag}: {r}")
print("full verdict:", sys.argv[1])
PY
exit $VRC

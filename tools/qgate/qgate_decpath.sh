#!/usr/bin/env bash
# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# qgate_decpath.sh -- DECODE-PATH lossless test (~5 min per run on a dedicated card).
# Exercises the T<=16 decode kernels (fused MoE / hyper-connection tails, skinny decode GEMV,
# speculative-verify graph shapes) that teacher-forced NLL -- a prefill measurement -- never reaches.
# Generation logprobs only: no prompt logprobs anywhere, so it is safe on a live lane.
#
#   qgate_decpath.sh run URL MODEL LABEL        # 48 real prompts (<500 tok), greedy 384 tok, 2 reps, conc 1 and 4
#   qgate_decpath.sh verdict BASE CAND [BASE2]  # labels of earlier runs
#
# BASE and BASE2 = two runs of the unmodified config on the same hardware class (BASE2 may be the same
# boot, run again). CAND = the candidate. Order does not matter. Verdict per concurrency:
# BITWISE | PASS | WARN | FAIL | UNCALIBRATED.
set -u
cd "$(dirname "$0")"
RUNS=${QGATE_RUNS:-runs}; mkdir -p "$RUNS"; PY=${PYTHON:-python3}
case "${1:-}" in
run)
  $PY qgate.py run --url "$2" --model "$3" --label "dec_$4" --out "$RUNS/dec_$4.json" \
    --suites decpath --prefix-n 0 --conc 4 --decpath-prompts 48 --decpath-tokens 384 --decpath-reps 2 --decpath-conc 1,4 \
    > "$RUNS/dec_$4.log" 2>&1
  rc=$?; tail -1 "$RUNS/dec_$4.log" | cut -c1-300; exit $rc ;;
verdict)
  NOISE=""; [ $# -ge 4 ] && NOISE="--noise $RUNS/dec_$4.json"
  $PY qgate.py compare --baseline "$RUNS/dec_$2.json" --candidate "$RUNS/dec_$3.json" $NOISE \
    --claim lossless --out "$RUNS/verdict_dec_$3.json" > /dev/null
  $PY - "$RUNS/verdict_dec_$3.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
m = d["metrics"]["decpath"]
print("DECPATH VERDICT:", m["verdict"], "| noise sources:", m["noise_sources"], "| invalid:", d.get("invalid"))
for conc, e in m["by_conc"].items():
    c, n = e["cand_vs_base"], e.get("noise") or {}
    print(f"  conc {conc}: {e['verdict']:12s} cand identical {c['identical']}/{c['prompts']} agree-len median {c['agree_len_median']} "
          f"mean|dlp| {c['mean_abs_dlogprob_agreeing']} | noise identical {n.get('identical')}/{n.get('prompts')} "
          f"median {n.get('agree_len_median')} mean|dlp| {n.get('mean_abs_dlogprob_agreeing')} "
          f"| p(shorter) {e.get('p_agree_shorter')} p(dlp larger) {e.get('p_dlogprob_larger')}")
dg = d["metrics"].get("degeneracy", {})
print("  degeneracy:", dg.get("verdict"), (dg.get("candidate") or {}).get("bad"), "bad of", (dg.get("candidate") or {}).get("scanned"))
print("full:", sys.argv[1])
PY
  ;;
*) echo "usage: $0 run URL MODEL LABEL | verdict BASE CAND [BASE2]"; exit 2 ;;
esac

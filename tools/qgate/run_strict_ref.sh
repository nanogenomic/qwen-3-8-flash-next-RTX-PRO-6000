#!/usr/bin/env bash
# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Two IDENTICAL-CONFIG reference runs for STRICT mode (about 3 h each at 4 concurrent requests).
# R2 teacher-forces R1's continuations, so R1-vs-R2 is the noise floor, including the measured
# minimum detectable drop per suite.
#   run_strict_ref.sh URL MODEL
# Then: QGATE_BASELINE=runs/strict_ref_R1.json QGATE_NOISE=runs/strict_ref_R2.json \
#       ./qgate_verdict.sh URL MODEL LABEL lossy strict
set -u
cd "$(dirname "$0")"
URL=$1; MODEL=$2; RUNS=${QGATE_RUNS:-runs}; mkdir -p "$RUNS"; PY=${PYTHON:-python3}
S="--suites refagree,mmlupro,gsm8k,heplus,mbppplus,ifeval,niah,nll,degen --max-tokens 16384 --n-gsm8k 1319 --n-ifeval 541 \
   --prefix-n 3 --conc ${QGATE_CONC:-4} --degen-conc ${QGATE_CONC:-4} --pool-frac 0.95 \
   --n-mmlupro ${QGATE_N_MMLUPRO:-1200} --degen-n 200 --degen-max-tokens 16384 --nll-window 64 \
   --niah-lengths ${QGATE_NIAH_LENGTHS:-4096,32768,131072} --niah-trials 4 --nll-ctx 512,4096,32768,131072"
$PY qgate.py run --url "$URL" --model "$MODEL" --label strict_ref_R1 --out "$RUNS/strict_ref_R1.json" $S > "$RUNS/strict_ref_R1.log" 2>&1
$PY qgate.py run --url "$URL" --model "$MODEL" --label strict_ref_R2 --out "$RUNS/strict_ref_R2.json" $S \
  --ref-file "$RUNS/strict_ref_R1.json" > "$RUNS/strict_ref_R2.log" 2>&1
$PY qgate.py compare --baseline "$RUNS/strict_ref_R1.json" --candidate "$RUNS/strict_ref_R2.json" --claim lossless --strict \
  --out "$RUNS/strict_noise_R1_vs_R2.json" > /dev/null
echo "noise floor: $RUNS/strict_noise_R1_vs_R2.json (per-suite mde_80pct / n_for_2pt_mde)"

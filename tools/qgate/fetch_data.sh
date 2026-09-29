#!/usr/bin/env bash
# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Download the PUBLIC evaluation data qgate.py uses into ./data (or $QGATE_DATA).
# Nothing here is bundled in the repository; every file comes from its original publisher.
# data_check.py compares CONTENT checksums against the files used for the published verdicts
# (data.content-sha256). A mismatch is reported, not fatal -- Gutenberg re-issues files -- but a
# different file means different items, so only compare runs made from the same content.
set -euo pipefail
cd "$(dirname "$0")"
D=${QGATE_DATA:-data}
mkdir -p "$D"
get() { [ -s "$D/$2" ] || { echo "fetch $2"; curl -fsSL --retry 3 -o "$D/$2" "$1"; }; }

# GSM8K test (OpenAI, MIT). Converted to add an "idx" field = line number.
if [ ! -s "$D/gsm8k_test.jsonl" ]; then
  curl -fsSL --retry 3 https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
    | python3 -c 'import json,sys
for i,l in enumerate(sys.stdin):
    d=json.loads(l); print(json.dumps({"idx":i,"question":d["question"],"answer":d["answer"]}))' > "$D/gsm8k_test.jsonl"
fi
# HumanEval (OpenAI, MIT)
if [ ! -s "$D/humaneval.jsonl" ]; then
  curl -fsSL --retry 3 https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz | gunzip > "$D/humaneval.jsonl"
fi
# IFEval input data (Google, Apache-2.0), the Hugging Face hub copy (the GitHub copy differs in content)
get https://huggingface.co/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl ifeval.jsonl
# MMLU-Pro test split (TIGER-Lab, MIT)
get https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet mmlu_pro_test.parquet
# EvalPlus HumanEval+ / MBPP+ (Apache-2.0)
get https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz HumanEvalPlus.jsonl.gz
get https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz MbppPlus.jsonl.gz
# Project Gutenberg haystacks / NLL prose (public domain in the US)
for id in 2600 2701 1342 98 1400; do get "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" "pg$id.txt"; done

echo "content checksums against the files used for the published verdicts:"
python3 data_check.py "$D" || echo "WARNING: some files differ from the published run (see above)."

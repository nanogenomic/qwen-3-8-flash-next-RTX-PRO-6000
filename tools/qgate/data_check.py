# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Content checksums for the public evaluation files.

JSONL files are hashed by CONTENT (each record canonicalised: sorted keys, None-valued kwargs
dropped), because the same dataset is published with different serialisations and with
None-padded kwargs (e.g. IFEval on GitHub vs on the Hugging Face hub). Binary files are hashed raw.

  python3 data_check.py [data_dir]          # verify against data.content-sha256
  python3 data_check.py --write [data_dir]  # (maintainers) regenerate the reference list
"""
import hashlib, json, os, sys

FILES = ["gsm8k_test.jsonl", "humaneval.jsonl", "ifeval.jsonl", "mmlu_pro_test.parquet",
         "HumanEvalPlus.jsonl.gz", "MbppPlus.jsonl.gz",
         "pg2600.txt", "pg2701.txt", "pg1342.txt", "pg98.txt", "pg1400.txt"]
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.content-sha256")


def canon(rec):
    if isinstance(rec, dict):
        return {k: canon(v) for k, v in rec.items() if not (k == "kwargs" and v is None)}
    if isinstance(rec, list):
        return [({k: v for k, v in x.items() if v is not None} if isinstance(x, dict) else canon(x)) for x in rec]
    return rec


def digest(path):
    h = hashlib.sha256()
    if path.endswith(".jsonl"):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                h.update(json.dumps(canon(json.loads(line)), sort_keys=True).encode() + b"\n")
    else:
        h.update(open(path, "rb").read())
    return h.hexdigest()


def main():
    args = [x for x in sys.argv[1:] if x != "--write"]
    d = args[0] if args else os.environ.get("QGATE_DATA", "data")
    if "--write" in sys.argv:
        with open(REF, "w") as fh:
            for f in FILES:
                fh.write(f"{digest(os.path.join(d, f))}  {f}\n")
        print("wrote", REF)
        return
    ref = dict(reversed(l.split()) for l in open(REF) if l.strip())
    bad = 0
    for f in FILES:
        p = os.path.join(d, f)
        if not os.path.exists(p):
            print(f"{f}: MISSING"); bad += 1; continue
        ok = digest(p) == ref.get(f)
        bad += not ok
        print(f"{f}: {'OK' if ok else 'DIFFERS from the published run'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

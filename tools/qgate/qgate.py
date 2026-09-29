#!/usr/bin/env python3
# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
qgate.py -- quality gate for serving-engine changes to an OpenAI-compatible SGLang server.

Two sub-commands:

  run      Evaluate ONE server and write a JSON result file.
  compare  Compare a candidate result file against a baseline result file (and,
           optionally, a second baseline run that defines the NOISE FLOOR) and
           emit a PASS / FAIL / WARN verdict per metric as JSON.

Suites (all real public data; no synthetic scoring data):
  gsm8k      math reasoning, exact numeric match              (short context)
  ifeval     instruction following, Google IFEval checkers     (short context)
  humaneval  code generation, pass@1, tests executed           (short context)
  niah       keyed needle retrieval with 3 distractor needles in real Gutenberg
             prose, at several lengths x depths                 (long context)
  nll        teacher-forced NLL of a fixed 128-token window of real prose at
             several context depths, via /generate input logprobs. Prompt token
             IDs come from the LOCAL tokenizer so baseline and candidate score
             byte-identical inputs.                             (short+long ctx)
  det        greedy decode repeated R times with per-token logprobs; measures
             within-server determinism and, in compare, cross-server agreement.

Load discipline (a baseline may be taken against a live, shared serving lane):
  * bounded concurrency (--conc, default 2, lane max_running_requests=4)
  * before every request: back off while the lane has queued requests or while
    the KV pool could not hold the request with --pool-frac headroom
  * long-context cells are skipped (recorded, not silently dropped) when they do
    not fit server context_len or the pool guard
  * nll scores only a 128-token window per request, so the logits tensor stays
    small (128 x 248k vocab), never the full prompt

Everything a number depends on is written into the output: server fingerprint,
item ids, raw outputs, per-token logprobs, timings, skips.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import random
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("QGATE_DATA", os.path.join(HERE, "data"))
CODE_CORPUS = os.environ.get("QGATE_CODE_CORPUS", "")  # optional: a text file NOT in any pretraining set
BOOKS = ["pg2600.txt", "pg2701.txt", "pg1342.txt", "pg98.txt", "pg1400.txt"]
MODEL_DEFAULT = os.environ.get("QGATE_MODEL", "default")


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------

def _post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


class Server:
    def __init__(self, url, model, pool_frac, timeout):
        self.url = url.rstrip("/")
        self.model = model
        self.pool_frac = pool_frac
        self.timeout = timeout
        self.lock = threading.Lock()
        self.backoffs = 0
        self.info = {}
        try:
            self.info = _get(self.url + "/get_server_info", 15)
        except Exception as e:  # noqa
            self.info = {"error": repr(e)}
        self.context_len = int(self.info.get("context_length") or 0) or None

    def fingerprint(self):
        keys = ["model_path", "served_model_name", "context_length", "max_total_num_tokens",
                "speculative_algorithm", "speculative_num_steps", "speculative_eagle_topk",
                "speculative_num_draft_tokens", "attention_backend", "kv_cache_dtype",
                "quantization", "json_model_override_args", "disable_cuda_graph",
                "enable_torch_compile", "mem_fraction_static", "max_running_requests",
                "base_gpu_id", "tp_size", "enable_deterministic_inference",
                "fp4_gemm_backend", "linear_attn_backend", "mamba_ssm_dtype",
                "chunked_prefill_size", "version", "weight_version"]
        fp = {k: self.info.get(k) for k in keys}
        fp["url"] = self.url
        return fp

    def loads(self):
        try:
            d = _get(self.url + "/v1/loads", 8)
            return (d.get("loads") or [{}])[0]
        except Exception:
            return {}

    def wait_for_room(self, need_tokens=0, max_wait=900):
        """Back off while the lane is queueing or the pool cannot take need_tokens."""
        t0 = time.time()
        while True:
            ld = self.loads()
            if not ld:
                return True  # observability down -> do not deadlock; request is bounded anyway
            used = ld.get("num_used_tokens") or 0
            cap = ld.get("max_total_num_tokens") or 0
            waiting = ld.get("num_waiting_reqs") or 0
            ok_queue = waiting == 0
            ok_pool = (cap == 0) or (used + need_tokens <= self.pool_frac * cap)
            if ok_queue and ok_pool:
                return True
            if need_tokens > self.pool_frac * cap:
                return False  # can never fit
            if time.time() - t0 > max_wait:
                return False
            with self.lock:
                self.backoffs += 1
            time.sleep(5)

    def chat(self, messages, max_tokens, temperature=0.0, logprobs=False, need_tokens=2048, sampling=None, think=1):
        if not self.wait_for_room(need_tokens):
            return {"skipped": "pool_guard"}
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "return_token_ids": True}
        if not think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if sampling:
            body.update(sampling)
        elif temperature == 0.0:
            body["top_p"] = 1.0
            body["top_k"] = 1
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 1
        t0 = time.time()
        try:
            d = _post(self.url + "/v1/chat/completions", body, self.timeout)
        except Exception as e:
            return {"error": repr(e)[:300], "wall_s": round(time.time() - t0, 3)}
        c = d["choices"][0]
        m = c.get("message") or {}
        out = {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
               "finish_reason": c.get("finish_reason"), "usage": d.get("usage"),
               "wall_s": round(time.time() - t0, 3)}
        out["degen"] = degeneracy(c.get("token_ids") or [], out["content"] + out["reasoning"],
                                  out["finish_reason"])
        if logprobs and c.get("logprobs"):
            out["tokens"] = [t["token"] for t in c["logprobs"]["content"]]
            out["logprobs"] = [round(t["logprob"], 5) for t in c["logprobs"]["content"]]
        return out

    def generate_out(self, input_ids, max_new_tokens=32):
        """Greedy continuation of raw input ids with OUTPUT-token logprobs only (no logprob_start_len,
        so no prompt logits are materialised and the radix prefix match is not truncated).
        Memory cost is the same as ordinary decoding, so it is safe on a live, memory-tight lane."""
        if not self.wait_for_room(len(input_ids) + max_new_tokens + 64):
            return {"skipped": "pool_guard"}
        body = {"input_ids": input_ids, "return_logprob": True,
                "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0, "top_k": 1}}
        t0 = time.time()
        try:
            d = _post(self.url + "/generate", body, self.timeout)
        except Exception as e:
            return {"error": repr(e)[:300]}
        m = d["meta_info"]
        otl = m.get("output_token_logprobs") or []
        return {"ids": [x[1] for x in otl], "lps": [round(x[0], 5) for x in otl],
                "cached_tokens": m.get("cached_tokens"), "wall_s": round(time.time() - t0, 3)}

    def input_logprobs(self, input_ids, start):
        # Prompt-logprob requests materialise (scored tokens x vocab) logits OUTSIDE the KV pool. On a live
        # lane with little VRAM headroom that can OOM the server, so servers listed in QGATE_NO_PROMPT_LOGPROBS
        # (comma-separated URL roots) refuse them, recorded as a designed skip.
        if self.url.rstrip("/") in [u.strip().rstrip("/") for u in
                                    os.environ.get("QGATE_NO_PROMPT_LOGPROBS", "").split(",") if u.strip()]:
            return {"skipped": "prompt_logprobs_forbidden"}
        if not self.wait_for_room(len(input_ids) + 64):
            return {"skipped": "pool_guard"}
        body = {"input_ids": input_ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "return_logprob": True, "logprob_start_len": start}
        t0 = time.time()
        try:
            d = _post(self.url + "/generate", body, self.timeout)
        except Exception as e:
            return {"error": repr(e)[:300]}
        m = d["meta_info"]
        lps = [x[0] for x in m["input_token_logprobs"] if x[0] is not None]
        return {"lps": [round(x, 5) for x in lps], "cached_tokens": m.get("cached_tokens"),
                "wall_s": round(time.time() - t0, 3)}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

VOCAB_REAL = int(os.environ.get("QGATE_VOCAB_SIZE", "248077"))  # real tokenizer vocab; ids >= this are
# embedding padding and can never be produced by a healthy model ("impossible"). 248077 = Qwen3.8 tokenizer.


def degeneracy(ids, text, finish):
    """Detectors for the corruption modes named in external-intel.md.

    * token-0 ('!') runs            -- SM120 FA2 varlen fallback (#36806) / PDL router NaN (#38290)
    * any id >= VOCAB_REAL          -- poisoned radix prefix emits impossible token 248319 (#38319)
    * longest single-token run      -- '!!!!' collapse and similar
    * periodic tail loop            -- long-generation repetition loops
    * U+FFFD / NaN-like text        -- decode of garbage ids
    """
    d = {"n": len(ids), "tok0": ids.count(0), "impossible": sum(1 for x in ids if x >= VOCAB_REAL),
         "max_run": 0, "loop_period": None, "replacement_chars": text.count("\ufffd")}
    run = best = 0
    prev = None
    for x in ids:
        run = run + 1 if x == prev else 1
        prev = x
        best = max(best, run)
    d["max_run"] = best
    if best >= 32:  # forensics: which token, and the text around the collapse
        run = 0
        prev = None
        for x in ids:
            run = run + 1 if x == prev else 1
            prev = x
            if run == best:
                d["run_token"] = x
                break
        d["tail"] = text[-240:]
    tail = ids[-256:]
    if len(tail) == 256:
        for pdl in range(1, 65):
            m = sum(1 for i in range(pdl, 256) if tail[i] == tail[i - pdl])
            if m >= 0.95 * (256 - pdl):
                d["loop_period"] = pdl
                break
    if not ids:  # no ids returned: fall back to text
        m = re.search(r"(.)\1{31,}", text)
        d["max_run"] = len(m.group(0)) if m else 0
    r0 = b0 = 0
    for x in ids:
        r0 = r0 + 1 if x == 0 else 0
        b0 = max(b0, r0)
    d["tok0_run"] = b0
    d["bad"] = is_bad(d)
    return d


def is_bad(d):
    """Degenerate = impossible id, a run of >=8 token-0 ('!') in a row, any single-token run >=32,
    a periodic tail loop, or >=4 U+FFFD. A TOTAL count of '!' is NOT a signal: IFEval prompts that ask
    for shouting legitimately emit 100+ '!' tokens (measured on IFEval item 3186)."""
    t0run = d.get("tok0_run")
    if t0run is None:  # records written before tok0_run existed: a run of 0 is bounded by max_run
        t0run = d.get("max_run", 0) if d.get("tok0", 0) >= 8 else 0
    return bool(d.get("impossible") or t0run >= 8 or d.get("max_run", 0) >= 32 or d.get("loop_period")
                or d.get("replacement_chars", 0) >= 4)

def jl(path):
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def pick(items, n, seed):
    """Deterministic, spread subset: fixed shuffle, first n. Same ids every run."""
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    return sorted(idx[:n]) if n < len(items) else idx


def pmap(fn, jobs, conc):
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        return list(ex.map(fn, jobs))


_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(os.environ.get("QGATE_TOKENIZER", os.path.join(HERE, "tokenizer", "tokenizer.json")))
    return _tok


_books = {}


def book_ids(name):
    """Token ids of a Gutenberg book body (license header/footer stripped)."""
    if name not in _books:
        path = CODE_CORPUS if name == "private_corpus" else os.path.join(DATA, name)
        if not path:
            raise FileNotFoundError("code corpus requested but QGATE_CODE_CORPUS is not set")
        txt = open(path, encoding="utf-8", errors="ignore").read()
        a = txt.find("*** START OF")
        b = txt.find("*** END OF")
        if a != -1:
            txt = txt[txt.find("\n", a) + 1:]
            b = txt.find("*** END OF")
        if b != -1:
            txt = txt[:b]
        txt = re.sub(r"\r", "", txt)
        _books[name] = tokenizer().encode(txt).ids
    return _books[name]


def haystack_ids(n_tokens, offset_seed):
    """n_tokens of real prose: concatenated books starting at a seeded offset."""
    out = []
    r = random.Random(offset_seed)
    order = BOOKS[:]
    r.shuffle(order)
    first = True
    while len(out) < n_tokens:
        for b in order:
            ids = book_ids(b)
            start = r.randrange(0, max(1, len(ids) // 2)) if first else 0
            first = False
            out.extend(ids[start:])
            if len(out) >= n_tokens:
                break
    return out[:n_tokens]


# ----------------------------------------------------------------------------
# Suites
# ----------------------------------------------------------------------------

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _num(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        v = float(s)
        return v
    except ValueError:
        return None


def suite_gsm8k(srv, a):
    items = jl(os.path.join(DATA, "gsm8k_test.jsonl"))
    ids = pick(items, a.n_gsm8k, 1001)

    def one(i):
        it = items[i]
        gold = _num(it["answer"].split("####")[-1])
        msg = [{"role": "user", "content": it["question"] +
                "\n\nSolve the problem. End your reply with a final line of the form '#### <number>'."}]
        r = srv.chat(msg, a.max_tokens)
        pred = None
        txt = r.get("content", "")
        m = re.findall(r"####\s*([^\n]+)", txt)
        if m:
            nm = NUM_RE.findall(m[-1])
            pred = _num(nm[0]) if nm else None
        if pred is None:
            nm = NUM_RE.findall(txt)
            pred = _num(nm[-1]) if nm else None
        ok = pred is not None and gold is not None and abs(pred - gold) < 1e-6
        return {"id": f"gsm8k/{it.get('idx', i)}", "ok": bool(ok), "gold": gold, "pred": pred,
                "finish": r.get("finish_reason"), "err": r.get("error") or r.get("skipped"),
                "ctoks": (r.get("usage") or {}).get("completion_tokens"), "content": txt[-400:], "degen": r.get("degen")}
    rows = pmap(one, ids, a.conc)
    return {"metric": "accuracy", "rows": rows}


def suite_ifeval(srv, a):
    sys.path.insert(0, HERE)
    from ifeval_lib import instructions_registry as reg  # Google IFEval checkers, vendored
    items = jl(os.path.join(DATA, "ifeval.jsonl"))
    ids = pick(items, a.n_ifeval, 2002)

    def check(it, resp):
        strict = []
        loose = []
        lines = resp.split("\n")
        variants = [resp, resp.replace("*", ""), "\n".join(lines[1:]).strip(),
                    "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
        variants += [v.replace("*", "") for v in variants[2:]]
        for iid, kw in zip(it["instruction_id_list"], it["kwargs"]):
            ins = reg.INSTRUCTION_DICT[iid](iid)
            kw = {k: v for k, v in (kw or {}).items() if v is not None}
            ins.build_description(**kw)
            args = ins.get_instruction_args()
            if args and "prompt" in args:
                ins.build_description(prompt=it["prompt"])
            s = bool(resp.strip()) and bool(ins.check_following(resp))
            l = any(v.strip() and ins.check_following(v) for v in variants)
            strict.append(s)
            loose.append(bool(l))
        return strict, loose

    def one(i):
        it = items[i]
        r = srv.chat([{"role": "user", "content": it["prompt"]}], a.max_tokens)
        resp = r.get("content", "") or ""
        try:
            s, l = check(it, resp)
        except Exception as e:  # checker crash is recorded, scored as fail
            s, l = [False] * len(it["instruction_id_list"]), [False] * len(it["instruction_id_list"])
            r["error"] = f"checker:{e!r}"[:200]
        return {"id": f"ifeval/{it['key']}", "ok": all(s), "ok_loose": all(l),
                "inst_strict": s, "inst_loose": l, "finish": r.get("finish_reason"),
                "err": r.get("error") or r.get("skipped"),
                "ctoks": (r.get("usage") or {}).get("completion_tokens"), "content": resp[:600], "degen": r.get("degen")}
    rows = pmap(one, ids, a.conc)
    return {"metric": "prompt_strict_accuracy", "rows": rows}


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
    os.setsid()


def he_exec(prompt, completion, test, entry):
    code = completion
    if f"def {entry}" not in code:  # model returned only a body
        code = prompt + code
    prog = code + "\n\n" + test + f"\n\ncheck({entry})\n"
    with tempfile.TemporaryDirectory(prefix="qgate_he_") as d:
        p = os.path.join(d, "t.py")
        open(p, "w").write(prog)
        try:
            cp = subprocess.run([sys.executable, "-I", p], cwd=d, capture_output=True, timeout=20,
                                preexec_fn=_limits, env={"PATH": "/usr/bin:/bin"})
            return cp.returncode == 0, cp.stderr.decode(errors="ignore")[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def suite_humaneval(srv, a):
    items = jl(os.path.join(DATA, "humaneval.jsonl"))
    ids = pick(items, a.n_humaneval, 3003)

    def one(i):
        it = items[i]
        msg = [{"role": "user", "content":
                "Complete the following Python function. Reply with the complete function "
                "(including the signature and any needed imports) in a single ```python code block.\n\n"
                "```python\n" + it["prompt"] + "```"}]
        r = srv.chat(msg, a.max_tokens)
        txt = r.get("content", "") or ""
        blocks = re.findall(r"```(?:python|py)?\n(.*?)```", txt, re.S)
        code = max(blocks, key=len) if blocks else txt
        if "import" not in code and "from typing" in it["prompt"]:
            code = "from typing import *\n" + code
        ok, err = he_exec(it["prompt"], code, it["test"], it["entry_point"])
        return {"id": it["task_id"], "ok": ok, "exec_err": err if not ok else "",
                "finish": r.get("finish_reason"), "err": r.get("error") or r.get("skipped"),
                "ctoks": (r.get("usage") or {}).get("completion_tokens"), "content": txt[:600], "degen": r.get("degen")}
    rows = pmap(one, ids, a.conc)
    return {"metric": "pass@1_greedy", "rows": rows}


def _words():
    ws = [w.strip() for w in open(os.environ.get("QGATE_WORDS", "/usr/share/dict/words")) if w.strip().isalpha()
          and w.strip().islower() and 5 <= len(w.strip()) <= 9]
    return ws


def suite_niah(srv, a):
    """Keyed needles: 1 target + 3 distractors (same template, different keys) in real prose.

    Scoring: exact 7-digit match of the target value. 'confused' = model returned a
    distractor's value (positional aliasing / wrong-needle retrieval).
    """
    tok = tokenizer()
    words = _words()
    lengths = [int(x) for x in a.niah_lengths.split(",")]
    depths = [float(x) for x in a.niah_depths.split(",")]
    jobs = []
    for L in lengths:
        for d in depths:
            for t in range(a.niah_trials):
                jobs.append((L, d, t))

    def one(job):
        L, d, t = job
        seed = int(hashlib.sha256(f"{L}-{d}-{t}".encode()).hexdigest()[:8], 16)
        r = random.Random(seed)
        keys = r.sample(words, 4)
        vals = [str(r.randrange(1_000_000, 9_999_999)) for _ in keys]
        needles = [f" The special magic number for {k} is: {v}. " for k, v in zip(keys, vals)]
        nid = [tok.encode(n).ids for n in needles]
        q = (f"\n\nWhat is the special magic number for {keys[0]} mentioned in the text above? "
             f"Answer with the number only.")
        overhead = 40 + len(tok.encode(q).ids) + sum(len(x) for x in nid)
        body_len = L - overhead
        if srv.context_len and L + a.niah_max_tokens > srv.context_len:
            return {"id": f"niah/{L}/{d}/{t}", "L": L, "depth": d, "skipped": "context_len"}
        hay = haystack_ids(body_len, seed)
        other_depths = [x for x in (0.05, 0.25, 0.45, 0.65, 0.85, 0.97) if abs(x - d) > 0.08]
        r.shuffle(other_depths)
        placements = sorted([(d, 0)] + [(other_depths[j], j + 1) for j in range(3)], reverse=True)
        for dep, j in placements:  # insert deepest first so earlier offsets stay valid
            pos = int(len(hay) * dep)
            hay[pos:pos] = nid[j]
        text = ("Below is a long document. Some sentences in it state special magic numbers.\n\n"
                + tok.decode(hay) + q)
        res = srv.chat([{"role": "user", "content": text}], a.niah_max_tokens, need_tokens=L + 1024)
        if res.get("skipped") or res.get("error"):
            return {"id": f"niah/{L}/{d}/{t}", "L": L, "depth": d, "skipped": res.get("skipped") or "error",
                    "err": res.get("error")}
        out = (res.get("content") or "")
        found = re.findall(r"\d{7}", out.replace(",", ""))
        ok = vals[0] in found
        confused = (not ok) and any(v in found for v in vals[1:])
        return {"id": f"niah/{L}/{d}/{t}", "L": L, "depth": d, "ok": ok, "confused": confused,
                "prompt_tokens": (res.get("usage") or {}).get("prompt_tokens"),
                "finish": res.get("finish_reason"), "err": res.get("error"),
                "wall_s": res.get("wall_s"), "content": out[:200], "degen": res.get("degen")}
    # long cells one at a time on a shared lane; short ones may overlap
    rows = []
    short = [j for j in jobs if j[0] <= 32768]
    long_ = [j for j in jobs if j[0] > 32768]
    rows += pmap(one, short, a.conc)
    rows += pmap(one, long_, 1)
    return {"metric": "retrieval_accuracy", "rows": rows,
            "known_weakness": "needle retrieval is a lexical-match proxy; it does not test "
                              "multi-hop reasoning or aggregation over long context"}


def suite_nll(srv, a):
    """Teacher-forced NLL of a 128-token window located at context depth C.

    Input ids come from the local tokenizer: identical bytes on every server.
    """
    W = a.nll_window
    ctxs = [int(x) for x in a.nll_ctx.split(",")]
    jobs = []
    sources = BOOKS[:a.nll_books] + (["private_corpus"] if (a.nll_code and CODE_CORPUS) else [])
    for b in sources:
        for C in ctxs:
            for j in range(a.nll_starts):
                jobs.append((b, C, j))

    def one(job):
        b, C, j = job
        ids = book_ids(b)
        if j:  # extra windows: same context length, different document offset
            off = (j * 97_531) % max(1, len(ids) - C - W - 1)
            ids = ids[off:]
        seed = int(hashlib.sha256(f"{b}-{C}".encode()).hexdigest()[:8], 16) + j
        if C + W > len(ids) and b == "private_corpus":
            return {"id": f"nll/{b}/{C}" + (f"/{j}" if j else ""), "C": C, "skipped": "corpus_too_short"}
        if C + W > len(ids):
            ids = haystack_ids(C + W, seed)
            src = "concat"
        else:
            src = b
            ids = ids[:C + W]
        if srv.context_len and C + W > srv.context_len:
            return {"id": f"nll/{b}/{C}" + (f"/{j}" if j else ""), "C": C, "skipped": "context_len"}
        res = srv.input_logprobs(ids, C - 1)
        if res.get("skipped") or res.get("error"):
            return {"id": f"nll/{b}/{C}" + (f"/{j}" if j else ""), "C": C, "skipped": res.get("skipped") or "error", "err": res.get("error")}
        lps = res["lps"][-W:]
        return {"id": f"nll/{b}/{C}" + (f"/{j}" if j else ""), "C": C, "src": src, "corpus": "code" if b == "private_corpus" else "prose", "nll": -sum(lps) / len(lps), "lps": lps,
                "n": len(lps), "cached_tokens": res.get("cached_tokens"), "wall_s": res.get("wall_s")}
    rows = pmap(one, jobs, 1)
    return {"metric": "mean_nll_nats_per_token", "rows": rows}


def suite_det(srv, a):
    """Greedy decode with per-token logprobs, repeated R times per prompt."""
    items = jl(os.path.join(DATA, "gsm8k_test.jsonl"))
    he = jl(os.path.join(DATA, "humaneval.jsonl"))
    prompts = []
    for i in pick(items, a.det_prompts // 2, 4004):
        prompts.append((f"gsm8k/{items[i]['idx']}", items[i]["question"]))
    for i in pick(he, a.det_prompts - a.det_prompts // 2, 5005):
        prompts.append((he[i]["task_id"], "Complete this function:\n```python\n" + he[i]["prompt"] + "```"))
    jobs = [(pid, p, rep) for rep in range(a.det_reps) for pid, p in prompts]

    def one(job):
        pid, p, rep = job
        r = srv.chat([{"role": "user", "content": p}], a.det_max_tokens, logprobs=True)
        return {"id": pid, "rep": rep, "tokens": r.get("tokens"), "logprobs": r.get("logprobs"),
                "degen": r.get("degen"),
                "err": r.get("error") or r.get("skipped")}
    rows = pmap(one, jobs, a.conc)
    return {"metric": "greedy_token_agreement", "rows": rows}


def suite_degen(srv, a):
    """Production sampling (generation_config: T=1.0, top_k=20, top_p=0.95), long generations,
    at the lane's full concurrency so batched MoE routing / MTP verify paths are exercised.
    Counts outputs flagged by degeneracy(). Real prompts: GSM8K + HumanEval + IFEval."""
    g = jl(os.path.join(DATA, "gsm8k_test.jsonl"))
    he = jl(os.path.join(DATA, "humaneval.jsonl"))
    ife = jl(os.path.join(DATA, "ifeval.jsonl"))
    prompts = []
    for i in pick(g, a.degen_n // 3, 6006):
        prompts.append((f"gsm8k/{g[i]['idx']}", g[i]["question"]))
    for i in pick(he, a.degen_n // 3, 7007):
        prompts.append((he[i]["task_id"], "Write the function, then write thorough pytest tests for it:\n```python\n" + he[i]["prompt"] + "```"))
    for i in pick(ife, a.degen_n - 2 * (a.degen_n // 3), 8008):
        prompts.append((f"ifeval/{ife[i]['key']}", ife[i]["prompt"]))
    samp = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 1234}

    def one(job):
        pid, p = job
        r = srv.chat([{"role": "user", "content": p}], a.degen_max_tokens, temperature=1.0, sampling=samp)
        return {"id": pid, "degen": r.get("degen"), "finish": r.get("finish_reason"),
                "err": r.get("error") or r.get("skipped"),
                "ctoks": (r.get("usage") or {}).get("completion_tokens"),
                "tail": ((r.get("content") or "") or (r.get("reasoning") or ""))[-200:]}
    rows = pmap(one, prompts, a.degen_conc)
    return {"metric": "degenerate_output_count", "rows": rows}


def suite_prefix(srv, a, phase):
    """Radix-cache poison probe (#38319/#38355), GENERATION-ONLY (prefix probe v2).

    Fixed shared prefixes (8k/24k/49k tokens of the frozen code corpus) -> greedy 32-token continuation
    with OUTPUT logprobs. Phases: cold (first sight), warm (immediate repeat; prefix should now be a
    radix hit -- cached_tokens proves it), late (after every other suite has churned the cache with
    chunked prefill). A poisoned cached prefix shows up as impossible ids or a continuation whose
    logprobs jump on the shared greedy prefix; plain kernel nondeterminism shows up as small drift.

    v1 (removed) scored INPUT logprobs of a window; measured, it never hit the cache (cached_tokens=0 on
    every warm/late request, on three engine builds): a prompt-logprob request caps its own prefix match,
    so it measured prefill recompute noise, not cache integrity."""
    # private code corpus if supplied (preferred: not memorised), else a fixed Gutenberg concatenation
    ids_all = book_ids("private_corpus") if CODE_CORPUS else haystack_ids(600_000, 424242)
    rows = []
    for k in range(a.prefix_n):
        L = [8192, 24576, 49152][k % 3]
        st = 400_000 + (k % 3) * 60_000 + (k // 3) * 7_000  # disjoint from nll windows
        ids = ids_all[st: st + L]
        res = srv.generate_out(ids, 32)
        rows.append({"id": f"prefix/{k}", "phase": phase, "L": L, "mode": "gen",
                     "out_ids": res.get("ids") or [], "out_lps": res.get("lps") or [],
                     "cached_tokens": res.get("cached_tokens"),
                     "err": res.get("error") or res.get("skipped"),
                     "degen": degeneracy(res.get("ids") or [], "", None) if res.get("ids") else None})
    return rows


def _gen_drift(c, x):
    a, b = c["out_ids"], x["out_ids"]
    n = min(len(a), len(b))
    k = next((j for j in range(n) if a[j] != b[j]), n)
    d = [abs(c["out_lps"][j] - x["out_lps"][j]) for j in range(k)]
    return {"first_div": k, "identical": a == b,
            "mean_abs": sum(d) / len(d) if d else 0.0, "max_abs": max(d) if d else 0.0}



# ============================================================================
# STRICT-mode suites (unsaturated, real published items)
# ============================================================================

MCQ_LETTERS = "ABCDEFGHIJ"


def suite_mmlupro(srv, a):
    """MMLU-Pro (TIGER-Lab, 12,032 items, 10 options). Seeded subset stratified by category.
    Thinking follows --think (production default on). Scored by the final 'Answer: X' letter."""
    import pandas as pd
    df = pd.read_parquet(os.path.join(DATA, "mmlu_pro_test.parquet"))
    items = df.to_dict("records")
    by = {}
    for i, it in enumerate(items):
        by.setdefault(it["category"], []).append(i)
    n = a.n_mmlupro
    ids = []
    for cat in sorted(by):  # proportional stratified, seeded
        k = max(1, round(n * len(by[cat]) / len(items)))
        pool = by[cat][:]
        random.Random(9009 + len(cat)).shuffle(pool)
        ids += pool[:k]
    ids = sorted(ids)[:n] if len(ids) > n else sorted(ids)

    def one(i):
        it = items[i]
        opts = "\n".join(f"{MCQ_LETTERS[j]}. {o}" for j, o in enumerate(it["options"]))
        msg = [{"role": "user", "content": f"{it['question']}\n\n{opts}\n\n"
                "Answer the multiple-choice question. End your reply with a final line 'Answer: X' "
                "where X is the letter of the correct option."}]
        r = srv.chat(msg, a.max_tokens, think=a.think)
        txt = r.get("content", "") or ""
        m = re.findall(r"[Aa]nswer\s*(?:is)?\s*[:：]?\s*\(?([A-J])\)?", txt)
        pred = m[-1] if m else None
        if pred is None:
            m2 = re.findall(r"\b([A-J])\b", txt[-20:])
            pred = m2[-1] if m2 else None
        return {"id": f"mmlupro/{it['question_id']}", "cat": it["category"], "ok": pred == it["answer"],
                "gold": it["answer"], "pred": pred, "finish": r.get("finish_reason"),
                "err": r.get("error") or r.get("skipped"),
                "ctoks": (r.get("usage") or {}).get("completion_tokens"), "content": txt[-300:],
                "degen": r.get("degen")}
    rows = pmap(one, ids, a.conc)
    return {"metric": "accuracy", "rows": rows}


PLUS_RUNNER = r"""
import json, sys, math, copy, signal
sys.setrecursionlimit(10000)
spec = json.load(open(sys.argv[1]))
ns_ref, ns_cand = {}, {}
exec(spec["ref"], ns_ref)
exec(spec["cand"], ns_cand)
f_ref, f_cand = ns_ref[spec["entry"]], ns_cand[spec["entry"]]
atol = spec["atol"] or 0
import re as _re
def close(x, y):
    if isinstance(x, _re.Match) or isinstance(y, _re.Match) or x is None and isinstance(y, _re.Match):
        return bool(x) == bool(y)   # MBPP regex tasks return Match objects: compare truthiness (EvalPlus does too)
    if isinstance(x, float) or isinstance(y, float):
        try: return math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=max(atol, 1e-6))
        except Exception: return False
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return len(x) == len(y) and all(close(u, v) for u, v in zip(x, y))
    return x == y
class TO(Exception): pass
def h(*_): raise TO()
signal.signal(signal.SIGALRM, h)
for inp in spec["inputs"]:
    try:
        signal.alarm(4); exp = f_ref(*copy.deepcopy(inp)); signal.alarm(0)
    except BaseException:
        signal.alarm(0); continue          # reference itself fails/timeouts on this input: skip it
    try:
        signal.alarm(4); got = f_cand(*copy.deepcopy(inp)); signal.alarm(0)
    except BaseException as e:
        print("EXC", type(e).__name__, file=sys.stderr); sys.exit(1)
    if not close(got, exp):
        print("MISMATCH", repr(inp)[:120], file=sys.stderr); sys.exit(1)
sys.exit(0)
"""


def plus_exec(ref_code, cand_code, entry, inputs, atol):
    with tempfile.TemporaryDirectory(prefix="qgate_plus_") as d:
        sp = os.path.join(d, "spec.json")
        json.dump({"ref": ref_code, "cand": cand_code, "entry": entry, "inputs": inputs, "atol": atol},
                  open(sp, "w"))
        rp = os.path.join(d, "run.py")
        open(rp, "w").write(PLUS_RUNNER)
        try:
            cp = subprocess.run([sys.executable, "-I", rp, sp], cwd=d, capture_output=True, timeout=120,
                                preexec_fn=_limits_plus, env={"PATH": "/usr/bin:/bin"})
            return cp.returncode == 0, cp.stderr.decode(errors="ignore")[-200:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def _limits_plus():
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (100, 100))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
    os.setsid()


def _extract_code(txt):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", txt, re.S)
    return max(blocks, key=len) if blocks else txt


def _plus_suite(srv, a, fname, prefix, n, make_prompt):
    import gzip
    items = [json.loads(l) for l in gzip.open(os.path.join(DATA, fname))]
    ids = pick(items, n, 1111)

    def one(i):
        it = items[i]
        r = srv.chat([{"role": "user", "content": make_prompt(it)}], a.max_tokens, think=a.think)
        txt = r.get("content", "") or ""
        code = _extract_code(txt)
        if f"def {it['entry_point']}" not in code:
            code = it["prompt"] + "\n" + code
        hdr = "import math, re, heapq, itertools, collections, functools, string, bisect\n" \
              "from typing import *\nfrom collections import *\nfrom itertools import *\n"
        ref = hdr + it["prompt"] + "\n" + it["canonical_solution"] if f"def {it['entry_point']}" not in it["canonical_solution"] \
            else hdr + it["canonical_solution"]
        inputs = list(it["base_input"]) + list(it["plus_input"])
        ok_base, _ = plus_exec(ref, hdr + code, it["entry_point"], list(it["base_input"]), it.get("atol") or 0)
        ok_plus, err = plus_exec(ref, hdr + code, it["entry_point"], inputs, it.get("atol") or 0) if ok_base else (False, "base failed")
        return {"id": f"{prefix}/{it['task_id']}", "ok": ok_plus, "ok_base": ok_base, "exec_err": err if not ok_plus else "",
                "finish": r.get("finish_reason"), "err": r.get("error") or r.get("skipped"),
                "ctoks": (r.get("usage") or {}).get("completion_tokens"), "content": txt[:400],
                "degen": r.get("degen")}
    rows = pmap(one, ids, a.conc)
    return {"metric": "pass@1_greedy_plus_tests", "rows": rows}


def suite_heplus(srv, a):
    """HumanEval+ (EvalPlus v0.1.10): 164 problems, base + ~80x augmented inputs, judged against the
    canonical solution's outputs."""
    return _plus_suite(srv, a, "HumanEvalPlus.jsonl.gz", "he+", a.n_heplus, lambda it:
                       "Complete the following Python function. Reply with the complete function "
                       "(including the signature and any needed imports) in a single ```python code block.\n\n"
                       "```python\n" + it["prompt"] + "```")


def suite_mbppplus(srv, a):
    """MBPP+ (EvalPlus v0.2.0): 378 problems, base + augmented inputs."""
    return _plus_suite(srv, a, "MbppPlus.jsonl.gz", "mbpp+", a.n_mbppplus, lambda it:
                       "Write a Python function for the task below. Reply with the complete function in a single "
                       "```python code block. It must satisfy the example assertion.\n\n" + it["prompt"])


def suite_refagree(srv, a):
    """Reference agreement at 4k / 32k / 128k (lossless primary test).

    Fixed real prompts: a document (optional private code corpus, or Gutenberg prose) followed by a
    question about it. Per prompt:
      1. greedy 128-token continuation with OUTPUT logprobs (generation-only; safe on a live lane)
      2. teacher-forced INPUT logprobs of the REFERENCE continuation (from --ref-file; the reference
         run uses its own continuation) -- identical token ids on every server, so the per-token
         delta is a direct measure of distribution shift, free of greedy-divergence artifacts.
         Needs prompt logprobs over a 128-token window (refused on QGATE_NO_PROMPT_LOGPROBS servers).
    """
    tok = tokenizer()
    ref = {}
    if a.ref_file:
        R = json.load(open(a.ref_file))
        for r in R["suites"].get("refagree", {}).get("rows", []):
            if r.get("gen_ids"):
                ref[r["id"]] = r
    lengths = [int(x) for x in a.refagree_lengths.split(",")]
    qs = ["Summarise what the preceding text is about in two sentences, then list three specific details from its final third.",
          "Quote the first sentence of the text verbatim, then explain in one paragraph how the text ends.",
          "Name the most frequently mentioned person, function, or entity in the text and describe its role."]
    jobs = []
    for L in lengths:
        for k in range(a.refagree_per_len):
            jobs.append((L, k))

    def one(job):
        L, k = job
        pid = f"refagree/{L}/{k}"
        seed = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)
        if srv.context_len and L + 256 > srv.context_len:
            return {"id": pid, "L": L, "skipped": "context_len"}
        if k % 2 == 0 and CODE_CORPUS:
            src = book_ids("private_corpus")
            st = random.Random(seed).randrange(0, max(1, len(src) - L))
            doc = src[st: st + L] if L <= len(src) else haystack_ids(L, seed)
        else:
            doc = haystack_ids(L, seed)
        q = qs[k % len(qs)]
        # raw ids with the chat framing of the model, built via the tokenizer (no server templating),
        # thinking disabled by an empty think block, so every server scores the same bytes
        head = tok.encode("<|im_start|>user\n").ids
        tail = tok.encode("\n\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").ids
        prompt = head + doc[: L - len(head) - len(tail)] + tail
        g = srv.generate_out(prompt, a.refagree_gen)
        row = {"id": pid, "L": L, "prompt_tokens": len(prompt), "gen_ids": g.get("ids"), "gen_lps": g.get("lps"),
               "cached_tokens": g.get("cached_tokens"), "err": g.get("error") or g.get("skipped"),
               "degen": degeneracy(g.get("ids") or [], "", None) if g.get("ids") else None}
        cont = (ref.get(pid) or {}).get("gen_ids") or g.get("ids")
        row["tf_source"] = "ref" if pid in ref else "self"
        if cont:
            tf = srv.input_logprobs(prompt + cont, len(prompt) - 1)
            if tf.get("lps"):
                row["tf_ids"] = cont
                row["tf_lps"] = tf["lps"][-len(cont):]
            else:
                row["tf_skipped"] = tf.get("skipped") or tf.get("error")
        return row
    rows = pmap(one, [j for j in jobs if j[0] <= 32768], a.conc) + pmap(one, [j for j in jobs if j[0] > 32768], 1)
    return {"metric": "reference_agreement", "rows": rows}



def suite_decpath(srv, a):
    """DECODE-PATH lossless test.

    NLL is teacher-forced PREFILL (T >> 16) and never runs decode-only kernels (fused SBMOE / HC at
    T <= 16, decode GEMV, spec-verify shapes). This suite exercises exactly those: fixed real prompts
    kept SHORT (< 512 tokens -> one prefill chunk, measured bitwise-deterministic), raw input
    ids built from the local tokenizer (identical bytes on every engine), greedy decode with OUTPUT
    logprobs only (never prompt logprobs -> safe on any lane), repeated R times at each
    concurrency in --decpath-conc (1 -> bs1 x verify tokens; 4 -> bs4 graph shapes)."""
    tok = tokenizer()
    g = jl(os.path.join(DATA, "gsm8k_test.jsonl"))
    he = jl(os.path.join(DATA, "humaneval.jsonl"))
    ife = jl(os.path.join(DATA, "ifeval.jsonl"))
    texts = []
    k = a.decpath_prompts
    for i in pick(g, k // 3, 12012):
        texts.append((f"gsm8k/{g[i]['idx']}", g[i]["question"]))
    for i in pick(he, k // 3, 13013):
        texts.append((he[i]["task_id"], "Complete this function:\n```python\n" + he[i]["prompt"] + "```"))
    for i in pick(ife, k - 2 * (k // 3), 14014):
        texts.append((f"ifeval/{ife[i]['key']}", ife[i]["prompt"]))
    prompts = []
    for pid, t in texts:
        ids = tok.encode("<|im_start|>user\n" + t + "<|im_end|>\n<|im_start|>assistant\n<think>\n").ids
        if len(ids) < 500:
            prompts.append((pid, ids))
    rows = []
    for conc in [int(x) for x in a.decpath_conc.split(",")]:
        for rep in range(a.decpath_reps):
            def one(job, conc=conc, rep=rep):
                pid, ids = job
                r = srv.generate_out(ids, a.decpath_tokens)
                return {"id": pid, "conc": conc, "rep": rep, "prompt_tokens": len(ids),
                        "gen_ids": r.get("ids"), "gen_lps": r.get("lps"),
                        "err": r.get("error") or r.get("skipped"),
                        "degen": degeneracy(r.get("ids") or [], "", None) if r.get("ids") else None}
            rows += pmap(one, prompts, conc)
    return {"metric": "decode_path_agreement", "rows": rows}


def _pair_stats(x, y):
    a_, b_ = x["gen_ids"] or [], y["gen_ids"] or []
    n = min(len(a_), len(b_))
    k = next((j for j in range(n) if a_[j] != b_[j]), n)
    d = [abs(x["gen_lps"][j] - y["gen_lps"][j]) for j in range(k)]
    return {"agree_len": k, "identical": a_ == b_ and bool(a_), "n_min": n, "abs_d": d}


def decpath_pairs(R1, R2, rep1=0, rep2=0):
    """Per (prompt, conc): stats of R1[rep1] vs R2[rep2]."""
    def idx(R, rep):
        return {(r["id"], r["conc"]): r for r in R["suites"].get("decpath", {}).get("rows", [])
                if r.get("rep") == rep and r.get("gen_ids")}
    A, B = idx(R1, rep1), idx(R2, rep2)
    return {k: _pair_stats(A[k], B[k]) for k in sorted(set(A) & set(B))}


def decpath_summary(pairs):
    by = {}
    for (pid, conc), st in pairs.items():
        by.setdefault(conc, []).append(st)
    out = {}
    for conc, v in sorted(by.items()):
        al = sorted(x["agree_len"] for x in v)
        toks = [d for x in v for d in x["abs_d"]]
        per_prompt = [sum(x["abs_d"]) / len(x["abs_d"]) for x in v if x["abs_d"]]
        out[str(conc)] = {"prompts": len(v), "identical": sum(x["identical"] for x in v),
                          "identical_rate": round(sum(x["identical"] for x in v) / len(v), 4),
                          "agree_len_median": al[len(al) // 2], "agree_len_mean": round(sum(al) / len(al), 1),
                          "agree_len_p10": al[len(al) // 10],
                          "mean_abs_dlogprob_agreeing": round(sum(toks) / max(1, len(toks)), 6),
                          "agreeing_tokens": len(toks),
                          "_agree": al, "_pp": per_prompt}
    return out


SUITES = {"gsm8k": suite_gsm8k, "ifeval": suite_ifeval, "humaneval": suite_humaneval,
          "niah": suite_niah, "nll": suite_nll, "det": suite_det, "degen": suite_degen,
          "mmlupro": suite_mmlupro, "heplus": suite_heplus, "mbppplus": suite_mbppplus,
          "refagree": suite_refagree, "decpath": suite_decpath}


DESIGNED_SKIPS = {"context_len", "corpus_too_short", "prompt_logprobs_forbidden", "no_code_corpus"}


def validity(res, invalid_frac):
    """A suite whose error/unplanned-skip fraction exceeds invalid_frac is INVALID, never FAIL/PASS:
    its numbers describe a broken transport or a dead instance, not the model."""
    rows = res["rows"]
    planned = [r for r in rows if r.get("skipped") not in DESIGNED_SKIPS]
    bad = [r for r in planned if r.get("err") or (r.get("skipped") and r.get("skipped") not in DESIGNED_SKIPS)]
    frac = len(bad) / max(1, len(planned))
    # every row a DESIGNED skip (context too long, prompt logprobs refused, no private corpus) -> the suite
    # is NOT MEASURED, which is not the same as broken transport: valid, but yields no verdict.
    return {"planned": len(planned), "errored_or_unplanned_skip": len(bad), "frac": round(frac, 4),
            "valid": frac <= invalid_frac, "not_measured": not planned}


def summarize(name, res, invalid_frac=0.05):
    rows = [r for r in res["rows"] if not r.get("skipped")]
    s = {"n": len(rows), "skipped": len(res["rows"]) - len(rows), "validity": validity(res, invalid_frac)}
    if name in ("gsm8k", "ifeval", "humaneval", "niah", "mmlupro", "heplus", "mbppplus"):
        s["acc"] = round(sum(r.get("ok", False) for r in rows) / max(1, len(rows)), 4)
        s["errors"] = sum(1 for r in rows if r.get("err"))
        s["truncated"] = sum(1 for r in rows if r.get("finish") == "length")
    if name == "ifeval":
        s["prompt_loose"] = round(sum(r["ok_loose"] for r in rows) / max(1, len(rows)), 4)
        inst = [x for r in rows for x in r["inst_strict"]]
        s["inst_strict"] = round(sum(inst) / max(1, len(inst)), 4)
    if name == "niah":
        by = {}
        for r in rows:
            by.setdefault(r["L"], []).append(r)
        s["by_length"] = {L: {"n": len(v), "acc": round(sum(x["ok"] for x in v) / len(v), 4),
                              "confused": sum(x["confused"] for x in v)} for L, v in sorted(by.items())}
    if name == "nll":
        by = {}
        for r in rows:
            by.setdefault(r["C"], []).append(r["nll"])
        s["by_ctx"] = {C: round(sum(v) / len(v), 5) for C, v in sorted(by.items())}
        for corp in ("prose", "code"):
            byc = {}
            for r in rows:
                if r.get("corpus") == corp:
                    byc.setdefault(r["C"], []).append(r["nll"])
            s[f"by_ctx_{corp}"] = {C: round(sum(v) / len(v), 5) for C, v in sorted(byc.items())}
        s["mean_nll"] = round(sum(r["nll"] for r in rows) / max(1, len(rows)), 5)
    if any(r.get("degen") for r in res["rows"]):
        s["degeneracy"] = degen_count(res)
    if name == "det":
        by = {}
        for r in rows:
            if r.get("tokens"):
                by.setdefault(r["id"], []).append(r["tokens"])
        same = sum(1 for v in by.values() if len(v) > 1 and all(x == v[0] for x in v))
        s["prompts"] = len(by)
        s["all_reps_identical"] = same
        s["within_server_identical_rate"] = round(same / max(1, len(by)), 4)
    return s


def summarize_prefix(rows):
    by = {}
    for r in rows:
        by.setdefault(r["id"], {})[r["phase"]] = r
    imp = sum((r.get("degen") or {}).get("impossible", 0) for r in rows)
    bad = sum(1 for r in rows if r.get("degen") and is_bad(r["degen"]))
    errs = sum(1 for r in rows if r.get("err"))
    if rows and rows[0].get("mode") == "gen":
        per = {}
        for pid, v in by.items():
            c = v.get("cold")
            for ph in ("warm", "late"):
                x = v.get(ph)
                if c and x and c["out_ids"] and x["out_ids"]:
                    dd = _gen_drift(c, x)
                    dd["cached_tokens"] = x.get("cached_tokens")
                    dd["L"] = c["L"]
                    per[f"{pid}/{ph}"] = dd
        hits = [d["cached_tokens"] or 0 for d in per.values()]
        return {"mode": "gen", "prefixes": len(by), "pairs": per,
                "cache_hit_pairs": sum(1 for h in hits if h > 0), "pairs_total": len(per),
                "max_mean_abs_dlogprob_shared": round(max([d["mean_abs"] for d in per.values()] or [0]), 4),
                "max_abs_dlogprob_shared": round(max([d["max_abs"] for d in per.values()] or [0]), 4),
                "identical_pairs": sum(1 for d in per.values() if d["identical"]),
                "degenerate": bad, "impossible_tokens": imp, "errors": errs}
    worst = {"warm": 0.0, "late": 0.0}
    wmean = {"warm": 0.0, "late": 0.0}
    by_L = {}
    hash_changes = 0
    hits = 0
    for v in by.values():
        c = v.get("cold")
        for ph in ("warm", "late"):
            x = v.get(ph)
            if c and x and c.get("lps") and x.get("lps"):
                n = min(len(c["lps"]), len(x["lps"]))
                dl = [abs(c["lps"][i] - x["lps"][i]) for i in range(n)]
                worst[ph] = max(worst[ph], max(dl))
                wmean[ph] = max(wmean[ph], sum(dl) / n)
                by_L.setdefault(str(c["L"]), []).append(round(sum(dl) / n, 4))
                hits += 1 if (x.get("cached_tokens") or 0) > 0 else 0
        if c and v.get("late") and c.get("greedy_ids_hash") != v["late"].get("greedy_ids_hash"):
            hash_changes += 1
    return {"mode": "input_window_v1", "prefixes": len(by),
            "max_abs_dlogprob_cold_vs_warm": round(worst["warm"], 4),
            "max_abs_dlogprob_cold_vs_late": round(worst["late"], 4),
            "max_mean_abs_dlogprob_cold_vs_warm": round(wmean["warm"], 4),
            "max_mean_abs_dlogprob_cold_vs_late": round(wmean["late"], 4),
            "mean_abs_by_L": by_L, "cache_hit_pairs": hits,
            "greedy_changed_cold_vs_late": hash_changes, "degenerate": bad, "impossible_tokens": imp}


def degen_count(res):
    rows = [r for r in res["rows"] if r.get("degen")]
    bad = [r for r in rows if is_bad(r["degen"])]
    return {"scanned": len(rows), "bad": len(bad),
            "tok0_total": sum(r["degen"].get("tok0", 0) for r in rows),
            "impossible_total": sum(r["degen"].get("impossible", 0) for r in rows),
            "loops": sum(1 for r in rows if r["degen"].get("loop_period")),
            "max_run": max([r["degen"].get("max_run", 0) for r in rows] or [0])}


def cmd_run(a):
    srv = Server(a.url, a.model, a.pool_frac, a.timeout)
    out = {"schema": "qgate/v1", "label": a.label, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "argv": sys.argv, "server": srv.fingerprint(), "suites": {}, "summary": {}}
    out["loads_snapshots"] = []

    def snap(tag):
        ld = srv.loads()
        out["loads_snapshots"].append({"t": time.time(), "tag": tag,
                                       "accept_length": (ld.get("speculative") or {}).get("accept_length"),
                                       "accept_rate": (ld.get("speculative") or {}).get("accept_rate"),
                                       "num_running_reqs": ld.get("num_running_reqs"),
                                       "num_used_tokens": ld.get("num_used_tokens"),
                                       "gen_throughput": ld.get("gen_throughput")})
    snap("start")
    do_prefix = a.prefix_n > 0
    if do_prefix:
        out["prefix_probe"] = suite_prefix(srv, a, "cold") + suite_prefix(srv, a, "warm")
    for name in [x for x in a.suites.split(",") if x]:
        snap(f"pre-{name}")
        t0 = time.time()
        print(f"[qgate] {a.label}: suite {name} ...", file=sys.stderr, flush=True)
        res = SUITES[name](srv, a)
        res["wall_s"] = round(time.time() - t0, 1)
        out["suites"][name] = res
        out["summary"][name] = summarize(name, res, a.invalid_frac)
        out["summary"][name]["wall_s"] = res["wall_s"]
        print(f"[qgate] {name}: {json.dumps(out['summary'][name])}", file=sys.stderr, flush=True)
        with open(a.out, "w") as fh:  # checkpoint after every suite
            json.dump(out, fh)
    if do_prefix:
        out["prefix_probe"] += suite_prefix(srv, a, "late")
        out["summary"]["prefix"] = summarize_prefix(out["prefix_probe"])
    snap("end")
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out["invalid_suites"] = [k for k, v in out["summary"].items() if isinstance(v, dict)
                             and v.get("validity") and not v["validity"]["valid"]]
    out["run_valid"] = not out["invalid_suites"]
    out["pool_guard_backoffs"] = srv.backoffs
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print(json.dumps({"label": a.label, "out": a.out, "run_valid": out["run_valid"],
                      "invalid_suites": out["invalid_suites"], "summary": out["summary"]}, indent=1))
    if not out["run_valid"]:
        print(f"[qgate] RUN INVALID: suites {out['invalid_suites']} exceed {a.invalid_frac:.0%} errors/unplanned skips",
              file=sys.stderr)
        sys.exit(2)


# ----------------------------------------------------------------------------
# Compare / verdict
# ----------------------------------------------------------------------------

def binom_sf(k, n, p=0.5):
    """P(X >= k), X ~ Bin(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def paired_binary(base_rows, cand_rows, key="ok"):
    b = {r["id"]: r for r in base_rows if not r.get("skipped")}
    c = {r["id"]: r for r in cand_rows if not r.get("skipped")}
    ids = sorted(set(b) & set(c))
    lost = sum(1 for i in ids if b[i].get(key) and not c[i].get(key))
    gained = sum(1 for i in ids if c[i].get(key) and not b[i].get(key))
    n = len(ids)
    ab = sum(b[i].get(key, False) for i in ids) / max(1, n)
    ac = sum(c[i].get(key, False) for i in ids) / max(1, n)
    p = binom_sf(lost, lost + gained) if lost + gained else 1.0  # one-sided sign test: cand worse
    return {"n_paired": n, "base_acc": round(ab, 4), "cand_acc": round(ac, 4),
            "delta": round(ac - ab, 4), "lost": lost, "gained": gained, "p_worse": round(p, 4)}


# Practical-significance floors (accuracy points) per suite. A FAIL needs BOTH a drop
# larger than max(floor, measured noise |delta|) AND sign-test p < 0.05, OR a drop
# larger than the hard ceiling regardless of p.
FLOORS = {"gsm8k": 0.02, "ifeval": 0.03, "humaneval": 0.04, "niah": 0.05, "mmlupro": 0.02, "heplus": 0.03, "mbppplus": 0.03}
CEIL = {"gsm8k": 0.06, "ifeval": 0.08, "humaneval": 0.10, "niah": 0.15, "mmlupro": 0.05, "heplus": 0.08, "mbppplus": 0.06}


def cmp_binary(name, B, C, N):
    pb = paired_binary(B["suites"][name]["rows"], C["suites"][name]["rows"])
    noise = None
    if N and name in N["suites"]:
        noise = paired_binary(B["suites"][name]["rows"], N["suites"][name]["rows"])
    nf = abs(noise["delta"]) if noise else 0.0
    thr = max(FLOORS[name], nf)
    drop = -pb["delta"]
    if drop > CEIL[name] and (pb["n_paired"] >= 100 or pb["p_worse"] < 0.05):
        v = "FAIL"  # ceiling without significance only at n>=100: a 60-item same-config pilot swung 6.7 pts
    elif drop > thr and pb["p_worse"] < 0.05:
        v = "FAIL"
    elif drop > thr or pb["p_worse"] < 0.05:
        v = "WARN"
    else:
        v = "PASS"
    return {"verdict": v, **pb, "threshold_drop": round(thr, 4), "noise": noise}


def cmp_niah_by_length(B, C):
    out = {}
    br = [r for r in B["suites"]["niah"]["rows"] if not r.get("skipped")]
    cr = [r for r in C["suites"]["niah"]["rows"] if not r.get("skipped")]
    Ls = sorted({r["L"] for r in br} | {r["L"] for r in cr})
    for L in Ls:
        pb = paired_binary([r for r in br if r["L"] == L], [r for r in cr if r["L"] == L])
        conf = sum(r.get("confused", False) for r in cr if r["L"] == L)
        # per-length cells are small: fail only on >=3 lost-net or any bucket with >=2 wrong-needle answers
        v = "PASS"
        if pb["n_paired"] == 0:
            v = "NOT_COMPARABLE"
        elif pb["lost"] - pb["gained"] >= 3 or conf >= 2:
            v = "FAIL"
        elif pb["lost"] - pb["gained"] >= 1:
            v = "WARN"
        out[str(L)] = {"verdict": v, **pb, "cand_confused": conf}
    only_c = sorted({r["L"] for r in cr} - {r["L"] for r in br})
    return out, only_c


def lp_rows(R):
    return {r["id"]: r for r in R["suites"]["nll"]["rows"] if not r.get("skipped") and r.get("lps")}


def nll_delta(B, C):
    b, c = lp_rows(B), lp_rows(C)
    ids = sorted(set(b) & set(c))
    per = []
    maxabs = 0.0
    for i in ids:
        n = min(len(b[i]["lps"]), len(c[i]["lps"]))
        d = [c[i]["lps"][k] - b[i]["lps"][k] for k in range(n)]
        maxabs = max(maxabs, max(abs(x) for x in d))
        # aligned: both windows start at token C; compare only the shared first n tokens
        per.append((b[i]["C"], -sum(d) / n, sum(abs(x) for x in d) / n))
    if not per:
        return None
    by = {}
    for C_, dn, mad in per:
        by.setdefault(C_, []).append((dn, mad))
    return {"n_windows": len(per), "mean_dnll": round(sum(p[1] for p in per) / len(per), 5),
            "mean_abs_token_dlogprob": round(sum(p[2] for p in per) / len(per), 5),
            "max_abs_token_dlogprob": round(maxabs, 4),
            "by_ctx": {str(k): {"dnll": round(sum(x[0] for x in v) / len(v), 5),
                                "mad": round(sum(x[1] for x in v) / len(v), 5)} for k, v in sorted(by.items())}}


def det_agreement(B, C):
    """Cross-server greedy agreement using rep 0 of each; prefix-match length."""
    b = {r["id"]: r for r in B["suites"]["det"]["rows"] if r.get("rep") == 0 and r.get("tokens")}
    c = {r["id"]: r for r in C["suites"]["det"]["rows"] if r.get("rep") == 0 and r.get("tokens")}
    ids = sorted(set(b) & set(c))
    same = 0
    div = []
    lpd = []
    for i in ids:
        x, y = b[i]["tokens"], c[i]["tokens"]
        n = min(len(x), len(y))
        k = next((j for j in range(n) if x[j] != y[j]), n)
        if x == y:
            same += 1
        div.append(k)
        lpd += [abs(b[i]["logprobs"][j] - c[i]["logprobs"][j]) for j in range(k)]
    div.sort()
    return {"n": len(ids), "identical": same, "identical_rate": round(same / max(1, len(ids)), 4),
            "median_first_divergence": div[len(div) // 2] if div else None,
            "min_first_divergence": div[0] if div else None,
            "mean_abs_logprob_diff_on_shared_prefix": round(sum(lpd) / max(1, len(lpd)), 5)}


def load_merged(spec):
    """Comma-separated result files -> one result; later files add suites the earlier lack."""
    if not spec:
        return None
    out = None
    for pth in spec.split(","):
        d = json.load(open(pth))
        if out is None:
            out = d
            out["merged_from"] = [pth]
            continue
        out["merged_from"].append(pth)
        for k, v in d["suites"].items():
            out["suites"].setdefault(k, v)
            out["summary"].setdefault(k, d["summary"].get(k))
        if d.get("prefix_probe") and not out.get("prefix_probe"):
            out["prefix_probe"] = d["prefix_probe"]
            out["summary"]["prefix"] = d["summary"].get("prefix")
        out.setdefault("loads_snapshots", []).extend(d.get("loads_snapshots", []))
    return out


def total_degen(R):
    tot = {"scanned": 0, "bad": 0, "impossible_total": 0, "tok0_total": 0, "loops": 0}
    for name, res in R["suites"].items():
        dc = degen_count(res)
        for k in tot:
            tot[k] += dc[k]
    return tot



STRICT_BINARY = ("mmlupro", "heplus", "mbppplus", "gsm8k", "ifeval", "humaneval")
Z_A, Z_B = 1.645, 0.842  # one-sided alpha 0.05, power 0.80


def mde(n, disc_rate):
    """Minimum detectable paired accuracy drop (fraction) at alpha .05 one-sided, 80% power,
    given the discordance rate measured between two identical-config runs."""
    if not n or disc_rate is None:
        return None
    return (Z_A + Z_B) * math.sqrt(max(disc_rate, 1.0 / n) / n)


def n_for_mde(target, disc_rate):
    return math.ceil(((Z_A + Z_B) ** 2) * disc_rate / target ** 2) if disc_rate else None


def paired_rows(R, name):
    return [r for r in R["suites"][name]["rows"] if not r.get("skipped")] if name in R["suites"] else []


def refagree_compare(B, C):
    """Teacher-forced per-token logprob deltas on IDENTICAL token ids (the reference continuation)."""
    b = {r["id"]: r for r in B["suites"].get("refagree", {}).get("rows", []) if r.get("tf_lps")}
    c = {r["id"]: r for r in C["suites"].get("refagree", {}).get("rows", []) if r.get("tf_lps")}
    by = {}
    for i in sorted(set(b) & set(c)):
        if b[i]["tf_ids"] != c[i]["tf_ids"]:
            continue  # candidate was not forced on the same continuation (missing --ref-file)
        d = [c[i]["tf_lps"][k] - b[i]["tf_lps"][k] for k in range(min(len(b[i]["tf_lps"]), len(c[i]["tf_lps"])))]
        g_b, g_c = b[i].get("gen_ids") or [], c[i].get("gen_ids") or []
        n = min(len(g_b), len(g_c))
        fd = next((k for k in range(n) if g_b[k] != g_c[k]), n)
        e = by.setdefault(str(b[i]["L"]), {"tokens": 0, "sum_d": 0.0, "sum_abs": 0.0, "max_abs": 0.0,
                                           "prompts": 0, "greedy_identical": 0, "first_div": []})
        e["tokens"] += len(d); e["sum_d"] += sum(d); e["sum_abs"] += sum(abs(x) for x in d)
        e["max_abs"] = max(e["max_abs"], max(abs(x) for x in d) if d else 0)
        e["prompts"] += 1; e["greedy_identical"] += int(g_b == g_c and bool(g_b)); e["first_div"].append(fd)
    out = {}
    for L, e in sorted(by.items(), key=lambda x: int(x[0])):
        fdv = sorted(e["first_div"])
        out[L] = {"prompts": e["prompts"], "tokens": e["tokens"],
                  "dnll": round(-e["sum_d"] / max(1, e["tokens"]), 5),
                  "mean_abs": round(e["sum_abs"] / max(1, e["tokens"]), 5), "max_abs": round(e["max_abs"], 3),
                  "greedy_identical": e["greedy_identical"], "median_first_div": fdv[len(fdv) // 2] if fdv else None}
    return out


def strict_block(B, C, N, rep, a):
    st = {"suites": {}, "lossless_equivalent": None, "lossy_acceptable": None, "reasons_not_equivalent": [],
          "reasons_not_acceptable": [], "not_measured": []}
    pooled = {"n": 0, "lost": 0, "gained": 0, "noise_disc": 0, "noise_n": 0}
    for name in STRICT_BINARY:
        if name not in B["suites"] or name not in C["suites"]:
            continue
        pb = paired_binary(B["suites"][name]["rows"], C["suites"][name]["rows"])
        nz = paired_binary(B["suites"][name]["rows"], N["suites"][name]["rows"]) if N and name in N["suites"] else None
        disc = (nz["lost"] + nz["gained"]) / nz["n_paired"] if nz and nz["n_paired"] else None
        n = pb["n_paired"]
        diff = (pb["gained"] - pb["lost"]) / max(1, n)
        var = max(0.0, (pb["lost"] + pb["gained"]) / max(1, n) ** 2 - diff ** 2 / max(1, n))
        lower = diff - Z_A * math.sqrt(var)
        drop = -diff
        e = {**pb, "noise_discordance": None if disc is None else round(disc, 4),
             "noise_delta": nz["delta"] if nz else None,
             "mde_80pct": None if disc is None else round(mde(n, disc), 4),
             "n_for_2pt_mde": n_for_mde(0.02, disc), "delta_lower95": round(lower, 4)}
        # strict per-suite: FAIL = significant drop larger than 2 points, or lower bound below -2 pts with drop>noise
        if drop > 0.02 and pb["p_worse"] < 0.05:
            e["verdict"] = "FAIL"
        elif lower < -0.02:
            e["verdict"] = "INCONCLUSIVE"  # cannot exclude a >2pt drop at this n
        else:
            e["verdict"] = "PASS"
        e["within_noise"] = pb["p_worse"] >= 0.05 and (nz is None or abs(pb["delta"]) <= max(abs(nz["delta"]), e["mde_80pct"] or 0))
        st["suites"][name] = e
        pooled["n"] += n; pooled["lost"] += pb["lost"]; pooled["gained"] += pb["gained"]
        if nz:
            pooled["noise_disc"] += nz["lost"] + nz["gained"]; pooled["noise_n"] += nz["n_paired"]
    if pooled["n"]:
        n = pooled["n"]; diff = (pooled["gained"] - pooled["lost"]) / n
        var = max(0.0, (pooled["lost"] + pooled["gained"]) / n ** 2 - diff ** 2 / n)
        k = pooled["lost"] + pooled["gained"]
        pooled["delta"] = round(diff, 4)
        pooled["delta_lower95"] = round(diff - Z_A * math.sqrt(var), 4)
        pooled["p_worse"] = round(binom_sf(pooled["lost"], k), 4) if k else 1.0
        disc = pooled["noise_disc"] / pooled["noise_n"] if pooled["noise_n"] else None
        pooled["mde_80pct"] = None if disc is None else round(mde(n, disc), 4)
        st["pooled"] = pooled
    # reference agreement
    ra = refagree_compare(B, C) if "refagree" in B["suites"] and "refagree" in C["suites"] else {}
    ran = refagree_compare(B, N) if N and "refagree" in N["suites"] else {}
    st["refagree"] = {"cand_vs_ref": ra, "noise_vs_ref": ran}
    # ---- lossless-equivalent: every measured quantity inside the measured noise
    ne = st["reasons_not_equivalent"]
    if not ra:
        st["not_measured"].append("refagree (needs a reference run and --ref-file on the candidate)")
    if not ran:
        st["not_measured"].append("refagree noise floor (needs a 2nd identical-config reference run)")
    for L, x in ra.items():
        nzL = ran.get(L)
        if not nzL:
            continue
        tol_abs = 1.25 * nzL["mean_abs"] + 0.005
        tol_nll = 3 * abs(nzL["dnll"]) + 0.005
        x["tol_mean_abs"], x["tol_dnll"] = round(tol_abs, 5), round(tol_nll, 5)
        if x["mean_abs"] > tol_abs:
            ne.append(f"refagree {L}: mean|dlogprob| {x['mean_abs']} > noise tol {tol_abs:.4f}")
        if abs(x["dnll"]) > tol_nll:
            ne.append(f"refagree {L}: dNLL {x['dnll']} outside noise tol {tol_nll:.4f}")
    for name, e in st["suites"].items():
        if not e["within_noise"]:
            ne.append(f"{name}: delta {e['delta']} (p_worse {e['p_worse']}) not within noise")
    if pooled.get("p_worse", 1) < 0.05:
        ne.append(f"pooled accuracy drop significant (delta {pooled['delta']}, p {pooled['p_worse']})")
    dg = rep["metrics"].get("degeneracy", {})
    if dg.get("verdict") in ("FAIL", "WARN"):
        ne.append(f"degeneracy {dg.get('verdict')}")
    for k in ("prefix_cache", "nll", "det"):
        v = rep["metrics"].get(k, {}).get("verdict")
        if v in ("FAIL", "WARN"):
            ne.append(f"{k} {v}")
    st["lossless_equivalent"] = (not ne) if (ra and ran and st["suites"]) else None
    # ---- lossy-acceptable: no suite drop > 2 pts established; pooled lower bound >= -2 pts;
    #      refagree dNLL <= 0.02 nats/token at every length; degeneracy/prefix/long-context not FAIL
    na = st["reasons_not_acceptable"]
    for name, e in st["suites"].items():
        if e["verdict"] == "FAIL":
            na.append(f"{name}: significant drop {e['delta']}")
        elif e["verdict"] == "INCONCLUSIVE":
            na.append(f"{name}: cannot exclude a >2pt drop (lower95 {e['delta_lower95']}, n {e['n_paired']})")
    if pooled.get("delta_lower95") is not None and pooled["delta_lower95"] < -0.02:
        na.append(f"pooled lower95 {pooled['delta_lower95']} < -0.02")
    for L, x in ra.items():
        if x["dnll"] > 0.02:
            na.append(f"refagree {L}: dNLL {x['dnll']} > 0.02 nats/token")
    for k in ("degeneracy", "prefix_cache", "long_context_claim", "niah"):
        v = rep["metrics"].get(k, {}).get("verdict")
        if v == "FAIL":
            na.append(f"{k} FAIL")
    for L, m in rep["metrics"].get("niah_by_length", {}).items():
        if m.get("verdict") == "FAIL":
            na.append(f"niah {L} FAIL")
    hard = [x for x in na if "cannot exclude" not in x and "lower95" not in x]
    if not st["suites"]:
        st["lossy_acceptable"] = None
    elif hard:
        st["lossy_acceptable"] = False
    elif na:
        st["lossy_acceptable"] = None  # no drop established, but n too small to exclude a 2-pt drop
        st["not_measured"].append("2-point resolution not reached: " + "; ".join(na))
    else:
        st["lossy_acceptable"] = True
    return st



def nll_windows(B, C):
    """Per-window dNLL (cand - base) on the shared first n tokens of each window."""
    if not C or "nll" not in C["suites"]:
        return []
    b, c = lp_rows(B), lp_rows(C)
    out = []
    for i in sorted(set(b) & set(c)):
        n = min(len(b[i]["lps"]), len(c[i]["lps"]))
        out.append(-sum(c[i]["lps"][k] - b[i]["lps"][k] for k in range(n)) / n)
    return out


def decpath_verdict(B, C, N):
    from scipy import stats
    cand = {}
    for rep in (0, 1):
        cand.update({(k[0], k[1], rep): v for k, v in decpath_pairs(B, C, rep, rep).items()})
    noise = {(k[0], k[1], "within"): v for k, v in decpath_pairs(B, B, 0, 1).items()}
    if N and "decpath" in N["suites"]:
        for rep in (0, 1):
            noise.update({(k[0], k[1], f"x{rep}"): v for k, v in decpath_pairs(B, N, rep, rep).items()})
    def regroup(dct):
        return {(k[0] + "#" + str(k[2]), k[1]): v for k, v in dct.items()}
    cs, ns = decpath_summary(regroup(cand)), decpath_summary(regroup(noise))
    out = {"by_conc": {}, "noise_sources": sorted({k[2] for k in noise}),
           "scope": "decode path: greedy tokens + output logprobs, single-chunk prompts"}
    verdicts = []
    for conc, c in cs.items():
        n = ns.get(conc)
        e = {"cand_vs_base": {k: v for k, v in c.items() if not k.startswith("_")},
             "noise": {k: v for k, v in n.items() if not k.startswith("_")} if n else None}
        if c["identical"] == c["prompts"]:
            v = "BITWISE"
        elif not n:
            v = "UNCALIBRATED"
        else:
            tol = max(1.5 * n["mean_abs_dlogprob_agreeing"], n["mean_abs_dlogprob_agreeing"] + 0.002, 0.003)
            p_len = float(stats.mannwhitneyu(c["_agree"], n["_agree"], alternative="less").pvalue) \
                if len(set(c["_agree"] + n["_agree"])) > 1 else 1.0
            p_pp = float(stats.mannwhitneyu(c["_pp"], n["_pp"], alternative="greater").pvalue) \
                if c["_pp"] and n["_pp"] and len(set(c["_pp"] + n["_pp"])) > 1 else 1.0
            ratio = c["agree_len_median"] / max(1, n["agree_len_median"])
            e.update({"tol_mean_abs_dlogprob": round(tol, 5), "p_agree_shorter": round(p_len, 4),
                      "p_dlogprob_larger": round(p_pp, 4), "agree_median_ratio": round(ratio, 3)})
            bad_lp = c["mean_abs_dlogprob_agreeing"] > tol and p_pp < 0.01
            bad_len = p_len < 0.01 and ratio < 0.75
            v = "FAIL" if (bad_lp or bad_len) else ("WARN" if (p_len < 0.05 or p_pp < 0.05) else "PASS")
        e["verdict"] = v
        verdicts.append(v)
        out["by_conc"][conc] = e
    out["verdict"] = ("FAIL" if "FAIL" in verdicts else "WARN" if "WARN" in verdicts else
                      "UNCALIBRATED" if "UNCALIBRATED" in verdicts else
                      "BITWISE" if verdicts and all(x == "BITWISE" for x in verdicts) else "PASS")
    return out



# Seeded base-vs-base NLL noise, measured on the reference build (identical config, same boot,
# 96 paired windows): per-window |dNLL| mean by context. ctx 512 is bit-deterministic. Only used
# when no repeated-base run is supplied; supply your own (--noise / --nll-noise) for your hardware.
NLL_NOISE_SEED = {512: 0.0, 4096: 0.0347, 16384: 0.0528, 65536: 0.0411}
SINGLE_CHUNK_MAX = 4096 - 128  # windows whose prompt fits one prefill chunk (chunked_prefill_size 4096)


def nll_windows_by_ctx(B, C):
    """{C: [(dNLL, max_abs_token_delta), ...]} for windows present in both runs (cand - base)."""
    out = {}
    if not C or "nll" not in C["suites"]:
        return out
    b, c = lp_rows(B), lp_rows(C)
    for i in sorted(set(b) & set(c)):
        n = min(len(b[i]["lps"]), len(c[i]["lps"]))
        d = [c[i]["lps"][k] - b[i]["lps"][k] for k in range(n)]
        out.setdefault(b[i]["C"], []).append((-sum(d) / n, max(abs(x) for x in d)))
    return out


def nll_verdict(B, C, N, a):
    """NLL verdict.
    * ctx <= SINGLE_CHUNK_MAX (512 in the standard set): prefill is ONE chunk and measured bit-deterministic
      (base-vs-base max token delta 0.0). HARD check: lossless claims FAIL on any token |dlogprob| > 1e-4;
      lossy claims FAIL if the mean dNLL there exceeds 0.02 nats/token.
    * ctx >= 4k: multi-chunk prefill is run-to-run nondeterministic, so ONLY noise-relative: per-bucket and
      pooled z-test of the candidate's signed mean dNLL against the base-vs-base spread (measured from
      --noise / --nll-noise when present, else the seeded per-context values), plus a practical floor.
    NLL is prefill only: it cannot certify decode-path (T<=16) levers -- decpath is the instrument there."""
    from scipy import stats
    cw = nll_windows_by_ctx(B, C)
    noise = {}
    srcs = []
    if N and "nll" in N["suites"]:
        for k, v in nll_windows_by_ctx(B, N).items():
            noise.setdefault(k, []).extend(v)
        srcs.append(a.noise)
    for pth in (a.nll_noise.split(",") if a.nll_noise else []):
        for k, v in nll_windows_by_ctx(B, json.load(open(pth))).items():
            noise.setdefault(k, []).extend(v)
        srcs.append(pth)
    floor = 0.003 if a.claim == "lossless" else 0.02
    m = {"scope": "prefill only; cannot see decode-path (T<=16) kernels -- use decpath",
         "claim": a.claim, "practical_floor": floor, "noise_sources": srcs or ["seeded:NLL_NOISE_SEED"],
         "single_chunk": {}, "by_ctx": {}}
    if not cw:
        m["verdict"] = "NOT_COMPARABLE"
        return m
    verdicts = []
    # --- hard single-chunk check
    sc = [x for Cc, v in cw.items() if Cc <= SINGLE_CHUNK_MAX for x in v]
    if sc:
        mx = max(x[1] for x in sc)
        md = sum(x[0] for x in sc) / len(sc)
        base_nd = [x[1] for Cc, v in noise.items() if Cc <= SINGLE_CHUNK_MAX for x in v]
        e = {"windows": len(sc), "max_abs_token_dlogprob": round(mx, 6), "mean_dnll": round(md, 6),
             "noise_max_abs_token_dlogprob": round(max(base_nd), 6) if base_nd else "seeded 0.0"}
        if a.claim == "lossless":
            v = "FAIL" if mx > 1e-4 else "PASS"
            e["rule"] = "lossless: any token |dlogprob| > 1e-4 at single-chunk ctx FAILs (base is bit-deterministic)"
        else:
            v = "FAIL" if md > 0.02 else ("WARN" if md > 0.01 else "PASS")
            e["rule"] = "lossy: mean dNLL at single-chunk ctx > 0.02 nats/token FAILs (deterministic, no noise needed)"
        e["verdict"] = v
        verdicts.append(v)
        m["single_chunk"] = e
    # --- noise-relative multi-chunk buckets
    zs = []
    pooled_c, pooled_var = [], 0.0
    for Cc, v in sorted(cw.items()):
        if Cc <= SINGLE_CHUNK_MAX:
            continue
        cv = [x[0] for x in v]
        nv = [x[0] for x in noise.get(Cc, [])]
        if len(nv) >= 8:
            sd_n = stats.tstd(nv)
            mu_n = sum(nv) / len(nv)
            src = "measured"
        else:
            seed = NLL_NOISE_SEED.get(Cc) or max(NLL_NOISE_SEED.values())
            sd_n, mu_n, src = 1.2533 * seed, 0.0, "seeded"  # half-normal: sd = mean|x| * sqrt(pi/2)
        mc = sum(cv) / len(cv)
        se = sd_n * math.sqrt(1 / len(cv) + (1 / len(nv) if src == "measured" else 0))
        z = (mc - mu_n) / se if se > 0 else 0.0
        p = float(stats.norm.sf(z))
        bv = "FAIL" if (p < 0.01 and mc - mu_n > floor) else ("WARN" if p < 0.05 else "PASS")
        m["by_ctx"][str(Cc)] = {"windows": len(cv), "mean_dnll": round(mc, 5), "noise_mean": round(mu_n, 5),
                                "noise_sd": round(sd_n, 5), "noise": src, "z": round(z, 2), "p_worse": round(p, 4),
                                "verdict": bv}
        verdicts.append(bv)
        pooled_c += cv
        pooled_var += (se ** 2) * (len(cv) ** 2)
        zs.append((mc - mu_n) * len(cv))
    if pooled_c:
        n = len(pooled_c)
        exc = sum(zs) / n
        se = math.sqrt(pooled_var) / n
        z = exc / se if se > 0 else 0.0
        p = float(stats.norm.sf(z))
        pv = "FAIL" if (p < 0.01 and exc > floor) else ("WARN" if p < 0.05 else "PASS")
        m["multi_chunk_pooled"] = {"windows": n, "excess_dnll": round(exc, 5), "z": round(z, 2),
                                   "p_worse": round(p, 4), "verdict": pv}
        verdicts.append(pv)
    m["verdict"] = "FAIL" if "FAIL" in verdicts else ("WARN" if "WARN" in verdicts else "PASS")
    return m


def cmd_compare(a):
    B = load_merged(a.baseline)
    C = load_merged(a.candidate)
    N = load_merged(a.noise)
    rep = {"schema": "qgate-verdict/v1", "baseline": a.baseline, "candidate": a.candidate,
           "noise": a.noise, "claim": a.claim, "baseline_server": B["server"],
           "candidate_server": C["server"], "metrics": {}}
    def valid(R, name):
        res = R["suites"].get(name)
        return res is not None and validity(res, a.invalid_frac)["valid"]
    rep["invalid"] = {}
    for R, tag in ((B, "baseline"), (C, "candidate"), (N, "noise")):
        if R:
            bad = [k for k in R["suites"] if not valid(R, k)]
            if bad:
                rep["invalid"][tag] = bad
                for k in bad:  # drop invalid suites so no verdict is computed from them
                    R["suites"].pop(k)
    for name in ("gsm8k", "ifeval", "humaneval", "niah", "mmlupro", "heplus", "mbppplus"):
        if name in B["suites"] and name in C["suites"]:
            rep["metrics"][name] = cmp_binary(name, B, C, N)
    if "niah" in B["suites"] and "niah" in C["suites"]:
        byL, only_c = cmp_niah_by_length(B, C)
        rep["metrics"]["niah_by_length"] = byL
        if only_c:
            rep["metrics"]["niah_candidate_only_lengths"] = {
                str(L): C["summary"]["niah"]["by_length"].get(L) or C["summary"]["niah"]["by_length"].get(str(L))
                for L in only_c}
    if "nll" in B["suites"] and "nll" in C["suites"]:
        rep["metrics"]["nll"] = nll_verdict(B, C, N, a)
    if "decpath" in B["suites"] and "decpath" in C["suites"]:
        rep["metrics"]["decpath"] = decpath_verdict(B, C, N)
    if "det" in B["suites"] and "det" in C["suites"]:
        d = det_agreement(B, C)
        nd = det_agreement(B, N) if N and "det" in N["suites"] else None
        # Measured: greedy decode on the reference stack is NOT bitwise deterministic (0/16 prompts identical across
        # repeats; median first divergence at token 13). Exact-match is therefore useless as a lossless test.
        # The usable signal is the logprob gap on the SHARED greedy prefix (same context, same next token).
        lim = max(0.08, 3 * nd["mean_abs_logprob_diff_on_shared_prefix"]) if nd else 0.08
        v = "INFO"
        if a.claim == "lossless" and d["n"]:
            v = "PASS" if d["mean_abs_logprob_diff_on_shared_prefix"] <= lim else "FAIL"
        rep["metrics"]["det"] = {"verdict": v, "limit_mean_abs_dlogprob": round(lim, 4),
                                 "cand_vs_base": d, "noise_vs_base": nd}
    # --- degeneracy (intel: #38290/#36811 '!' collapse, #36806 token-0 loops, #38319 impossible ids)
    cd, bd = total_degen(C), total_degen(B)
    nd_ = total_degen(N) if N else None
    refs = [x for x in (bd, nd_) if x and x["scanned"]]
    ref_rate = max([x["bad"] / x["scanned"] for x in refs] or [0.0])
    allowed = math.ceil(ref_rate * cd["scanned"]) + 1
    v = "PASS"
    if cd["impossible_total"] > 0:
        v = "FAIL"
    elif cd["scanned"] == 0 or not refs:
        v = "NOT_COMPARABLE"
    elif cd["bad"] > allowed:
        v = "FAIL"
    elif cd["bad"] > math.ceil(ref_rate * cd["scanned"]):
        v = "WARN"
    rep["metrics"]["degeneracy"] = {"verdict": v, "candidate": cd, "baseline": bd, "noise": nd_,
                                    "reference_bad_rate": round(ref_rate, 5), "allowed_bad": allowed,
                                    "rule": "FAIL on any impossible id, or bad > ceil(ref_rate x scanned) + 1"}
    # --- radix-cache poison probe (#38319 / #38355)
    def prefix_summary(R):
        return summarize_prefix(R["prefix_probe"]) if R and R.get("prefix_probe") else None
    cps = prefix_summary(C)
    if cps:
        refs = [x for x in (prefix_summary(B), prefix_summary(N)) if x]
        for pth in (a.prefix_ref.split(",") if a.prefix_ref else []):
            x = prefix_summary(json.load(open(pth)))
            if x:
                x["source"] = pth
                refs.append(x)
        refs = [x for x in refs if x["mode"] == cps["mode"]]
        m = {"candidate": cps, "references": refs, "mode": cps["mode"]}
        if cps["impossible_tokens"] > 0:
            v = "FAIL"
        elif cps["mode"] == "gen":
            ref_mean = max([x["max_mean_abs_dlogprob_shared"] for x in refs] or [0.0])
            ref_bad = max([x["degenerate"] for x in refs] or [0])
            lim = max(0.1, 3 * ref_mean)
            m["limit_mean_abs_dlogprob_shared"] = round(lim, 4)
            if not refs:
                v = "NOT_COMPARABLE"
            elif cps["cache_hit_pairs"] == 0:
                v = "NOT_COMPARABLE"
                m["note"] = "no warm/late request reported cached_tokens > 0: the cache was not exercised"
            elif cps["max_mean_abs_dlogprob_shared"] > lim or cps["degenerate"] > ref_bad + 1:
                v = "FAIL"
            elif cps["degenerate"] > ref_bad:
                v = "WARN"
            else:
                v = "PASS"
            m["rule"] = ("gen mode: FAIL on any impossible id, a degenerate continuation beyond the reference +1, "
                         "or mean |dlogprob| on the shared greedy continuation > max(0.1, 3 x reference); "
                         "NOT_COMPARABLE if no reference or the cache was never hit")
        else:
            # v1 input-window probe. Measured: warm/late never hit the cache (cached_tokens=0), so this is
            # PREFILL-RECOMPUTE drift, not cache integrity. Per-token max is heavy-tailed and grows with
            # context (old engine, 49k: max 1.42 nats cold-vs-warm), so gate on the per-length window mean.
            ref_by_L = {}
            for x in refs:
                for L, vals in x["mean_abs_by_L"].items():
                    ref_by_L[L] = max(ref_by_L.get(L, 0.0), max(vals))
            worst = {}
            for L, vals in cps["mean_abs_by_L"].items():
                ref = ref_by_L.get(L)
                worst[L] = {"cand_max_window_mean": max(vals), "ref_max_window_mean": ref,
                            "ratio": round(max(vals) / ref, 3) if ref else None}
            m["by_L"] = worst
            m["cache_exercised"] = cps["cache_hit_pairs"] > 0
            ratios = [w["ratio"] for w in worst.values() if w["ratio"] is not None]
            if not ratios:
                v = "NOT_COMPARABLE"
            elif max(ratios) > 3:
                v = "WARN"  # numerics drift beyond reference; nll/det gate numerics, this cannot see the cache
            else:
                v = "PASS"
            m["rule"] = ("v1 input-window mode (cache NOT exercised): FAIL only on impossible ids; WARN if any "
                         "length's worst window-mean |dlogprob| exceeds 3x the reference at the same length")
        m["verdict"] = v
        rep["metrics"]["prefix_cache"] = m
    # --- YaRN actually applied: a context claim past native 262144 must be PROVEN by retrieval there
    NATIVE = 262144
    cl = (C.get("server") or {}).get("context_length") or 0
    if cl > NATIVE:
        rows = [r for r in C["suites"].get("niah", {}).get("rows", []) if not r.get("skipped") and r["L"] > NATIVE]
        acc = sum(r["ok"] for r in rows) / len(rows) if rows else None
        v = "FAIL" if (acc is None or acc < 0.8) else "PASS"
        pool = (C.get("server") or {}).get("max_total_num_tokens") or 0
        if "niah" in rep["invalid"].get("candidate", []):
            v = "INVALID"
        elif not rows and pool and pool < NATIVE + 1024:
            v = "NOT_MEASURABLE"  # the instance's KV pool cannot admit any request past native
            rep.setdefault("claims_unverified", []).append(
                f"context_length {cl}: KV pool {pool} tokens cannot admit a request past {NATIVE}")
        nll_far = [r for r in C["suites"].get("nll", {}).get("rows", []) if not r.get("skipped") and r["C"] > NATIVE]
        rep["metrics"]["long_context_claim"] = {
            "verdict": v, "context_length": cl, "niah_cells_past_native": len(rows),
            "niah_acc_past_native": None if acc is None else round(acc, 4),
            "nll_windows_past_native": {r["id"]: round(r["nll"], 4) for r in nll_far},
            "override": (C.get("server") or {}).get("json_model_override_args"),
            "rule": "context_length > 262144 requires NIAH cells beyond 262144 with acc >= 0.8; "
                    "a no-op rope override serves the length but cannot retrieve there"}
    # --- speculative acceptance / uptime confound (#37326): recorded, never gated
    rep["accept_length"] = {
        "baseline": [x.get("accept_length") for x in B.get("loads_snapshots", [])],
        "candidate": [x.get("accept_length") for x in C.get("loads_snapshots", [])]}
    vs = [m["verdict"] for k, m in rep["metrics"].items() if isinstance(m, dict) and "verdict" in m]
    vs += [m["verdict"] for m in rep["metrics"].get("niah_by_length", {}).values()]
    rep["overall"] = "FAIL" if "FAIL" in vs else ("WARN" if "WARN" in vs else "PASS")
    if a.strict:
        rep["strict"] = strict_block(B, C, N, rep, a)
        s_ = rep["strict"]
        rep["strict_verdict"] = ("LOSSLESS-EQUIVALENT" if s_["lossless_equivalent"] else
                                 "LOSSY-ACCEPTABLE" if s_["lossy_acceptable"] else
                                 "NOT-ACCEPTABLE" if s_["lossy_acceptable"] is False else "NOT-CERTIFIABLE")
    if rep["invalid"]:
        rep["overall"] = "INVALID" if rep["overall"] != "FAIL" else "FAIL+INVALID"
    s = json.dumps(rep, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    print(s)
    sys.exit(1 if rep["overall"].startswith("FAIL") else (2 if rep["overall"] == "INVALID" else 0))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--url", required=True, help="server root, e.g. http://127.0.0.1:30000")
    r.add_argument("--model", default=MODEL_DEFAULT)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--suites", default="gsm8k,ifeval,humaneval,niah,nll,det,degen")
    r.add_argument("--conc", type=int, default=2)
    r.add_argument("--pool-frac", type=float, default=0.7)
    r.add_argument("--timeout", type=float, default=900)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--n-gsm8k", type=int, default=150)
    r.add_argument("--n-ifeval", type=int, default=120)
    r.add_argument("--n-humaneval", type=int, default=80)
    r.add_argument("--niah-lengths", default="4096,16384,32768,65536,131072")
    r.add_argument("--niah-depths", default="0.1,0.3,0.5,0.7,0.9")
    r.add_argument("--niah-trials", type=int, default=2)
    r.add_argument("--niah-max-tokens", type=int, default=1024)
    r.add_argument("--nll-ctx", default="512,4096,16384,65536,131072")
    r.add_argument("--nll-books", type=int, default=5)
    r.add_argument("--nll-window", type=int, default=128,
                   help="scored tokens per request; logits memory = window x 248k vocab. Lower on a nearly-full card")
    r.add_argument("--invalid-frac", type=float, default=0.05)
    r.add_argument("--nll-code", type=int, default=1, help="also score a private code corpus (QGATE_CODE_CORPUS; skipped if unset)")
    r.add_argument("--det-prompts", type=int, default=16)
    r.add_argument("--det-reps", type=int, default=2)
    r.add_argument("--det-max-tokens", type=int, default=256)
    r.add_argument("--n-mmlupro", type=int, default=2000)
    r.add_argument("--n-heplus", type=int, default=164)
    r.add_argument("--n-mbppplus", type=int, default=378)
    r.add_argument("--think", type=int, default=1, help="1 = production thinking mode (default)")
    r.add_argument("--ref-file", help="reference run whose refagree continuations are teacher-forced here")
    r.add_argument("--refagree-lengths", default="4096,32768,131072")
    r.add_argument("--refagree-per-len", type=int, default=8)
    r.add_argument("--refagree-gen", type=int, default=128)
    r.add_argument("--decpath-prompts", type=int, default=48)
    r.add_argument("--decpath-tokens", type=int, default=384)
    r.add_argument("--decpath-reps", type=int, default=2)
    r.add_argument("--decpath-conc", default="1,4")
    r.add_argument("--nll-starts", type=int, default=1, help="windows per (corpus, ctx) at different document offsets")
    r.add_argument("--degen-n", type=int, default=60)
    r.add_argument("--degen-conc", type=int, default=2, help="2 on a shared live lane; = max_running_requests on a dedicated instance")
    r.add_argument("--degen-max-tokens", type=int, default=6144)
    r.add_argument("--prefix-n", type=int, default=6, help="0 disables the radix-poison probe")
    c = sp.add_parser("compare")
    c.add_argument("--baseline", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--noise", help="second baseline run of the SAME server config (defines noise floor)")
    c.add_argument("--claim", choices=["lossless", "lossy"], default="lossless")
    c.add_argument("--out")
    c.add_argument("--invalid-frac", type=float, default=0.05)
    c.add_argument("--nll-noise", help="comma list of extra runs of a PREFILL-IDENTICAL config (repeated base runs); "
                                       "their per-window dNLL vs --baseline is pooled into the NLL noise distribution")
    c.add_argument("--strict", action="store_true", help="add the strict block: lossless-equivalent / lossy-acceptable")
    c.add_argument("--prefix-ref", help="extra result file(s), comma-separated, whose prefix_probe is the "
                                        "reference when baseline/noise carry none")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    main()

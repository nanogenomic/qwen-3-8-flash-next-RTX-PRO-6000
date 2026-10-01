# Copyright © 2025 Ligandal, Inc. All rights reserved.
"""Hot-cache correctness + timing for QSA host K/V.

Kernel-equivalence test on one small GPU. Tensor contents and the step-to-step
selection drift are SYNTHETIC (seeded); the real hit rate comes only from the
model run (SGLANG_QSA_HOST_KV_STATS=1). What this proves: the cached gather is
bit-identical to the uncached host gather under speculative rewrites of the
newest tokens, verify rows sharing a request, set conflicts, and request-slot
reuse; and what a given hit rate costs/saves in time.
"""
import json
import sys

import torch

sys.path.insert(0, sys.argv[1])
from sglang.srt.mem_cache.qsa_host_kv import allocate_host_kv_buffers  # noqa: E402
from sglang.srt.layers.attention.qsa.host_kv_cache import QSAHostKVCache  # noqa: E402
from sglang.srt.layers.attention.qsa.sparse_attn import (  # noqa: E402
    qwen_sparse_kv_extraction_compact_triton,
)

dev = torch.device("cuda:0")
H, D, PAGE, R, NB = 2, 256, 64, 4, 512
TOPK = NB * R + R - 1
STRIDE = (TOPK + 63) // 64 * 64
g = torch.Generator(device=dev).manual_seed(1)


def randn(*shape):
    return torch.randn(*shape, device=dev, generator=g, dtype=torch.float32).to(torch.bfloat16)


def make_rows(reqs, lens, sel_blocks):
    """4 verify rows per request: lengths L+1..L+4, same block selection + tail."""
    idx_rows, req_rows, len_rows = [], [], []
    for r, L, blocks in zip(reqs, lens, sel_blocks):
        for j in range(4):
            n = L + j + 1
            full = n // R
            b = blocks[blocks < full]
            if b.numel() > NB:
                b = b[:NB]
            toks = (b[:, None] * R + torch.arange(R, device=dev)).flatten()
            tail = torch.arange(full * R, n, device=dev)
            row = torch.full((TOPK,), -1, dtype=torch.int64, device=dev)
            row[: toks.numel()] = toks
            row[toks.numel(): toks.numel() + tail.numel()] = tail
            idx_rows.append(row)
            req_rows.append(r)
            len_rows.append(n)
    return (
        torch.stack(idx_rows).int().contiguous(),
        torch.tensor(req_rows, dtype=torch.int32, device=dev),
        torch.tensor(len_rows, dtype=torch.int32, device=dev),
    )


def gather(fn_cache, layer, kb, vb, r2t, req_idx, idx, lens):
    batch = idx.shape[0]
    cu = torch.arange(batch + 1, dtype=torch.int32, device=dev) * STRIDE
    ok = torch.full((batch * STRIDE, H, D), 3.0, dtype=torch.bfloat16, device=dev)
    ov = torch.full_like(ok, 3.0)
    if fn_cache is None:
        qwen_sparse_kv_extraction_compact_triton(
            kb, vb, r2t, req_idx, idx, lens, cu, ok, ov, batch, TOPK, zero_fill_cols=STRIDE
        )
    else:
        fn_cache.gather(layer, kb, vb, r2t, req_idx, idx, lens, cu, ok, ov, batch, TOPK,
                        zero_fill_cols=STRIDE)
    return ok, ov


def drift(blocks, full, keep, n=NB):
    """Keep `keep` of the previous selection, redraw the rest (synthetic)."""
    nkeep = int(round(keep * n))
    kept = blocks[torch.randperm(blocks.numel(), device=dev, generator=g)[:nkeep]]
    kept = kept[kept < full]
    fresh = torch.randperm(full, device=dev, generator=g)
    fresh = fresh[~torch.isin(fresh, kept)][: n - kept.numel()]
    # always include the two newest complete groups (recency is typical and it
    # exercises the speculative-rewrite margin)
    newest = torch.tensor([full - 1, full - 2], device=dev)
    out = torch.unique(torch.cat([kept, fresh, newest]))[:n]
    return out


def correctness(kv_dtype=torch.bfloat16):
    size, slots, layers = 65536, 6, 2
    shape = (size + PAGE, H, D)
    kh, vh, owner = allocate_host_kv_buffers(layer_num=layers, k_shape=shape, v_shape=shape,
                                             dtype=kv_dtype, device=dev)
    for l in range(layers):
        kh[l].copy_(randn(*shape).to(kv_dtype))
        vh[l].copy_(randn(*shape).to(kv_dtype))
    cache = QSAHostKVCache(num_layers=layers, num_req_slots=slots, heads=H, dim=D,
                           dtype=kv_dtype, ratio=R, num_sets=1024, margin=16,
                           device=dev)
    r2t = torch.zeros(slots, size + PAGE, dtype=torch.int32, device=dev)
    pages = torch.randperm(size // PAGE, device=dev, generator=g)
    # request slots 1..3 each own a disjoint third of the pages
    per = pages.numel() // 3
    reqs = [1, 2, 3]
    for i, r in enumerate(reqs):
        own = pages[i * per:(i + 1) * per]
        m = (own[:, None] * PAGE + torch.arange(PAGE, device=dev) + PAGE).flatten()
        r2t[r, : m.numel()] = m.int()
    cap = per * PAGE - 64
    lens = [cap - 400, cap - 900, cap - 1500]
    sel = [torch.randperm(L // R, device=dev, generator=g)[:NB] for L in lens]
    res = {"steps": 0, "mismatch_steps": 0, "hit_tokens": 0, "sel_tokens": 0}
    for step in range(40):
        # speculative writes: positions L..L+3 get fresh K/V (drafts), overwriting
        # whatever the previous step's rejected drafts left there
        for r, L in zip(reqs, lens):
            pos = torch.arange(L, L + 4, device=dev)
            for l in range(layers):
                kh[l][r2t[r, pos].long()] = randn(4, H, D).to(kv_dtype)
                vh[l][r2t[r, pos].long()] = randn(4, H, D).to(kv_dtype)
        idx, req_idx, rl = make_rows(reqs, lens, sel)
        cache.stats.zero_()
        for l in range(layers):
            a = gather(None, l, kh[l], vh[l], r2t, req_idx, idx, rl)
            b = gather(cache, l, kh[l], vh[l], r2t, req_idx, idx, rl)
            if not (torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])):
                res["mismatch_steps"] += 1
        res["steps"] += 1
        # accept 1..4 tokens, drift the selection 85% (synthetic)
        acc = [1 + (step + i) % 4 for i in range(3)]
        lens = [L + a_ for L, a_ in zip(lens, acc)]
        sel = [drift(s_, L // R, 0.85) for s_, L in zip(sel, lens)]
    torch.cuda.synchronize()
    # stats kernel counters are off unless the env is set; count hits directly
    # request-slot reuse: slot 2 gets a new request whose positions map to other
    # pages with different contents. With invalidation: exact. Without: stale.
    new_pages = pages[3 * per:] if pages.numel() > 3 * per else pages[:per]
    # re-map slot 2 onto slot 1's pages shifted (different physical slots per position)
    m = r2t[1, : per * PAGE].flip(0).clone()
    idx, req_idx, rl = make_rows([2], [lens[1]], [sel[1]])
    r2t_new = r2t.clone()
    r2t_new[2, : m.numel()] = m
    a = gather(None, 0, kh[0], vh[0], r2t_new, req_idx, idx, rl)
    b_stale = gather(cache, 0, kh[0], vh[0], r2t_new, req_idx, idx, rl)
    res["negative_control_stale_without_invalidate"] = not torch.equal(a[0], b_stale[0])
    cache.invalidate(0, torch.tensor([2], device=dev))
    b = gather(cache, 0, kh[0], vh[0], r2t_new, req_idx, idx, rl)
    res["reuse_exact_after_invalidate"] = torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    res["pass"] = (res["mismatch_steps"] == 0 and res["reuse_exact_after_invalidate"]
                   and res["negative_control_stale_without_invalidate"])
    del kh, vh, owner, cache
    return res


def timing():
    size, layers = 131072, 1
    shape = (size + PAGE, H, D)
    kh, vh, owner = allocate_host_kv_buffers(layer_num=layers, k_shape=shape, v_shape=shape,
                                             dtype=torch.bfloat16, device=dev)
    kh[0].copy_(randn(*shape))
    vh[0].copy_(randn(*shape))
    out = {}
    for nreq in (1, 4):
        slots = nreq + 1
        cache = QSAHostKVCache(num_layers=1, num_req_slots=slots, heads=H, dim=D,
                               dtype=torch.bfloat16, ratio=R, num_sets=4096, margin=16,
                               device=dev)
        r2t = torch.zeros(slots, size + PAGE, dtype=torch.int32, device=dev)
        pages = torch.randperm(size // PAGE, device=dev, generator=g)
        for r in range(1, slots):
            r2t[r, :size] = (pages[:, None] * PAGE + torch.arange(PAGE, device=dev) + PAGE).flatten().int()
        reqs = list(range(1, slots))
        L = size - 100
        for keep in (1.0, 0.9, 0.7, 0.0):
            sel = [torch.randperm(L // R, device=dev, generator=g)[:NB] for _ in reqs]
            # warm the cache with the previous selection, then time the next step
            times = {"host": 0.0, "cached": 0.0}
            n = 0
            for it in range(12):
                idx, req_idx, rl = make_rows(reqs, [L] * nreq, sel)
                if it >= 4:
                    for name, c in (("host", None), ("cached", cache)):
                        s_, e_ = torch.cuda.Event(True), torch.cuda.Event(True)
                        if name == "cached":
                            # time the cached path on a state warmed by the
                            # previous selection: undo nothing, just measure
                            pass
                        s_.record()
                        gather(c, 0, kh[0], vh[0], r2t, req_idx, idx, rl)
                        e_.record()
                        torch.cuda.synchronize()
                        times[name] += s_.elapsed_time(e_)
                    n += 1
                else:
                    gather(cache, 0, kh[0], vh[0], r2t, req_idx, idx, rl)
                sel = [drift(s_, L // R, keep) for s_ in sel]
            out[f"reqs{nreq}_rows{4*nreq}_keep{keep}"] = {
                "host_ms_per_layer": round(times["host"] / n, 4),
                "cached_ms_per_layer": round(times["cached"] / n, 4),
            }
        del cache
    return out


if __name__ == "__main__":
    res = {"gpu": torch.cuda.get_device_name(0)}
    res["correctness"] = correctness()
    res["correctness_fp8"] = correctness(torch.float8_e4m3fn)
    res["timing_synthetic_drift"] = timing()
    res["peak_device_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["correctness"]["pass"] and res["correctness_fp8"]["pass"] else 1)

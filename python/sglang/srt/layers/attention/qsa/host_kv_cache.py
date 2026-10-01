# Copyright © 2025 Ligandal, Inc. All rights reserved.
"""GPU hot cache in front of host-resident QSA K/V.

With ``SGLANG_QSA_HOST_KV`` the full-attention K/V live in mapped host memory,
so every decode row would pull its whole top-budget selection (2051 tokens x
2 KiB = 4.2 MB per layer) across PCIe. Consecutive decode steps -- and the
MTP verify rows of one step -- select mostly the same compressed groups, so a
small per-request cache on the GPU turns most of those reads into HBM reads.

Design: direct-mapped, one entry per compressed group (``ratio`` tokens), per
(layer, request slot). Set = group % num_sets, tag = group id.

Three kernels per layer call:
  1. gather   : selected token -> packed scratch, from the cache on a tag hit,
                from host memory otherwise (drop-in for ``_compact_kv``).
  2. claim    : every cacheable missed group bids for its set (atomic max), so
                exactly one group owns a contested set.
  3. fill     : the winning groups copy their freshly packed rows (already in
                HBM) into the cache and publish the tag.
Tags are only read by a LATER kernel launch than the one that writes them, so
data and tag are always consistent at read time.

Correctness invariants:
  * Only committed, immutable tokens are cached: a group is cacheable iff
    ``(g + 1) * ratio + margin <= row_seq_len``, with ``margin`` above the
    speculative draft window, so draft-token K/V that a later step rewrites is
    never cached.
  * A request slot's tags are invalidated on every real EXTEND of that slot
    (new request, chunked prefill, re-prefill after retraction), which is how
    a reused request slot starts empty.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_SETS_ENV = "SGLANG_QSA_HOST_KV_CACHE_SETS"


def host_kv_cache_sets() -> int:
    return int(os.environ.get(_SETS_ENV, "4096"))


@triton.jit
def _gather_cached(
    k,
    v,
    ck,
    cv,
    tags,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    stats,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    pad_cols,
    num_sets: tl.constexpr,
    margin: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ZERO_FILL: tl.constexpr,
    STATS: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch).to(tl.int64)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    safe_pos = tl.where(valid, positions, 0)
    group = safe_pos // RATIO
    sets = group % num_sets
    tag = tl.load(tags + req * num_sets + sets, mask=valid, other=-1)
    cacheable = (group + 1) * RATIO + margin <= length
    hit = valid & cacheable & (tag == group)
    miss = valid & (~hit)
    slots = tl.load(req_to_token + req * req_stride + safe_pos, mask=miss, other=0)
    dmask = dims[None, :] < dim
    src_host = slots.to(tl.int64)[:, None] * heads * dim + head * dim + dims[None, :]
    cache_row = req * (num_sets * RATIO) + sets.to(tl.int64) * RATIO + safe_pos % RATIO
    src_cache = cache_row[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (
        (pack_start + cols).to(tl.int64)[:, None] * heads * dim
        + head * dim
        + dims[None, :]
    )
    if ZERO_FILL:
        store_mask = (cols < pad_cols)[:, None] & dmask
    else:
        store_mask = valid[:, None] & dmask
    out_dtype = out_k.dtype.element_ty
    kh = tl.load(k + src_host, mask=miss[:, None] & dmask, other=0.0).to(out_dtype)
    kc = tl.load(ck + src_cache, mask=hit[:, None] & dmask, other=0.0).to(out_dtype)
    tl.store(out_k + dst, tl.where(hit[:, None], kc, kh), mask=store_mask)
    vh = tl.load(v + src_host, mask=miss[:, None] & dmask, other=0.0).to(out_dtype)
    vc = tl.load(cv + src_cache, mask=hit[:, None] & dmask, other=0.0).to(out_dtype)
    tl.store(out_v + dst, tl.where(hit[:, None], vc, vh), mask=store_mask)
    if STATS:
        if head == 0:
            tl.atomic_add(stats + 0, tl.sum(hit.to(tl.int64), axis=0))
            tl.atomic_add(stats + 1, tl.sum(valid.to(tl.int64), axis=0))
            tl.atomic_add(stats + 2, tl.sum((valid & cacheable).to(tl.int64), axis=0))


@triton.jit
def _claim_sets(
    tags,
    claim,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    topk: tl.constexpr,
    idx_stride: tl.constexpr,
    num_sets: tl.constexpr,
    margin: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    batch, block = tl.program_id(0), tl.program_id(1)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch).to(tl.int64)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    safe_pos = tl.where(valid, positions, 0)
    group = safe_pos // RATIO
    sets = group % num_sets
    tag = tl.load(tags + req * num_sets + sets, mask=valid, other=-1)
    cacheable = (group + 1) * RATIO + margin <= length
    bid = valid & cacheable & (tag != group) & (safe_pos % RATIO == 0)
    tl.atomic_max(claim + req * num_sets + sets, group.to(tl.int32), mask=bid)


@triton.jit
def _fill_cache(
    ck,
    cv,
    tags,
    claim,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    idx_stride: tl.constexpr,
    num_sets: tl.constexpr,
    margin: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch).to(tl.int64)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    safe_pos = tl.where(valid, positions, 0)
    group = safe_pos // RATIO
    sets = group % num_sets
    cacheable = (group + 1) * RATIO + margin <= length
    # `claim` holds only groups that MISSED in this call (see _claim_sets), so a
    # group that owns its set never overwrites data another row just hit on.
    owner = tl.load(claim + req * num_sets + sets, mask=valid, other=-1)
    win = valid & cacheable & (owner == group)
    dmask = dims[None, :] < dim
    src = (
        (pack_start + cols).to(tl.int64)[:, None] * heads * dim
        + head * dim
        + dims[None, :]
    )
    cache_row = req * (num_sets * RATIO) + sets.to(tl.int64) * RATIO + safe_pos % RATIO
    dst = cache_row[:, None] * heads * dim + head * dim + dims[None, :]
    cdtype = ck.dtype.element_ty
    m = win[:, None] & dmask
    tl.store(ck + dst, tl.load(out_k + src, mask=m, other=0.0).to(cdtype), mask=m)
    tl.store(cv + dst, tl.load(out_v + src, mask=m, other=0.0).to(cdtype), mask=m)
    if head == 0:
        tl.store(
            tags + req * num_sets + sets,
            group.to(tl.int32),
            mask=win & (safe_pos % RATIO == 0),
        )


class QSAHostKVCache:
    """Per-layer GPU hot cache for one host-resident QSA KV pool."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_req_slots: int,
        heads: int,
        dim: int,
        dtype: torch.dtype,
        ratio: int,
        num_sets: int,
        margin: int,
        device,
    ):
        self.num_layers = num_layers
        self.num_req_slots = num_req_slots
        self.num_sets = int(num_sets)
        self.ratio = int(ratio)
        self.margin = int(margin)
        self.heads, self.dim = heads, dim
        self.elem = torch.empty((), dtype=dtype).element_size()
        rows = num_req_slots * self.num_sets * self.ratio
        self.k: List[torch.Tensor] = []
        self.v: List[torch.Tensor] = []
        self.tags: List[torch.Tensor] = []
        for _ in range(num_layers):
            self.k.append(torch.zeros((rows, heads, dim), dtype=dtype, device=device))
            self.v.append(torch.zeros((rows, heads, dim), dtype=dtype, device=device))
            self.tags.append(
                torch.full(
                    (num_req_slots, self.num_sets), -1, dtype=torch.int32, device=device
                )
            )
        self.claim = torch.full(
            (num_req_slots, self.num_sets), -1, dtype=torch.int32, device=device
        )
        # [hits, selected tokens, cacheable selected tokens], cumulative.
        self.stats = torch.zeros(3, dtype=torch.int64, device=device)
        self.stats_enabled = os.environ.get("SGLANG_QSA_HOST_KV_STATS", "0") == "1"
        if self.stats_enabled:
            self._start_stats_thread(device)
        nbytes = 2 * num_layers * rows * heads * dim * torch.empty((), dtype=dtype).element_size()
        logger.info(
            "QSA host KV hot cache: %d layers x %d request slots x %d sets x %d tokens "
            "= %.2f GiB GPU (margin %d)",
            num_layers,
            num_req_slots,
            self.num_sets,
            self.ratio,
            nbytes / (1 << 30),
            self.margin,
        )

    def _start_stats_thread(self, device) -> None:
        import threading
        import time

        interval = float(os.environ.get("SGLANG_QSA_HOST_KV_STATS_INTERVAL", "30"))
        stream = torch.cuda.Stream(device=device)
        host = torch.zeros(3, dtype=torch.int64).pin_memory()

        def run():
            last = (0, 0, 0)
            while True:
                time.sleep(interval)
                try:
                    with torch.cuda.stream(stream):
                        host.copy_(self.stats, non_blocking=True)
                    stream.synchronize()
                    hits, sel, cach = (int(x) for x in host.tolist())
                except Exception:  # shutdown
                    return
                d = (hits - last[0], sel - last[1], cach - last[2])
                last = (hits, sel, cach)
                if d[1] > 0:
                    logger.info(
                        "QSA host KV cache: last %.0fs hit %.4f of selected tokens "
                        "(%.4f of cacheable), %d selected; cumulative hit %.4f "
                        "over %d; host bytes/token read %.0f",
                        interval,
                        d[0] / d[1],
                        d[0] / max(d[2], 1),
                        d[1],
                        hits / max(sel, 1),
                        sel,
                        (d[1] - d[0]) / d[1] * 2 * self.heads * self.dim * self.elem,
                    )

        threading.Thread(target=run, name="qsa-host-kv-stats", daemon=True).start()

    def invalidate(self, layer: int, req_pool_indices: torch.Tensor) -> None:
        self.tags[layer].index_fill_(0, req_pool_indices.long(), -1)

    def gather(
        self,
        layer: int,
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        batch: int,
        topk: int,
        zero_fill_cols: int = 0,
    ) -> None:
        _, heads, dim = k.shape
        block_topk = 16
        zero_fill = zero_fill_cols > 0
        num_cols = zero_fill_cols if zero_fill else topk
        block_d = triton.next_power_of_2(dim)
        ck, cv, tags = self.k[layer], self.v[layer], self.tags[layer]
        _gather_cached[(batch, heads, triton.cdiv(num_cols, block_topk))](
            k, v, ck, cv, tags, req_to_token, req_indices, indices, seq_lens, cu_k,
            out_k, out_v, self.stats, topk, heads, dim, req_to_token.stride(0),
            indices.stride(0), num_cols, self.num_sets, self.margin,
            RATIO=self.ratio, BLOCK_TOPK=block_topk, BLOCK_D=block_d,
            ZERO_FILL=zero_fill, STATS=self.stats_enabled, num_warps=8,
        )
        self.claim.fill_(-1)
        _claim_sets[(batch, triton.cdiv(topk, 128))](
            tags, self.claim, req_indices, indices, seq_lens, cu_k, topk,
            indices.stride(0), self.num_sets, self.margin,
            RATIO=self.ratio, BLOCK_TOPK=128, num_warps=4,
        )
        _fill_cache[(batch, heads, triton.cdiv(topk, block_topk))](
            ck, cv, tags, self.claim, req_indices, indices, seq_lens, cu_k,
            out_k, out_v, topk, heads, dim, indices.stride(0), self.num_sets,
            self.margin, RATIO=self.ratio, BLOCK_TOPK=block_topk, BLOCK_D=block_d,
            num_warps=8,
        )


__all__ = ["QSAHostKVCache", "host_kv_cache_sets"]

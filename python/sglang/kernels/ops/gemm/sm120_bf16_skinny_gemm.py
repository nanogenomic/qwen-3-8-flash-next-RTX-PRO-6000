# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Skinny BF16 GEMM for SM12x decode/verify row counts (M <= 32).

y[M, N] = x[M, K] @ w[N, K]^T, bf16 in/out, fp32 accumulation.

On SM120 cuBLAS serves these shapes with SM80 WMMA tiles plus a separate split-K
reduce kernel, which leaves 15-45% of DRAM bandwidth unused for the mid-sized
per-layer weights. This kernel streams each weight exactly once with evict-first
loads and computes (W @ X^T) so the wide N dimension sits on the MMA M side; the
activation rows are padded to a 16/32-row MMA tile.

Split-K is reduced inside the same launch: each split writes an fp32 partial to a
per-stream workspace, the last split to arrive for an N tile (atomic ticket) sums
the partials in a fixed order and writes bf16, then resets its ticket. One kernel
per GEMM, deterministic run to run, and CUDA-graph safe: the tickets are back at
zero after every call, and each stream owns its workspace so GEMMs overlapped on
two streams never share one.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

MAX_M = 32
_MAX_N = 65536  # lm_head-sized N stays on cuBLAS (already at the bandwidth ceiling)
_MIN_BLOCK_K = 128
_MIN_GEMV_WEIGHT_ELEMS = 8 << 20  # M=1 below this stays on cuBLAS gemv
_WS_FLOATS = 1 << 20  # 4 MiB fp32 split-K workspace per stream
_N_TICKETS = 1 << 14


@triton.jit
def _skinny_gemm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    ws_ptr,
    tickets_ptr,
    M,
    N,
    stride_xm,
    stride_wn,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    mask_n = offs_n < N
    mask_m = offs_m < M
    k_begin = pid_k * K_PER_SPLIT

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = k_begin + kk + tl.arange(0, BLOCK_K)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=mask_n[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        acc = tl.dot(w, tl.trans(x), acc)

    y_offs = offs_m[None, :] * stride_ym + offs_n[:, None]
    y_mask = mask_m[None, :] & mask_n[:, None]
    if SPLIT_K == 1:
        tl.store(y_ptr + y_offs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)
    else:
        # workspace layout [SPLIT_K, BLOCK_M, N] fp32
        ws_offs = offs_m[None, :] * N + offs_n[:, None]
        tl.store(ws_ptr + pid_k * (BLOCK_M * N) + ws_offs, acc, mask=y_mask)
        tl.debug_barrier()
        ticket = tl.atomic_add(tickets_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if ticket == SPLIT_K - 1:
            total = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                total += tl.load(
                    ws_ptr + s * (BLOCK_M * N) + ws_offs,
                    mask=y_mask,
                    other=0.0,
                    cache_modifier=".cg",
                )
            tl.store(y_ptr + y_offs, total.to(y_ptr.dtype.element_ty), mask=y_mask)
            tl.atomic_xchg(tickets_ptr + pid_n, 0, sem="relaxed", scope="gpu")


_num_sms: Dict[int, int] = {}
_scratch: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
_configs: Dict[Tuple[int, int, int], Tuple[int, int, int, int, int]] = {}


def _get_num_sms(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _num_sms:
        _num_sms[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _num_sms[idx]


def _get_scratch(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Workspace + tickets owned by the current stream (kept alive for graph replay)."""
    key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    buf = _scratch.get(key)
    if buf is None:
        buf = (
            torch.empty(_WS_FLOATS, dtype=torch.float32, device=device),
            torch.zeros(_N_TICKETS, dtype=torch.int32, device=device),
        )
        _scratch[key] = buf
    return buf


def pick_config(m: int, n: int, k: int, num_sms: int) -> Tuple[int, int, int, int, int]:
    """(block_n, block_k, split_k, num_warps, num_stages).

    Measured on sm_120 (RTX PRO 4000, 70 SMs) at native N and with N scaled by
    70/188 to reproduce the RTX PRO 6000's per-SM work: a weight strip of
    BLOCK_M rows per CTA (square MMA tile), split-K doubled until there are >= 2
    (16-row tiles) or >= 1 (32-row tiles) CTAs per SM while each split keeps >= 256
    K elements, and the partial workspace fits.
    """
    block_m = 16 if m <= 16 else 32
    bn = block_m
    tiles = triton.cdiv(n, bn)
    # CTAs per SM to reach before splitting stops: 2 for 16-row tiles, 1 for 32-row
    target = 2 if block_m == 16 else 1
    split = 1
    while (
        tiles * split < target * num_sms
        and split < 8
        and k % (split * 2 * _MIN_BLOCK_K) == 0
        and k // (split * 2) >= 256
        and (split * 2) * block_m * n <= _WS_FLOATS
    ):
        split *= 2
    kps = k // split
    if m <= 16 and kps % 256 == 0 and kps >= 1024:
        return bn, 256, split, 4, 3
    return bn, 128, split, 4, 4


def use_sm120_skinny_gemm(x: torch.Tensor, weight: torch.Tensor) -> bool:
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if weight.dim() != 2 or weight.stride(1) != 1 or x.stride(-1) != 1:
        return False
    k = weight.shape[1]
    m = x.numel() // k if k else 0
    n = weight.shape[0]
    if m == 1 and n * k < _MIN_GEMV_WEIGHT_ELEMS:
        return False  # cuBLAS gemv is as fast for small single-row GEMMs
    return (
        1 <= m <= MAX_M
        and x.shape[-1] == k
        and k % _MIN_BLOCK_K == 0
        and 16 <= weight.shape[0] <= _MAX_N
    )


def sm120_skinny_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """y = x @ weight^T (+ bias). ``x`` may be [..., K]; returns [..., N] (or ``out``)."""
    k = weight.shape[1]
    n = weight.shape[0]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if out is None:
        y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    else:
        y = out
    key = (m, n, k)
    cfg = _configs.get(key)
    if cfg is None:
        cfg = pick_config(m, n, k, _get_num_sms(x.device))
        _configs[key] = cfg
    bn, bk, split, num_warps, num_stages = cfg
    n_tiles = triton.cdiv(n, bn)
    ws, tickets = _get_scratch(x.device)
    assert n_tiles <= tickets.numel()
    _skinny_gemm_kernel[(n_tiles, split)](
        x2,
        weight,
        y,
        ws,
        tickets,
        m,
        n,
        x2.stride(0),
        weight.stride(0),
        y.stride(0),
        BLOCK_M=16 if m <= 16 else 32,
        BLOCK_N=bn,
        BLOCK_K=bk,
        SPLIT_K=split,
        K_PER_SPLIT=k // split,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if bias is not None:
        y.add_(bias)
    if out is not None:
        return out
    return y.view(*x.shape[:-1], n)

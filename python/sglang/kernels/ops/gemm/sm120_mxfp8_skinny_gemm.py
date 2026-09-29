# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Skinny MXFP8 (W8A8, 1x32 ue8m0 blocks) GEMM for SM12x decode/verify row counts (M <= 32).

y[M, N] = mxfp8(x)[M, K] @ mxfp8_w[N, K]^T, bf16 out, fp32 accumulation, on the native
block-scaled MMA (tl.dot_scaled, e4m3 x e4m3 with e8m0 scales).

FlashInfer's CUTLASS mm_mxfp8 serves these row counts on SM120 at 45-55% of DRAM bandwidth
plus a separate activation-quantize launch, which is slower than the BF16 GEMM it replaces.
Here the activations are quantized in-kernel with flashinfer mxfp8_quantize's rule
(scale = 2^ceil(log2(amax/448)), RNE-saturating e4m3; measured 100% scale and value match),
so the product equals CUTLASS's up to fp32 summation order. The weight is read exactly once
(evict-first) from the linear [N, K/32] ue8m0 scale layout (layer.weight_scale_inv).
Split-K is reduced in-kernel with per-tile tickets and a per-stream workspace, as in
sm120_bf16_skinny_gemm, so it is one launch and CUDA-graph safe.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

MAX_M = 32
_WS_FLOATS = 1 << 20
_N_TICKETS = 1 << 14


@triton.jit
def _pow2_from_biased(e):
    # 2^(e-127) for integer e in [1, 254], exact, built from the exponent bits
    return (e.to(tl.int32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _mxfp8_skinny_kernel(
    x_ptr,
    w_ptr,
    ws_ptr,
    y_ptr,
    part_ptr,
    tickets_ptr,
    M,
    N,
    stride_xm,
    stride_wn,
    stride_wsn,
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
    G: tl.constexpr = BLOCK_K // 32

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        k0 = k_begin + kk
        offs_k = k0 + tl.arange(0, BLOCK_K)
        w8 = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=mask_n[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        offs_g = k0 // 32 + tl.arange(0, G)
        wse = tl.load(
            ws_ptr + offs_n[:, None] * stride_wsn + offs_g[None, :],
            mask=mask_n[:, None],
            other=127,
            eviction_policy="evict_first",
        )
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        # MXFP8 activation quantization (flashinfer mxfp8_quantize rule)
        xf = tl.reshape(x.to(tl.float32), (BLOCK_M, G, 32))
        amax = tl.max(tl.abs(xf), axis=2)
        bits = tl.math.div_rn(amax, 448.0).to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        e = tl.minimum(tl.maximum(e, 1), 254)
        xq = tl.reshape(
            (xf / _pow2_from_biased(e)[:, :, None]).to(tl.float8e4nv), (BLOCK_M, BLOCK_K)
        )
        acc = tl.dot_scaled(w8, wse, "e4m3", tl.trans(xq), e.to(tl.uint8), "e4m3", acc)

    y_offs = offs_m[None, :] * stride_ym + offs_n[:, None]
    y_mask = mask_m[None, :] & mask_n[:, None]
    if SPLIT_K == 1:
        tl.store(y_ptr + y_offs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)
    else:
        p_offs = offs_m[None, :] * N + offs_n[:, None]
        tl.store(part_ptr + pid_k * (BLOCK_M * N) + p_offs, acc, mask=y_mask)
        tl.debug_barrier()
        ticket = tl.atomic_add(tickets_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if ticket == SPLIT_K - 1:
            total = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                total += tl.load(
                    part_ptr + s * (BLOCK_M * N) + p_offs,
                    mask=y_mask,
                    other=0.0,
                    cache_modifier=".cg",
                )
            tl.store(y_ptr + y_offs, total.to(y_ptr.dtype.element_ty), mask=y_mask)
            tl.atomic_xchg(tickets_ptr + pid_n, 0, sem="relaxed", scope="gpu")


_num_sms: Dict[int, int] = {}
_scratch: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
_configs: Dict[Tuple[int, int, int], Tuple[int, int, int, int, int]] = {}


def _get_scratch(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
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
    """(block_n, block_k, split_k, num_warps, num_stages); same occupancy rule as the
    BF16 skinny kernel (>= 2 CTAs/SM for 16-row tiles, >= 1 for 32-row)."""
    block_m = 16 if m <= 16 else 32
    bn = block_m
    tiles = triton.cdiv(n, bn)
    target = 2 if block_m == 16 else 1
    split = 1
    while (
        tiles * split < target * num_sms
        and split < 8
        and k % (split * 2 * 128) == 0
        and k // (split * 2) >= 256
        and (split * 2) * block_m * n <= _WS_FLOATS
    ):
        split *= 2
    kps = k // split
    if kps % 256 == 0 and kps >= 1024:
        return bn, 256, split, 4, 3
    return bn, 128, split, 4, 4


def use_sm120_mxfp8_skinny_gemm(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> bool:
    if x.dtype != torch.bfloat16 or weight.dtype != torch.float8_e4m3fn:
        return False
    if weight_scale.dtype != torch.uint8 or weight.dim() != 2 or weight_scale.dim() != 2:
        return False
    n, k = weight.shape
    if weight_scale.shape != (n, k // 32) or weight_scale.stride(1) != 1:
        return False
    if weight.stride(1) != 1 or x.stride(-1) != 1 or x.shape[-1] != k:
        return False
    m = x.numel() // k
    return 1 <= m <= MAX_M and k % 128 == 0 and 16 <= n <= 65536


def sm120_mxfp8_skinny_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """y = mxfp8(x) @ mxfp8_w^T (+ bias). x [..., K] bf16 -> [..., N] bf16."""
    n, k = weight.shape
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    if idx not in _num_sms:
        _num_sms[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    key = (m, n, k)
    cfg = _configs.get(key)
    if cfg is None:
        cfg = _configs[key] = pick_config(m, n, k, _num_sms[idx])
    bn, bk, split, num_warps, num_stages = cfg
    ws, tickets = _get_scratch(x.device)
    _mxfp8_skinny_kernel[(triton.cdiv(n, bn), split)](
        x2,
        weight,
        weight_scale,
        y,
        ws,
        tickets,
        m,
        n,
        x2.stride(0),
        weight.stride(0),
        weight_scale.stride(0),
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
    return y.view(*x.shape[:-1], n)

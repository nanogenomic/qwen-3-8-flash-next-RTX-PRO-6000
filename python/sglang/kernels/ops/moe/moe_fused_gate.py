from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    is_arch_support_pdl,
    load_jit,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.kernels.ops.moe import moe_route_radix

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SCORING_FUNC_MAP = {
    "sigmoid": 0,
    "sqrtsoftplus": 1,
    "softmax": 2,
}


@cache_once
def _jit_moe_fused_gate_module() -> Module:
    return load_jit(
        "moe_fused_gate",
        cuda_files=["moe/moe_fused_gate.cuh"],
        cuda_wrappers=[("moe_fused_gate", "MoEFusedGateKernel::run")],
    )


@cache_once
def can_use_moe_fused_gate() -> bool:
    logger = logging.getLogger(__name__)
    try:
        _jit_moe_fused_gate_module()
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT MoE fused gate kernel: {e}")
        return False


def moe_fused_gate_jit(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scoring_func_int = _SCORING_FUNC_MAP.get(scoring_func.lower())
    assert scoring_func_int is not None, (
        f"Unknown scoring_func '{scoring_func}', must be one of {list(_SCORING_FUNC_MAP.keys())}"
    )

    assert input.dtype == torch.float32, "input must be float32"
    assert bias.dtype == torch.float32, "bias must be float32"
    assert input.ndim == 2, "input must be 2D"
    assert bias.ndim == 1, "bias must be 1D"
    assert input.size(1) == bias.size(0), "input and bias must have same num_experts"
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"

    num_rows, _ = input.shape
    device = input.device

    output = torch.empty(num_rows, topk, dtype=torch.float32, device=device)
    indices = torch.empty(num_rows, topk, dtype=torch.int32, device=device)

    module = _jit_moe_fused_gate_module()
    module.moe_fused_gate(
        input,
        bias,
        output,
        indices,
        topk,
        scoring_func_int,
        num_fused_shared_experts,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )

    return output, indices


@triton.jit
def _router_triton_kernel(
    scores_ptr,  # [M, N] raw logits, fp32/fp16/bf16 (upcast to fp32 on load)
    bias_ptr,  # [N]    fp32/fp16/bf16 (upcast to fp32 on load)
    bias_alt_ptr,
    input_ids_ptr,
    num_token_non_padded_ptr,
    out_weights_ptr,  # [M, K] fp32
    out_indices_ptr,  # [M, K] int32
    out_packed_ptr,  # [M, K] int32 (HAS_PACKED)
    M,
    routed_scaling_factor,
    moe_softcapping,
    N: tl.constexpr,
    K: tl.constexpr,  # total topk (includes fused shared experts)
    K_ROUTED: tl.constexpr,  # K - num_fused_shared_experts
    BLOCK_M: tl.constexpr,  # rows processed per program (row tiling)
    BLOCK_N: tl.constexpr,  # >= N, power of 2
    BLOCK_K: tl.constexpr,  # >= K, power of 2
    N_GROUP: tl.constexpr,  # expert groups (1 = ungrouped)
    TOPK_GROUP: tl.constexpr,  # groups kept per token (grouped routing)
    EXPERTS_PER_GROUP: tl.constexpr,  # N // N_GROUP
    BLOCK_G: tl.constexpr,  # >= N_GROUP, power of 2
    SCORING_FUNC: tl.constexpr,  # 0 = sigmoid, 1 = sqrtsoftplus, 2 = softmax
    SQRTSOFTPLUS_LOG1P: tl.constexpr,  # sqrtsoftplus via log1p (V4.1 numerics)
    HAS_SOFTCAP: tl.constexpr,  # tanh softcapping (softmax only)
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,  # apply_routed_scaling_factor_on_output
    HAS_BIAS: tl.constexpr,
    HAS_TOKEN_BIAS: tl.constexpr,
    BIAS_ALT_TOKEN_ID: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    HAS_PACKED: tl.constexpr,
    RENORMALIZE_EPSILON: tl.constexpr,
    USE_PDL: tl.constexpr,
    stride_bias,
    stride_bias_alt,
    stride_input_ids,
    stride_sm,
    stride_sn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    stride_pm,
    stride_pk,
    # --- qwenopt dynamic-k; DYNK_MODE == 0 compiles all of it out ---
    dynk_tau_ptr,  # [64] fp32 per-slot tau (cumprob) / log10 error budget (learned)
    dynk_mlp_ptr,  # [64, P] fp32 packed learned-gate params (DYNK_MODE == 3)
    dynk_slot,
    dynk_k,
    dynk_kk_ptr,  # [64, 2] int32 per-slot (k_min, k_max), device-resident (modes 2, 3)
    dynk_cnt_ptr,  # [64, 2] int32 (kept slots, live rows) per slot, DYNK_COUNT only
    DYNK_COUNT: tl.constexpr,
    DYNK_MODE: tl.constexpr,  # 0 off, 1 fixed-k, 2 cumulative-prob, 3 learned gate
    DYNK_TOPN: tl.constexpr,  # unused (kept for signature stability)
    DYNK_F: tl.constexpr,  # learned-gate input features (16, power of 2)
    DYNK_H: tl.constexpr,  # hidden width (32)
    DYNK_OUT: tl.constexpr,  # outputs: predicted log10 rel. error for k = 3 .. 3+OUT-1
) -> None:
    # Row-tiled: each program handles BLOCK_M rows; all reductions run along the
    # expert (N) axis. Tiling rows keeps CTAs large enough to stay occupancy-bound
    # rather than launch-bound at small N (many tiny 1-warp CTAs otherwise).
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_m = offs_m < M
    mask_n = offs_n < N

    # PDL may start this grid before prior kernel stores are visible. Bias can
    # be produced by a preceding cast or fill kernel, so wait before loading
    # either bias or scores.
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    # Plain softmax routing has no bias, so keep the zero value in registers
    # rather than materializing and clearing a device tensor per call.
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n * stride_bias, mask=mask_n, other=0.0).to(
            tl.float32
        )
    else:
        bias = tl.zeros([BLOCK_N], dtype=tl.float32)
    if HAS_TOKEN_BIAS:
        bias_alt = tl.load(
            bias_alt_ptr + offs_n * stride_bias_alt, mask=mask_n, other=0.0
        ).to(tl.float32)

    live_m = mask_m
    if HAS_PADDING:
        live_m = live_m & (offs_m < tl.load(num_token_non_padded_ptr))
    row_bias = bias[None, :]
    if HAS_TOKEN_BIAS:
        input_ids = tl.load(
            input_ids_ptr + offs_m * stride_input_ids, mask=live_m, other=0
        )
        row_bias = tl.where(
            (input_ids == BIAS_ALT_TOKEN_ID)[:, None], bias_alt[None, :], row_bias
        )

    row_ptr = scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    mask2d = live_m[:, None] & mask_n[None, :]
    scores = tl.load(row_ptr, mask=mask2d, other=0.0).to(
        tl.float32
    )  # [BLOCK_M, BLOCK_N]

    if SCORING_FUNC == 0:
        # sigmoid(x) = 1 / (1 + exp(-x)); bias is for ranking only, weight is bias-free.
        activated = tl.sigmoid(scores)
        biased = activated + row_bias
    elif SCORING_FUNC == 1:
        if SQRTSOFTPLUS_LOG1P:
            # log1p preserves small positive scores for negative logits.
            sp = tl.where(scores > 20.0, scores, libdevice.log1p(libdevice.exp(scores)))
            activated = libdevice.sqrt(sp)
        else:
            # Open-coded log1p; reproduces the DeepSeek-V4 sqrtsoftplus numerics.
            z = tl.exp(-tl.abs(scores))
            u = 1.0 + z
            exact = u == 1.0
            log1p_z = tl.where(exact, z, z * tl.log(u) / tl.where(exact, 1.0, u - 1.0))
            sp = tl.maximum(scores, 0.0) + log1p_z
            activated = tl.sqrt(sp)
        biased = activated + row_bias
    else:
        # softmax over the row: weight is the softmax probability (bias kept), with
        # optional tanh softcapping. Ranking by the (softcapped, biased) logit is
        # monotonic with the softmax prob, so the topk loop below ranks on `biased`.
        logit = scores
        if HAS_SOFTCAP:
            # tanh(z) = 2*sigmoid(2z) - 1 (avoids relying on tl.math.tanh availability).
            z = logit / moe_softcapping
            logit = moe_softcapping * (2.0 * tl.sigmoid(2.0 * z) - 1.0)
        biased = logit + row_bias
        biased = tl.where(mask_n[None, :], biased, -float("inf"))
        row_max = tl.max(biased, axis=1)[:, None]  # [BLOCK_M, 1]
        exp_row = tl.where(mask_n[None, :], tl.exp(biased - row_max), 0.0)
        row_sum = tl.sum(exp_row, axis=1)[:, None]  # [BLOCK_M, 1]
        activated = exp_row / row_sum

    biased = tl.where(mask_n[None, :], biased, -float("inf"))  # [BLOCK_M, BLOCK_N]

    if SCORING_FUNC == 1 and SQRTSOFTPLUS_LOG1P:
        # Rank NaNs above finite scores, matching torch.topk.
        biased = tl.where(biased == biased, biased, float("inf"))
    else:
        biased = tl.where(biased == biased, biased, -1e30)

    # Grouped routing (DeepSeek-V3 noaux_tc): per-group score = sum of the top-2
    # biased values; keep TOPK_GROUP groups (lowest group id wins ties); mask the
    # experts of dropped groups to -inf before the top-k below. Weight is still the
    # bias-free `activated`. Constexpr N_GROUP <= 1 skips this entirely (ungrouped).
    if N_GROUP > 1:
        offs_g = tl.arange(0, BLOCK_G)  # [BLOCK_G]
        group_of_n = offs_n // EXPERTS_PER_GROUP  # [BLOCK_N]
        group_score = tl.full([BLOCK_M, BLOCK_G], -float("inf"), dtype=tl.float32)
        for g in tl.static_range(N_GROUP):
            in_g = (group_of_n[None, :] == g) & mask_n[None, :]
            vals = tl.where(in_g, biased, -float("inf"))
            top1 = tl.max(vals, axis=1)[:, None]  # [BLOCK_M, 1]
            vals2 = tl.where(vals >= top1, -float("inf"), vals)
            top2 = tl.max(vals2, axis=1)[:, None]  # [BLOCK_M, 1]
            group_score = tl.where(offs_g[None, :] == g, top1 + top2, group_score)

        gcur = group_score
        keep = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for _i in tl.static_range(TOPK_GROUP):
            gmax = tl.max(gcur, axis=1)[:, None]  # [BLOCK_M, 1]
            glane = tl.where(gcur == gmax, offs_g[None, :], N_GROUP + 1)
            win_g = tl.min(glane, axis=1)[:, None]  # [BLOCK_M, 1] lowest-id on ties
            keep = tl.where(group_of_n[None, :] == win_g, 1.0, keep)
            gcur = tl.where(offs_g[None, :] == win_g, -float("inf"), gcur)
        biased = tl.where(keep > 0.0, biased, -float("inf"))

    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    mask_k_total = offs_k < K
    mask_k_routed = offs_k < K_ROUTED
    selected_vals = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)

    cur = biased  # [BLOCK_M, BLOCK_N]
    remaining = tl.broadcast_to(mask_n[None, :], (BLOCK_M, BLOCK_N))
    if DYNK_MODE == 3:
        offs_f = tl.arange(0, DYNK_F)
        feat = tl.zeros([BLOCK_M, DYNK_F], dtype=tl.float32)
    for k in tl.static_range(K_ROUTED):
        max_val = tl.max(cur, axis=1)[:, None]  # [BLOCK_M, 1]
        is_max = remaining & (cur == max_val)
        lane_id = tl.where(is_max, offs_n[None, :], N + 1)  # lowest expert id wins ties
        win_lane = tl.min(lane_id, axis=1)[:, None].to(tl.int32)  # [BLOCK_M, 1]
        win_activated = tl.sum(
            tl.where(offs_n[None, :] == win_lane, activated, 0.0), axis=1
        )[:, None]  # [BLOCK_M, 1]
        slot = offs_k[None, :] == k  # [1, BLOCK_K]
        selected_vals = tl.where(slot, win_activated, selected_vals)
        selected_idx = tl.where(slot, win_lane, selected_idx)
        remaining = remaining & (offs_n[None, :] != win_lane)
        cur = tl.where(remaining, cur, -float("inf"))
        if DYNK_MODE == 3:
            feat = tl.where(offs_f[None, :] == k, win_activated, feat)


    routed_sum = tl.sum(tl.where(mask_k_routed[None, :], selected_vals, 0.0), axis=1)[
        :, None
    ]  # [BLOCK_M, 1]

    # Fill fused-shared-expert slots: weight = routed_sum / routed_scaling_factor,
    # id = num_experts + (slot - K_ROUTED).
    if K_ROUTED < K:
        is_shared = (offs_k[None, :] >= K_ROUTED) & mask_k_total[None, :]
        shared_weight = routed_sum / routed_scaling_factor  # [BLOCK_M, 1]
        shared_idx = (N + (offs_k - K_ROUTED)).to(tl.int32)[None, :]  # [1, BLOCK_K]
        selected_vals = tl.where(is_shared, shared_weight, selected_vals)
        selected_idx = tl.where(is_shared, shared_idx, selected_idx)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    if RENORMALIZE:
        if RENORMALIZE_EPSILON > 0.0:
            norm = routed_sum + RENORMALIZE_EPSILON
        else:
            norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)  # [BLOCK_M, 1]
        selected_vals = selected_vals / norm
    if APPLY_SCALE:
        selected_vals = selected_vals * routed_scaling_factor
    if DYNK_MODE != 0:
        # Drop tail slots: id -1, weight 0; kept slots keep their top-K weight
        # (no renormalisation). Slots are in descending weight order here.
        if DYNK_MODE >= 2:
            dynk_kmin = tl.load(dynk_kk_ptr + dynk_slot * 2)
            dynk_kmax = tl.load(dynk_kk_ptr + dynk_slot * 2 + 1)
        if DYNK_MODE == 1:
            keep = offs_k[None, :] < dynk_k
        elif DYNK_MODE == 2:
            tau = tl.load(dynk_tau_ptr + dynk_slot)
            rv = tl.where(mask_k_routed[None, :], selected_vals, 0.0)
            tot = tl.sum(rv, axis=1)[:, None]
            cum_excl = tl.cumsum(rv, axis=1) - rv
            keep = (cum_excl < tau * tot) | (offs_k[None, :] < dynk_kmin)
            keep = keep & (offs_k[None, :] < dynk_kmax)
        else:
            # Learned per-slot gate: features -> MLP -> predicted log10 rel. error
            # for each k; choose the smallest k in [kmin, kmax] under the budget.
            # Features (F=16), all from the K_ROUTED argmax rounds already done:
            #  0..9  sorted top-10 probs (softmax over all experts)
            #  10    tail mass beyond top-10 (= 1 - top-10 mass; no extra ranking rounds)
            #  11    entropy of the top-10 renormalised
            #  12    log p1      13 log(p1/p2)      14 log(tail)      15 top-5 share of top-10
            p1 = tl.sum(tl.where(offs_f[None, :] == 0, feat, 0.0), axis=1)[:, None]
            p2 = tl.sum(tl.where(offs_f[None, :] == 1, feat, 0.0), axis=1)[:, None]
            top_n = tl.where(offs_f[None, :] < K_ROUTED, feat, 0.0)
            top_mass = tl.sum(top_n, axis=1)[:, None]
            top5 = tl.sum(tl.where(offs_f[None, :] < 5, feat, 0.0), axis=1)[:, None]
            tail = tl.maximum(1.0 - top_mass, 0.0)
            qn = top_n / tl.maximum(top_mass, 1e-20)
            ent = -tl.sum(tl.where(qn > 0.0, qn * tl.log(tl.maximum(qn, 1e-20)), 0.0), axis=1)[:, None]
            lp1 = tl.log(tl.maximum(p1, 1e-20))
            feat = top_n
            feat = tl.where(offs_f[None, :] == K_ROUTED, tail, feat)
            feat = tl.where(offs_f[None, :] == K_ROUTED + 1, ent, feat)
            feat = tl.where(offs_f[None, :] == K_ROUTED + 2, lp1, feat)
            feat = tl.where(offs_f[None, :] == K_ROUTED + 3, lp1 - tl.log(tl.maximum(p2, 1e-20)), feat)
            feat = tl.where(offs_f[None, :] == K_ROUTED + 4, tl.log(tail + 1e-6), feat)
            feat = tl.where(offs_f[None, :] == K_ROUTED + 5, top5 / tl.maximum(top_mass, 1e-20), feat)
            base = dynk_mlp_ptr + dynk_slot * (DYNK_H * DYNK_F + DYNK_H + DYNK_OUT * DYNK_H + DYNK_OUT)
            offs_h = tl.arange(0, DYNK_H)
            w1 = tl.load(base + offs_h[:, None] * DYNK_F + offs_f[None, :])  # [H, F]
            b1 = tl.load(base + DYNK_H * DYNK_F + offs_h)  # [H]
            hid = tl.sum(feat[:, None, :] * w1[None, :, :], axis=2) + b1[None, :]
            hid = tl.maximum(hid, 0.0)  # [BLOCK_M, H]
            offs_o = tl.arange(0, 8)
            w2 = tl.load(
                base + DYNK_H * DYNK_F + DYNK_H + offs_o[:, None] * DYNK_H + offs_h[None, :],
                mask=offs_o[:, None] < DYNK_OUT,
                other=0.0,
            )  # [8, H]
            b2 = tl.load(
                base + DYNK_H * DYNK_F + DYNK_H + DYNK_OUT * DYNK_H + offs_o,
                mask=offs_o < DYNK_OUT,
                other=0.0,
            )
            pred = tl.sum(hid[:, None, :] * w2[None, :, :], axis=2) + b2[None, :]  # [BLOCK_M, 8]
            budget = tl.load(dynk_tau_ptr + dynk_slot)
            kval = offs_o[None, :] + 3  # k represented by output o
            ok = (pred <= budget) & (kval >= dynk_kmin) & (kval <= dynk_kmax) & (offs_o[None, :] < DYNK_OUT)
            k_sel = tl.min(tl.where(ok, kval, dynk_kmax), axis=1)[:, None]  # [BLOCK_M, 1]
            keep = offs_k[None, :] < k_sel
        if DYNK_COUNT:
            kept_n = tl.sum(
                tl.sum((keep & mask_k_routed[None, :] & live_m[:, None]).to(tl.int32), axis=1),
                axis=0,
            )
            rows_n = tl.sum(live_m.to(tl.int32), axis=0)
            tl.atomic_add(dynk_cnt_ptr + dynk_slot * 2, kept_n)
            tl.atomic_add(dynk_cnt_ptr + dynk_slot * 2 + 1, rows_n)
        keep = keep | (offs_k[None, :] >= K_ROUTED)  # never drop fused shared slots
        selected_vals = tl.where(keep, selected_vals, 0.0)
        selected_idx = tl.where(keep, selected_idx, -1)

    if HAS_PADDING:
        selected_vals = tl.where(live_m[:, None], selected_vals, 0.0)
        selected_idx = tl.where(live_m[:, None], selected_idx, -1)

    out_w_ptr = (
        out_weights_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk
    )
    out_i_ptr = (
        out_indices_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    )
    store_mask = mask_m[:, None] & mask_k_total[None, :]
    tl.store(out_w_ptr, selected_vals, mask=store_mask)
    tl.store(out_i_ptr, selected_idx, mask=store_mask)
    if HAS_PACKED:
        # Must stay bitwise identical to fused_pack_topk.
        w_bits = selected_vals.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
        packed = (selected_idx << 16) | (w_bits & 0xFFFF)
        out_p_ptr = (
            out_packed_ptr + offs_m[:, None] * stride_pm + offs_k[None, :] * stride_pk
        )
        tl.store(out_p_ptr, packed, mask=store_mask)


_DUMMY_I32: Dict[torch.device, torch.Tensor] = {}


def _dummy_i32(device: torch.device) -> torch.Tensor:
    # Placeholder pointer for kernel args whose constexpr flag is off.
    t = _DUMMY_I32.get(device)
    if t is None:
        t = _DUMMY_I32[device] = torch.empty(1, dtype=torch.int32, device=device)
    return t


@debug_kernel_api
def moe_fused_gate(
    scores: torch.Tensor,
    bias: Optional[torch.Tensor],
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    moe_softcapping: float = 0.0,
    num_expert_group: int = 1,
    topk_group: int = 1,
    *,
    bias_alt: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    bias_alt_token_id: Optional[int] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    renormalize_epsilon: float = 0.0,
    packed_out: Optional[torch.Tensor] = None,
    sqrtsoftplus_log1p: bool = False,
    dynk=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton fused router: scoring + bias + topk + (optional) renorm/scale.

    Mirrors :func:`moe_fused_gate_jit` (the CUDA JIT kernel) for the shared
    parameters; the keyword-only extras are Triton-only.
    With ``num_expert_group > 1`` it performs DeepSeek-V3 grouped routing
    (per-group top-2-sum group scores, keep ``topk_group`` groups, then top-k
    within). ``scores`` contains raw GEMM logits.

    Rows past the device scalar ``num_token_non_padded`` return zero weights and -1 ids.
    Positive ``renormalize_epsilon`` uses ``sum + epsilon`` instead of the zero-sum guard.
    ``sqrtsoftplus_log1p`` evaluates sqrtsoftplus through ``log1p`` and ranks NaNs first
    (DeepSeek-V4.1); off, the DeepSeek-V4 formula and NaN order are kept.
    ``packed_out`` ([M, topk] int32, optional) receives the FlashInfer routed-MoE form
    ``(id << 16) | bf16_bits(weight)``, bitwise identical to ``fused_pack_topk``.
    """
    scoring_func_int = _SCORING_FUNC_MAP.get(scoring_func.lower())
    assert scoring_func_int is not None, (
        f"Unknown scoring_func '{scoring_func}', must be one of {list(_SCORING_FUNC_MAP.keys())}"
    )
    assert scores.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), "scores must be float32/float16/bfloat16"
    assert scores.ndim == 2, "scores must be 2D"
    if bias is None:
        assert scoring_func.lower() == "softmax", (
            "bias is required for non-softmax routing"
        )
    else:
        # The kernel loads the bias and upcasts it to fp32 in-register (see
        # _router_triton_kernel), so a non-fp32 bias (DeepSeek-V4 stores the
        # correction bias in bf16) needs no host-side cast/copy.
        assert bias.dtype in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ), "bias must be float32/float16/bfloat16"
        assert bias.ndim == 1, "bias must be 1D"
        assert scores.size(1) == bias.size(0), (
            "scores and bias must have same num_experts"
        )
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"
    if input_ids is not None:
        assert bias_alt is not None and bias_alt_token_id is not None
        assert bias is not None and bias_alt.shape == bias.shape
        assert input_ids.shape == (scores.size(0),)
    if packed_out is not None:
        assert packed_out.dtype == torch.int32, "packed_out must be int32"
        assert packed_out.shape == (scores.size(0), topk), (
            "packed_out must be [M, topk]"
        )
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    # K3 radix-select fast path: native-CUDA radix-select replaces the 16
    # dependent argmax rounds (single CTA per token; ids bit-identical to this
    # triton kernel incl. ties).
    # The radix kernel keeps keys register-resident and returns winners in
    # expert-id order (skipping the biased-descending sort; downstream MoE
    # kernels are order-insensitive). It is 3.1-3.5x faster than the Triton
    # kernel at [1..8192, 896] top-16 on B200.
    if (
        scoring_func.lower() == "sigmoid"
        and num_fused_shared_experts == 0
        and num_expert_group <= 1
        and moe_softcapping == 0.0
        and input_ids is None
        and num_token_non_padded is None
        and renormalize_epsilon == 0.0
        and packed_out is None
        and bias.stride(0) == 1
    ):
        radix_args = (
            scores,
            bias,
            topk,
            renormalize,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
        if dynk is None and moe_route_radix.covered(scores, bias, topk):
            return moe_route_radix.route_radix(*radix_args, sorted=False)

    M, N = scores.shape
    K = topk
    K_routed = topk - num_fused_shared_experts
    if num_expert_group > 1:
        assert N % num_expert_group == 0, "num_experts must be divisible by group count"
        assert 1 <= topk_group <= num_expert_group, "invalid topk_group"
    experts_per_group = N // num_expert_group
    BLOCK_G = triton.next_power_of_2(num_expert_group)

    weights = torch.empty((M, K), dtype=torch.float32, device=scores.device)
    indices = torch.empty((M, K), dtype=torch.int32, device=scores.device)
    if M == 0:
        return weights, indices

    BLOCK_N = triton.next_power_of_2(N)  # 256 -> 256, 384 -> 512
    BLOCK_K = triton.next_power_of_2(K)  # 6 -> 8, 8 -> 8
    # Single warp per program keeps the per-row top-k reductions on cheap warp
    # shuffles; pack a few rows per program only when N is small so tiny launches
    # stay occupancy-bound. Swept on H100/B200: this beats the AOT kernels across
    # shapes, whereas larger tiles / more warps regress (register pressure).
    BLOCK_M = max(1, min(4, 256 // BLOCK_N))
    # For wide rows (e.g. Kimi K3: 896 experts, BLOCK_N 1024) the K sequential
    # argmax passes dominate and benefit from more warps despite the
    # cross-warp reduction cost.
    num_warps = 1 if BLOCK_N <= 512 else 4
    grid = (triton.cdiv(M, BLOCK_M),)
    use_pdl = is_arch_support_pdl()
    if use_pdl and scoring_func_int == 1 and N == 384 and K == 6 and M <= 8:
        # On SM103, early-launching the small DSV4.1 target router increases
        # latency when it overlaps with mHC/shared-expert work. Use ordinary
        # stream dependencies; keep PDL for the draft router and larger batches.
        arch = get_jit_cuda_arch()
        use_pdl = (arch.major, arch.minor) != (10, 3)
    extra = {"launch_pdl": True} if use_pdl else {}
    # Dynamo cannot analyze the kernel (PDL inline asm), so it writes back every
    # pointer arg; aliasing an output as an unused arg's fallback clobbers it.
    _unused_i32 = _dummy_i32(scores.device)
    if dynk is not None:
        # qwenopt dynamic-k: dropped slots become id -1 (never duplicates); the
        # packed FlashInfer form would encode -1 << 16, so refuse that combination.
        assert packed_out is None, "dynamic-k is not supported with packed_out"
        from sglang.srt.layers.moe import dynk as _dynk_mod

        if dynk.mode == _dynk_mod.MODE_LEARNED:
            assert K - num_fused_shared_experts == 10, "learned gate features assume top-10"
            assert dynk.mlp is not None, "learned dynamic-k gate weights not loaded"
        dynk_args = dict(
            dynk_tau_ptr=dynk.tau,
            dynk_mlp_ptr=dynk.mlp if dynk.mlp is not None else dynk.tau,
            dynk_slot=int(dynk.slot),
            dynk_k=int(dynk.k),
            dynk_kk_ptr=dynk.kk,
            dynk_cnt_ptr=dynk.cnt if dynk.cnt is not None else _unused_i32,
            DYNK_COUNT=dynk.cnt is not None,
            DYNK_MODE=int(dynk.mode),
            DYNK_TOPN=_dynk_mod.LEARNED_TOPN,
            DYNK_F=_dynk_mod.LEARNED_F_IN,
            DYNK_H=_dynk_mod.LEARNED_H,
            DYNK_OUT=_dynk_mod.LEARNED_OUT,
        )
    else:
        dynk_args = dict(
            dynk_tau_ptr=_unused_i32,
            dynk_mlp_ptr=_unused_i32,
            dynk_slot=0,
            dynk_k=0,
            dynk_kk_ptr=_unused_i32,
            dynk_cnt_ptr=_unused_i32,
            DYNK_COUNT=False,
            DYNK_MODE=0,
            DYNK_TOPN=16,
            DYNK_F=16,
            DYNK_H=32,
            DYNK_OUT=8,
        )
    _router_triton_kernel[grid](
        scores,
        bias if bias is not None else scores,
        bias_alt,
        input_ids,
        num_token_non_padded,
        weights,
        indices,
        packed_out if packed_out is not None else _unused_i32,
        M,
        float(routed_scaling_factor),
        float(moe_softcapping),
        N=N,
        K=K,
        K_ROUTED=K_routed,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        N_GROUP=num_expert_group,
        TOPK_GROUP=topk_group,
        EXPERTS_PER_GROUP=experts_per_group,
        BLOCK_G=BLOCK_G,
        SCORING_FUNC=scoring_func_int,
        SQRTSOFTPLUS_LOG1P=bool(sqrtsoftplus_log1p),
        HAS_SOFTCAP=bool(moe_softcapping != 0.0),
        RENORMALIZE=bool(renormalize),
        APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
        HAS_BIAS=bias is not None,
        HAS_TOKEN_BIAS=input_ids is not None,
        BIAS_ALT_TOKEN_ID=bias_alt_token_id,
        HAS_PADDING=num_token_non_padded is not None,
        HAS_PACKED=packed_out is not None,
        RENORMALIZE_EPSILON=renormalize_epsilon,
        USE_PDL=use_pdl,
        stride_bias=bias.stride(0) if bias is not None else 0,
        stride_bias_alt=bias_alt.stride(0) if bias_alt is not None else 0,
        stride_input_ids=input_ids.stride(0) if input_ids is not None else 0,
        stride_sm=scores.stride(0),
        stride_sn=scores.stride(1),
        stride_wm=weights.stride(0),
        stride_wk=weights.stride(1),
        stride_im=indices.stride(0),
        stride_ik=indices.stride(1),
        stride_pm=packed_out.stride(0) if packed_out is not None else 0,
        stride_pk=packed_out.stride(1) if packed_out is not None else 0,
        num_warps=num_warps,
        **dynk_args,
        **extra,
    )
    return weights, indices

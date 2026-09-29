"""Fused HC low-rank mix for decode-size batches.

One persistent kernel replaces the five-kernel `GatedResidual._mix_compute` chain.
One CTA per SM keeps every CTA resident, so the software grid barrier cannot deadlock;
the last CTA to finish resets the barrier counters,
so a captured CUDA graph replays with them in their initial state.
Row counts beyond ``_FUSED_MIX_MAX_ROWS`` stay on the torch.compile path.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_FUSED_MIX_MAX_ROWS = 16
# Prefetch variant: phase 2's up-weight tile is loaded before the phase barrier and the
# fp32 accumulator is persistent (re-zeroed by the last CTA), so there is no zeroing
# barrier. SGLANG_HC_MIX_PREFETCH=0 restores the original kernel.
_PREFETCH_ENABLED = os.environ.get("SGLANG_HC_MIX_PREFETCH", "1") != "0"
_PREFETCH_BLOCK_J = 8


@triton.jit
def _grid_barrier(counter_ptr, num_ctas):
    tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    while tl.atomic_add(counter_ptr, 0, sem="acq_rel", scope="gpu") < num_ctas:
        pass


@triton.jit
def _hc_mix_persistent_kernel(
    x_ptr,
    w_down_ptr,
    w_up_ptr,
    t_raw_ptr,
    out_ptr,
    counters_ptr,
    K,
    LOWRANK,
    HS,
    num_rows,
    num_ctas,
    inv_hc,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows

    zero_span = ROWS * LOWRANK
    offs_z = tl.arange(0, 256)
    for z0 in range(pid * 256, zero_span, num_ctas * 256):
        idx = z0 + offs_z
        tl.store(t_raw_ptr + idx, 0.0, mask=idx < zero_span)
    _grid_barrier(counters_ptr + 0, num_ctas)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    n_blocks = tl.cdiv(LOWRANK, BLOCK_N)
    k_chunks = tl.cdiv(K, BLOCK_K)
    for tile in range(pid, n_blocks * k_chunks, num_ctas):
        nb = tile % n_blocks
        kc = tile // n_blocks
        n = nb * BLOCK_N + offs_n
        k = kc * BLOCK_K + offs_k
        mask_n = n < LOWRANK
        xt = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        acc = tl.dot(xt, tl.trans(w))
        tl.atomic_add(
            t_raw_ptr + offs_m[:, None] * LOWRANK + n[None, :],
            acc,
            mask=mask_n[None, :],
            sem="relaxed",
            scope="gpu",
        )
    _grid_barrier(counters_ptr + 1, num_ctas)

    offs_j = tl.arange(0, BLOCK_J)
    offs_r = tl.arange(0, BLOCK_R)
    offs_g = tl.arange(0, HC)
    j_blocks = tl.cdiv(HS, BLOCK_J)
    for jb in range(pid, j_blocks, num_ctas):
        j = jb * BLOCK_J + offs_j
        mask_j = j < HS
        gj = offs_g[:, None] * HS + j[None, :]
        gj_flat = tl.reshape(gj, (HC * BLOCK_J,))
        mask_gj = tl.reshape(
            tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,)
        )
        acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
        for r0 in range(0, LOWRANK, BLOCK_R):
            r = r0 + offs_r
            mask_r = r < LOWRANK
            a = tl.load(
                t_raw_ptr + offs_m[:, None] * LOWRANK + r[None, :],
                mask=mask_r[None, :],
                other=0.0,
            )
            a = a * inv_hc
            t = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty)
            w = tl.load(
                w_up_ptr + gj_flat[:, None] * LOWRANK + r[None, :],
                mask=mask_gj[:, None] & mask_r[None, :],
                other=0.0,
            )
            acc = tl.dot(t, tl.trans(w), acc)
        gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
        xg = tl.load(
            x_ptr
            + offs_m[:, None, None] * (HC * HS)
            + offs_g[None, :, None] * HS
            + j[None, None, :],
            mask=mask_m[:, None, None] & mask_j[None, None, :],
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(gate * xg, axis=1) * inv_hc
        tl.store(
            out_ptr + offs_m[:, None] * HS + j[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_j[None, :],
        )

    ticket = tl.atomic_add(counters_ptr + 2, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        tl.store(counters_ptr + 0, 0)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)


@triton.jit
def _hc_mix_prefetch_kernel(
    x_ptr,
    w_down_ptr,
    w_up_ptr,
    t_raw_ptr,
    out_ptr,
    counters_ptr,
    K,
    HS,
    num_rows,
    num_ctas,
    inv_hc,
    LOWRANK: tl.constexpr,
    RA: tl.constexpr,
    RB: tl.constexpr,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows

    # ---- prefetch phase-2 operands for this CTA's first J block (independent of phase 1)
    j_blocks = tl.cdiv(HS, BLOCK_J)
    offs_j = tl.arange(0, BLOCK_J)
    offs_g = tl.arange(0, HC)
    j = pid * BLOCK_J + offs_j
    mask_j = j < HS
    gj = offs_g[:, None] * HS + j[None, :]
    gj_flat = tl.reshape(gj, (HC * BLOCK_J,))
    mask_gj = tl.reshape(tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,))
    offs_ra = tl.arange(0, RA)
    offs_rb = tl.arange(0, RB)
    w_a = tl.load(
        w_up_ptr + gj_flat[:, None] * LOWRANK + offs_ra[None, :],
        mask=mask_gj[:, None],
        other=0.0,
        eviction_policy="evict_first",
    )
    w_b = tl.load(
        w_up_ptr + gj_flat[:, None] * LOWRANK + RA + offs_rb[None, :],
        mask=mask_gj[:, None],
        other=0.0,
        eviction_policy="evict_first",
    )
    xg = tl.load(
        x_ptr + offs_m[:, None, None] * (HC * HS) + offs_g[None, :, None] * HS + j[None, None, :],
        mask=mask_m[:, None, None] & mask_j[None, None, :],
        other=0.0,
    ).to(tl.float32)

    # ---- phase 0 removed: t_raw is persistent and all-zero on entry (see exit)

    # ---- phase 1: t_raw += x @ Wd^T, tiles over (N block, K chunk)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    n_blocks = tl.cdiv(LOWRANK, BLOCK_N)
    k_chunks = tl.cdiv(K, BLOCK_K)
    for tile in range(pid, n_blocks * k_chunks, num_ctas):
        nb = tile % n_blocks
        kc = tile // n_blocks
        n = nb * BLOCK_N + offs_n
        k = kc * BLOCK_K + offs_k
        mask_n = n < LOWRANK
        xt = tl.load(x_ptr + offs_m[:, None] * K + k[None, :], mask=mask_m[:, None], other=0.0)
        w = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        acc = tl.dot(xt, tl.trans(w))
        tl.atomic_add(
            t_raw_ptr + offs_m[:, None] * LOWRANK + n[None, :],
            acc,
            mask=mask_n[None, :],
            sem="relaxed",
            scope="gpu",
        )
    _grid_barrier(counters_ptr + 1, num_ctas)

    # ---- phase 2: first J block uses the prefetched operands
    if pid < j_blocks:
        a = tl.load(t_raw_ptr + offs_m[:, None] * LOWRANK + offs_ra[None, :], cache_modifier=".cg") * inv_hc
        ta = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty)
        b = tl.load(t_raw_ptr + offs_m[:, None] * LOWRANK + RA + offs_rb[None, :], cache_modifier=".cg") * inv_hc
        tb = (b * tl.sigmoid(b)).to(x_ptr.dtype.element_ty)
        acc = tl.dot(ta, tl.trans(w_a))
        acc = tl.dot(tb, tl.trans(w_b), acc)
        gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
        out = tl.sum(gate * xg, axis=1) * inv_hc
        tl.store(out_ptr + offs_m[:, None] * HS + j[None, :], out.to(out_ptr.dtype.element_ty),
                 mask=mask_m[:, None] & mask_j[None, :])
        # remaining J blocks (only when j_blocks > num_ctas)
        for jb in range(pid + num_ctas, j_blocks, num_ctas):
            j2 = jb * BLOCK_J + offs_j
            mask_j2 = j2 < HS
            gj2 = tl.reshape(offs_g[:, None] * HS + j2[None, :], (HC * BLOCK_J,))
            mask_gj2 = tl.reshape(tl.broadcast_to(mask_j2[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,))
            wa2 = tl.load(w_up_ptr + gj2[:, None] * LOWRANK + offs_ra[None, :], mask=mask_gj2[:, None], other=0.0)
            wb2 = tl.load(w_up_ptr + gj2[:, None] * LOWRANK + RA + offs_rb[None, :], mask=mask_gj2[:, None], other=0.0)
            acc2 = tl.dot(ta, tl.trans(wa2))
            acc2 = tl.dot(tb, tl.trans(wb2), acc2)
            gate2 = tl.sigmoid(tl.reshape(acc2, (ROWS, HC, BLOCK_J)))
            xg2 = tl.load(
                x_ptr + offs_m[:, None, None] * (HC * HS) + offs_g[None, :, None] * HS + j2[None, None, :],
                mask=mask_m[:, None, None] & mask_j2[None, None, :], other=0.0,
            ).to(tl.float32)
            out2 = tl.sum(gate2 * xg2, axis=1) * inv_hc
            tl.store(out_ptr + offs_m[:, None] * HS + j2[None, :], out2.to(out_ptr.dtype.element_ty),
                     mask=mask_m[:, None] & mask_j2[None, :])

    # every CTA has finished reading t_raw once it takes a ticket; the last one re-zeroes
    # the accumulator and the counters, restoring the entry invariant for the next call
    # (and for CUDA-graph replay).
    ticket = tl.atomic_add(counters_ptr + 2, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        offs_z = tl.arange(0, 1024)
        for z0 in range(0, ROWS * LOWRANK, 1024):
            idx = z0 + offs_z
            tl.store(t_raw_ptr + idx, 0.0, mask=idx < ROWS * LOWRANK)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)


_counters_cache = {}
_acc_cache = {}


def _split_lowrank(r: int):
    """LOWRANK = RA + RB with both powers of two (320 = 256 + 64), else None."""
    ra = 1 << (r.bit_length() - 1)
    rb = r - ra
    if rb == 0:
        return ra // 2, ra // 2
    if rb < 16 or rb & (rb - 1):
        return None
    return ra, rb


def _get_acc(device: torch.device, lowrank: int) -> torch.Tensor:
    buf = _acc_cache.get((device, lowrank))
    if buf is None:
        buf = torch.zeros((16, lowrank), dtype=torch.float32, device=device)
        _acc_cache[(device, lowrank)] = buf
    return buf


def _get_counters(device: torch.device) -> torch.Tensor:
    buf = _counters_cache.get(device)
    if buf is None:
        buf = torch.zeros(3, dtype=torch.int32, device=device)
        _counters_cache[device] = buf
    return buf


def _deterministic_inference() -> bool:
    from sglang.srt.runtime_context import get_exec

    try:
        exec_cfg = get_exec()
    except ValueError:
        return False
    return bool(exec_cfg.deterministic.enable_deterministic_inference)


def fused_hc_mix_supported(
    hyper_input_normed: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> bool:
    # The persistent kernel accumulates the down projection with
    # device-scope atomics, so summation order varies across replays.
    if _deterministic_inference():
        return False
    return (
        hyper_input_normed.is_cuda
        and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
        and w_down.dtype == hyper_input_normed.dtype
        and w_up.dtype == hyper_input_normed.dtype
        and hyper_input_normed.shape[0] <= _FUSED_MIX_MAX_ROWS
        and hyper_input_normed.dim() == 2
        and hyper_input_normed.shape[1] % 2048 == 0
        and hyper_input_normed.is_contiguous()
        and w_down.is_contiguous()
        and w_up.is_contiguous()
    )


def fused_hc_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
) -> torch.Tensor:
    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    rows_pad = 16
    device = hyper_input_normed.device
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    split = _split_lowrank(lowrank) if _PREFETCH_ENABLED else None
    if split is not None:
        out = torch.empty((rows, hs), dtype=hyper_input_normed.dtype, device=device)
        if rows == 0:
            return out
        # Same stream-ordering contract as the original kernel's shared counters:
        # calls on one device must not run concurrently.
        _hc_mix_prefetch_kernel[(num_ctas,)](
            hyper_input_normed,
            w_down,
            w_up,
            _get_acc(device, lowrank),
            out,
            _get_counters(device),
            k,
            hs,
            rows,
            num_ctas,
            1.0 / hc,
            LOWRANK=lowrank,
            RA=split[0],
            RB=split[1],
            ROWS=rows_pad,
            HC=hc,
            BLOCK_N=32,
            BLOCK_K=256,
            BLOCK_J=_PREFETCH_BLOCK_J,
            num_warps=8,
        )
        return out
    t_raw = torch.empty((rows_pad, lowrank), dtype=torch.float32, device=device)
    out = torch.empty((rows, hs), dtype=hyper_input_normed.dtype, device=device)
    if rows == 0:
        return out
    _hc_mix_persistent_kernel[(num_ctas,)](
        hyper_input_normed,
        w_down,
        w_up,
        t_raw,
        out,
        _get_counters(device),
        k,
        lowrank,
        hs,
        rows,
        num_ctas,
        1.0 / hc,
        ROWS=rows_pad,
        HC=hc,
        BLOCK_N=32,
        BLOCK_K=256,
        BLOCK_J=32,
        BLOCK_R=64,
        num_warps=8,
    )
    return out


# ---------------------------------------------------------------------------------------
# qwen-opt fuse: HC mix + combine-gate logits in one launch.
# Used only when SGLANG_QWENOPT_FUSE_HC=1 (see srt/layers/hyperconnection.py); the kernels
# above are untouched. The combine gate 2*sigmoid(normed . W_inject^T / hc) depends only on
# the mix input, so its dot products ride along phase 1 of the mix (same x tiles, 4 extra
# weight rows) and the [rows, hc] fp32 logits are handed to the combine, which then needs
# no gate launch. Summation order of the logits differs from hc_combine_gate (atomics over
# K chunks), exactly like the mix's own low-rank accumulator.
# ---------------------------------------------------------------------------------------
_GATE_PAD = 16  # inject rows padded to the minimum tl.dot N


@triton.jit
def _hc_gate_tile(
    x_ptr,
    w_inj_ptr,
    g_raw_ptr,
    kc,
    K,
    offs_m,
    mask_m,
    HC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GP: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_K)
    offs_c = tl.arange(0, GP)
    mask_c = offs_c < HC
    k = kc * BLOCK_K + offs_k
    xt = tl.load(x_ptr + offs_m[:, None] * K + k[None, :], mask=mask_m[:, None], other=0.0)
    wi = tl.load(
        w_inj_ptr + offs_c[:, None] * K + k[None, :],
        mask=mask_c[:, None],
        other=0.0,
        eviction_policy="evict_first",
    )
    acc = tl.dot(xt, tl.trans(wi))
    tl.atomic_add(
        g_raw_ptr + offs_m[:, None] * GP + offs_c[None, :],
        acc,
        mask=mask_c[None, :],
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def _hc_gate_emit(g_raw_ptr, logits_ptr, offs_m, mask_m, HC: tl.constexpr, GP: tl.constexpr):
    offs_c = tl.arange(0, GP)
    mask_c = offs_c < HC
    g = tl.load(
        g_raw_ptr + offs_m[:, None] * GP + offs_c[None, :], cache_modifier=".cg"
    )
    tl.store(
        logits_ptr + offs_m[:, None] * HC + offs_c[None, :],
        g,
        mask=mask_m[:, None] & mask_c[None, :],
    )


@triton.jit
def _hc_mix_prefetch_gate_kernel(
    x_ptr,
    w_down_ptr,
    w_up_ptr,
    w_inj_ptr,
    t_raw_ptr,
    g_raw_ptr,
    out_ptr,
    logits_ptr,
    counters_ptr,
    K,
    HS,
    num_rows,
    num_ctas,
    inv_hc,
    LOWRANK: tl.constexpr,
    RA: tl.constexpr,
    RB: tl.constexpr,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    GP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows

    j_blocks = tl.cdiv(HS, BLOCK_J)
    offs_j = tl.arange(0, BLOCK_J)
    offs_g = tl.arange(0, HC)
    j = pid * BLOCK_J + offs_j
    mask_j = j < HS
    gj = offs_g[:, None] * HS + j[None, :]
    gj_flat = tl.reshape(gj, (HC * BLOCK_J,))
    mask_gj = tl.reshape(tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,))
    offs_ra = tl.arange(0, RA)
    offs_rb = tl.arange(0, RB)
    w_a = tl.load(
        w_up_ptr + gj_flat[:, None] * LOWRANK + offs_ra[None, :],
        mask=mask_gj[:, None],
        other=0.0,
        eviction_policy="evict_first",
    )
    w_b = tl.load(
        w_up_ptr + gj_flat[:, None] * LOWRANK + RA + offs_rb[None, :],
        mask=mask_gj[:, None],
        other=0.0,
        eviction_policy="evict_first",
    )
    xg = tl.load(
        x_ptr + offs_m[:, None, None] * (HC * HS) + offs_g[None, :, None] * HS + j[None, None, :],
        mask=mask_m[:, None, None] & mask_j[None, None, :],
        other=0.0,
    ).to(tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    n_blocks = tl.cdiv(LOWRANK, BLOCK_N)
    k_chunks = tl.cdiv(K, BLOCK_K)
    n_tiles = n_blocks * k_chunks
    for tile in range(pid, n_tiles, num_ctas):
        nb = tile % n_blocks
        kc = tile // n_blocks
        n = nb * BLOCK_N + offs_n
        k = kc * BLOCK_K + offs_k
        mask_n = n < LOWRANK
        xt = tl.load(x_ptr + offs_m[:, None] * K + k[None, :], mask=mask_m[:, None], other=0.0)
        w = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        acc = tl.dot(xt, tl.trans(w))
        tl.atomic_add(
            t_raw_ptr + offs_m[:, None] * LOWRANK + n[None, :],
            acc,
            mask=mask_n[None, :],
            sem="relaxed",
            scope="gpu",
        )
    # Gate tiles go to the CTAs that follow the main tiles round-robin, i.e. gate tile g runs on
    # CTA (n_tiles + g) % num_ctas: the ones with one main tile fewer. A separate loop keeps the
    # main loop's body (and its software pipelining) identical to the original kernel.
    g_first = (pid - n_tiles % num_ctas + num_ctas) % num_ctas
    for g in range(g_first, k_chunks, num_ctas):
        _hc_gate_tile(x_ptr, w_inj_ptr, g_raw_ptr, g, K, offs_m, mask_m, HC, BLOCK_K, GP)
    _grid_barrier(counters_ptr + 1, num_ctas)

    # The last CTA runs the fewest phase-2 J blocks, so it has slack for the logits copy-out.
    if pid == num_ctas - 1:
        _hc_gate_emit(g_raw_ptr, logits_ptr, offs_m, mask_m, HC, GP)

    if pid < j_blocks:
        a = tl.load(t_raw_ptr + offs_m[:, None] * LOWRANK + offs_ra[None, :], cache_modifier=".cg") * inv_hc
        ta = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty)
        b = tl.load(t_raw_ptr + offs_m[:, None] * LOWRANK + RA + offs_rb[None, :], cache_modifier=".cg") * inv_hc
        tb = (b * tl.sigmoid(b)).to(x_ptr.dtype.element_ty)
        acc = tl.dot(ta, tl.trans(w_a))
        acc = tl.dot(tb, tl.trans(w_b), acc)
        gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
        out = tl.sum(gate * xg, axis=1) * inv_hc
        tl.store(out_ptr + offs_m[:, None] * HS + j[None, :], out.to(out_ptr.dtype.element_ty),
                 mask=mask_m[:, None] & mask_j[None, :])
        for jb in range(pid + num_ctas, j_blocks, num_ctas):
            j2 = jb * BLOCK_J + offs_j
            mask_j2 = j2 < HS
            gj2 = tl.reshape(offs_g[:, None] * HS + j2[None, :], (HC * BLOCK_J,))
            mask_gj2 = tl.reshape(tl.broadcast_to(mask_j2[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,))
            wa2 = tl.load(w_up_ptr + gj2[:, None] * LOWRANK + offs_ra[None, :], mask=mask_gj2[:, None], other=0.0)
            wb2 = tl.load(w_up_ptr + gj2[:, None] * LOWRANK + RA + offs_rb[None, :], mask=mask_gj2[:, None], other=0.0)
            acc2 = tl.dot(ta, tl.trans(wa2))
            acc2 = tl.dot(tb, tl.trans(wb2), acc2)
            gate2 = tl.sigmoid(tl.reshape(acc2, (ROWS, HC, BLOCK_J)))
            xg2 = tl.load(
                x_ptr + offs_m[:, None, None] * (HC * HS) + offs_g[None, :, None] * HS + j2[None, None, :],
                mask=mask_m[:, None, None] & mask_j2[None, None, :], other=0.0,
            ).to(tl.float32)
            out2 = tl.sum(gate2 * xg2, axis=1) * inv_hc
            tl.store(out_ptr + offs_m[:, None] * HS + j2[None, :], out2.to(out_ptr.dtype.element_ty),
                     mask=mask_m[:, None] & mask_j2[None, :])

    # Same exit protocol as _hc_mix_prefetch_kernel, plus the gate accumulator.
    ticket = tl.atomic_add(counters_ptr + 2, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        offs_z = tl.arange(0, 1024)
        for z0 in range(0, ROWS * LOWRANK, 1024):
            idx = z0 + offs_z
            tl.store(t_raw_ptr + idx, 0.0, mask=idx < ROWS * LOWRANK)
        offs_zg = tl.arange(0, ROWS * GP)
        tl.store(g_raw_ptr + offs_zg, 0.0)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)


@triton.jit
def _hc_mix_persistent_gate_kernel(
    x_ptr,
    w_down_ptr,
    w_up_ptr,
    w_inj_ptr,
    t_raw_ptr,
    g_raw_ptr,
    out_ptr,
    logits_ptr,
    counters_ptr,
    K,
    LOWRANK,
    HS,
    num_rows,
    num_ctas,
    inv_hc,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows

    zero_span = ROWS * LOWRANK
    offs_z = tl.arange(0, 256)
    for z0 in range(pid * 256, zero_span, num_ctas * 256):
        idx = z0 + offs_z
        tl.store(t_raw_ptr + idx, 0.0, mask=idx < zero_span)
    if pid == 0:
        tl.store(g_raw_ptr + tl.arange(0, ROWS * GP), 0.0)
    _grid_barrier(counters_ptr + 0, num_ctas)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    n_blocks = tl.cdiv(LOWRANK, BLOCK_N)
    k_chunks = tl.cdiv(K, BLOCK_K)
    n_tiles = n_blocks * k_chunks
    for tile in range(pid, n_tiles, num_ctas):
        nb = tile % n_blocks
        kc = tile // n_blocks
        n = nb * BLOCK_N + offs_n
        k = kc * BLOCK_K + offs_k
        mask_n = n < LOWRANK
        xt = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        acc = tl.dot(xt, tl.trans(w))
        tl.atomic_add(
            t_raw_ptr + offs_m[:, None] * LOWRANK + n[None, :],
            acc,
            mask=mask_n[None, :],
            sem="relaxed",
            scope="gpu",
        )
    # Gate tiles go to the CTAs that follow the main tiles round-robin, i.e. gate tile g runs on
    # CTA (n_tiles + g) % num_ctas: the ones with one main tile fewer. A separate loop keeps the
    # main loop's body (and its software pipelining) identical to the original kernel.
    g_first = (pid - n_tiles % num_ctas + num_ctas) % num_ctas
    for g in range(g_first, k_chunks, num_ctas):
        _hc_gate_tile(x_ptr, w_inj_ptr, g_raw_ptr, g, K, offs_m, mask_m, HC, BLOCK_K, GP)
    _grid_barrier(counters_ptr + 1, num_ctas)

    # The last CTA runs the fewest phase-2 J blocks, so it has slack for the logits copy-out.
    if pid == num_ctas - 1:
        _hc_gate_emit(g_raw_ptr, logits_ptr, offs_m, mask_m, HC, GP)

    offs_j = tl.arange(0, BLOCK_J)
    offs_r = tl.arange(0, BLOCK_R)
    offs_g = tl.arange(0, HC)
    j_blocks = tl.cdiv(HS, BLOCK_J)
    for jb in range(pid, j_blocks, num_ctas):
        j = jb * BLOCK_J + offs_j
        mask_j = j < HS
        gj = offs_g[:, None] * HS + j[None, :]
        gj_flat = tl.reshape(gj, (HC * BLOCK_J,))
        mask_gj = tl.reshape(
            tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,)
        )
        acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
        for r0 in range(0, LOWRANK, BLOCK_R):
            r = r0 + offs_r
            mask_r = r < LOWRANK
            a = tl.load(
                t_raw_ptr + offs_m[:, None] * LOWRANK + r[None, :],
                mask=mask_r[None, :],
                other=0.0,
            )
            a = a * inv_hc
            t = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty)
            w = tl.load(
                w_up_ptr + gj_flat[:, None] * LOWRANK + r[None, :],
                mask=mask_gj[:, None] & mask_r[None, :],
                other=0.0,
            )
            acc = tl.dot(t, tl.trans(w), acc)
        gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
        xg = tl.load(
            x_ptr
            + offs_m[:, None, None] * (HC * HS)
            + offs_g[None, :, None] * HS
            + j[None, None, :],
            mask=mask_m[:, None, None] & mask_j[None, None, :],
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(gate * xg, axis=1) * inv_hc
        tl.store(
            out_ptr + offs_m[:, None] * HS + j[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_j[None, :],
        )

    ticket = tl.atomic_add(counters_ptr + 2, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        tl.store(counters_ptr + 0, 0)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)


_gacc_cache = {}


def _get_gacc(device: torch.device) -> torch.Tensor:
    buf = _gacc_cache.get(device)
    if buf is None:
        buf = torch.zeros((16, _GATE_PAD), dtype=torch.float32, device=device)
        _gacc_cache[device] = buf
    return buf


def fused_hc_mix_gate_supported(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_inject: torch.Tensor,
    hc: int,
) -> bool:
    return (
        fused_hc_mix_supported(hyper_input_normed, w_down, w_up)
        and w_inject.dtype == hyper_input_normed.dtype
        and w_inject.is_contiguous()
        and w_inject.dim() == 2
        and w_inject.shape[0] == hc
        and w_inject.shape[1] == hyper_input_normed.shape[1]
        and hc <= _GATE_PAD
    )


def fused_hc_mix_gate(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_inject: torch.Tensor,
    hc: int,
    hs: int,
):
    """``fused_hc_mix`` plus the combine-gate logits.

    Returns ``(mixed [rows, hs], logits [rows, hc] fp32)`` where
    ``logits[m, c] = dot(hyper_input_normed[m], w_inject[c])`` accumulated in fp32
    (the combine applies ``2 * sigmoid(logit / hc)``).
    """
    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    rows_pad = 16
    device = hyper_input_normed.device
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    out = torch.empty((rows, hs), dtype=hyper_input_normed.dtype, device=device)
    logits = torch.empty((rows, hc), dtype=torch.float32, device=device)
    if rows == 0:
        return out, logits
    block_k = 256
    split = _split_lowrank(lowrank) if _PREFETCH_ENABLED else None
    if split is not None:
        _hc_mix_prefetch_gate_kernel[(num_ctas,)](
            hyper_input_normed,
            w_down,
            w_up,
            w_inject,
            _get_acc(device, lowrank),
            _get_gacc(device),
            out,
            logits,
            _get_counters(device),
            k,
            hs,
            rows,
            num_ctas,
            1.0 / hc,
            LOWRANK=lowrank,
            RA=split[0],
            RB=split[1],
            ROWS=rows_pad,
            HC=hc,
            BLOCK_N=32,
            BLOCK_K=block_k,
            BLOCK_J=_PREFETCH_BLOCK_J,
            GP=_GATE_PAD,
            num_warps=8,
        )
        return out, logits
    t_raw = torch.empty((rows_pad, lowrank), dtype=torch.float32, device=device)
    g_raw = torch.empty((rows_pad, _GATE_PAD), dtype=torch.float32, device=device)
    _hc_mix_persistent_gate_kernel[(num_ctas,)](
        hyper_input_normed,
        w_down,
        w_up,
        w_inject,
        t_raw,
        g_raw,
        out,
        logits,
        _get_counters(device),
        k,
        lowrank,
        hs,
        rows,
        num_ctas,
        1.0 / hc,
        ROWS=rows_pad,
        HC=hc,
        BLOCK_N=32,
        BLOCK_K=block_k,
        BLOCK_J=32,
        BLOCK_R=64,
        GP=_GATE_PAD,
        num_warps=8,
    )
    return out, logits

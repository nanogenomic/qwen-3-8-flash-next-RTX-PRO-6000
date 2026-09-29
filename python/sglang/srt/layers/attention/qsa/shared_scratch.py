# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One max-size QSA scratch set shared by every ``QwenSparseAttnBackend``.

Why this exists
---------------
Adaptive speculation builds one ``SpecRuntimeState`` per candidate step count
(``eagle_worker_v2.build_adaptive_runtime_state``), and each state constructs its
own QSA backends: one for target verify, one for draft-extend, and
``speculative_num_steps - 1`` for the draft decode chain
(``QwenSparseMultiStepDraftBackend.__init__``). Every one of those instances used
to own

  * a private 128 MiB trtllm-gen workspace, and
  * a private packed-KV gather buffer sized from *its own* verify window,

so the runtime-state cost grew with tier depth and the deep tiers ran the GPU out
of memory before the KV pool was even sized. Only one tier is active at a time
and the state swap happens between forwards, so a single max-size set of scratch
serves all of them.

This mirrors what the DSV4 / DSA backends already do for the same kernel family
(``deepseek_v4_trtllm_backend._get_trtllm_workspace_buffer``,
``dsa_backend`` ``"dsa_trtllm_workspace"``): one named persistent buffer per
process via ``runtime_context.get_buffer``. The QSA backend was the outlier.

The non-aliasing argument
-------------------------
Both shared regions are **pure scratch with no state that crosses a forward
boundary**:

  * the trtllm workspace is written and consumed inside one
    ``trtllm_batch_decode_with_kv_cache`` launch;
  * ``packed_k`` / ``packed_v`` are fully rewritten by
    ``qwen_sparse_kv_extraction_compact_triton`` (with ``zero_fill_cols=stride``,
    so even the padding columns are re-established) immediately before the decode
    kernel reads them.

A single backend already reuses both across every QSA layer of a forward and
across every forward, so reuse across *instances* is the same access pattern with
a different owner — provided no two QSA forwards are in flight at once. Two
things could break that, and both disable sharing here
(:func:`qsa_scratch_sharing_enabled`): two-batch overlap and PDMux, which are the
only mechanisms that put model forwards on more than one stream.

The pointer-stability rule
--------------------------
A captured CUDA graph bakes in the *address* of the packed-KV buffer. Growing the
buffer after a capture would free the block the earlier tier's graph still points
at — a use-after-free that reads as silent corruption, not an error. So the
buffer is **planned before it is allocated**: every backend calls
:func:`qsa_reserve_packed_kv_rows` from ``init_cuda_graph_state`` with the row
count the *deepest reachable* tier needs, the first real use allocates at the max
of every reservation, and a later request that exceeds the allocation raises
instead of reallocating.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Page size the sparse gather packs to; must match
# ``qwen_sparse_attn_backend._TRTLLM_SPARSE_PAGE_SIZE``.
TRTLLM_SPARSE_PAGE_SIZE = 64
# trtllm-gen scratch size, same constant DSV4 uses.
TRTLLM_SPARSE_WORKSPACE_BYTES = 128 * 1024 * 1024

_PLAN_KEY = "qsa_shared_scratch_plan"
_WORKSPACE_KEY = "qsa_trtllm_sparse_workspace"


def qsa_scratch_sharing_enabled() -> bool:
    """False when a second stream could run a QSA forward concurrently.

    Sharing is only safe while QSA forwards are serialized. Two-batch overlap and
    PDMux are the two features that put model forwards on more than one stream;
    under either, each backend keeps its own scratch.
    """
    from sglang.srt.environ import envs

    if not envs.SGLANG_QSA_SHARE_SCRATCH.get():
        return False
    try:
        from sglang.srt.runtime_context import get_disagg, get_exec

        if get_exec().overlap.enable_two_batch_overlap:
            return False
        # PDMux replays on one attention backend per stream. Adaptive spec
        # already refuses it (adaptive_spec_params.py:84-86); the guard is here
        # so a non-adaptive PDMux boot keeps private scratch regardless.
        if get_disagg().enable_pdmux:
            return False
    except Exception:
        # No runtime context (unit harnesses, tooling): nothing to overlap with.
        return True
    return True


def packed_kv_rows(num_tokens: int, expanded_topk: int) -> int:
    """Rows the packed-KV gather needs for ``num_tokens`` query rows.

    The gather writes one page-aligned run of ``pages_per_row * page`` rows per
    query row; this is the same expression ``_forward_trtllm_sparse`` uses to
    size its request.
    """
    pages_per_row = (
        expanded_topk + TRTLLM_SPARSE_PAGE_SIZE - 1
    ) // TRTLLM_SPARSE_PAGE_SIZE
    return int(num_tokens) * pages_per_row * TRTLLM_SPARSE_PAGE_SIZE


def _normalize_device(device) -> torch.device:
    """Resolve an index-less device so it cannot key a second allocation.

    ``torch.device("cuda")`` and ``torch.device("cuda", 0)`` are different dict
    keys but the same physical device. A caller that passes the bare form (the
    forward path passes ``k_buffer.device``, which always carries an index, but a
    reservation or a test may not) would otherwise mint a second max-size buffer
    and undo the sharing. Caught by the GPU equivalence run, which reported
    ``packed_allocations=2``.
    """
    d = torch.device(device)
    if d.index is not None or d.type == "cpu":
        return d
    try:
        return torch.device(d.type, torch.get_device_module(d.type).current_device())
    except Exception:
        return d


class QSASharedScratchPlan:
    """Reservations, allocations and the pointer-stability invariant.

    Held as one named persistent buffer so every backend in the process — across
    every adaptive runtime state — addresses the same instance.
    """

    def __init__(self) -> None:
        self.reserved_rows: int = 0
        # (num_kv_heads, head_dim, dtype, device) -> (k, v) at allocated_rows
        self._packed: Dict[Tuple[int, int, torch.dtype, torch.device], Tuple] = {}
        self._allocated_rows: Dict[Tuple[int, int, torch.dtype, torch.device], int] = {}
        # (batch, pages_per_row, device) -> (cu_strided, block_tables)
        self._sparse_tables: Dict[Tuple[int, int, torch.device], Tuple] = {}
        self.reserve_calls: int = 0
        self.packed_allocations: int = 0
        self.handout_calls: int = 0
        self.overflow_calls: int = 0
        # True once any buffer was handed out while a CUDA graph was capturing;
        # from then on the buffer address is baked into a graph and must not move.
        self._captured: bool = False

    # -- planning -------------------------------------------------------
    def reserve_rows(self, rows: int) -> int:
        """Record a row requirement. Safe to call any number of times."""
        rows = int(rows)
        self.reserve_calls += 1
        if rows > self.reserved_rows:
            # A buffer may already exist before any reservation: FlashInfer
            # autotune runs a forward (and so a handout) before
            # init_cuda_graph_state reserves. A reservation the existing buffer
            # already covers is satisfied as-is (idempotent); only one that
            # would need a LARGER buffer is refused, because growing it would
            # strand graphs that baked in the old pointer.
            covered = min(self._allocated_rows.values()) if self._allocated_rows else None
            if covered is not None and rows > covered and not self._captured:
                # Nothing has been handed out under CUDA-graph capture yet (e.g.
                # only the autotune warmup allocated), so no graph holds the old
                # pointer: drop the undersized buffers; the next handout
                # allocates at the new reservation.
                logger.info(
                    "QSA shared packed-KV scratch regrown before capture: %d -> %d rows",
                    covered,
                    rows,
                )
                self._packed.clear()
                self._allocated_rows.clear()
                covered = None
            if covered is not None and rows > covered:
                raise ValueError(
                    "QSA shared packed-KV scratch was already allocated at "
                    f"{max(self._allocated_rows.values())} rows when a backend "
                    f"reserved {rows}. Reservations must be made at the deepest "
                    "reachable speculative width before the first CUDA graph "
                    "capture (see QwenSparseAttnBackend.init_cuda_graph_state)."
                )
            self.reserved_rows = rows
        return self.reserved_rows

    # -- handout --------------------------------------------------------
    def packed_kv(
        self,
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Views over the shared buffer, or None when it is too small.

        None is the signal to the caller that this request cannot be served from
        shared storage. The shared buffer is never grown: a captured graph holds
        its address, so growing it would free the block that graph replays from.
        """
        capacity = int(capacity)
        device = _normalize_device(device)
        key = (int(num_kv_heads), int(head_dim), dtype, device)
        self.handout_calls += 1
        if (
            not self._captured
            and torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        ):
            self._captured = True
        buffers = self._packed.get(key)
        if buffers is None:
            rows = max(capacity, self.reserved_rows)
            shape = (rows, int(num_kv_heads), int(head_dim))
            buffers = (
                torch.empty(shape, dtype=dtype, device=device),
                torch.empty(shape, dtype=dtype, device=device),
            )
            self._packed[key] = buffers
            self._allocated_rows[key] = rows
            self.packed_allocations += 1
            logger.info(
                "QSA shared packed-KV scratch allocated: %d rows x %d heads x %d "
                "dim %s (%.1f MiB for k+v), reserved=%d requested=%d",
                rows,
                num_kv_heads,
                head_dim,
                dtype,
                2 * rows * num_kv_heads * head_dim * buffers[0].element_size() / 2**20,
                self.reserved_rows,
                capacity,
            )
        elif capacity > self._allocated_rows[key]:
            self.overflow_calls += 1
            return None
        return buffers[0][:capacity], buffers[1][:capacity]

    def sparse_tables(
        self, batch: int, pages_per_row: int, page: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read-only arange block tables — a pure function of the key."""
        device = _normalize_device(device)
        key = (int(batch), int(pages_per_row), device)
        cached = self._sparse_tables.get(key)
        if cached is None:
            stride = pages_per_row * page
            cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * stride
            block_tables = (
                torch.arange(batch, dtype=torch.int32, device=device)[:, None]
                * pages_per_row
                + torch.arange(pages_per_row, dtype=torch.int32, device=device)[None, :]
            ).contiguous()
            cached = (cu, block_tables)
            self._sparse_tables[key] = cached
        return cached

    # -- introspection (tests, boot log) --------------------------------
    def describe(self) -> dict:
        return {
            "reserved_rows": self.reserved_rows,
            "reserve_calls": self.reserve_calls,
            "packed_allocations": self.packed_allocations,
            "handout_calls": self.handout_calls,
            "overflow_calls": self.overflow_calls,
            "allocated_rows": dict(self._allocated_rows),
            "packed_bytes": sum(
                2 * k.numel() * k.element_size() for k, _ in self._packed.values()
            ),
            "sparse_table_keys": len(self._sparse_tables),
        }


def qsa_shared_scratch_plan() -> QSASharedScratchPlan:
    from sglang.srt.runtime_context import get_buffer

    return get_buffer(_PLAN_KEY, QSASharedScratchPlan)


def qsa_reserve_packed_kv_rows(rows: int) -> int:
    """Reserve ``rows`` of packed-KV scratch before any graph capture."""
    if not qsa_scratch_sharing_enabled():
        return 0
    return qsa_shared_scratch_plan().reserve_rows(rows)


def qsa_packed_kv(
    capacity: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Shared packed-KV views, or None when the request exceeds the plan."""
    return qsa_shared_scratch_plan().packed_kv(
        capacity, num_kv_heads, head_dim, dtype, device
    )


def qsa_sparse_block_tables(
    batch: int, pages_per_row: int, page: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    return qsa_shared_scratch_plan().sparse_tables(batch, pages_per_row, page, device)


def qsa_trtllm_workspace(device: torch.device) -> torch.Tensor:
    """The one trtllm-gen scratch buffer every QSA backend hands the kernel."""
    from sglang.srt.runtime_context import get_buffer

    return get_buffer(
        _WORKSPACE_KEY,
        lambda: torch.zeros(
            TRTLLM_SPARSE_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=_normalize_device(device),
        ),
    )


def reset_qsa_shared_scratch() -> None:
    """Drop the shared plan and workspace (tests only)."""
    from sglang.srt.runtime_context import get_context

    buffers = get_context().resources.buffers
    buffers.pop(_PLAN_KEY, None)
    buffers.pop(_WORKSPACE_KEY, None)


def deepest_packed_kv_rows(
    *,
    max_bs: int,
    max_num_tokens: int,
    expanded_topk: int,
    widest_num_draft_tokens: Optional[int] = None,
    widest_batch: Optional[int] = None,
) -> int:
    """Packed-KV rows the widest reachable (batch x verify window) needs.

    ``init_cuda_graph_state`` is called once per backend with that backend's own
    geometry, and BOTH axes are narrower than the process-wide worst case:

      * the window is the *active* tier's under adaptive speculation, so it is
        scaled up by ``widest_num_draft_tokens``;
      * the batch is that runner's FILTERED capture list. The target verify runner
        filters the configured bs by ``bs * W % alignment`` while the draft runner
        filters by ``bs * 1``, so the two can resolve different ``max_bs`` --- and
        whichever reserved second would be asking to GROW a buffer the first one's
        graphs had already captured, which is refused. ``widest_batch`` (the
        configured graph-bs list unioned with the admission cap) removes that
        ordering dependence, and covering the admission cap also means an eager
        verify above every captured graph batch is served from shared storage
        instead of falling back to a private buffer.
    """
    max_bs = max(1, int(max_bs))
    max_num_tokens = max(1, int(max_num_tokens))
    per_req = max(1, max_num_tokens // max_bs)
    widest_w = max(per_req, int(widest_num_draft_tokens or 0))
    widest_bs = max(max_bs, int(widest_batch or 0))
    tokens = max(max_num_tokens, widest_bs * widest_w)
    return packed_kv_rows(tokens, expanded_topk)


def expanded_topk_from_pool(pool) -> int:
    """Selection columns the gather sees: top-k blocks plus the uncompressed tail."""
    token_topk = int(getattr(pool, "qsa_token_topk", 0) or 0)
    ratio = int(getattr(pool, "qsa_compress_ratio", 1) or 1)
    if token_topk <= 0:
        raise ValueError("QSA pool does not publish qsa_token_topk")
    return token_topk + ratio - 1

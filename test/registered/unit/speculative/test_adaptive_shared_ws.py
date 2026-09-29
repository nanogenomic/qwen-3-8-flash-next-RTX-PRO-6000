# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared QSA scratch across adaptive runtime states: the allocation plan.

CPU only, no CUDA device required. Everything here drives the PRODUCTION plan
object and the production row-sizing helpers, so a regression in them fails here
rather than in a serving boot.

What it pins:
  * the plan allocates ONCE and hands out views, so every tier's captured graph
    reads the same pointer (the pointer-stability invariant);
  * a reservation made at the deepest reachable width is enough for every
    shallower tier, in the exact build order the adaptive controller uses;
  * a late reservation that would have grown an already-allocated buffer raises
    instead of stranding the graphs that captured the old address;
  * an over-capacity request returns None (the caller's private-fallback signal)
    rather than reallocating;
  * the B3c-4 byte budget: what the shipped tier set costs shared vs unshared.
"""

from __future__ import annotations

import math

try:
    import pytest
except ModuleNotFoundError:  # the serving venv has no pytest and must not be modified

    class _Skipped(Exception):
        pass

    class _Mark:
        name = "parametrize"

        def __init__(self, *args):
            self.args = args

        def __call__(self, fn):
            marks = list(getattr(fn, "pytestmark", []))
            marks.append(self)
            fn.pytestmark = marks
            return fn

    class _MarkFactory:
        def parametrize(self, *args):
            return _Mark(*args)

    class _PytestShim:
        Skipped = _Skipped
        mark = _MarkFactory()

        @staticmethod
        def skip(reason):
            raise _Skipped(reason)

        @staticmethod
        def raises(exc):
            class _Ctx:
                def __enter__(self):
                    return self

                def __exit__(self, et, ev, tb):
                    if et is None:
                        raise AssertionError(f"{exc.__name__} not raised")
                    return issubclass(et, exc)

            return _Ctx()

    pytest = _PytestShim()

import torch

from sglang.srt.layers.attention.qsa.shared_scratch import (
    TRTLLM_SPARSE_PAGE_SIZE,
    TRTLLM_SPARSE_WORKSPACE_BYTES,
    QSASharedScratchPlan,
    deepest_packed_kv_rows,
    expanded_topk_from_pool,
    packed_kv_rows,
)

# ---------------------------------------------------------------------------
# The B3c-4 profile, all source-verified.
#   checkpoint config.json (qwen38-flash-next-nvfp4-radixark-20260826):
#     num_key_value_heads=2, head_dim=256, indexer_budget=2048,
#     indexer_compress_ratio=4
#   serving launcher:  --max-running-requests 4
#   B3c-4: graph-bs 4  ->  target/draft-extend graphs capture bs=[1..4]
#   adaptive_flashnext.json candidate steps {1,3,5,7}; launch width 3/4
# ---------------------------------------------------------------------------
KV_HEADS = 2
HEAD_DIM = 256
DTYPE = torch.bfloat16
ELEM = 2
# qsa_token_topk IS the indexer budget, not budget//ratio: kv_cache_configurator.py:1936
# passes ``qsa_token_topk=qsa_profile.budget``; qsa_kv_pool.py:125 then derives
# ``qsa_block_topk = token_topk // ratio`` (512). The selection the gather sees is
# the EXPANDED one, kernel.py:211 ``final_topk = token_topk + compress_ratio - 1``.
QSA_TOKEN_TOPK = 2048  # indexer_budget
RATIO = 4
EXPANDED_TOPK = QSA_TOKEN_TOPK + RATIO - 1  # 2051: expanded token selection
MAX_BS = 4
CANDIDATE_STEPS = [1, 3, 5, 7]
LAUNCH_STEPS = 3
WIDEST_W = max(CANDIDATE_STEPS) + 1  # 8


class _FakePool:
    """Only the two attributes the row planner reads off the QSA KV pool."""

    qsa_token_topk = QSA_TOKEN_TOPK
    qsa_compress_ratio = RATIO


def _w(steps: int) -> int:
    """Chain speculation: verify window = num_steps + 1."""
    return steps + 1


def _stride() -> int:
    return (
        math.ceil(EXPANDED_TOPK / TRTLLM_SPARSE_PAGE_SIZE) * TRTLLM_SPARSE_PAGE_SIZE
    )


def _pair_bytes(rows: int) -> int:
    return 2 * rows * KV_HEADS * HEAD_DIM * ELEM


def _cpu() -> torch.device:
    return torch.device("cpu")


def _import_or_skip(mod_name: str):
    """Import an engine module, or skip.

    A few checks in this file read PRODUCTION source to pin an invariant that
    lives outside the plan object. Those imports pull the whole runtime, which
    only resolves in the serving venv -- so they run there and skip in a plain
    test venv rather than being dropped.
    """
    try:
        return __import__(mod_name, fromlist=["x"])
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{mod_name} not importable here: {type(exc).__name__}: {exc}")


# -- the geometry the plan is built on --------------------------------------


def test_expanded_topk_and_stride_match_the_backend_expression():
    assert expanded_topk_from_pool(_FakePool()) == EXPANDED_TOPK == 2051
    # 2051 columns pack to 33 pages of 64 -> a 2112-row run per query row.
    assert _stride() == 2112
    assert packed_kv_rows(1, EXPANDED_TOPK) == 2112
    assert packed_kv_rows(32, EXPANDED_TOPK) == 32 * 2112


def test_trtllm_page_size_and_workspace_match_the_backend_constants():
    qsab = _import_or_skip("sglang.srt.layers.attention.qwen_sparse_attn_backend")

    assert qsab._TRTLLM_SPARSE_PAGE_SIZE == TRTLLM_SPARSE_PAGE_SIZE
    assert TRTLLM_SPARSE_WORKSPACE_BYTES == 128 * 1024 * 1024


@pytest.mark.parametrize("steps", CANDIDATE_STEPS)
def test_deepest_rows_is_tier_independent(steps):
    """Every tier reserves the SAME rows, which is what makes one alloc enough.

    ``init_cuda_graph_state`` sees the active tier's window, so the planner has
    to scale it back up to the widest reachable one.
    """
    rows = deepest_packed_kv_rows(
        max_bs=MAX_BS,
        max_num_tokens=MAX_BS * _w(steps),
        expanded_topk=EXPANDED_TOPK,
        widest_num_draft_tokens=WIDEST_W,
    )
    assert rows == MAX_BS * WIDEST_W * _stride() == 4 * 8 * 2112 == 67584


def test_draft_decode_backends_reserve_the_same_rows():
    """The draft chain captures at topk=1 (one row per request), so its own need
    is 8x smaller -- it must still reserve the target's width, because it shares
    the buffer with the target verify backend."""
    draft_rows = deepest_packed_kv_rows(
        max_bs=MAX_BS,
        max_num_tokens=MAX_BS * 1,  # captured_req_width = topk = 1
        expanded_topk=EXPANDED_TOPK,
        widest_num_draft_tokens=WIDEST_W,
    )
    assert draft_rows == MAX_BS * WIDEST_W * _stride()


def test_reservation_is_independent_of_the_runners_filtered_capture_list():
    """Target and draft runners can resolve DIFFERENT max_bs, and the one that
    reserves second must not be asking to grow an already-captured buffer.

    ``get_batch_sizes_to_capture`` filters the configured graph bs by
    ``bs * captured_req_width % alignment``; the target verify runner passes
    ``captured_req_width = W`` and the draft runner passes 1, so with an alignment
    > 1 their surviving lists differ. ``widest_batch`` is what makes both
    reservations land on the same number.
    """
    widest_bs = 8  # configured graph bs list max / admission cap
    rows = [
        deepest_packed_kv_rows(
            max_bs=bs,
            max_num_tokens=bs * per_req,
            expanded_topk=EXPANDED_TOPK,
            widest_num_draft_tokens=WIDEST_W,
            widest_batch=widest_bs,
        )
        # target verify at its own bs and W, and the draft chain at bs x topk=1
        for bs, per_req in [(2, 4), (4, 4), (8, 8), (4, 1), (8, 1), (1, 1)]
    ]
    assert len(set(rows)) == 1
    assert rows[0] == widest_bs * WIDEST_W * _stride()

    # Without the bound, the wider runner reserves more and the plan refuses it.
    plan = QSASharedScratchPlan()
    plan.reserve_rows(
        deepest_packed_kv_rows(
            max_bs=4,
            max_num_tokens=4 * 4,
            expanded_topk=EXPANDED_TOPK,
            widest_num_draft_tokens=WIDEST_W,
        )
    )
    plan.packed_kv(plan.reserved_rows, KV_HEADS, HEAD_DIM, DTYPE, _cpu())
    plan._captured = True  # the first runner's graphs were captured
    with pytest.raises(ValueError):
        plan.reserve_rows(
            deepest_packed_kv_rows(
                max_bs=8,
                max_num_tokens=8 * 8,
                expanded_topk=EXPANDED_TOPK,
                widest_num_draft_tokens=WIDEST_W,
            )
        )


def test_widest_hint_missing_falls_back_to_the_backend_window():
    """No adaptive capacity published (static boot): the backend's own window."""
    rows = deepest_packed_kv_rows(
        max_bs=MAX_BS,
        max_num_tokens=MAX_BS * 4,
        expanded_topk=EXPANDED_TOPK,
        widest_num_draft_tokens=None,
    )
    assert rows == MAX_BS * 4 * _stride()


# -- the pointer-stability invariant ---------------------------------------


def _reserve_every_tier(plan: QSASharedScratchPlan) -> int:
    """Exactly what the adaptive build order does: the launch tier's backends
    first, then init_states() over the remaining candidates in sorted order."""
    order = [LAUNCH_STEPS] + [s for s in CANDIDATE_STEPS if s != LAUNCH_STEPS]
    for steps in order:
        w = _w(steps)
        # target verify, draft-extend, then steps-1 draft decode backends
        for max_num_tokens in (
            [MAX_BS * w, MAX_BS * w] + [MAX_BS * 1] * max(0, steps - 1)
        ):
            plan.reserve_rows(
                deepest_packed_kv_rows(
                    max_bs=MAX_BS,
                    max_num_tokens=max_num_tokens,
                    expanded_topk=EXPANDED_TOPK,
                    widest_num_draft_tokens=WIDEST_W,
                )
            )
    return plan.reserved_rows


def test_one_allocation_serves_every_tier_and_the_pointer_never_moves():
    plan = QSASharedScratchPlan()
    reserved = _reserve_every_tier(plan)
    assert reserved == MAX_BS * WIDEST_W * _stride()
    assert plan.reserve_calls == sum(2 + max(0, s - 1) for s in CANDIDATE_STEPS) == 20

    base_ptr = None
    for steps in [LAUNCH_STEPS] + [s for s in CANDIDATE_STEPS if s != LAUNCH_STEPS]:
        capacity = MAX_BS * _w(steps) * _stride()
        k, v = plan.packed_kv(capacity, KV_HEADS, HEAD_DIM, DTYPE, _cpu())
        assert k.shape == (capacity, KV_HEADS, HEAD_DIM)
        assert v.shape == k.shape
        if base_ptr is None:
            base_ptr = k.data_ptr()
        # Views into one allocation: the address a captured graph baked in.
        assert k.data_ptr() == base_ptr
    assert plan.packed_allocations == 1
    assert plan.overflow_calls == 0


def test_k_and_v_do_not_alias_each_other():
    """The gather writes keys and values separately; one buffer would corrupt."""
    plan = QSASharedScratchPlan()
    plan.reserve_rows(MAX_BS * WIDEST_W * _stride())
    k, v = plan.packed_kv(576, KV_HEADS, HEAD_DIM, DTYPE, _cpu())
    assert k.data_ptr() != v.data_ptr()
    k.fill_(0)
    v.fill_(1)
    assert torch.all(k == 0) and torch.all(v == 1)


def test_late_reservation_that_would_grow_the_buffer_raises():
    """A tier that reserves AFTER the first capture is a config error, not a
    resize: the earlier tier's graph still points at the current block."""
    plan = QSASharedScratchPlan()
    plan.reserve_rows(MAX_BS * 4 * _stride())  # launch width only -- the bug
    plan.packed_kv(MAX_BS * 4 * _stride(), KV_HEADS, HEAD_DIM, DTYPE, _cpu())
    plan._captured = True  # the first tier's graph was captured
    with pytest.raises(ValueError):
        plan.reserve_rows(MAX_BS * WIDEST_W * _stride())


def test_allocate_then_reserve_equal_succeeds():
    """Boot order on GPU1: FlashInfer autotune hands out the buffer before any
    reservation, then init_cuda_graph_state reserves the SAME size. Regression for
    "already allocated at 33792 rows when a backend reserved 33792"."""
    plan = QSASharedScratchPlan()
    rows = MAX_BS * WIDEST_W * _stride()
    k0, _ = plan.packed_kv(rows, KV_HEADS, HEAD_DIM, DTYPE, _cpu())
    assert plan.reserve_rows(rows) == rows
    k1, _ = plan.packed_kv(rows, KV_HEADS, HEAD_DIM, DTYPE, _cpu())
    assert k1.data_ptr() == k0.data_ptr() and plan.packed_allocations == 1


def test_allocate_then_reserve_smaller_succeeds():
    plan = QSASharedScratchPlan()
    rows = MAX_BS * WIDEST_W * _stride()
    plan.packed_kv(rows, KV_HEADS, HEAD_DIM, DTYPE, _cpu())
    plan.reserve_rows(rows // 2)
    plan.reserve_rows(rows)
    assert plan.packed_allocations == 1 and plan.overflow_calls == 0


def test_boot_order_autotune_then_every_tier_reserves_then_capture():
    """Smoke of the real boot order: autotune handout at the widest width, then
    every tier's backend reserves, then each tier's capture hands out its own
    width. One allocation, one pointer, no raise."""
    plan = QSASharedScratchPlan()
    widest = MAX_BS * WIDEST_W * _stride()
    k0, _ = plan.packed_kv(widest, KV_HEADS, HEAD_DIM, DTYPE, _cpu())
    for s in CANDIDATE_STEPS:
        plan.reserve_rows(MAX_BS * _w(s) * _stride())
    for s in CANDIDATE_STEPS:
        k, _ = plan.packed_kv(
            MAX_BS * _w(s) * _stride(), KV_HEADS, HEAD_DIM, DTYPE, _cpu()
        )
        assert k.data_ptr() == k0.data_ptr()
    assert plan.packed_allocations == 1 and plan.overflow_calls == 0


def test_allocate_smaller_then_reserve_larger_before_capture_regrows():
    """Autotune may hand out at the ACTIVE tier's width, below the widest
    reservation. Before any capture that is a regrow, not an error."""
    plan = QSASharedScratchPlan()
    small = MAX_BS * 4 * _stride()
    widest = MAX_BS * WIDEST_W * _stride()
    plan.packed_kv(small, KV_HEADS, HEAD_DIM, DTYPE, _cpu())
    assert plan.reserve_rows(widest) == widest
    k, _ = plan.packed_kv(widest, KV_HEADS, HEAD_DIM, DTYPE, _cpu())
    assert k.shape[0] == widest and plan.overflow_calls == 0


def test_allocate_then_reserve_larger_still_raises():
    plan = QSASharedScratchPlan()
    rows = MAX_BS * 4 * _stride()
    plan.packed_kv(rows, KV_HEADS, HEAD_DIM, DTYPE, _cpu())
    plan._captured = True  # a graph already holds the pointer
    with pytest.raises(ValueError):
        plan.reserve_rows(rows + 1)


def test_reservation_is_idempotent_and_order_free():
    plan_a = QSASharedScratchPlan()
    plan_b = QSASharedScratchPlan()
    widths = [MAX_BS * _w(s) * _stride() for s in CANDIDATE_STEPS]
    for rows in widths:
        plan_a.reserve_rows(rows)
    for rows in reversed(widths):
        plan_b.reserve_rows(rows)
    assert plan_a.reserved_rows == plan_b.reserved_rows == max(widths)


def test_over_capacity_request_returns_none_instead_of_reallocating():
    """An eager verify wider than anything captured (max_running_requests above
    the widest graph batch). The caller falls back to private scratch; the shared
    block must not move."""
    plan = QSASharedScratchPlan()
    plan.reserve_rows(MAX_BS * WIDEST_W * _stride())
    k, _ = plan.packed_kv(576, KV_HEADS, HEAD_DIM, DTYPE, _cpu())
    ptr = k.data_ptr()
    assert (
        plan.packed_kv(
            2 * MAX_BS * WIDEST_W * _stride(), KV_HEADS, HEAD_DIM, DTYPE, _cpu()
        )
        is None
    )
    assert plan.overflow_calls == 1
    assert plan.packed_allocations == 1
    k2, _ = plan.packed_kv(576, KV_HEADS, HEAD_DIM, DTYPE, _cpu())
    assert k2.data_ptr() == ptr


def test_indexless_and_indexed_device_are_one_key():
    """``cuda`` and ``cuda:0`` must not mint two max-size buffers.

    The forward path passes ``k_buffer.device`` (always indexed) but a reservation
    or a test may pass the bare form. Keying on the raw torch.device split them
    and allocated twice -- found by the GPU equivalence run
    (``packed_allocations=2``), fixed by normalizing the key.
    """
    from sglang.srt.layers.attention.qsa.shared_scratch import _normalize_device

    # CPU has no index to resolve and must be left alone.
    assert _normalize_device(torch.device("cpu")) == torch.device("cpu")
    assert _normalize_device("cpu") == torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device to resolve an index against")
    assert _normalize_device(torch.device("cuda")).index is not None
    assert _normalize_device(torch.device("cuda")) == _normalize_device(
        torch.device("cuda", torch.cuda.current_device())
    )
    plan = QSASharedScratchPlan()
    plan.reserve_rows(576)
    k1, _ = plan.packed_kv(576, KV_HEADS, HEAD_DIM, DTYPE, torch.device("cuda"))
    k2, _ = plan.packed_kv(
        576, KV_HEADS, HEAD_DIM, DTYPE, torch.device("cuda", torch.cuda.current_device())
    )
    assert k1.data_ptr() == k2.data_ptr()
    assert plan.packed_allocations == 1


def test_block_tables_are_pure_functions_of_their_key():
    plan = QSASharedScratchPlan()
    pages = math.ceil(EXPANDED_TOPK / TRTLLM_SPARSE_PAGE_SIZE)
    cu1, bt1 = plan.sparse_tables(MAX_BS, pages, TRTLLM_SPARSE_PAGE_SIZE, _cpu())
    cu2, bt2 = plan.sparse_tables(MAX_BS, pages, TRTLLM_SPARSE_PAGE_SIZE, _cpu())
    assert cu1.data_ptr() == cu2.data_ptr() and bt1.data_ptr() == bt2.data_ptr()
    assert cu1.tolist() == [i * pages * TRTLLM_SPARSE_PAGE_SIZE for i in range(MAX_BS + 1)]
    assert bt1.shape == (MAX_BS, pages)
    assert bt1[0].tolist() == list(range(pages))
    assert bt1[1].tolist() == list(range(pages, 2 * pages))
    assert plan.describe()["sparse_table_keys"] == 1


# -- the byte budget -------------------------------------------------------


def test_shipped_tier_set_byte_budget():
    """What the shipped {1,3,5,7} tier set costs, shared vs unshared, at B3c-4.

    Unshared is the code before this change: one 128 MiB trtllm workspace and one
    packed-KV pair PER QwenSparseAttnBackend instance, each sized from its own
    tier's window.
    """
    stride = _stride()
    instances = 0
    unshared_packed = 0
    for steps in CANDIDATE_STEPS:
        w = _w(steps)
        # target verify + draft-extend: bs x W query rows each
        unshared_packed += 2 * _pair_bytes(MAX_BS * w * stride)
        # draft decode chain: steps-1 backends at topk=1 -> bs query rows
        unshared_packed += max(0, steps - 1) * _pair_bytes(MAX_BS * 1 * stride)
        instances += 2 + max(0, steps - 1)

    assert instances == 20
    unshared_ws = instances * TRTLLM_SPARSE_WORKSPACE_BYTES
    shared_ws = TRTLLM_SPARSE_WORKSPACE_BYTES
    shared_packed = _pair_bytes(MAX_BS * WIDEST_W * stride)

    # Pin the numbers the run spec prices, in MiB.
    mib = 1024 * 1024
    assert round(unshared_ws / mib) == 2560
    assert round(shared_ws / mib) == 128
    assert round(unshared_packed / mib) == 858
    assert round(shared_packed / mib) == 132
    # Per-tier unshared cost, against the executor's measured per-state mem= line:
    #   steps=1 -> 2 ws + 66 MiB = 0.31 GiB   [measured 0.35]
    #   steps=5 -> 6 ws + 264 MiB = 1.01 GiB  [measured 1.10]
    per_tier = {}
    for steps in CANDIDATE_STEPS:
        w = _w(steps)
        per_tier[steps] = round(
            (
                2 * _pair_bytes(MAX_BS * w * stride)
                + max(0, steps - 1) * _pair_bytes(MAX_BS * stride)
            )
            / mib
        )
    assert per_tier == {1: 66, 3: 165, 5: 264, 7: 363}

    saved = (unshared_ws + unshared_packed) - (shared_ws + shared_packed)
    # >= 2.5 GiB recovered across the four tiers.
    assert round(saved / mib) == 3158
    assert round(saved / (1024**3), 2) == 3.08

    # The three EXTRA tiers are what the OOM was paying for; the launch tier's
    # cost is the pre-adaptive baseline.
    extra_instances = sum(2 + max(0, s - 1) for s in CANDIDATE_STEPS if s != LAUNCH_STEPS)
    assert extra_instances == 16
    extra_packed = 0
    for steps in CANDIDATE_STEPS:
        if steps == LAUNCH_STEPS:
            continue
        w = _w(steps)
        extra_packed += 2 * _pair_bytes(MAX_BS * w * stride)
        extra_packed += max(0, steps - 1) * _pair_bytes(MAX_BS * 1 * stride)
    launch_packed = _pair_bytes(MAX_BS * _w(LAUNCH_STEPS) * stride)
    extra_saved = (
        extra_instances * TRTLLM_SPARSE_WORKSPACE_BYTES
        + extra_packed
        - (shared_packed - launch_packed)
    )
    assert round(extra_saved / mib) == 2675
    assert round(extra_saved / (1024**3), 2) == 2.61
    # Priced as bf16 KV pool tokens at 26 KB/token.
    assert round(extra_saved / (26 * 1024)) == 105354


def test_describe_reports_what_the_boot_log_prints():
    plan = QSASharedScratchPlan()
    _reserve_every_tier(plan)
    plan.packed_kv(576, KV_HEADS, HEAD_DIM, DTYPE, _cpu())
    d = plan.describe()
    for key in (
        "reserved_rows",
        "reserve_calls",
        "packed_allocations",
        "handout_calls",
        "overflow_calls",
        "packed_bytes",
    ):
        assert key in d
    assert d["packed_allocations"] == 1
    assert d["overflow_calls"] == 0
    assert d["packed_bytes"] == _pair_bytes(MAX_BS * WIDEST_W * _stride())


# -- the graph memory pool is already shared -------------------------------


def test_graph_memory_pool_is_process_global():
    """The prompt's 'single graph mempool' item: the engine already does this.

    Every decode/draft/draft-extend runner resolves its capture backend through
    ``resolve_decode_backend``, and each of those backends takes the pool from
    ``get_or_create_global_graph_memory_pool``, which memoizes one handle on the
    process resources bag. So graph intermediates are already max-over-graphs,
    not sum-over-graphs, and no per-tier pool exists to merge.
    """
    import inspect

    pool_mod = _import_or_skip("sglang.srt.model_executor.runner_utils.pool")

    src = inspect.getsource(pool_mod.get_or_create_global_graph_memory_pool)
    assert "resources.graph_memory_pool" in src
    assert "graph_pool_handle()" in src

    for mod_name in (
        "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend",
        "sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend",
    ):
        mod = _import_or_skip(mod_name)
        assert "get_or_create_global_graph_memory_pool" in inspect.getsource(mod)


def test_logits_output_buffer_is_already_max_sized_and_shared():
    """The 'graph static output buffer' item: also already shared.

    ``GraphSharedOutput.get_logits_buffer`` allocates once at ``max_rows`` per
    vocab and hands out row views, and it asserts rather than growing -- so a
    deeper tier cannot strand a shallower tier's captured pointer.
    """
    import inspect

    gso = _import_or_skip("sglang.srt.model_executor.graph_shared_output")

    src = inspect.getsource(gso.GraphSharedOutput.get_logits_buffer)
    assert "assert rows <= self.max_rows" in src
    assert "(self.max_rows, vocab_size)" in src
    assert "return buffer[:rows]" in src


# --------------------------------------------------------------------------
# Plain-script runner (the serving venv has no pytest):
#   PYTHONPATH=<worktree>/python CUDA_VISIBLE_DEVICES=9 \
#       python test_adaptive_shared_ws.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import traceback

    CASES = []
    for _name, _fn in sorted(list(globals().items())):
        if not _name.startswith("test_") or not callable(_fn):
            continue
        marks = getattr(_fn, "pytestmark", [])
        argsets = None
        for m in marks:
            if m.name == "parametrize":
                names = [n.strip() for n in m.args[0].split(",")]
                argsets = [
                    dict(zip(names, v if isinstance(v, tuple) else (v,)))
                    for v in m.args[1]
                ]
        CASES.extend(
            [(_name, _fn, a) for a in argsets] if argsets else [(_name, _fn, {})]
        )

    failures = 0
    skipped = 0
    for name, fn, kwargs in CASES:
        label = name + (f"[{list(kwargs.values())}]" if kwargs else "")
        try:
            fn(**kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001
            # pytest's own Skipped derives from BaseException, so a bare
            # ``except Exception`` silently aborts the whole script run.
            if exc.__class__.__name__ in ("Skipped", "_Skipped"):
                skipped += 1
                print(f"SKIP {label}: {exc}")
                continue
            failures += 1
            print(f"FAIL {label}")
            traceback.print_exc()
        else:
            print(f"ok   {label}")
    print(
        f"\n{len(CASES) - failures - skipped} passed, "
        f"{failures} failed, {skipped} skipped"
    )
    sys.exit(1 if failures else 0)

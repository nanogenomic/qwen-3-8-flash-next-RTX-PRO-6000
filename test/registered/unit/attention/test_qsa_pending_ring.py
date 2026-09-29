"""Pure-index tests for the QSA pending index-key ring.

The ring stores one raw index key per *live* position and is addressed by
``req_pool_idx * ring_size + position % ring_size``.  Everything below drives
the production builders (``build_pending_ring_slots`` /
``build_group_ring_slots``) on CPU tensors and checks the one property the
compression step depends on: when a forward compresses the group ending at
position ``p``, the ring rows named for positions ``p - ratio + 1 .. p`` still
hold *those* positions' keys and no others.

The simulation writes a position tag instead of a key, so a collision is
observable exactly where the real pool would silently compress the wrong keys.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
)
from sglang.srt.layers.attention.qsa.config import (
    qsa_max_draft_tokens,
    qsa_pending_ring_size,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

RATIO = 4
# Request pool slot 0 is never allocated; it is the inert dump.
REQ = 3
EMPTY = -1


class RingSim:
    """One request's pending ring, driven by the production slot builders."""

    def __init__(self, ring_size: int, ratio: int = RATIO, num_requests: int = 8):
        self.ring_size = ring_size
        self.ratio = ratio
        self.slots = torch.full((num_requests * ring_size,), EMPTY, dtype=torch.long)
        self.compressions: list[tuple[int, list[int]]] = []

    def _pending_slots(self, positions, lengths, is_extend):
        rows = torch.arange(positions.numel(), dtype=torch.long)
        return build_pending_ring_slots(
            token_to_batch_idx=rows,
            req_pool_indices=torch.full((positions.numel(),), REQ, dtype=torch.long),
            sequence_lengths=lengths,
            logical_positions=positions,
            compress_ratio=self.ratio,
            is_extend=is_extend,
            ring_size=self.ring_size,
        )

    def _group_slots(self, group_end_positions):
        n = group_end_positions.numel()
        return build_group_ring_slots(
            req_pool_indices=torch.full((n,), REQ, dtype=torch.long),
            group_end_positions=group_end_positions,
            sequence_ids=torch.arange(n, dtype=torch.long),
            compress_ratio=self.ratio,
            ring_size=self.ring_size,
        )

    def forward(self, first_position: int, width: int, is_extend: bool = False):
        """One forward over ``width`` consecutive positions.

        Mirrors ``QSAIndexer.update_key_state_and_compress``: every row's state
        is stored first, then every completed group is compressed out of the
        ring.  Returns the list of (group_end_position, member_tags) it read.
        """
        positions = torch.arange(
            first_position, first_position + width, dtype=torch.long
        )
        lengths = positions + 1
        if is_extend:
            # Extend rows carry the whole request length, not a per-row length.
            lengths = torch.full_like(positions, first_position + width)

        slots = self._pending_slots(positions, lengths, is_extend)
        self.slots[slots] = positions

        # A paged row completes the group its length ends; extend reads its
        # members from the packed chunk instead of the ring, so only the paged
        # (decode / speculative) path is a ring consumer.
        read = []
        if not is_extend:
            boundary = positions[(positions + 1) % self.ratio == 0]
            if boundary.numel():
                group_slots = self._group_slots(boundary)
                for row, end in enumerate(boundary.tolist()):
                    members = self.slots[group_slots[row]].tolist()
                    read.append((end, members))
                    self.compressions.append((end, members))
        return read

    @staticmethod
    def expected_members(group_end: int, ratio: int = RATIO) -> list[int]:
        return list(range(group_end - ratio + 1, group_end + 1))


def run_stream(ring_size, prefill_len, window, accepts, ratio=RATIO):
    """Prefill, then verify windows committing ``accepts[i]`` tokens each."""
    sim = RingSim(ring_size, ratio=ratio)
    sim.forward(0, prefill_len, is_extend=True)
    committed = prefill_len
    for accept in accepts:
        sim.forward(committed, window)
        committed += accept
    return sim


def assert_all_groups_correct(sim, ratio=RATIO):
    for end, members in sim.compressions:
        assert members == RingSim.expected_members(end, ratio), (
            f"group ending at {end} compressed positions {members}, "
            f"expected {RingSim.expected_members(end, ratio)}"
        )


# --------------------------------------------------------------------------
# ring size formula
# --------------------------------------------------------------------------


@pytest.mark.parametrize("window", list(range(1, 17)))
def test_ring_size_covers_window_plus_group(window):
    ring = qsa_pending_ring_size(RATIO, window)
    # The window and the group it completes span at most W + ratio - 1
    # consecutive positions, every one of which needs its own row.
    assert ring >= window + RATIO - 1
    assert ring % RATIO == 0
    # And it is the tightest such multiple of the ratio.
    assert ring - RATIO < window + RATIO - 1


def test_ring_size_is_unchanged_without_speculation():
    assert qsa_pending_ring_size(RATIO, 0) == RATIO
    assert qsa_pending_ring_size(RATIO, 1) == RATIO


def test_declared_window_matches_ring():
    for window in range(1, 17):
        ring = qsa_pending_ring_size(RATIO, window)
        assert qsa_max_draft_tokens(RATIO, ring) >= window


# --------------------------------------------------------------------------
# collision freedom of the addressing itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("window", list(range(1, 9)))
def test_live_positions_have_distinct_rows(window):
    """Every position a forward may still need maps to its own ring row."""
    ring = qsa_pending_ring_size(RATIO, window)
    for first in range(0, 4 * RATIO + window + 1):
        # Window positions, plus the earlier members of a group the window
        # completes (at most ratio - 1 of them).
        live = list(range(first - RATIO + 1, first + window))
        live = [position for position in live if position >= 0]
        rows = {position % ring for position in live}
        assert len(rows) == len(live), (
            f"window={window} ring={ring} first={first}: "
            f"{len(live)} live positions share {len(rows)} rows"
        )


# --------------------------------------------------------------------------
# end-to-end streams
# --------------------------------------------------------------------------


@pytest.mark.parametrize("window", list(range(1, 9)))
@pytest.mark.parametrize("prefill_len", [1, 2, 3, 4, 5, 7, 8, 9, 16, 17])
def test_every_group_compresses_its_own_members(window, prefill_len):
    ring = qsa_pending_ring_size(RATIO, window)
    # Walk every accept length, so the committed length visits every residue.
    accepts = [1 + (step % window) for step in range(12)]
    sim = run_stream(ring, prefill_len, window, accepts)
    assert sim.compressions, "the stream compressed no group at all"
    assert_all_groups_correct(sim)


@pytest.mark.parametrize("window", [4, 5, 6, 8])
def test_exhaustive_accept_patterns(window):
    """Four cycles of every accept pattern, for the widths we intend to run."""
    import itertools

    ring = qsa_pending_ring_size(RATIO, window)
    for prefill_len in (1, 4, 6):
        for accepts in itertools.product(range(1, window + 1), repeat=3):
            sim = run_stream(ring, prefill_len, window, accepts)
            assert_all_groups_correct(sim)


@pytest.mark.parametrize("window", [4, 5, 6, 8])
def test_rejected_tokens_never_corrupt_the_committed_prefix(window):
    """A rejected draft must not damage the group its committed neighbours are in.

    After a verify that accepts ``k < W``, positions ``L+k .. L+W-1`` were
    written to the ring and then abandoned.  The next verify re-writes them with
    the newly drafted tokens, and any group that completes afterwards must still
    read the committed positions -- never a leftover from the rejected chain.
    """
    ring = qsa_pending_ring_size(RATIO, window)
    for prefill_len in range(1, 2 * RATIO + 1):
        for accept in range(1, window + 1):
            sim = RingSim(ring)
            sim.forward(0, prefill_len, is_extend=True)
            committed = prefill_len
            # Cycle 1: write the full window, commit only `accept` of it.
            sim.forward(committed, window)
            committed += accept
            # Cycles 2-4: the abandoned rows are still in the ring.
            for _ in range(3):
                sim.forward(committed, window)
                committed += accept
            assert_all_groups_correct(sim)


def test_draft_extend_window_shares_the_ring_bound():
    """DRAFT_EXTEND_V2 writes a full ``num_draft_tokens`` window per request,
    so the MTP layer's ring needs the same size as the target layers'."""
    for window in range(1, 9):
        ring = qsa_pending_ring_size(RATIO, window)
        sim = RingSim(ring)
        sim.forward(0, 6, is_extend=True)
        committed = 6
        for accept in (1, window, 2, 3):
            accept = min(accept, window)
            # Draft-extend covers the same W positions the verify did.
            sim.forward(committed, window)
            committed += accept
        assert_all_groups_correct(sim)


# --------------------------------------------------------------------------
# the regression this sizing exists to prevent
# --------------------------------------------------------------------------


def test_ratio_sized_ring_aliases_once_the_window_is_wider_than_one():
    """A ring of exactly ``ratio`` rows cannot serve a multi-token window.

    This pins the reason the default ring grew.  With ``ring == ratio`` a verify
    window of W == ratio covers every residue class, so the members of the group
    that completes mid-window are overwritten -- by positions from the *next*
    group -- before the compression reads them.  It is silent: the compressed
    key is well-formed, just built from the wrong tokens.
    """
    window = RATIO
    corrupted = []
    for first in range(1, RATIO):  # committed length not on a group boundary
        sim = RingSim(RATIO)
        sim.forward(0, first, is_extend=True)
        sim.forward(first, window)
        for end, members in sim.compressions:
            if members != RingSim.expected_members(end):
                corrupted.append((first, end, members))
    assert corrupted, (
        "expected the ratio-sized ring to alias for every unaligned committed "
        "length; if this no longer reproduces, the ring sizing can be reverted"
    )
    assert len(corrupted) == RATIO - 1

    # The same streams are clean once the ring is sized for the window.
    ring = qsa_pending_ring_size(RATIO, window)
    for first in range(1, RATIO):
        sim = RingSim(ring)
        sim.forward(0, first, is_extend=True)
        sim.forward(first, window)
        assert_all_groups_correct(sim)


def test_extend_dump_rows_belong_to_no_request():
    """Non-pending extend tokens land in rows [0, ring), which request slot 0
    owns and no allocated request ever addresses."""
    for window in (1, 4, 8):
        ring = qsa_pending_ring_size(RATIO, window)
        sim = RingSim(ring)
        positions = torch.arange(11, dtype=torch.long)
        lengths = torch.full_like(positions, 11)
        slots = sim._pending_slots(positions, lengths, is_extend=True)
        pending = positions >= (11 // RATIO) * RATIO
        assert (slots[~pending] < ring).all()
        assert (slots[pending] >= REQ * ring).all()
        assert (slots[pending] < (REQ + 1) * ring).all()

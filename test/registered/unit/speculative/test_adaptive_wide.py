# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive speculation x wide QSA verify: the couplings that must hold.

CPU only. Everything here drives PRODUCTION functions -- the adaptive policy, the
config loader and the QSA ring sizing -- so a regression in them fails here rather
than in a serving boot. qwen-opt.
"""

from __future__ import annotations

import json
import os

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

    pytest = _PytestShim()

from sglang.srt.layers.attention.qsa.config import (
    qsa_max_draft_tokens,
    qsa_pending_ring_size,
)
import sglang.srt.speculative.adaptive_spec_params
from sglang.srt.speculative.adaptive_spec_params import (
    DEFAULT_ADAPTIVE_CONFIG,
    AdaptiveSpeculativeParams,
    resolve_candidate_steps_from_config,
)

# The two tier configs ship in-tree next to adaptive_spec_params.py. Override the
# directory with QWENOPT_ADAPTIVE_CONFIG_DIR to test a hand-tuned pair.
_IN_TREE_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(sglang.srt.speculative.adaptive_spec_params.__file__)),
    "configs",
)
CONFIG_DIR = os.environ.get("QWENOPT_ADAPTIVE_CONFIG_DIR", _IN_TREE_CONFIG_DIR)
SHIPPED = os.path.join(CONFIG_DIR, "adaptive_flashnext.json")
LEAN = os.path.join(CONFIG_DIR, "adaptive_flashnext_lean.json")
RATIO = 4  # qwen38-flash-next indexer_compress_ratio [source-verified, config.json]
CUDA_GRAPH_BS = [1, 2, 3, 4, 5, 6, 7, 8]


def _shipped_configs():
    paths = [p for p in (SHIPPED, LEAN) if os.path.exists(p)]
    if not paths:
        pytest.skip("shipped adaptive configs not present on this host")
    return paths


# --------------------------------------------------------------------------
# 1. The ring must serve the DEEPEST tier, not the launch width.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cfg_path", [None] + _shipped_configs())
def test_ring_sized_from_max_candidate_serves_every_tier(cfg_path):
    """qsa_pending_ring_size(ratio, max(candidates)+1) is collision-free for the
    verify window of EVERY candidate tier -- this is what lets the pool allocate
    once at startup while the active width moves."""
    candidates = resolve_candidate_steps_from_config(cfg_path)
    max_draft = max(candidates) + 1
    ring = qsa_pending_ring_size(RATIO, max_draft)
    assert ring % RATIO == 0
    for steps in candidates:
        width = max(steps + 1, 1)
        # window U group spans width + ratio - 1 consecutive positions
        assert ring >= width + RATIO - 1, (steps, width, ring)
        assert width <= qsa_max_draft_tokens(RATIO, ring)


def test_launch_width_ring_would_alias_the_deep_tiers():
    """The point of sizing from the max: a ring sized for the LAUNCH width 4
    cannot serve a 6- or 8-token window. If this ever passes, sizing from the
    launch value stopped being a bug and something else changed."""
    launch_ring = qsa_pending_ring_size(RATIO, 4)
    assert launch_ring == 8
    assert qsa_max_draft_tokens(RATIO, launch_ring) == 5
    for deep_steps in (5, 7):
        assert deep_steps + 1 > qsa_max_draft_tokens(RATIO, launch_ring)


def test_shipped_config_ring_is_12_at_ratio_4():
    """Concrete: the shipped tiers top out at 7 steps -> W=8 -> ring 12."""
    candidates = resolve_candidate_steps_from_config(SHIPPED)
    assert max(candidates) == 7
    assert qsa_pending_ring_size(RATIO, max(candidates) + 1) == 12


# --------------------------------------------------------------------------
# 2. Per-state CUDA-graph pruning has to agree with runtime routing.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cfg_path", [None] + _shipped_configs())
def test_every_graph_bs_has_its_active_step_captured(cfg_path):
    """The invariant that makes cuda_graph_bs_for_step() safe: for every graph
    batch size, the tier the router would activate at that batch size is one the
    pruned capture list includes. A violation is a runtime graph miss (silent
    fall back to eager, or a missing-state ValueError)."""
    params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=cfg_path)
    params.set_cuda_graph_bs(CUDA_GRAPH_BS)
    for bs in CUDA_GRAPH_BS:
        step = params.get_steps_for_batch(bs)
        captured = params.cuda_graph_bs_for_step(step)
        assert captured is not None and bs in captured, (bs, step, captured)


@pytest.mark.parametrize("cfg_path", [None] + _shipped_configs())
def test_capture_budget_is_bounded(cfg_path):
    """Graph VRAM scales with the number of (state, bs) captures, so count them.
    The static config captures len(CUDA_GRAPH_BS); this asserts the shipped
    configs stay within 2x that, which is the measured VRAM headroom
    (0.65 GB for 4 bs at W=4 against 2.82 GB spare)."""
    params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=cfg_path)
    params.set_cuda_graph_bs(CUDA_GRAPH_BS)
    total = sum(
        len(params.cuda_graph_bs_for_step(step) or []) for step in params.candidate_steps
    )
    if cfg_path is None:
        # The DEFAULT config is the one that does not fit: recorded, not asserted.
        assert total > len(CUDA_GRAPH_BS)
        return
    assert total <= 2 * len(CUDA_GRAPH_BS), (cfg_path, total)


def test_shipped_capture_counts_are_the_documented_ones():
    p = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=SHIPPED)
    p.set_cuda_graph_bs(CUDA_GRAPH_BS)
    per_step = {s: len(p.cuda_graph_bs_for_step(s) or []) for s in p.candidate_steps}
    assert per_step == {1: 1, 3: 8, 5: 3, 7: 1}, per_step
    assert sum(per_step.values()) == 13
    # steps=1 must be reachable ONLY from the bs8 slot: bs 4..7 route to slot 4,
    # so adding 1 there would put the steps=1 state on 5 batch sizes.
    assert p.cuda_graph_bs_for_step(1) == [8]

    lean = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=LEAN)
    lean.set_cuda_graph_bs(CUDA_GRAPH_BS)
    lean_per_step = {s: len(lean.cuda_graph_bs_for_step(s) or []) for s in lean.candidate_steps}
    assert lean_per_step == {3: 8, 7: 1}, lean_per_step


# --------------------------------------------------------------------------
# 3. Hysteresis must place the switch at the derived break-even and not chatter.
# --------------------------------------------------------------------------


def _drive(params, ema_target: float, batch_size: int, batches: int = 400):
    """Feed a constant accepted-draft count and return the step trajectory."""
    trajectory = []
    for _ in range(batches):
        steps = params.get_steps_for_batch(batch_size)
        # num_correct_drafts is clamped by the active width, exactly as the
        # verify kernel would clamp it.
        per_req = min(ema_target, float(max(steps, 0)))
        params.on_verify_complete([per_req], batch_size=batch_size)
        trajectory.append(params.get_steps_for_batch(batch_size))
    return trajectory


@pytest.mark.parametrize(
    "ema,expect_top",
    [
        (1.0, False),   # below the bs1 n3->n5 break-even (1.29): stay at 3
        (3.6, True),    # well above every break-even: climb
    ],
)
def test_bs1_switches_on_the_right_side_of_the_breakeven(ema, expect_top):
    params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=SHIPPED)
    params.set_cuda_graph_bs(CUDA_GRAPH_BS)
    traj = _drive(params, ema, batch_size=1)
    final = traj[-1]
    if expect_top:
        assert final > 3, traj[-20:]
    else:
        assert final == 3, traj[-20:]


def test_no_oscillation_at_a_steady_accept_rate():
    """A steady acceptance must settle: at most a handful of switches over 400
    batches. Chatter costs an apply_runtime_state (backend + graph-runner swap)
    every update_interval and would confound any A/B."""
    for ema in (0.5, 1.0, 1.4, 2.0, 3.0, 4.0, 6.0):
        params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=SHIPPED)
        params.set_cuda_graph_bs(CUDA_GRAPH_BS)
        traj = _drive(params, ema, batch_size=1)
        switches = sum(1 for a, b in zip(traj, traj[1:]) if a != b)
        assert switches <= 4, (ema, switches, traj[:40])
        tail = traj[-50:]
        assert len(set(tail)) == 1, (ema, set(tail))


def test_aggregate_slots_never_widen_past_the_static_width():
    """The hard constraint is aggregate >= config B at conc 4/8, so the bs>=4
    slots must never select a width deeper than the static 3 no matter how high
    acceptance goes."""
    params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=SHIPPED)
    params.set_cuda_graph_bs(CUDA_GRAPH_BS)
    for bs in (4, 8):
        traj = _drive(params, 7.0, batch_size=bs)
        assert max(traj) <= 3, (bs, set(traj))


def test_collapsed_acceptance_drops_the_aggregate_slots_to_one_step():
    params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=SHIPPED)
    params.set_cuda_graph_bs(CUDA_GRAPH_BS)
    traj = _drive(params, 0.0, batch_size=8)
    assert traj[-1] == 1, set(traj)


# --------------------------------------------------------------------------
# 4. Per-tier acceptance accounting (the run spec reads these).
# --------------------------------------------------------------------------


def test_tier_stats_attribute_to_the_width_that_produced_the_drafts():
    params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=SHIPPED)
    params.set_cuda_graph_bs(CUDA_GRAPH_BS)
    _drive(params, 3.6, batch_size=1, batches=300)
    stats = params.tier_stats()
    assert "bs1/steps3" in stats
    assert any(k.startswith("bs1/steps5") or k.startswith("bs1/steps7") for k in stats)
    for name, v in stats.items():
        steps = int(name.split("steps")[1])
        # accept_len = accepted drafts + the always-committed bonus token
        assert 1.0 <= v["accept_len"] <= steps + 1 + 1e-9, (name, v)
        assert 0.0 <= v["accept_rate"] <= 1.0 + 1e-9, (name, v)
    assert params.tier_stats_line()


def test_tier_stats_separate_batch_sizes():
    params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=SHIPPED)
    params.set_cuda_graph_bs(CUDA_GRAPH_BS)
    _drive(params, 2.0, batch_size=1, batches=60)
    _drive(params, 2.0, batch_size=8, batches=60)
    keys = set(params.tier_stats())
    assert any(k.startswith("bs1/") for k in keys)
    assert any(k.startswith("bs8/") for k in keys)


# --------------------------------------------------------------------------
# 5. The shipped configs must load under the production validator.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cfg_path", _shipped_configs())
def test_shipped_configs_load_and_are_monotone(cfg_path):
    raw = json.load(open(cfg_path))
    for key, entry in raw.items():
        if not key.isdigit():
            continue
        steps = entry["candidate_steps"]
        assert steps == sorted(set(steps)), (cfg_path, key, steps)
        assert all(s >= 0 for s in steps)
    params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=cfg_path)
    # initial_steps must be a candidate in every slot, or a slot silently starts
    # at its middle candidate and the first A/B point is not the launch config.
    for bs in (1, 2, 4, 8):
        assert params.get_steps_for_batch(bs) == 3, bs


def test_default_config_is_unchanged():
    """The shipped files are additions, not edits to the engine default."""
    assert set(DEFAULT_ADAPTIVE_CONFIG) == {"1", "8", "32", "64"}
    assert DEFAULT_ADAPTIVE_CONFIG["1"]["candidate_steps"] == [1, 3, 5, 7]


# --------------------------------------------------------------------------
# No-pytest driver: the serving venv has no pytest and must not be
# modified, so this file also runs as a plain script there.
#   PYTHONPATH=<worktree>/python CUDA_VISIBLE_DEVICES=9 python test_adaptive_wide.py
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
        except Exception as exc:  # noqa: BLE001
            if exc.__class__.__name__ in ("Skipped", "_Skipped"):
                skipped += 1
                print(f"SKIP {label}: {exc}")
                continue
            failures += 1
            print(f"FAIL {label}")
            traceback.print_exc()
        else:
            print(f"ok   {label}")
    print(f"\n{len(CASES) - failures - skipped} passed, {failures} failed, {skipped} skipped")
    sys.exit(1 if failures else 0)

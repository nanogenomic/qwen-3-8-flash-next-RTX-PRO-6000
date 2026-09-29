# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic expert count (dynamic-k) for softmax top-k MoE routing.

Off by default. Enabled only by ``SGLANG_QWENOPT_DYNK`` (inline JSON) or
``SGLANG_QWENOPT_DYNK_FILE`` (path to JSON). Schema::

    {"target": ARM, "draft": ARM}          # either key may be omitted (= off)
    ARM = {"mode": "off"}
        | {"mode": "fixed",   "k": 8}                                   # top-k truncation
        | {"mode": "cumprob", "tau": 0.9 | [tau_0 .. tau_47], "k_min": 4, "k_max": 10}
        | {"mode": "learned", "weights": "/path/gate.pt", "budget": -2.0 | [..],
           "k_min": 4, "k_max": 10}

Semantics (all modes): the router still selects the model's full top-K (K=10) and
normalises weights over that top-K exactly as today. Dynamic-k then DROPS tail slots:
dropped slots get id -1 and weight 0, kept slots keep their ORIGINAL top-K weight (no
renormalisation, the "k2 = K" rule of arXiv 2609.04575). The output shape stays [M, K],
so CUDA graphs are unchanged. flashinfer cutlass_fused_moe treats id -1 as "expert not
on this rank": the slot is excluded from the permutation, grouped GEMM and finalize,
so dropped experts cost zero bytes (GATE 1: bitwise equal to the narrow reference).

fixed:   keep slots 0..k-1 (k is a launch constant).
cumprob: keep the shortest prefix of the descending top-K weights whose mass is
         >= tau * (top-K mass), clamped to [k_min, k_max].
learned: per-slot MLP on router statistics predicts log10 relative output error for
         each k in 3..10; keep the smallest k in [k_min, k_max] with prediction <= budget.
For cumprob/learned, tau/budget AND k_min/k_max are per-slot DEVICE tensors, so they
can be changed while serving (no graph recapture) via ``SGLANG_QWENOPT_DYNK_CONTROL``:
a JSON file {"target": {...}, "draft": {...}} with any of tau/budget, k_min, k_max,
polled every 2 s. cumprob with tau=1, k_min=k_max=k is exactly fixed-k; with k=10 it is
exactly off (bitwise), so one boot can sweep every arm.

Slots: target MoE layers use their layer id (0..62); the MTP/draft MoE uses slot 63.
Duplicate expert ids within a token are a correctness hazard for cutlass_fused_moe
(the duplicate's unpermuted->permuted row is never written; measured wrong by up to
15.5 abs). Dynamic-k only ever writes -1, never a duplicate; ``check_no_duplicates``
asserts it outside graph capture when ``SGLANG_QWENOPT_DYNK_CHECK=1``.

``SGLANG_QWENOPT_DYNK_STATS_FILE=/path.json`` turns on in-kernel per-slot counters
(kept slots, live rows; padded graph rows excluded), written every 5 s, so mean k per
layer is measured on live traffic rather than assumed.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

MODE_OFF, MODE_FIXED, MODE_CUMPROB, MODE_LEARNED = 0, 1, 2, 3
_MODE_NAMES = {"off": MODE_OFF, "fixed": MODE_FIXED, "cumprob": MODE_CUMPROB, "learned": MODE_LEARNED}
NUM_SLOTS = 64
DRAFT_SLOT = NUM_SLOTS - 1
TARGET_SLOTS = range(0, DRAFT_SLOT)
# Learned gate: per-slot MLP  F_IN -> H -> OUT predicted log10 rel. error for k = 3..10.
# Features (all from the router's own top-10 rounds, no extra ranking):
#   p1..p10 (softmax over all experts, sorted), tail = 1 - sum(p1..p10), entropy of the
#   renormalised top-10, log p1, log(p1/p2), log(tail + 1e-6), top-5 share of top-10.
LEARNED_TOPN = 10
LEARNED_F_IN = 16
LEARNED_H = 32
LEARNED_OUT = 8


@dataclass
class DynKArm:
    mode: int
    k: int = 10
    k_min: int = 1
    k_max: int = 10
    tau: Optional[list] = None  # cumprob tau / learned budget, per target layer or scalar
    weights_path: Optional[str] = None


@dataclass
class DynKCallParams:
    """What one router call needs. Tensors are device-resident and allocated once."""

    mode: int
    slot: int
    k: int
    tau: torch.Tensor  # [NUM_SLOTS] fp32 (cumprob tau / learned log10 budget)
    kk: torch.Tensor  # [NUM_SLOTS, 2] int32 (k_min, k_max)
    mlp: Optional[torch.Tensor] = None  # [NUM_SLOTS, P] fp32 packed learned-gate params
    cnt: Optional[torch.Tensor] = None  # [NUM_SLOTS, 2] int32 (kept slots, live rows)


_cfg: Optional[Dict[str, DynKArm]] = None
_dev: Dict[str, dict] = {}
_thread = None


def _as_list(v, n):
    if isinstance(v, list):
        v = [float(x) for x in v]
        return v + [v[-1]] * (n - len(v)) if len(v) < n else v[:n]
    return [float(v)] * n


def _parse_arm(d: Optional[dict], name: str) -> DynKArm:
    if not d:
        return DynKArm(MODE_OFF)
    mode = _MODE_NAMES[d.get("mode", "off")]
    arm = DynKArm(mode)
    if mode == MODE_FIXED:
        arm.k = int(d["k"])
        assert 1 <= arm.k <= 16, f"dynk {name}: k={arm.k}"
    elif mode in (MODE_CUMPROB, MODE_LEARNED):
        arm.k_min = int(d.get("k_min", 4))
        arm.k_max = int(d.get("k_max", 10))
        assert 1 <= arm.k_min <= arm.k_max, f"dynk {name}: k_min/k_max"
        if mode == MODE_CUMPROB:
            arm.tau = _as_list(d["tau"], len(TARGET_SLOTS))
            assert all(0.0 < t <= 1.0 for t in arm.tau), f"dynk {name}: tau in (0,1]"
        else:
            arm.weights_path = d["weights"]
            arm.tau = _as_list(d.get("budget", -2.0), len(TARGET_SLOTS))
    return arm


def get_config() -> Dict[str, DynKArm]:
    global _cfg
    if _cfg is None:
        raw = os.environ.get("SGLANG_QWENOPT_DYNK", "").strip()
        path = os.environ.get("SGLANG_QWENOPT_DYNK_FILE", "").strip()
        if not raw and path:
            with open(path) as f:
                raw = f.read()
        d = json.loads(raw) if raw else {}
        _cfg = {"target": _parse_arm(d.get("target"), "target"), "draft": _parse_arm(d.get("draft"), "draft")}
        if any(a.mode != MODE_OFF for a in _cfg.values()):
            logger.warning("qwenopt dynamic-k ENABLED (lossy): %s", d)
    return _cfg


def is_enabled() -> bool:
    return any(a.mode != MODE_OFF for a in get_config().values())


def _arm_for_slot(slot: int) -> DynKArm:
    return get_config()["draft" if slot == DRAFT_SLOT else "target"]


def slot_for_layer(layer_id: Optional[int], is_draft: bool) -> Optional[int]:
    if is_draft:
        return DRAFT_SLOT
    if layer_id is None:
        return None
    assert 0 <= layer_id < DRAFT_SLOT, f"dynk: layer_id {layer_id} out of slot range"
    return layer_id


def learned_param_count() -> int:
    # W1 [H, F_IN] + b1 [H] + W2 [OUT, H] + b2 [OUT]
    return LEARNED_H * LEARNED_F_IN + LEARNED_H + LEARNED_OUT * LEARNED_H + LEARNED_OUT


def _host_tables():
    cfg = get_config()
    tau = [1.0] * NUM_SLOTS
    kk = [[1, 10] for _ in range(NUM_SLOTS)]
    t, dr = cfg["target"], cfg["draft"]
    for s in TARGET_SLOTS:
        if t.tau is not None:
            tau[s] = t.tau[s]
        kk[s] = [t.k_min, t.k_max]
    if dr.tau is not None:
        tau[DRAFT_SLOT] = dr.tau[0]
    kk[DRAFT_SLOT] = [dr.k_min, dr.k_max]
    return tau, kk


def prepare(slot: Optional[int], device: torch.device) -> None:
    """Allocate the device tensors eagerly (model init, before graph capture)."""
    if slot is None or not is_enabled():
        return
    key = str(device)
    if key not in _dev:
        tau, kk = _host_tables()
        st = {
            "tau": torch.tensor(tau, dtype=torch.float32, device=device),
            "kk": torch.tensor(kk, dtype=torch.int32, device=device),
            "cnt": None,
            "mlp": None,
            "tau_host": list(tau),
            "kk_host": [list(x) for x in kk],
        }
        if os.environ.get("SGLANG_QWENOPT_DYNK_STATS_FILE", "").strip():
            st["cnt"] = torch.zeros((NUM_SLOTS, 2), dtype=torch.int32, device=device)
        _dev[key] = st
        _start_thread(device)
    st = _dev[key]
    cfg = get_config()
    if st["mlp"] is None and any(a.mode == MODE_LEARNED for a in cfg.values()):
        packed = torch.zeros((NUM_SLOTS, learned_param_count()), dtype=torch.float32)
        for name, arm in cfg.items():
            if arm.mode != MODE_LEARNED:
                continue
            blob = torch.load(arm.weights_path, map_location="cpu", weights_only=True)
            p = blob["packed"].to(torch.float32)  # [L, P] (target) or [1, P] (draft)
            assert p.shape[1] == learned_param_count(), "learned gate: packed width mismatch"
            if name == "draft":
                packed[DRAFT_SLOT] = p[-1]
            else:
                for s in TARGET_SLOTS:
                    packed[s] = p[min(s, p.shape[0] - 1)]
        st["mlp"] = packed.to(device)


def call_params(slot: Optional[int], device: torch.device) -> Optional[DynKCallParams]:
    if slot is None or not is_enabled():
        return None
    arm = _arm_for_slot(slot)
    if arm.mode == MODE_OFF:
        return None
    key = str(device)
    if key not in _dev or (arm.mode == MODE_LEARNED and _dev[key]["mlp"] is None):
        # Not prepared at init: allocating now would be illegal inside graph capture.
        assert not torch.cuda.is_current_stream_capturing(), (
            "dynk: device params were not prepared before CUDA graph capture"
        )
        prepare(slot, device)
    st = _dev[key]
    return DynKCallParams(mode=arm.mode, slot=slot, k=arm.k, tau=st["tau"], kk=st["kk"],
                          mlp=st["mlp"], cnt=st["cnt"])


def _start_thread(device: torch.device) -> None:
    """Side thread: (a) every 2 s apply SGLANG_QWENOPT_DYNK_CONTROL if it changed;
    (b) every 5 s write the kept-slot counters to SGLANG_QWENOPT_DYNK_STATS_FILE.
    All device traffic is on a private side stream; the compute stream is never synced.
    A control update lands between (or, rarely, within) steps; each value is one aligned
    4-byte word, so a router reads either the old or the new value."""
    global _thread
    ctl = os.environ.get("SGLANG_QWENOPT_DYNK_CONTROL", "").strip()
    stats = os.environ.get("SGLANG_QWENOPT_DYNK_STATS_FILE", "").strip()
    if _thread is not None or not (ctl or stats):
        return
    import threading
    import time

    st = _dev[str(device)]

    def apply_control(d):
        tau = list(st["tau_host"])  # host mirrors: never read the device here
        kk = [list(x) for x in st["kk_host"]]
        for name, slots in (("target", list(TARGET_SLOTS)), ("draft", [DRAFT_SLOT])):
            c = d.get(name)
            if not c:
                continue
            if _arm_for_slot(slots[0]).mode not in (MODE_CUMPROB, MODE_LEARNED):
                logger.warning("dynk control: %s arm is not cumprob/learned; ignored", name)
                continue
            if "tau" in c or "budget" in c:
                v = _as_list(c.get("tau", c.get("budget")), len(slots))
                for i, s in enumerate(slots):
                    tau[s] = v[i]
            for s in slots:
                kmin = int(c.get("k_min", kk[s][0]))
                kmax = int(c.get("k_max", kk[s][1]))
                assert 1 <= kmin <= kmax <= 16
                kk[s] = [kmin, kmax]
        side = torch.cuda.Stream(device=device)
        with torch.cuda.stream(side):
            st["tau"].copy_(torch.tensor(tau, dtype=torch.float32).pin_memory(), non_blocking=True)
            st["kk"].copy_(torch.tensor(kk, dtype=torch.int32).pin_memory(), non_blocking=True)
        side.synchronize()
        st["tau_host"], st["kk_host"] = tau, kk
        logger.warning("dynk control applied: %s", d)

    def run():
        torch.cuda.set_device(device)
        side = torch.cuda.Stream(device=device)
        host = torch.empty((NUM_SLOTS, 2), dtype=torch.int32, pin_memory=True)
        last_ctl, last_stats = None, 0.0
        while True:
            time.sleep(2.0)
            try:
                if ctl and os.path.exists(ctl):
                    raw = open(ctl).read()
                    if raw != last_ctl:
                        apply_control(json.loads(raw) if raw.strip() else {})
                        last_ctl = raw
                        if stats:
                            with open(stats + ".control", "a") as f:
                                f.write(json.dumps({"t": time.time(), "control": raw}) + "\n")
                if stats and st["cnt"] is not None and time.time() - last_stats >= 5.0:
                    last_stats = time.time()
                    with torch.cuda.stream(side):
                        host.copy_(st["cnt"], non_blocking=True)
                    side.synchronize()
                    h = host.tolist()
                    out = {"t": last_stats, "slots": {str(i): h[i] for i in range(NUM_SLOTS) if h[i][1] > 0}}
                    with open(stats + ".tmp", "w") as f:
                        json.dump(out, f)
                    os.replace(stats + ".tmp", stats)
            except Exception as e:  # never take the server down
                logger.warning("dynk side thread: %s", e)

    _thread = threading.Thread(target=run, name="dynk-side", daemon=True)
    _thread.start()


def apply_torch(topk_weights: torch.Tensor, topk_ids: torch.Tensor, p: DynKCallParams,
                k_routed: int) -> None:
    """Reference / fallback (in place) for router paths other than the fused Triton gate.
    Matches the kernel for fixed and cumprob. Learned mode is kernel-only."""
    K = topk_ids.shape[1]
    ar = torch.arange(K, device=topk_ids.device)[None, :]
    routed = ar < k_routed
    if p.mode == MODE_FIXED:
        keep = ar < p.k
    elif p.mode == MODE_CUMPROB:
        kmin, kmax = p.kk[p.slot, 0], p.kk[p.slot, 1]
        # rank by weight descending (the fused kernel emits slots already sorted)
        w = torch.where(routed, topk_weights, torch.zeros_like(topk_weights))
        order = torch.argsort(w, dim=1, descending=True, stable=True)
        ws = torch.gather(w, 1, order)
        tot = ws.sum(1, keepdim=True)
        cum_excl = ws.cumsum(1) - ws
        keep_sorted = (cum_excl < p.tau[p.slot] * tot) | (ar < kmin)
        keep_sorted = keep_sorted & (ar < kmax)
        keep = torch.zeros_like(keep_sorted)
        keep.scatter_(1, order, keep_sorted)
    else:
        raise NotImplementedError("learned dynamic-k exists only in the fused router kernel")
    keep = keep | ~routed
    topk_weights.masked_fill_(~keep, 0.0)
    topk_ids.masked_fill_(~keep, -1)


def check_no_duplicates(topk_ids: torch.Tensor) -> None:
    if os.environ.get("SGLANG_QWENOPT_DYNK_CHECK", "0") != "1":
        return
    if torch.cuda.is_current_stream_capturing() or topk_ids.numel() == 0:
        return
    ids = topk_ids.to(torch.int64)
    s, _ = torch.sort(ids, dim=1)
    dup = (s[:, 1:] == s[:, :-1]) & (s[:, 1:] >= 0)
    assert not bool(dup.any()), "dynk: duplicate expert id within a token (cutlass_fused_moe drops it)"

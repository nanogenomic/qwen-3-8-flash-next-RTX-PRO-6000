# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hidden-state dump for MTP draft-head training (qwen-opt).

Writes, from inside the scheduler process, the exact tensors the MTP consumes:

  target.hc   [T, hc_count*hidden]  the target's hyper-connection hidden state
                                    (Qwen4ExpModel.forward's hc_hidden_states,
                                    which Qwen4ExpForConditionalGeneration.forward
                                    publishes as LogitsProcessorOutput.hidden_states
                                    and eagle_worker_v2 hands to the MTP)
  ids         [T] int32 input_ids, [T] int32 positions

and, when the equivalence-gate channel is on, SGLang's OWN MTP outputs at every
prefill position:

  mtp.hc      [T, hc_count*hidden]  the MTP layer's hc output
  mtp.mixed   [T, hidden]           hyper_connection_mixer.mix(...) == lm_head in

Why a patch and not --return-hidden-states: the supported path converts the
tensor with `.cpu().clone().tolist()` per request
(scheduler_components/batch_result_processor.py::_append_prefill_hidden_states)
and ships Python floats over ZMQ.  At 10240 floats/token that is unusable for a
multi-million-token dump.  This writer does a pinned D2H copy and a raw
sequential write on a background thread.

Enable with:
  SGLANG_MTP_HIDDEN_DUMP_DIR=/path        (target hc channel; off when unset)
  SGLANG_MTP_HIDDEN_DUMP_GATE=1           (also dump SGLang's own MTP outputs)
  SGLANG_MTP_HIDDEN_DUMP_MAX_TOKENS=N     (stop after N tokens, 0 = unlimited)
  SGLANG_MTP_HIDDEN_DUMP_SHARD_GIB=8      (shard rotation size)

Tail-window control.  Dumping every position of a 16k-context sequence costs
20 KB x 16k = 320 MB per sequence; the training rows only need the generated
span plus enough preceding context to rebuild the MTP's own KV.  The client
writes the integer file {dump_dir}/KEEP_FROM before each request and the dumper
drops every row with position < that value.  The client is strictly serial (it
waits for each response), so there is no race.  KEEP_FROM absent or 0 keeps
everything, which is what the equivalence-gate run uses.

Layout under the dump dir:
  target-00000.bf16   raw little-endian bfloat16, row-major [T, hc_dim]
  target-00000.i32    raw int32, [T] input_ids then [T] positions per record
  mtp-00000.bf16      (gate channel) [T, hc_dim] then [T, hidden] per record
  index.jsonl         one JSON object per forward batch, in write order
  meta.json           written at close
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from typing import Optional

import torch

_ENV_DIR = "SGLANG_MTP_HIDDEN_DUMP_DIR"
_ENV_GATE = "SGLANG_MTP_HIDDEN_DUMP_GATE"
_ENV_MAX = "SGLANG_MTP_HIDDEN_DUMP_MAX_TOKENS"
_ENV_SHARD = "SGLANG_MTP_HIDDEN_DUMP_SHARD_GIB"
_ENV_QUEUE = "SGLANG_MTP_HIDDEN_DUMP_QUEUE"


class _Dumper:
    def __init__(self, path: str):
        self.dir = path
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        self.gate = os.environ.get(_ENV_GATE, "0") == "1"
        self.max_tokens = int(os.environ.get(_ENV_MAX, "0"))
        self.shard_bytes = int(float(os.environ.get(_ENV_SHARD, "8")) * (1 << 30))
        self.q: "queue.Queue" = queue.Queue(maxsize=int(os.environ.get(_ENV_QUEUE, "8")))
        self.n_tokens = 0
        self.n_batches = 0
        self.stopped = False
        self.t0 = time.time()
        self._shard = -1
        self._fh_t = self._fh_i = self._fh_m = None
        self._off_t = self._off_i = self._off_m = 0
        self._index = open(os.path.join(self.dir, "index.jsonl"), "a", buffering=1)
        # One scheduler step = one target extend forward followed (spec on) by
        # one MTP draft-extend forward, per prefill CHUNK
        # (eagle_worker_v2._forward_prefill_batch calls
        # _draft_extend_for_prefill unconditionally, and
        # _eagle_prefill_tail_tokens keeps non-final chunks chained).  The two
        # records carry the same `step` so the loader can pair them.
        self._step = 0
        self._rotate()
        # Proof-of-life for the dump client: it refuses to send traffic unless
        # a fresh SERVER.json exists, so it can never aim at a non-dump
        # instance (i.e. at GPU0's live serving).
        with open(os.path.join(self.dir, "SERVER.json"), "w") as fh:
            json.dump(
                {
                    "pid": os.getpid(),
                    "host": os.uname().nodename,
                    "started": self.t0,
                    "gate": self.gate,
                    "max_tokens": self.max_tokens,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
                fh,
                indent=2,
            )
        self._thread = threading.Thread(target=self._run, daemon=True, name="mtp-dump")
        self._thread.start()
        atexit.register(self.close)

    # -- shard files ----------------------------------------------------
    def _rotate(self):
        for fh in (self._fh_t, self._fh_i, self._fh_m):
            if fh is not None:
                fh.close()
        self._shard += 1
        s = f"{self._shard:05d}"
        self._fh_t = open(os.path.join(self.dir, f"target-{s}.bf16"), "wb", buffering=0)
        self._fh_i = open(os.path.join(self.dir, f"target-{s}.i32"), "wb", buffering=0)
        self._fh_m = (
            open(os.path.join(self.dir, f"mtp-{s}.bf16"), "wb", buffering=0)
            if self.gate else None
        )
        self._off_t = self._off_i = self._off_m = 0

    # -- producer side (called from the forward, must be cheap) ---------
    def enabled(self) -> bool:
        return not self.stopped

    def _keep_from(self, first_pos: int) -> int:
        """Re-read KEEP_FROM at each sequence start (position 0)."""
        if first_pos == 0:
            self._keep = 0
            try:
                with open(os.path.join(self.dir, "KEEP_FROM")) as fh:
                    self._keep = int(fh.read().strip() or 0)
            except (OSError, ValueError):
                self._keep = 0
        return getattr(self, "_keep", 0)

    def submit_target(self, input_ids, positions, hc, extend_seq_lens, bs):
        if self.stopped:
            return
        positions = positions.reshape(-1)
        input_ids = input_ids.reshape(-1)
        first_pos = int(positions[0].item())
        keep = self._keep_from(first_pos)
        last_pos = int(positions[-1].item())
        if keep and last_pos < keep:
            self._step += 1  # keep step ids aligned with the MTP records
            return
        if keep and first_pos < keep:
            cut = keep - first_pos
            positions = positions[cut:]
            input_ids = input_ids[cut:]
            hc = hc[cut:]
        n = hc.shape[0]
        self._step += 1
        rec = {
            "kind": "target",
            "step": self._step,
            "n": int(n),
            "hc_dim": int(hc.shape[1]),
            "bs": int(bs),
            "extend_seq_lens": (
                [int(x) for x in extend_seq_lens.tolist()]
                if extend_seq_lens is not None else None
            ),
        }
        rec["p0"] = int(positions[0].item())
        payload = (
            hc.detach().to(torch.bfloat16).cpu(),
            input_ids.detach().to(torch.int32).cpu(),
            positions.detach().to(torch.int32).cpu(),
            None,
        )
        self._enqueue(rec, payload)
        self.n_tokens += n
        if self.max_tokens and self.n_tokens >= self.max_tokens:
            self.stopped = True

    def submit_mtp(self, hc, mixed):
        """Separate record, same `step` as the target forward it followed."""
        if self.stopped or not self.gate:
            return
        rec = {
            "kind": "mtp",
            "step": self._step,
            "n": int(hc.shape[0]),
            "hc_dim": int(hc.shape[1]),
            "hidden": int(mixed.shape[1]),
        }
        payload = (
            None, None, None,
            (hc.detach().to(torch.bfloat16).cpu(),
             mixed.detach().to(torch.bfloat16).cpu()),
        )
        self._enqueue(rec, payload)

    def _enqueue(self, rec, payload):
        try:
            self.q.put((rec, payload), timeout=120)
        except queue.Full:  # pragma: no cover
            self.stopped = True

    # -- consumer side --------------------------------------------------
    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            rec, (hc, ids, pos, mtp) = item
            try:
                if self._off_t >= self.shard_bytes:
                    self._rotate()
                rec["shard"] = self._shard
                if hc is not None:
                    b = hc.contiguous().view(torch.uint8).numpy().tobytes()
                    rec["t_off"] = self._off_t
                    self._fh_t.write(b)
                    self._off_t += len(b)
                    bi = (
                        ids.contiguous().numpy().tobytes()
                        + pos.contiguous().numpy().tobytes()
                    )
                    rec["i_off"] = self._off_i
                    self._fh_i.write(bi)
                    self._off_i += len(bi)
                if mtp is not None:
                    mhc, mmix = mtp
                    bm = (
                        mhc.contiguous().view(torch.uint8).numpy().tobytes()
                        + mmix.contiguous().view(torch.uint8).numpy().tobytes()
                    )
                    rec["m_off"] = self._off_m
                    self._fh_m.write(bm)
                    self._off_m += len(bm)
                self._index.write(json.dumps(rec) + "\n")
                self.n_batches += 1
            finally:
                self.q.task_done()

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.stopped = True
        try:
            self.q.put(None, timeout=30)
            self._thread.join(timeout=300)
        except Exception:
            pass
        for fh in (self._fh_t, self._fh_i, self._fh_m):
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
        try:
            with open(os.path.join(self.dir, "meta.json"), "w") as fh:
                json.dump(
                    {
                        "tokens": self.n_tokens,
                        "batches": self.n_batches,
                        "shards": self._shard + 1,
                        "gate": self.gate,
                        "seconds": time.time() - self.t0,
                    },
                    fh,
                    indent=2,
                )
            self._index.close()
        except Exception:
            pass


_DUMPER: Optional[_Dumper] = None
_INIT = False


def get_dumper() -> Optional[_Dumper]:
    global _DUMPER, _INIT
    if not _INIT:
        _INIT = True
        d = os.environ.get(_ENV_DIR)
        if d:
            _DUMPER = _Dumper(d)
    return _DUMPER


def dump_target_hidden(input_ids, positions, hc_hidden_states, forward_batch) -> None:
    """Hook point for Qwen4ExpForConditionalGeneration.forward (extend only)."""
    d = get_dumper()
    if d is None or not d.enabled() or hc_hidden_states is None:
        return
    # is_extend() also covers TARGET_VERIFY; the prefill set is exactly
    # EXTEND | MIXED | SPLIT_PREFILL (forward_batch_info.py:262).
    mode = forward_batch.forward_mode
    if mode.is_idle() or not mode.is_extend_or_draft_extend_or_mixed():
        return
    d.submit_target(
        input_ids,
        positions,
        hc_hidden_states,
        getattr(forward_batch, "extend_seq_lens", None),
        getattr(forward_batch, "batch_size", 1),
    )


def dump_mtp_hidden(hidden_states, hc_hidden_states, forward_batch) -> None:
    """Hook point for Qwen4ExpForCausalLMMTP.forward (gate channel, extend only)."""
    d = get_dumper()
    if d is None or not d.enabled() or not d.gate or hc_hidden_states is None:
        return
    # _draft_extend_for_prefill runs the MTP with the batch still in EXTEND
    # mode (DRAFT_EXTEND_V2 is only set by prepare_for_draft_extend, i.e. the
    # per-decode 4-slot pass, which the gate does not need).
    mode = forward_batch.forward_mode
    if mode.is_idle() or not mode.is_extend_or_draft_extend_or_mixed():
        return
    if hidden_states.shape[0] != hc_hidden_states.shape[0]:
        return
    d.submit_mtp(hc_hidden_states, hidden_states)

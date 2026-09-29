# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""qwen-opt fuse: HyperConnection combine tail in one launch.

    [shared]  y = Float(sigmoid(h . w_sg) * s + f)          (MoE shared-expert epilogue)
    hyper     = Float(R + y (x) 2*sigmoid(logits / hc))      (hc_combine_apply)
    [norm]    normed = grouped_gemma_rmsnorm(hyper, w_next)  (next sublayer's hc_norm)

``logits`` are the combine-gate dot products emitted by
``sglang.kernels.ops.gemm.hc_mix.fused_hc_mix_gate``. Kernel: hc_fused_tail.cuh (next to
this file). Only reached with SGLANG_QWENOPT_FUSE_HC=1.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_CUH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hc_fused_tail.cuh")


class DeferredSharedAdd(NamedTuple):
    """MoE block output with the shared-expert gate/add left for the HC tail.

    Materialised value: ``routed + sigmoid(gate_input . gate_weight) * shared``.
    """

    routed: torch.Tensor  # [M, H]
    shared: torch.Tensor  # [M, H] ungated shared-expert output
    gate_input: torch.Tensor  # [M, H] MoE block input
    gate_weight: torch.Tensor  # [H]

    def materialize(self) -> torch.Tensor:
        from sglang.kernels.ops.elementwise.elementwise import (
            fused_gate_sigmoid_mul_add,
        )

        fused_gate_sigmoid_mul_add(
            self.gate_input, self.gate_weight, self.shared, self.routed
        )
        return self.routed


@cache_once
def _jit_hc_fused_tail_module(
    hc_count: int, hidden_size: int, dtype: torch.dtype, shared: bool, norm: bool
) -> Module:
    if dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"Unsupported dtype {dtype}. Supported: bfloat16, float16")
    if hidden_size <= 0 or hidden_size % 512 != 0:
        raise RuntimeError(f"Unsupported hidden_size {hidden_size}: multiple of 512")
    args = make_cpp_args(hc_count, hidden_size, is_arch_support_pdl(), dtype, shared, norm)
    return load_jit(
        "qwenopt_hc_fused_tail",
        *args,
        cuda_files=[_CUH],
        cuda_wrappers=[("hc_fused_tail", f"HcFusedTailKernel<{args}>::run")],
    )


def hc_fused_tail_supported(hc_count: int, hidden_size: int, dtype: torch.dtype) -> bool:
    return dtype in (torch.bfloat16, torch.float16) and hidden_size % 512 == 0 and hc_count > 0


def hc_fused_tail(
    block_output,
    residual: torch.Tensor,
    logits: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    norm_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Returns ``(hyper [M, hc*H], normed [M, hc*H] or None)``.

    ``block_output`` is a [M, H] tensor or a :class:`DeferredSharedAdd`.
    """
    shared = isinstance(block_output, DeferredSharedAdd)
    if shared:
        y = block_output.routed.reshape(-1, hidden_size).contiguous()
        s = block_output.shared.reshape(-1, hidden_size).contiguous()
        h = block_output.gate_input.reshape(-1, hidden_size).contiguous()
        w_sg = block_output.gate_weight.reshape(hidden_size).contiguous()
    else:
        y = block_output.reshape(-1, hidden_size).contiguous()
    r = residual.reshape(-1, hc_count * hidden_size)
    if not shared:
        s = h = y
        w_sg = r  # unused, not verified
    lg = logits.reshape(-1, hc_count).contiguous()
    out_hyper = torch.empty_like(r)
    norm = norm_weight is not None
    if norm:
        out_normed = torch.empty_like(r)
        nw = norm_weight
    else:
        out_normed = None
    module = _jit_hc_fused_tail_module(hc_count, hidden_size, r.dtype, shared, norm)
    module.hc_fused_tail(
        y,
        s,
        h,
        w_sg,
        r,
        lg,
        nw if norm else r,
        out_hyper,
        out_normed if norm else out_hyper,
        float(eps),
    )
    return out_hyper.reshape(residual.shape), (
        out_normed.reshape(residual.shape) if norm else None
    )

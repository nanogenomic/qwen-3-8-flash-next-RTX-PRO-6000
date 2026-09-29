# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""qwen-opt fuse: hc_fused_tail + fused_hc_mix_gate vs the unfused kernels.

Correctness test on arbitrary tensors. Real-weight / real-activation checks and
end-to-end timing run separately against a loaded engine, and are not part of
this file.
"""
import pytest
import torch

from sglang.kernels.ops.elementwise import hc_combine as hcc
from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
from sglang.kernels.ops.elementwise.hc_fused_tail import DeferredSharedAdd, hc_fused_tail
from sglang.kernels.ops.gemm.hc_mix import fused_hc_mix, fused_hc_mix_gate
from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import grouped_gemma_rmsnorm

HC, HS, LR, EPS = 4, 2560, 320, 1e-6


def _inputs(T, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rn = lambda *s, sc=1.0: (torch.randn(*s, generator=g, device="cuda") * sc).to(torch.bfloat16)
    return dict(
        hyper=rn(T, HC * HS, sc=0.7), y=rn(T, HS, sc=0.7), routed=rn(T, HS, sc=0.7), shared=rn(T, HS, sc=0.7),
        norm_w=rn(HC * HS, sc=0.1), next_w=rn(HC * HS, sc=0.1), inj=rn(HC, HC * HS, sc=0.01),
        wd=rn(LR, HC * HS, sc=0.02), wu=rn(HC * HS, LR, sc=0.02), sg=rn(HS, sc=0.02),
    )


def _split_logits(T, device):
    p = hcc._get_partials(HC, device, T)[:T]
    tot = p[:, 0, :].clone()
    for s in range(1, p.shape[1]):
        tot = tot + p[:, s, :]
    return tot.contiguous()


@pytest.mark.parametrize("T", [1, 2, 4, 8, 16])
def test_tail_bitwise_vs_apply_and_norm(T):
    d = _inputs(T, T)
    normed = grouped_gemma_rmsnorm(d["hyper"], d["norm_w"], HS, EPS)
    ref = hcc.hc_combine_split(d["y"], d["hyper"], normed, d["inj"], HC, HS)
    lg = _split_logits(T, d["hyper"].device)
    new, nn_ = hc_fused_tail(d["y"], d["hyper"], lg, HC, HS, norm_weight=d["next_w"], eps=EPS)
    assert torch.equal(new, ref)
    assert torch.equal(nn_, grouped_gemma_rmsnorm(ref, d["next_w"], HS, EPS))
    new2, none = hc_fused_tail(d["y"], d["hyper"], lg, HC, HS)
    assert none is None and torch.equal(new2, ref)


@pytest.mark.parametrize("T", [1, 4, 16])
def test_tail_shared_within_one_ulp(T):
    d = _inputs(T, 100 + T)
    normed = grouped_gemma_rmsnorm(d["hyper"], d["norm_w"], HS, EPS)
    h = d["y"]
    yb = d["routed"].clone()
    fused_gate_sigmoid_mul_add(h, d["sg"], d["shared"], yb)
    ref = hcc.hc_combine_split(yb, d["hyper"], normed, d["inj"], HC, HS)
    lg = _split_logits(T, d["hyper"].device)
    new, _ = hc_fused_tail(DeferredSharedAdd(d["routed"].clone(), d["shared"], h, d["sg"]), d["hyper"], lg, HC, HS)
    # shared-gate dot is summed in a different order: allow 1 bf16 ulp of the magnitude
    torch.testing.assert_close(new.float(), ref.float(), rtol=2**-7, atol=2**-7 * float(ref.float().abs().max()))


@pytest.mark.parametrize("T", [1, 4, 16])
def test_mix_gate(T):
    d = _inputs(T, 200 + T)
    x = grouped_gemma_rmsnorm(d["hyper"], d["norm_w"], HS, EPS)
    m_ref = fused_hc_mix(x, d["wd"], d["wu"], HC, HS)
    m, lg = fused_hc_mix_gate(x, d["wd"], d["wu"], d["inj"], HC, HS)
    torch.testing.assert_close(m.float(), m_ref.float(), rtol=1e-2, atol=5e-3)
    ref64 = x.double() @ d["inj"].double().t()
    assert (lg.double() - ref64).abs().max() < 1e-4 * max(1.0, float(ref64.abs().max()))

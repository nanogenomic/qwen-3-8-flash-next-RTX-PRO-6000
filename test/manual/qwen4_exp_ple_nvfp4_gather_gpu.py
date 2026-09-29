# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Manual GPU numerics probe: the native NVFP4 PLE gather kernel vs sglang's
dequantize_nvfp4 (fp32) on real rows of a QAD checkpoint's PLE shard_0, and the
error an FP8 re-code of the same table carries on the same rows.

Checkpoint locations come from the environment -- no paths are baked in:
  QWENOPT_QAD_CKPT         NVFP4/QAD checkpoint dir with a native NVFP4 PLE table
  QWENOPT_QAD_FP8_OVERLAY  the FP8 PLE overlay to compare against
"""
import json, os, sys, torch
from safetensors import safe_open


def _need(name, what):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"{__file__}: set {name} -- {what}")
    return v


Q = _need("QWENOPT_QAD_CKPT", "checkpoint dir holding the native NVFP4 PLE table")
F8 = _need("QWENOPT_QAD_FP8_OVERLAY", "FP8 PLE overlay dir to compare against")
P = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
qi = json.load(open(f"{Q}/model.safetensors.index.json"))["weight_map"]
fi = json.load(open(f"{F8}/model.safetensors.index.json"))["weight_map"]
def get(root, idx, k):
    with safe_open(f"{root}/{idx[k]}", framework="pt") as f:
        return f.get_tensor(k)
codes = get(Q, qi, P + "shard_0.weight")          # U8 [R, 80]
bsc = get(Q, qi, P + "shard_0.weight_scale")      # F8 [R, 10]
gs = get(Q, qi, P + "weight_scale_2").float()     # F32 [1]
R, D = codes.shape[0], codes.shape[1] * 2
dev = torch.device("cuda:0")
table = torch.cat([codes, bsc.view(torch.uint8)], dim=1).contiguous().to(dev)  # [R, 90]
from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel
import triton
g = torch.Generator().manual_seed(0)
ids = torch.randint(0, R, (200_000,), generator=g)
ids = torch.cat([ids, torch.tensor([0, R - 1, R, R + 5])])  # edges + out-of-range
ids_d = ids.to(dev)
out = torch.empty(ids.numel(), D, dtype=torch.bfloat16, device=dev)
_gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
    table.data_ptr(), ids_d, out, embedding_dim=D, tp_vocab_start=0, tp_vocab_end=R,
    global_scale_ptr=gs.to(dev), is_fp8=False, is_nvfp4=True, BLOCK_D=triton.next_power_of_2(D))
torch.cuda.synchronize()
from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4
inr = ids < R
sel = ids[inr]
ref = dequantize_nvfp4(codes[sel], bsc[sel], gs, out_dtype=torch.float32)
got = out[inr.to(dev)].float().cpu()
ref_bf16 = ref.to(torch.bfloat16).float()
exact = (got == ref_bf16).float().mean().item()
rel = ((got - ref).norm() / ref.norm()).item()
oob_zero = bool((out[~inr.to(dev)] == 0).all().item())
print(f"rows={sel.numel()} D={D} exact-bf16-match={exact*100:.4f}% relerr_vs_fp32={rel:.2e} max_abs_diff_vs_bf16ref={(got-ref_bf16).abs().max().item():.3e} oob_rows_zero={oob_zero}")
# the FP8 re-code the overlay carries, same rows
fp8 = get(F8, fi, P + "shard_0.weight")
fscale = get(F8, fi, P + "weight_scale").float() if (P + "weight_scale") in fi else torch.ones(1)
f8 = fp8[sel].float() * fscale
print(f"FP8 re-code rel-RMS vs NVFP4 truth on same rows: {((f8 - ref).norm() / ref.norm()).item():.4f}")
print("peak GiB", torch.cuda.max_memory_allocated() / 2**30)
print("PASS" if exact > 0.9999 and oob_zero else "FAIL")

# Copyright © 2025 Ligandal, Inc. All rights reserved.
"""Correctness + transfer benchmark for QSA host-resident K/V.

Runs on ONE small GPU (<2 GiB device memory). Shapes are the real
qwen38-flash-next full-attention layer: 2 KV heads x 256 dims, bf16, page 64,
indexer budget 2048 tokens = 512 compressed groups of 4 (+3 pending tail).
Tensor CONTENTS are arbitrary (seeded randn); this is a kernel-equivalence and
bandwidth test, not a model-quality test.
"""
import json
import sys
import time

import torch

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
from sglang.srt.mem_cache.qsa_host_kv import allocate_host_kv_buffers  # noqa: E402
from sglang.srt.layers.attention.qsa.sparse_attn import (  # noqa: E402
    qwen_sparse_valid_counts_triton,
    qwen_sparse_kv_extraction_compact_triton,
)

dev = torch.device("cuda:0")
H, D, PAGE, TOPK_BLOCKS, RATIO = 2, 256, 64, 512, 4
TOPK = TOPK_BLOCKS * RATIO + RATIO - 1  # 2051, as expand_qsa_block_indices emits
out = {"gpu": torch.cuda.get_device_name(0)}
torch.manual_seed(0)


def ev_time(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


# ---------------------------------------------------------------- correctness
def correctness():
    size = 8192
    k_shape = (size + PAGE, H, D)
    kh, vh, owner = allocate_host_kv_buffers(
        layer_num=2, k_shape=k_shape, v_shape=k_shape, dtype=torch.bfloat16, device=dev
    )
    assert kh[0].is_cuda and kh[0].data_ptr() == owner.dev_ptr
    kd = [torch.zeros(k_shape, dtype=torch.bfloat16, device=dev) for _ in range(2)]
    vd = [torch.zeros(k_shape, dtype=torch.bfloat16, device=dev) for _ in range(2)]
    res = {}
    # 1) zero init
    res["zero_init"] = bool((kh[0].float().abs().sum() == 0).item())
    # 2) scatter writes (the set_kv_buffer index_put path) incl. repeated slots
    for layer in range(2):
        loc = torch.randperm(size, device=dev)[:3000] + PAGE
        kk = torch.randn(3000, H, D, device=dev, dtype=torch.bfloat16)
        vv = torch.randn(3000, H, D, device=dev, dtype=torch.bfloat16)
        kh[layer][loc] = kk
        vh[layer][loc] = vv
        kd[layer][loc] = kk
        vd[layer][loc] = vv
    torch.cuda.synchronize()
    res["scatter_write_equal"] = all(
        torch.equal(kh[i], kd[i]) and torch.equal(vh[i], vd[i]) for i in range(2)
    )
    # 3) host-side view sees the same bytes as the device view
    cpu = owner.cpu_view(0, k_shape, torch.bfloat16)
    res["cpu_view_equal"] = torch.equal(cpu, kd[0].cpu())
    # 4) index_select full-prefix read (chunked-prefill path)
    idx = torch.randperm(size, device=dev) + PAGE
    res["index_select_equal"] = torch.equal(
        kh[1].index_select(0, idx), kd[1].index_select(0, idx)
    )
    # 5) the QSA decode gather kernel, strided (trtllm) and compact layouts
    batch, seq = 4, size - 7
    req_to_token = torch.zeros(8, size + PAGE, dtype=torch.int32, device=dev)
    for r in range(batch):
        req_to_token[r + 1, :seq] = (torch.randperm(size, device=dev)[:seq] + PAGE).int()
    req_idx = torch.arange(1, batch + 1, dtype=torch.int32, device=dev)
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=dev)
    blocks = torch.stack(
        [torch.randperm(seq // RATIO, device=dev)[:TOPK_BLOCKS] for _ in range(batch)]
    ).sort(dim=1).values
    toks = (blocks[:, :, None] * RATIO + torch.arange(RATIO, device=dev)).flatten(1)
    tail = torch.tensor([seq - 3, seq - 2, seq - 1], device=dev).expand(batch, 3)
    indices = torch.cat([toks, tail], 1).int().contiguous()
    counts = torch.empty(batch, dtype=torch.int32, device=dev)
    qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, TOPK)
    stride = (TOPK + 63) // 64 * 64
    cu = torch.arange(batch + 1, dtype=torch.int32, device=dev) * stride
    ok = True
    for layer in range(2):
        outs = []
        for kb, vb in ((kh[layer], vh[layer]), (kd[layer], vd[layer])):
            ok_ = torch.full((batch * stride, H, D), 7.0, dtype=torch.bfloat16, device=dev)
            ov_ = torch.full_like(ok_, 7.0)
            qwen_sparse_kv_extraction_compact_triton(
                kb, vb, req_to_token, req_idx, indices, seq_lens, cu, ok_, ov_,
                batch, TOPK, zero_fill_cols=stride,
            )
            outs.append((ok_, ov_))
        ok = ok and torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
        # reference from explicit gather
        slots = req_to_token[req_idx.long()[:, None], indices.long()]
        ref = kd[layer][slots.long()]
        got = outs[0][0].view(batch, stride, H, D)[:, :TOPK]
        ok = ok and torch.equal(got, ref)
    res["decode_gather_equal"] = bool(ok)
    # 6) fp8 storage view (uint8 store dtype, as --kv-cache-dtype fp8_e4m3 uses)
    k8, v8, own8 = allocate_host_kv_buffers(
        layer_num=1, k_shape=k_shape, v_shape=k_shape, dtype=torch.uint8, device=dev
    )
    x = torch.randn(100, H, D, device=dev).to(torch.float8_e4m3fn)
    loc = torch.arange(100, device=dev) + PAGE
    k8[0][loc] = x.view(torch.uint8)
    res["fp8_roundtrip_equal"] = torch.equal(k8[0][loc].view(torch.float8_e4m3fn).float(), x.float())
    del kh, vh, owner, k8, v8, own8
    return res


# ----------------------------------------------------------------- bandwidth
def bandwidth():
    res = {}
    size = 262144  # one 256k-token layer: 256 MiB K + 256 MiB V
    k_shape = (size + PAGE, H, D)
    kh, vh, owner = allocate_host_kv_buffers(
        layer_num=1, k_shape=k_shape, v_shape=k_shape, dtype=torch.bfloat16, device=dev
    )
    kd = torch.randn(k_shape, dtype=torch.bfloat16, device=dev)
    vd = torch.randn(k_shape, dtype=torch.bfloat16, device=dev)
    kh[0].copy_(kd)
    vh[0].copy_(vd)
    nbytes = kd.numel() * 2
    # A) bulk cudaMemcpy pinned host -> device (DMA engine)
    dst = torch.empty_like(kd)
    cpu = owner.cpu_view(0, k_shape, torch.bfloat16)
    ms = ev_time(lambda: dst.copy_(cpu, non_blocking=True), iters=10)
    res["memcpy_h2d_GBps"] = round(nbytes / ms / 1e6, 2)
    ms = ev_time(lambda: cpu.copy_(dst, non_blocking=True), iters=10)
    res["memcpy_d2h_GBps"] = round(nbytes / ms / 1e6, 2)
    # B) zero-copy streaming read by a kernel (full-prefix index_select, prefill path)
    idx = torch.arange(PAGE, size + PAGE, device=dev)
    ms = ev_time(lambda: kh[0].index_select(0, idx), iters=10)
    res["zerocopy_seq_read_GBps"] = round(size * H * D * 2 / ms / 1e6, 2)
    ms = ev_time(lambda: kd.index_select(0, idx), iters=10)
    res["hbm_seq_read_GBps"] = round(size * H * D * 2 / ms / 1e6, 2)
    # C) zero-copy scattered write (set_kv_buffer, 8192-token prefill chunk)
    loc = torch.arange(PAGE, PAGE + 8192, device=dev)
    src = torch.randn(8192, H, D, dtype=torch.bfloat16, device=dev)

    def w():
        kh[0][loc] = src
    ms = ev_time(w, iters=20)
    res["zerocopy_write_8192tok_ms"] = round(ms, 3)
    res["zerocopy_write_GBps"] = round(8192 * H * D * 2 / ms / 1e6, 2)
    # D) the QSA decode gather: per layer, `rows` query rows x 2051 tokens
    seq = size - 5
    req_to_token = torch.zeros(9, size + PAGE, dtype=torch.int32, device=dev)
    # slots page-contiguous like the paged allocator (pages shuffled)
    pages = torch.randperm(size // PAGE, device=dev)
    slot_of_pos = (pages[:, None] * PAGE + torch.arange(PAGE, device=dev) + PAGE).flatten()
    for r in range(1, 9):
        req_to_token[r, :size] = slot_of_pos.int()
    gather = {}
    for rows in (1, 4, 8, 16):
        batch = rows
        req_idx = ((torch.arange(batch, device=dev) % 8) + 1).int()
        seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=dev)
        blocks = torch.stack(
            [torch.randperm(seq // RATIO, device=dev)[:TOPK_BLOCKS] for _ in range(batch)]
        ).sort(dim=1).values
        toks = (blocks[:, :, None] * RATIO + torch.arange(RATIO, device=dev)).flatten(1)
        tail = torch.tensor([seq - 3, seq - 2, seq - 1], device=dev).expand(batch, 3)
        indices = torch.cat([toks, tail], 1).int().contiguous()
        stride = (TOPK + 63) // 64 * 64
        cu = torch.arange(batch + 1, dtype=torch.int32, device=dev) * stride
        ok_ = torch.empty((batch * stride, H, D), dtype=torch.bfloat16, device=dev)
        ov_ = torch.empty_like(ok_)
        entry = {}
        for name, kb, vb in (("host", kh[0], vh[0]), ("hbm", kd, vd)):
            def g():
                qwen_sparse_kv_extraction_compact_triton(
                    kb, vb, req_to_token, req_idx, indices, seq_lens, cu, ok_, ov_,
                    batch, TOPK, zero_fill_cols=stride,
                )
            ms = ev_time(g, iters=30)
            entry[f"{name}_ms_per_layer"] = round(ms, 4)
        moved = batch * TOPK * H * D * 2 * 2
        entry["bytes_per_layer_MB"] = round(moved / 1e6, 2)
        entry["host_GBps"] = round(moved / entry["host_ms_per_layer"] / 1e6, 2)
        gather[str(rows)] = entry
    res["decode_gather"] = gather
    del kh, vh, owner
    return res


if __name__ == "__main__":
    out["correctness"] = correctness()
    out["bandwidth"] = bandwidth()
    out["peak_device_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    print(json.dumps(out, indent=1))
    if not all(out["correctness"].values()):
        sys.exit(1)

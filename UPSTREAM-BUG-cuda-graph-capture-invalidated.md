# DRAFT upstream report: nondeterministic `cudaErrorStreamCaptureInvalidated` during decode CUDA-graph capture

> ⛔⛔ **RESOLVED — DO NOT FILE. This is not an upstream bug.** The cause is in this fork: the
> host-KV hot cache's statistics thread (`SGLANG_QSA_HOST_KV_STATS=1`) issued a device-to-host copy
> plus `synchronize()` every 30 s from a side thread, and under PyTorch's default
> `capture_error_mode="global"` that invalidates any CUDA-graph capture in flight. Timing (a tick
> due at 23:28:57; capture began 23:28:56 and died 23:28:58), incidence (the only capture
> invalidation in the whole production log, and the reporter exists only with a host-resident
> pool) and an out-of-engine reproduction (7 of 24 captures fail with that thread, 0 of 24 under
> `"relaxed"`, without the thread, or with the fixed reporter) agree. Mitigation:
> `SGLANG_QSA_HOST_KV_STATS=0`. The fix — counters published by the forward thread into a pinned
> host mirror, reporter makes no CUDA call — is on a development branch and not yet in this patch
> series. Full write-up: [BENCHMARKS §3.19](BENCHMARKS.md#319-nondeterministic-cuda-graph-capture-failure-on-the-host-kv-build).
>
> The original draft follows unchanged except for one correction (the "~13 h of uptime" figure),
> because its ruled-out list is still right and its prior — *"the honest prior is that this is
> ours"* — turned out to be correct.

> ⛔ **THIS IS A DRAFT AND HAS NOT BEEN FILED.** It is published here so the evidence is on
> record, not because it is ready. It describes **one occurrence**, has **no minimal
> reproduction**, **no identified cause**, and **no fix**. Filing it as-is would be filing a
> "sometimes capture fails" report, which is not usually a useful issue. Read
> [What would make this filable](#what-would-make-this-filable) before sending it anywhere.
>
> It is also **not established** that this is an upstream bug at all. The observation is on a
> fork, and the one new thing in that fork's boot path is a host-resident KV pool. The honest
> prior is that this is **ours**.

- **Reported against:** upstream SGLang `6fa3fe69e2e5e19b75cadd9fc285b72634551992` plus this
  fork's change set, running Qwen3.8-Flash-Next NVFP4 on one RTX PRO 6000 Blackwell Max-Q
  (SM120), PyTorch's bundled CUDA allocator.
- **Symptom:** server exits during startup, 2 s into `Capture target verify CUDA graph`.
- **Frequency:** **1 failure in 2 boot attempts**, then 2 h 38 min of uptime until a commanded restart (an earlier revision said "~13 h"; wrong) on the same binary,
  same flags, same model. Nondeterministic.
- **Severity:** startup only; no observed effect on a server that reaches ready. Self-heals
  under a restart policy at ~5 min 44 s per failed attempt.
- **Fix in this fork:** **none** at the time of drafting. *(Now: mitigation `SGLANG_QSA_HOST_KV_STATS=0`; fix on a development branch — see the banner.)*

---

## The error

```
RuntimeError: info.status != cudaStreamCaptureStatusInvalidated INTERNAL ASSERT FAILED
  at "c10/cuda/CUDACachingAllocator.cpp":2213, please report a bug to PyTorch.
  Invalid stream capture status
Search for `cudaErrorStreamCaptureInvalidated' in the CUDA runtime API docs for more information.

During handling of the above exception, another exception occurred:
  decode_cuda_graph_runner.py:490  __init__ → self.capture()
  decode_cuda_graph_runner.py:1074 capture → self._capture_one_stream()
  decode_cuda_graph_runner.py:1134 _capture_one_stream → self.capture_one_shape(...)
  decode_cuda_graph_runner.py:1257 capture_one_shape → self.backend.capture_one(...)
  runner_backend/full_cuda_graph_backend.py:181 capture_one → out = forward_fn()

Exception: Capture cuda graph failed: CUDA error: operation failed due to a previous error
  during capture
```

Immediately preceded by:

```
Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=4, bs=[1,2,3,4],
  avail mem=8.47 GB
Capturing batches (bs=4 avail_mem=8.47 GB):  25%|██▌  | 1/4
Capturing batches (bs=3 avail_mem=8.34 GB):  50%|█████| 2/4
Capturing batches (bs=2 avail_mem=8.33 GB):  50%|█████| 2/4   ← died here
```

So `bs=4` and `bs=3` captured cleanly and `bs=2` did not.

## ⛔ The traceback frame is the detection point, not the cause

This is the most important thing in the report, and the reason a naive reading of the
traceback sends you to the wrong file.

The innermost frame is an allocation:

```
models/qwen3_5.py:1049            output, _ = self.out_proj(core_attn_out)
  layers/linear.py:1654           → self.quant_method.apply(...)
  quantization/unquant.py:522     → _bf16_gemm_dispatch_impl(x, layer.weight, bias)
  quantization/unquant.py:346     → _sm120_skinny_gemm(x, weight, bias)
  kernels/.../sm120_bf16_skinny_gemm.py:186
                                  y = torch.empty((m, n), dtype=x.dtype, device=x.device)
```

`torch.empty` is simply **the first allocator call after the capture was already
invalidated**, which is where PyTorch's caching allocator asserts. CUDA surfaces
`cudaErrorStreamCaptureInvalidated` to the *next* API call on the stream, not to the operation
that invalidated it — and the engine's own secondary message says exactly that: *"operation
failed due to a previous error during capture"*.

Two consequences:

1. **The frame does not localise the bug.** The invalidating operation is somewhere earlier in
   the same capture.
2. **It is specifically not evidence for the host-KV path.** This frame is inside
   `self.linear_attn` — the **GDN linear-attention** layer — which does not touch the
   host-resident full-attention K/V at all. Any hypothesis that blames the UVA path has to
   explain why the detection point is in the one attention family that is *not* host-resident.

A capture is invalidated by an illegal operation during capture: a synchronising call, an
allocation on an uncaptured stream, an event query, a `cudaMemcpy` from unregistered host
memory, and so on. None has been identified here.

## What was ruled out, and how

The deployment booted twice in 36 seconds, once failing and once succeeding, with the same
binary, flags and model. That pairing is most of the evidence.

| Candidate | Evidence against |
|---|---|
| **VRAM exhaustion** | `avail mem=8.47 GB` at `Capture … begin` on **both** boots — identical to two decimals. The successful boot completed the same phase in 2.33 s for 0.15 GB. And the assert is a capture-status assert, not an allocation failure. |
| **Host-RAM exhaustion / OOM killer** | `dmesg \| grep -ci "out of memory"` → **0**. Both host-KV pinning steps logged success (15.00 GiB and 1.25 GiB) three phases earlier, and both GPU hot caches allocated after them. The failed attempt's cgroup swap peak was **67.9 MiB**. |
| **A misreported "202 GiB memory peak"** | systemd's cgroup peak for this unit is dominated by file-backed and shared pages: live breakdown while serving is `memory.current` 98.72 GiB, `file` 91.63 GiB, `shmem` 80.33 GiB, against `anon` 6.69 GiB and `unevictable` 0. Not an anonymous-allocation ceiling. |
| **Start timeout** | The unit is `Type=simple`, so the 90 s start timeout is never armed against readiness. The boot that *succeeded* took 285 s. |
| **A deterministic bug in the changed code** | It is not deterministic: 1 in 2, then 2 h 38 min of uptime to a commanded restart. |
| **FlashInfer autotune state** | Autotune completed normally on both boots (59 s failed / 61 s succeeded) and logged completion before capture began. Not excluded as a *contributor*, but it did not fail. |

## The one thing that changed

The failing boot was the **first** boot of a build that moves the 12 full-attention layers' K/V
into **pinned, UVA-mapped host memory**, read zero-copy, with a small on-GPU direct-mapped hot
cache. `[hypothesis, untested]` Something on that path being capture-illegal is the leading
candidate because it is the only new thing in the boot sequence — but as noted above, the single
available frame argues against the simplest version of that story, and nothing here establishes
it.

**It was not checked whether the same assert ever occurred on the pre-cutover build.** That is
the cheapest missing piece of evidence and it is a log audit, not an experiment.

## Mitigation that works today

`Restart=always`, `RestartSec=30`. A failed attempt costs ~344 s — 314 s of attempt plus the
restart delay — and the next attempt succeeded. Anyone running a host-resident KV pool should
have a restart policy in place before cutting over; a one-shot launcher leaves the card dark.

## What would make this filable

In rough order of value:

1. **A second occurrence**, ideally with `CUDA_LAUNCH_BLOCKING=1` and
   `TORCH_CUDA_SANITIZER`-style capture diagnostics enabled, so the invalidating call is
   identified rather than the detecting one.
2. **A log audit of the pre-cutover build's boots** for the same assert. If it appears there,
   this is not about host-resident KV at all and the report changes completely.
3. **A paired boot loop with the host-KV path disabled**, to establish whether the failure rate
   differs. That is the experiment that would turn the hypothesis into evidence; it costs
   serving time, which is why it has not been run.
4. **A minimal reproduction**, which nobody has attempted.

Until at least (1) and (2) exist, this document is a record, not a report.

# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only dry-run of Qwen4-Exp load_weights over a real checkpoint.

No GPU is touched: CUDA_VISIBLE_DEVICES is emptied, torch.cuda is faked to
report one SM120 device so SGLang picks its CUDA code paths, and every model
parameter lives on the meta device. Checkpoint tensors are fed as meta tensors
built from the safetensors headers (real bytes only for tiny tensors, which the
loaders may .item()).

Acceptance:
  * every checkpoint tensor is consumed by exactly one parameter/buffer write
    (no silent skip; mtp.* must be consumed by the draft model);
  * every model parameter receives at least one write, and fused parameters get
    every shard id they are built for;
  * no write changes dtype (an F8/U8 tensor landing in a BF16 param means the
    layer was built unquantized and would be silently up-cast).

Usage: qad_dryrun.py <ckpt_dir> [extra server flags...]
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("SGLANG_RUST_BUILD_MODE", "never")

import torch  # noqa: E402

# ---- fake a single SM120 device (no CUDA context is ever created) ----------
torch.cuda.is_available = lambda: True
torch.cuda.device_count = lambda: 1
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_capability = lambda *a, **k: (12, 0)
torch.cuda.get_device_name = lambda *a, **k: "NVIDIA RTX PRO 6000 Blackwell (dry-run)"


class _Props:
    major, minor = 12, 0
    total_memory = 96 * 2**30
    name = "NVIDIA RTX PRO 6000 Blackwell (dry-run)"
    multi_processor_count = 188
    L2_cache_size = 128 * 2**20
    uuid = "dry-run"


torch.cuda.get_device_properties = lambda *a, **k: _Props()
torch.cuda.mem_get_info = lambda *a, **k: (90 * 2**30, 96 * 2**30)
for _n in ("set_device", "synchronize", "empty_cache", "reset_peak_memory_stats"):
    setattr(torch.cuda, _n, lambda *a, **k: None)
torch.cuda.memory_allocated = lambda *a, **k: 0
torch.cuda.max_memory_allocated = lambda *a, **k: 0


class _FakeStream:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def wait_stream(self, *a, **k):
        pass

    def synchronize(self):
        pass


torch.cuda.Stream = _FakeStream
torch.cuda.current_stream = lambda *a, **k: _FakeStream()
torch.cuda.stream = lambda *a, **k: _FakeStream()

_orig_copy = torch.Tensor.copy_
shape_errors = []


def _meta_copy(self, src, non_blocking=False):
    if self.is_meta or (isinstance(src, torch.Tensor) and src.is_meta):
        if isinstance(src, torch.Tensor) and tuple(self.shape) != tuple(src.shape):
            try:
                torch.broadcast_shapes(tuple(src.shape), tuple(self.shape))
                if torch.broadcast_shapes(tuple(src.shape), tuple(self.shape)) != tuple(self.shape):
                    raise RuntimeError
            except Exception:
                raise RuntimeError(f"dry-run copy shape mismatch dst={tuple(self.shape)} src={tuple(src.shape)}")
        return self
    return _orig_copy(self, src, non_blocking)


torch.Tensor.copy_ = _meta_copy

import json  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import collections  # noqa: E402

from safetensors import safe_open  # noqa: E402

CKPT = sys.argv[1]
EXTRA = sys.argv[2:]

FLAGS = [
    "--model-path", CKPT, "--quantization", "modelopt_mixed", "--trust-remote-code",
    "--context-length", "262144", "--attention-backend", "flashinfer",
    "--sampling-backend", "flashinfer", "--moe-runner-backend", "flashinfer_cutlass",
    "--mamba-ssm-dtype", "bfloat16", "--page-size", "64",
    "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3",
    "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4",
    "--ple-offload-embedding", "--fp8-gemm-backend", "flashinfer_cutlass",
    "--enable-return-routed-experts", "--max-running-requests", "16",
    "--mem-fraction-static", "0.94",
] + EXTRA

from sglang.srt.server_args import prepare_server_args, set_global_server_args_for_scheduler  # noqa: E402

sa = prepare_server_args(FLAGS)
set_global_server_args_for_scheduler(sa)

from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402
os.environ["CUDA_VISIBLE_DEVICES"] = "9"  # test_utils indexes [0] at import; never a real device
from sglang.test.test_utils import publish_build_topology  # noqa: E402
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sglang.srt.distributed.parallel_state as _ps  # noqa: E402
_orig_ica = _ps.is_cuda_alike
_ps.is_cuda_alike = lambda: False  # process groups on CPU/gloo
init_distributed_environment(
    backend="gloo", world_size=1, rank=0, local_rank=0,
    distributed_init_method=f"tcp://127.0.0.1:{29000 + os.getpid() % 2000}",
)
publish_build_topology(tp_size=1)
initialize_model_parallel()
_ps.is_cuda_alike = _orig_ica

import sglang.srt.models.qwen4_exp as q4  # noqa: E402

q4.allocate_ple_host_table = lambda shape, dtype, **k: torch.empty(shape, dtype=dtype, device="meta")
q4.make_ple_file_prefetcher = lambda *a, **k: None
q4.make_ple_file_rss_trimmer = lambda *a, **k: None

from sglang.srt.configs.load_config import LoadConfig  # noqa: E402
from sglang.srt.configs.model_config import ModelConfig  # noqa: E402
from sglang.srt.model_loader.loader import _get_quantization_config, _initialize_model  # noqa: E402
from sglang.srt.model_loader.loader import set_default_torch_dtype  # noqa: E402

# ---- checkpoint inventory (headers only) ------------------------------------
idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
wmap = idx["weight_map"]
_DT = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8, "I64": torch.int64,
    "I32": torch.int32, "I8": torch.int8, "BOOL": torch.bool,
}
headers = {}
for f in sorted(set(wmap.values())):
    with open(os.path.join(CKPT, f), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    headers[f] = {k: v for k, v in h.items() if k != "__metadata__" and wmap.get(k) == f}

SMALL = 64  # tensors this small are read for real (loaders may .item() them)
state = {"cur": None}
consumed = collections.Counter()   # ckpt name -> number of writes attributed
writes = collections.defaultdict(list)  # param name -> [(shard_id, expert_id, ckpt, shape, dtype)]
dtype_mismatch = []


def iter_weights(filter_fn):
    for f, h in headers.items():
        small = [k for k, v in h.items() if filter_fn(k) and _numel(v["shape"]) <= SMALL]
        real = {}
        if small:
            with safe_open(os.path.join(CKPT, f), framework="pt") as so:
                for k in small:
                    real[k] = so.get_tensor(k)
        for k, v in h.items():
            if not filter_fn(k):
                continue
            state["cur"] = k
            t = real.get(k)
            if t is None:
                t = torch.empty(v["shape"], dtype=_DT[v["dtype"]], device="meta")
            yield k, t
    state["cur"] = None


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


instrument_fail = []


def instrument(model, tag):
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    names_by_id = collections.defaultdict(list)
    params_by_id = {}
    for pname, p in model.named_parameters(remove_duplicate=False):
        names_by_id[id(p)].append(pname)
        params_by_id[id(p)] = p
    for pid, p in params_by_id.items():
        pnames = names_by_id[pid]
        inner = getattr(p, "weight_loader", default_weight_loader)

        def wrapped(param, loaded_weight, *args, _inner=inner, _pname=pnames[0], **kw):
            ck = state["cur"]
            consumed[ck] += 1
            shard = kw.get("shard_id", args[1] if len(args) >= 2 else (args[0] if len(args) == 1 else None))
            expert = kw.get("expert_id", args[2] if len(args) >= 3 else None)
            writes[f"{tag}:{_pname}"].append((repr(shard), expert, ck, tuple(loaded_weight.shape), loaded_weight.dtype))
            if loaded_weight.dtype != param.dtype and not _dtype_ok(param.dtype, loaded_weight.dtype, ck):
                dtype_mismatch.append((tag, _pname, ck, str(param.dtype), str(loaded_weight.dtype)))
            return _inner(param, loaded_weight, *args, **kw)

        try:
            if hasattr(p, "_weight_loader"):  # BasevLLMParameter: read-only property
                p._weight_loader = wrapped
            else:
                p.weight_loader = wrapped
            if p.weight_loader is not wrapped:
                raise RuntimeError("attribute did not stick")
        except Exception as e:
            instrument_fail.append((tag, pnames[0], repr(e)))
    print(f"[{tag}] instrumented {len(params_by_id)} params, failures {len(instrument_fail)}", flush=True)
    instrument_fail.clear()
    for t in instrument_fail[:10]:
        print("   INSTRUMENT-FAIL", t)
    return names_by_id


def _dtype_ok(pd, ld, ck):
    # Scalar global scales are allowed to change float width (f32 ckpt -> f32/bf16 param).
    if pd.is_floating_point and ld.is_floating_point and pd.itemsize >= 2 and ld.itemsize >= 2:
        if ld == torch.float32 and ck.endswith(("input_scale", "weight_scale_2")):
            return True
        # lossless widening of bf16 SSM parameters (A_log, dt_bias) to fp32
        if ld == torch.bfloat16 and pd == torch.float32 and ck.endswith(("A_log", "dt_bias")):
            return True
    return False


def build(model_config, qc_name):
    load_config = LoadConfig()
    qc = _get_quantization_config(model_config, load_config)
    with set_default_torch_dtype(model_config.dtype):
        with torch.device("meta"):
            model = _initialize_model(model_config, load_config, qc)
    print(f"[{qc_name}] quant_config={type(qc).__name__ if qc else None}", flush=True)
    return model, qc


def quant_summary(model, tag):
    c = collections.Counter()
    for name, m in model.named_modules():
        qm = getattr(m, "quant_method", None)
        if qm is None:
            continue
        key = re.sub(r"\.\d+\.", ".N.", name)
        c[(key, type(qm).__name__, getattr(qm, "use_mxfp8", None))] += 1
    for (k, t, mx), n in sorted(c.items()):
        print(f"  [{tag}] {n:4d} {k:70s} {t}{' mxfp8' if mx else ''}")


# ---- target model ------------------------------------------------------------
mc = ModelConfig.from_server_args(sa)
# Mirror load_model_with_memory_saver: the offload flags reach the text config there.
from sglang.srt.runtime_context import get_exec  # noqa: E402
_off = get_exec().offload.ple_offload_embedding
if _off is None:
    _off = getattr(sa, "ple_offload_embedding", None)
if _off is None:
    _off = "--ple-offload-embedding" in FLAGS
mc.hf_text_config.ple_offload_embedding = bool(_off)
mc.hf_text_config.ple_offload_backend = get_exec().offload.ple_offload_backend
mc.hf_text_config.ple_offload_dir = get_exec().offload.ple_offload_dir
print("ple_offload_embedding =", mc.hf_text_config.ple_offload_embedding,
      "ple_embedding_dtype =", mc.hf_text_config.ple_embedding_dtype, flush=True)
model, qc = build(mc, "target")
quant_summary(model, "target")
instrument(model, "target")
import time as _t
_t0 = _t.time()
target_loaded = model.load_weights(iter_weights(lambda k: not k.startswith("mtp.")))

# direct writes (PLE shards / buffers) are reported by load_weights' return value
for n in target_loaded:
    pass

# ---- draft (MTP) model ------------------------------------------------------
dmc = ModelConfig.from_server_args(sa, model_path=CKPT, is_draft_model=True)
draft, dqc = build(dmc, "draft")
quant_summary(draft, "draft")
instrument(draft, "draft")
draft_loaded = draft.load_weights(iter_weights(lambda k: k.startswith("mtp.")))

# ---- verdict ----------------------------------------------------------------
all_names = [k for h in headers.values() for k in h]
ple_direct = [k for k in all_names if ".ngram_embedding.shard_" in k or ".ple.ple_embedding." in k and not k.endswith(".weight")]
# Vision-tower quantized groups are dequantized into one bf16 write, which the
# recorder attributes to the group's last tensor; credit the whole group.
_written_mods = {k.rpartition(".")[0] for k, n in consumed.items() if k and n}
_vis_algos = {k for k in getattr(qc, "quantized_layers", {}) if k.startswith("model.visual.")}
unconsumed = [k for k in all_names if consumed[k] == 0 and k not in set(ple_direct)
              and "rotary_emb.inv_freq" not in k
              and not (k.rpartition(".")[0] in _vis_algos and k.rpartition(".")[0] in _written_mods)]
multi = [(k, n) for k, n in consumed.items() if k is not None and n > 1]

missing = []
for tag, m, loaded in (("target", model, target_loaded), ("draft", draft, draft_loaded)):
    for pname, _ in m.named_parameters():
        if pname.endswith("_swizzled"):
            continue  # derived in process_weights_after_loading, never in a checkpoint
        if tag == "draft" and pname in ("model.embed_tokens.weight", "lm_head.weight"):
            continue  # NEXTN draft shares the target's embed/lm_head (set_embed_and_head)
        if tag == "draft" and pname.endswith(("w13_input_scale", "w2_input_scale")):
            # W4A16_NVFP4 checkpoints carry no activation scale; the param keeps
            # its neutral 1.0 fill. Correct only on a W4A16 runner (marlin).
            be = getattr(sa, "speculative_moe_runner_backend", None)
            if str(be) != "marlin":
                missing.append(f"{tag}:{pname} (W4A16 experts on {be} would run W4A4; use --speculative-moe-runner-backend marlin)")
            continue
        if f"{tag}:{pname}" not in writes and pname not in (loaded or set()):
            missing.append(f"{tag}:{pname}")

# shard coverage for fused params
shard_gaps = []
EXPECT = {
    "qkv_proj": {"'q'", "'k'", "'v'"},
    "gate_up_proj": {"0", "1"},
    "in_proj_qkvz": {"(0, 1, 2)", "3"},
    "in_proj_ba": {"0", "1"},
}
for key, ws in writes.items():
    for frag, exp in EXPECT.items():
        if f".{frag}." in key:
            got = {w[0] for w in ws}
            if "None" in got:
                continue  # fused checkpoint tensor loaded whole
            if not exp <= got:
                shard_gaps.append((key, sorted(got)))
    if ".w13_" in key or ".w2_" in key:
        per = collections.defaultdict(set)
        for s, e, *_ in ws:
            per[e].add(s)
        want = {"'w1'", "'w3'"} if ".w13_" in key else {"'w2'"}
        bad = [e for e, ss in per.items() if not want <= ss]
        n_exp = len(per)
        if bad:
            shard_gaps.append((key, f"experts missing shards: {bad[:5]}"))
        writes_key_nexp = n_exp
        if ".w13_" in key or ".w2_" in key:
            if n_exp not in (512,):
                shard_gaps.append((key, f"only {n_exp} experts written"))

# PLE table: shard row coverage + storage identity
ple_problems = []
for mname, mod in model.named_modules():
    if type(mod).__name__ != "Qwen4ExpNGramEmbedding":
        continue
    emb = mod.ngram_embedding
    for part in (("weight", "weight_scale") if getattr(emb, "nvfp4", False) else ("weight",)):
        rows = sum(v["shape"][0] for h in headers.values() for k, v in h.items()
                   if re.search(r"\.ngram_embedding\.shard_\d+\." + part + "$", k))
        if rows != emb.org_vocab_size:
            ple_problems.append(f"{mname} {part}: shard rows {rows} != table rows {emb.org_vocab_size}")
    print(f"PLE {mname}: {type(emb).__name__} nvfp4={getattr(emb, 'nvfp4', False)} host table {tuple(emb.weight.shape)} "
          f"{emb.weight.dtype} = {emb.weight.numel() * emb.weight.element_size() / 2**30:.2f} GiB")
print("\n==== DRY-RUN VERDICT ====")
print(f"PLE problems: {len(ple_problems)}")
for t in ple_problems:
    print("   PLE", t)
print(f"checkpoint tensors: {len(all_names)}  PLE direct: {len(ple_direct)}")
print(f"unconsumed (silently skipped): {len(unconsumed)}")
_uc = collections.Counter(re.sub(r"\.\d+\.", ".N.", k) for k in unconsumed)
for k, n in sorted(_uc.items()):
    print("   UNCONSUMED", n, k)
print(f"consumed more than once: {len(multi)}")
for k, n in multi[:20]:
    print("   MULTI", k, n)
print(f"params never written: {len(missing)}")
for k in missing[:40]:
    print("   MISSING", k)
print(f"dtype-changing writes: {len(dtype_mismatch)}")
for t in dtype_mismatch[:40]:
    print("   DTYPE", t)
print(f"shard coverage gaps: {len(shard_gaps)}")
for t in shard_gaps[:40]:
    print("   GAP", t)
ok = not (unconsumed or multi or missing or dtype_mismatch or shard_gaps or ple_problems)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

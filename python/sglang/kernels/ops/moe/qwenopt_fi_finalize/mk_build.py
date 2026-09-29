# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Emits a build.ninja that rebuilds ONE object of flashinfer's fused_moe_120 module
# (flashinfer 0.6.18) with the qwen-opt ILP finalizeMoeRouting patch applied.
# Only cutlass_fused_moe_instantiation.cu is recompiled; every other object is reused
# read-only from flashinfer's own JIT cache directory.
#
# This is an OPTIONAL lever. It is the fallback for SGLANG_QWENOPT_FUSE_MOE_FINALIZE_ILP=1
# and is redundant when SGLANG_QWENOPT_FUSE_SBMOE=1 (SBMOE removes the finalize at T <= 16).
#
# Configure with environment variables -- there are no baked-in paths:
#   QWENOPT_FI_BUILD_DIR   scratch + output dir            (required)
#   QWENOPT_VENV           virtualenv holding flashinfer   (default: the running interpreter's prefix)
#   QWENOPT_FI_CACHED_OPS  flashinfer JIT cache dir for fused_moe_120, the one holding
#                          build.ninja and the cached .o files                (required)
#   QWENOPT_FI_CUTLASS_BACKEND  override the cutlass_backend source dir       (optional)
#
# Usage (about 2.5 min on one core):
#   export QWENOPT_FI_BUILD_DIR=/path/to/scratch
#   export QWENOPT_FI_CACHED_OPS="$HOME/.cache/flashinfer/0.6.18/120f/cached_ops/fused_moe_120"
#   python mk_build.py
#   ninja -j1 -C "$QWENOPT_FI_BUILD_DIR/build"
#
# Output: $QWENOPT_FI_BUILD_DIR/build/fused_moe_120.so. Point the engine at it with
#   SGLANG_QWENOPT_FI_MOE_SO=$QWENOPT_FI_BUILD_DIR/build/fused_moe_120.so
#   SGLANG_QWENOPT_FUSE_MOE_FINALIZE_ILP=1
import os, subprocess, sys


def _need(name, what):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"mk_build.py: set {name} -- {what}")
    return v


W = _need("QWENOPT_FI_BUILD_DIR", "scratch + output directory for the patched module")
V = os.environ.get("QWENOPT_VENV", sys.prefix)
CB = os.environ.get(
    "QWENOPT_FI_CUTLASS_BACKEND",
    f"{V}/lib/python{sys.version_info.major}.{sys.version_info.minor}"
    "/site-packages/flashinfer/data/csrc/fused_moe/cutlass_backend",
)
EX = _need(
    "QWENOPT_FI_CACHED_OPS",
    "flashinfer JIT cache dir for fused_moe_120 (holds build.ninja and the cached .o files)",
)
os.makedirs(f"{W}/src", exist_ok=True); os.makedirs(f"{W}/build", exist_ok=True)
subprocess.check_call(["cp", f"{CB}/cutlass_fused_moe_instantiation.cu", f"{W}/src/"])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "patch_cuh.py"),
                       f"{CB}/cutlass_fused_moe_kernels.cuh", f"{W}/src/cutlass_fused_moe_kernels.cuh"])
lines = open(f"{EX}/build.ninja").read().split("\n")
out = []
inst_o = f"{EX}/cutlass_backend_cutlass_fused_moe_instantiation.cuda.o"
for l in lines:
    if l.startswith("build "):
        if l.startswith(f"build {inst_o}:"):
            l = l.replace(inst_o, f"{W}/build/inst.cuda.o").replace(f"{CB}/cutlass_fused_moe_instantiation.cu",
                                                                     f"{W}/src/cutlass_fused_moe_instantiation.cu")
            out.append(l)
        elif l.startswith(f"build {EX}/fused_moe_120.so:"):
            out.append(l.replace(inst_o, f"{W}/build/inst.cuda.o").replace(f"{EX}/fused_moe_120.so", f"{W}/build/fused_moe_120.so"))
        continue
    if l.startswith("default "):
        l = f"default {W}/build/fused_moe_120.so"
    if l.startswith("cuda_cflags = $common_cflags $"):
        l = f"cuda_cflags = $common_cflags -I{W}/src -I{CB} $"
    out.append(l)
open(f"{W}/build/build.ninja", "w").write("\n".join(out))
print([l[:160] for l in out if l.startswith("build ")])

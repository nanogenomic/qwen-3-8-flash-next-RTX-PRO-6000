# Copyright © 2025 Ligandal, Inc. All rights reserved.
"""Host-resident (UVA zero-copy) K/V storage for QSA full-attention layers.

QSA decode reads only the indexer's top-``indexer_budget`` tokens (+ the
pending compress tail) of each full-attention layer, and the selection is made
from the compressed indexer keys, which stay on the GPU. So the full K/V of a
long context does not need to be GPU-resident: it can live in pinned, mapped
host memory and be read over PCIe by the very same gather kernels.

This module provides that storage. A buffer is allocated with
``cudaHostAlloc(Mapped | Portable)`` and exposed to PyTorch as a *CUDA* tensor
(via ``__cuda_array_interface__``) whose data pointer is the UVA device
pointer of the pinned host pages. Every existing torch op / Triton kernel that
reads or writes the K/V pool therefore keeps working unchanged, only the bytes
now cross PCIe instead of HBM.

Enable with ``SGLANG_QSA_HOST_KV``:
  * unset / ``0`` / ``off``  - disabled (default; nothing changes)
  * ``1`` / ``target``       - target model's full-attention K/V on host;
                               the MTP draft layer's K/V stays on the GPU
  * ``all``                  - target and draft K/V on host

``SGLANG_QSA_HOST_KV_MAX_GB`` (default 100) is a fail-fast guard on the pinned
bytes one pool may allocate, so an oversized ``--max-total-tokens`` dies at
startup instead of taking the host OOM killer down with it.


"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import weakref
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

_MODE_ENV = "SGLANG_QSA_HOST_KV"
_MAX_GB_ENV = "SGLANG_QSA_HOST_KV_MAX_GB"


def qsa_host_kv_mode() -> str:
    raw = os.environ.get(_MODE_ENV, "").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "target"):
        return "target"
    if raw == "all":
        return "all"
    raise ValueError(f"{_MODE_ENV} must be off|target|all, got {raw!r}")


def qsa_host_kv_enabled(is_draft_worker: bool) -> bool:
    mode = qsa_host_kv_mode()
    if mode == "off":
        return False
    return mode == "all" or not is_draft_worker


def _check(err, what: str):
    # cuda-python returns (cudaError_t, *values)
    code = int(err[0]) if isinstance(err, tuple) else int(err)
    if code != 0:
        raise RuntimeError(f"{what} failed with cudaError {code}")
    return err[1:] if isinstance(err, tuple) else ()


class _CudaArrayInterface:
    """Minimal ``__cuda_array_interface__`` exporter for a raw device pointer."""

    def __init__(self, ptr: int, nbytes: int, owner):
        self._owner = owner  # keep the mapped allocation alive
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }


class MappedHostBuffer:
    """One pinned + device-mapped host allocation (freed with the object)."""

    def __init__(self, nbytes: int, device: torch.device):
        from cuda.bindings import runtime as cudart

        if nbytes <= 0:
            raise ValueError("MappedHostBuffer needs a positive size")
        self.nbytes = int(nbytes)
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            (host_ptr,) = _check(
                cudart.cudaHostAlloc(
                    self.nbytes,
                    cudart.cudaHostAllocMapped | cudart.cudaHostAllocPortable,
                ),
                f"cudaHostAlloc({self.nbytes} B)",
            )
            host_ptr = int(host_ptr)
            (dev_ptr,) = _check(
                cudart.cudaHostGetDevicePointer(host_ptr, 0),
                "cudaHostGetDevicePointer",
            )
        self.host_ptr = host_ptr
        self.dev_ptr = int(dev_ptr)
        self._finalizer = weakref.finalize(self, _free_host, host_ptr)
        # cudaHostAlloc does not zero memory. Zero it once on the host side so a
        # never-written slot reads as 0.0 exactly like the torch.zeros GPU pool.
        ctypes.memset(self.host_ptr, 0, self.nbytes)

    def cuda_view(self, offset: int, shape, dtype: torch.dtype) -> torch.Tensor:
        numel = math.prod(shape)
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        if offset < 0 or offset + nbytes > self.nbytes:
            raise ValueError("cuda_view out of range")
        iface = _CudaArrayInterface(self.dev_ptr + offset, nbytes, self)
        raw = torch.as_tensor(iface, device=self.device)
        if raw.data_ptr() != self.dev_ptr + offset:
            raise RuntimeError("torch copied the mapped buffer instead of aliasing it")
        return raw.view(dtype).view(tuple(shape))

    def cpu_view(self, offset: int, shape, dtype: torch.dtype) -> torch.Tensor:
        numel = math.prod(shape)
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        buf = (ctypes.c_uint8 * nbytes).from_address(self.host_ptr + offset)
        return torch.frombuffer(buf, dtype=torch.uint8).view(dtype).view(tuple(shape))


def _free_host(ptr: int) -> None:
    try:
        from cuda.bindings import runtime as cudart

        cudart.cudaFreeHost(ptr)
    except Exception:  # interpreter shutdown
        pass


def _max_host_bytes() -> int:
    return int(float(os.environ.get(_MAX_GB_ENV, "100")) * (1 << 30))


def allocate_host_kv_buffers(
    *,
    layer_num: int,
    k_shape,
    v_shape,
    dtype: torch.dtype,
    device,
    label: str = "",
):
    """Allocate [layer_num] K and V buffers as device-mapped host memory.

    Returns (k_buffers, v_buffers, owner). The owner must be kept referenced
    for the buffers' lifetime.
    """
    itemsize = torch.empty((), dtype=dtype).element_size()
    k_bytes = math.prod(k_shape) * itemsize
    v_bytes = math.prod(v_shape) * itemsize
    # 256 B alignment for every per-layer view.
    align = 256
    k_stride = (k_bytes + align - 1) // align * align
    v_stride = (v_bytes + align - 1) // align * align
    total = layer_num * (k_stride + v_stride)
    limit = _max_host_bytes()
    if total > limit:
        raise RuntimeError(
            f"QSA host KV{label}: {total / (1 << 30):.2f} GiB of pinned host memory "
            f"requested, above {_MAX_GB_ENV}={limit / (1 << 30):.1f} GiB. Lower "
            "--max-total-tokens or raise the guard deliberately."
        )
    owner = MappedHostBuffer(total, torch.device(device))
    k_buffers: List[torch.Tensor] = []
    v_buffers: List[torch.Tensor] = []
    offset = 0
    for _ in range(layer_num):
        k_buffers.append(owner.cuda_view(offset, k_shape, dtype))
        offset += k_stride
        v_buffers.append(owner.cuda_view(offset, v_shape, dtype))
        offset += v_stride
    logger.info(
        "QSA host KV%s: %d layers x (K %s + V %s) %s = %.2f GiB pinned+mapped host "
        "memory (UVA zero-copy)",
        label,
        layer_num,
        tuple(k_shape),
        tuple(v_shape),
        dtype,
        total / (1 << 30),
    )
    return k_buffers, v_buffers, owner


class HostKVPoolMixin:
    """Mixin for an MHA KV pool class: full K/V live in mapped host memory."""

    _qsa_host_kv_owner: Optional[MappedHostBuffer] = None

    def _create_buffers_normal(self):  # overrides MHATokenToKVPool
        if getattr(self, "kv_cache_layout", None) == "vectorized_5d":
            raise NotImplementedError("QSA host KV supports the NHD/HND layouts only")
        k_shape, v_shape = self._kv_buffer_shapes()
        self.k_buffer, self.v_buffer, self._qsa_host_kv_owner = (
            allocate_host_kv_buffers(
                layer_num=self.layer_num,
                k_shape=k_shape,
                v_shape=v_shape,
                dtype=self.store_dtype,
                device=self.device,
            )
        )

    def _alloc_post_capture_buffers(self):
        raise NotImplementedError("QSA host KV is incompatible with post-capture KV")

    @property
    def qsa_host_kv_bytes(self) -> int:
        owner = self._qsa_host_kv_owner
        return 0 if owner is None else owner.nbytes


_HOST_CLASS_CACHE = {}


def host_kv_pool_class(base_cls: type) -> type:
    """Return a subclass of ``base_cls`` whose K/V buffers are host-resident."""
    cls = _HOST_CLASS_CACHE.get(base_cls)
    if cls is None:
        cls = type(f"HostKV{base_cls.__name__}", (HostKVPoolMixin, base_cls), {})
        _HOST_CLASS_CACHE[base_cls] = cls
    return cls


__all__ = [
    "MappedHostBuffer",
    "allocate_host_kv_buffers",
    "host_kv_pool_class",
    "qsa_host_kv_enabled",
    "qsa_host_kv_mode",
]

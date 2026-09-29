"""QSA profile parsing for Qwen4-Exp compressed indexing."""

from __future__ import annotations

from typing import Optional

import msgspec

_COMPRESSED_FIELDS = (
    "indexer_n_heads",
    "indexer_kv_heads",
    "indexer_head_dim",
    "indexer_budget",
    "indexer_compress_ratio",
)
# fast_topk_v2 only supports these compressed block top-k widths.
_COMPRESSED_BLOCK_TOPK = frozenset({512, 2048})


class QSAProfile(msgspec.Struct, frozen=True):
    """Compressed sparse-attention indexer configuration."""

    n_heads: int  # index query heads
    kv_heads: int  # index key/value heads
    head_dim: int  # per-head index dimension
    budget: int  # tokens selected per query row
    compress_ratio: int

    @property
    def block_topk(self) -> int:
        """Compressed blocks selected per query row."""

        return self.budget // self.compress_ratio


def qsa_pending_ring_size(compress_ratio: int, num_draft_tokens: int = 0) -> int:
    """Rows per request in the pending index-key ring.

    The ring is addressed by ``position % ring_size``, so it is collision-free
    exactly when every position that must be live at the same instant has a
    distinct residue.  A forward writes a window of ``W`` consecutive positions
    and compresses the group whose last member lands inside that window; the
    group's ``compress_ratio`` members may start up to ``compress_ratio - 1``
    positions before the window.  Window and group therefore cover at most
    ``W + compress_ratio - 1`` consecutive positions, and that span is the
    minimum ring size.  Round up to whole groups so request rows stay
    group-aligned.

    ``num_draft_tokens <= 1`` (plain decode) reproduces the historical
    ``ring_size == compress_ratio`` layout, where the window is one token and
    the ring holds exactly the pending group.
    """
    if compress_ratio <= 0:
        raise ValueError(f"QSA compress ratio must be positive, got {compress_ratio}")
    window = max(int(num_draft_tokens), 1)
    span = window + compress_ratio - 1
    return compress_ratio * -(-span // compress_ratio)


def qsa_max_draft_tokens(compress_ratio: int, ring_size: int) -> int:
    """Widest verify window a ring of ``ring_size`` rows can serve."""
    return max(1, ring_size - compress_ratio + 1)


def resolve_qsa_pending_ring_size(compress_ratio: int) -> int:
    """Ring size under the published spec config.

    The pool allocates by this and the attention backend addresses by it, so
    both must call the same function.  ``SGLANG_QSA_PENDING_RING_SLOTS``
    pins the ring for A/B runs against an older build; it is an override, not a
    tuning knob, and a value below ``compress_ratio`` is rejected.
    """
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_spec, max_speculative_num_draft_tokens

    override = int(envs.SGLANG_QSA_PENDING_RING_SLOTS.get() or 0)
    if override > 0:
        if override < compress_ratio:
            raise ValueError(
                "SGLANG_QSA_PENDING_RING_SLOTS must be at least the QSA compress "
                f"ratio ({compress_ratio}), got {override}"
            )
        return override
    try:
        spec = get_spec()
        algorithm = spec.speculative_algorithm
    except Exception:
        # No published configuration (unit tests, offline pool construction):
        # the decode-only ring is the safe minimum, and a speculative runtime
        # always has the config bag by the time it builds a pool.
        return qsa_pending_ring_size(compress_ratio, 0)
    if algorithm is None:
        return qsa_pending_ring_size(compress_ratio, 0)
    num_draft_tokens = max_speculative_num_draft_tokens() or (
        spec.speculative_num_draft_tokens or 0
    )
    return qsa_pending_ring_size(compress_ratio, num_draft_tokens)


def _text_config(config):
    return getattr(config, "text_config", config)


def _require_fields(config, fields) -> dict:
    missing = [name for name in fields if getattr(config, name, None) is None]
    if missing:
        raise ValueError(f"QSA config is missing required fields: {missing}")
    return {name: int(getattr(config, name)) for name in fields}


def _parse_compressed(text_config) -> QSAProfile:
    values = _require_fields(text_config, _COMPRESSED_FIELDS)
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"QSA config values must be positive: {values}")
    if values["indexer_kv_heads"] != 1:
        raise ValueError("the QSA MQA operators require indexer_kv_heads=1")
    ratio = values["indexer_compress_ratio"]
    budget = values["indexer_budget"]
    if ratio < 2:
        # Padding rows carry logical length 1, which must never reach a
        # compression boundary; ratio >= 2 guarantees that.
        raise ValueError(f"QSA requires indexer_compress_ratio >= 2, got {ratio}")
    if budget % ratio != 0:
        raise ValueError(
            "indexer_budget must be divisible by indexer_compress_ratio, got "
            f"{budget} / {ratio}"
        )
    if budget // ratio not in _COMPRESSED_BLOCK_TOPK:
        raise ValueError(
            "fast_topk_v2 requires indexer_budget / indexer_compress_ratio "
            f"to be one of {sorted(_COMPRESSED_BLOCK_TOPK)}, got {budget // ratio}"
        )
    return QSAProfile(
        n_heads=values["indexer_n_heads"],
        kv_heads=values["indexer_kv_heads"],
        head_dim=values["indexer_head_dim"],
        budget=budget,
        compress_ratio=ratio,
    )


def parse_qsa_profile(config) -> Optional[QSAProfile]:
    """QSA profile of config, None if absent; malformed schemas raise ValueError."""

    if config is None:
        return None
    text_config = _text_config(config)
    if text_config is None:
        return None
    if getattr(text_config, "indexer_n_heads", None) is not None:
        return _parse_compressed(text_config)
    return None


def is_qwen_qsa(config) -> bool:
    """Return whether the config describes Qwen compressed QSA."""

    return parse_qsa_profile(config) is not None


__all__ = [
    "QSAProfile",
    "qsa_max_draft_tokens",
    "qsa_pending_ring_size",
    "resolve_qsa_pending_ring_size",
    "is_qwen_qsa",
    "parse_qsa_profile",
]

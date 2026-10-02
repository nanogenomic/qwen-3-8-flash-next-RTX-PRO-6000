# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Provider contract.

A provider observes ONE class of thing and emits zero or more Panels. Emitting
zero panels is how a provider says "this is not running" -- providers never
invent a panel to fill space, and never synthesize a value. An absent reading is
reported as absent, not as zero.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


@dataclass
class Panel:
    """One renderable unit. `view` names the JS renderer that draws it."""

    provider: str
    view: str
    title: str
    subtitle: str = ""
    # Ranking. Higher wins the pinned pane.
    priority: float = 0.0
    # Stable identity across polls, so the UI can animate rather than re-mount.
    key: str = ""
    # Panels sharing a group describe ONE subject, and are kept adjacent.
    group: str = ""
    # Order within the group: 0 is the primary/overview view.
    rank: int = 0
    # Status word rendered in the panel chrome: live | idle | stale | error
    state: str = "live"
    data: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d["key"]:
            d["key"] = f"{self.provider}:{self.view}"
        if not d["group"]:
            d["group"] = self.provider
        return d


class Provider:
    """Polled on its own cadence in a background thread."""

    id: str = "base"
    title: str = "Base"
    interval: float = 5.0

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.conf = cfg.get(self.id, {}) or {}
        self.interval = float(self.conf.get("interval", self.interval))
        self.last_error: str | None = None

    def poll(self) -> list[Panel]:  # pragma: no cover - interface
        return []


def human_dt(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    if seconds < 86400:
        return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d {(seconds % 86400) // 3600:02d}h"

# Pinning the upstream 0.20.4 baseline

The fork's `pyproject.toml` declares `version = "0.20.4"`, `license = "MIT"`,
`license-files = ["LICENSE"]`. Locating the matching upstream tree took a
moment, because **there is no `v0.20.4` tag upstream and no 0.20.4 release on
PyPI**:

- `https://api.github.com/repos/NousResearch/hermes-agent/git/refs/tags/v0.20.4`
  → `404`.
- PyPI `hermes-agent` publishes 11 releases; the newest is `0.19.0`. No `0.20.x`.
- Upstream's public tags are date-shaped (`v2026.8.18`, `v2026.9.24`, …) plus
  `v0.21.4+canary.*` builds. `git tag -l | grep 0\.20` → empty.
- `git log -S'version = "0.20.4"' -- pyproject.toml` finds nothing, so the
  string never appears as a committed edit in that form.

Reading the version out of the date tags directly resolves it:

| upstream tag     | `pyproject.toml` version |
|------------------|--------------------------|
| `v2026.8.16`     | `0.20.2`                 |
| `v2026.8.16.2`   | `0.20.3`                 |
| **`v2026.8.18`** | **`0.20.4`**             |
| `v2026.8.19`     | `0.20.5`                 |

**Baseline: `NousResearch/hermes-agent` at tag `v2026.8.18`.**

Corroboration: the fork's untouched upstream files carry an mtime of
2026-08-18, and `v2026.8.18:LICENSE` is byte-identical to the fork's `LICENSE`
(md5 `1864e6648a4c302b984b2efd215c7918`) — which is the file this directory
ships.

## What was verified against that baseline

- `tools/delegate_tool.py` — `git am` of patch 0002 applies to the pristine
  file; the result byte-matches `upstream-modified/tools/delegate_tool.py` and
  compiles under `py_compile`.
- Patch 0002's dependencies are all upstream at 0.20.4: `json`, `os`,
  `threading`, `time`, `typing.{Any,Dict,Optional}` are already imported in
  `tools/delegate_tool.py`, and `_load_config()` is already defined there.
- `tools/async_delegation.active_task_count()` — the counter patch 0001 reads —
  **is upstream at 0.20.4**, unmodified by this fork. Patch 0001 therefore
  introduces no new dependency; it only changes which existing counter the
  status bar reports and how it is labelled.
- `cli.py` — the `delegation_streams` / `delegation_slots` snapshot keys patch
  0001 sits among **do not exist in pristine 0.20.4** (`grep -c` → 0). They are
  this fork's earlier work. This is why patch 0001 carries a prerequisite and
  will not `git am` onto pristine; see the README.

## Deriving patch 0001

Because its baseline is a fork state rather than an upstream tag, patch 0001 was
derived from the on-disk pre-change snapshot of `cli.py` taken immediately
before the edit, giving the isolated 39-line delta (+37 −2) rather than the
fork's whole ~85 KB divergence from upstream in that file. Its hunk offsets
(`@@ -6519` and `@@ -7552`) are fork line numbers, not upstream ones.

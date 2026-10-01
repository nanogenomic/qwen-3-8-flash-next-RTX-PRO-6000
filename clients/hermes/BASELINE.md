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

---

# Bases for patches 0003–0012

Patches 0003–0012 are a **series**, not ten independent patches. Their base is
**this fork at commit `5a3c9f57354c9c6d0ad5687deb40cc1f0fe26cf0`** (short
`5a3c9f5`, 2026-09-30 13:31 −0400), and each patch applies to the tree the
previous one leaves. `git am` the ten in numeric order, or `git apply` them in
numeric order; both were verified against a clean checkout of that commit (see
*What was verified* below).

| patch | applies to | squashed from fork commits | new files it adds |
|---|---|---|---|
| 0003 | `5a3c9f5` | `301b96a` | — |
| 0004 | 0003's tree | `d231466` | — |
| 0005 | 0004's tree | `f8ae3f1`, `1b84f9e` | `tests/agent/test_aux_yields_to_mains.py` |
| 0006 | 0005's tree | `b54d548`, `7d81775` | — |
| 0007 | 0006's tree | `6a428aa` | `tests/cli/test_cli_loop_tick_interrupt.py`, `tests/hermes_cli/test_goal_judge_never_stops_midflight.py` |
| 0008 | 0007's tree | `3e1599f` | `hermes_cli/pool_share.py`, `tests/cli/test_cli_status_bar_pool_share.py`, `tests/hermes_cli/test_pool_share.py` |
| 0009 | 0008's tree | `92dff5f` | — |
| 0010 | 0009's tree | `62b6f3f` | — |
| 0011 | 0010's tree | `e83ccfb`, `952f227`, `7493255` | `hermes_cli/goal_watchers.py`, `tests/hermes_cli/test_goal_wait_is_delegated.py` |
| 0012 | 0011's tree | `25f1873` | — |

Dependencies worth knowing before you cherry-pick rather than apply the series:
**0009 extends 0007**, **0010 fixes a field choice made in 0008**, **0011 builds
on 0007 and 0009**, and **0012 edits lines 0003 and 0004 wrote**. 0004 is in the
series only because 0012 edits lines it rewrote; skip it and 0012 needs a
three-way merge.

One ordering note. The series is in fork-history order with one exception: 0011
squashes a test-fixture correction (`7493255`) that landed *after* 0012's commit
(`25f1873`). The two touch disjoint files, so the printed order applies; it was
checked, not assumed.

## Why the base is a fork commit, and what that does and does not buy you

Unlike patch 0002, these do not apply to pristine upstream 0.20.4, and the
honest reason is that **most of what they touch does not exist upstream**:
`agent/pool_broker.py`, `agent/pool_policy.py`, `agent/overflow_router.py`,
`hermes_cli/pool_share.py` and `hermes_cli/goal_watchers.py` are this fork's own
modules, and `cli.py`, `hermes_cli/goals.py` and `tools/goal_resume_supervisor.py`
carry heavy fork divergence in exactly the regions these hunks sit in. A patch
claiming an upstream base here would be a fiction that failed on first contact.

So what the stated base buys you is an **exact, named base and a series proven
to apply onto it in order** — not a base you can `git fetch`. `5a3c9f5` is a
commit in a private tree. On a public checkout:

- the **five new files** in the table above apply anywhere, because a new-file
  hunk has no pre-image to match. They are the largest single pieces of new
  mechanism here (`pool_share.py`, `goal_watchers.py`).
- everything else will need hand placement. `git apply --reject` will land the
  hunks it can and leave `.rej` files for the rest; `git apply --3way` is **not**
  available, because its blobs are in the private tree. The `docs/` write-ups
  exist for precisely this case and name the integration points.

## What was verified

Against a clean checkout of `5a3c9f5`, with no fuzz and no three-way merge:

- `git apply --check` passes for each of 0003–0012 **in numeric order**, each
  against the tree its predecessor left.
- `git apply` of all ten in order succeeds, and the resulting tree differs from
  the fork's own `7493255` **only** by the publication substitutions listed in
  [`NOTICE`](NOTICE) — 82 substituted patch lines, each replacing one line with
  one line, giving 81 line differences in the resulting tree (one of the 82 is a
  context line in 0011 restating a line 0007 already substituted) — plus the one
  change held back as documented-only, fork commit `3272a15` against
  `agent/overflow_router.py` (+10 −2 in the fork; absent here).
- `git am` of all ten in order succeeds on a fresh branch off `5a3c9f5`, so the
  series is well-formed as mail-shaped patches and not only as raw diffs.
- Every Python file the series touches compiles under `py_compile` on the
  **sanitised** tree (27 files).
- The test suites the series adds or edits pass on the **sanitised** tree:
  **498 tests**, 0 failures, across
  `tests/hermes_cli/test_goal_judge_never_stops_midflight.py`,
  `tests/hermes_cli/test_goal_wait_is_delegated.py`,
  `tests/hermes_cli/test_pool_share.py`,
  `tests/agent/test_pool_broker.py`, `tests/agent/test_vram_throttle.py`,
  `tests/agent/test_aux_yields_to_mains.py`, `tests/agent/test_title_generator.py`,
  `tests/tools/test_goal_resume_supervisor.py`,
  `tests/cli/test_cli_goal_interrupt.py`, `tests/cli/test_cli_loop_tick_interrupt.py`,
  `tests/cli/test_cli_status_bar_pool_share.py` and
  `tests/cli/test_subagent_thinking.py`.

  This is the check that matters for the substituted test fixtures, not a
  formality: the judge-verdict corpus in patch 0007 is the input to a classifier,
  and the quoted park text in patch 0011 is the input to a quote-verification
  filter. If a substitution had broken the shape those depend on, those suites
  fail. They pass.
- Patch 0004's claim is reproduced on the applied tree: the fork's own
  `scripts/check-windows-footguns.py --all` reports **1** finding across 997
  files — the documented baseline, in a file this series does not touch — and
  **0** across the three files 0004 edits.

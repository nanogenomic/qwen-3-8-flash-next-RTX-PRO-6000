# Patch series

The 38 commits of this fork as `git am`-able patches against upstream SGLang
`6fa3fe69e2e5e19b75cadd9fc285b72634551992`.

Use these to apply the change set onto an upstream checkout you already have, without cloning
this repository:

```bash
cd /path/to/sglang
git checkout 6fa3fe69e2e5e19b75cadd9fc285b72634551992
git am /path/to/patches/00*.patch
```

Together they are exactly this fork's change set against the base commit, split by commit.
Every patch is authored `Andre Watson <dre@ligandal.com>`. The series has been verified to
reproduce the tree byte-for-byte: applying it to a clean checkout of the base commit yields
the same tree hash as the source files published in `python/` and `test/` here.

**If you are cherry-picking for an upstream submission**, the correctness fix is `0001` plus
its GPU test in `0002` — see [`../UPSTREAM-BUG-qsa-index-key-ring.md`](../UPSTREAM-BUG-qsa-index-key-ring.md).
`0030` and `0031` are the two boot-order follow-ups to the shared-scratch work in `0021`, and
are needed with it.

**`0035` is the other one worth lifting on its own.** It bounds the QSA prefill gather, which
before it allocated four full-context transients per full-attention layer per chunk — so peak
non-pool VRAM scaled with a request's total context rather than with `--chunked-prefill-size`.
It is bit-exact, it depends on nothing else in this series, and upstream has already fixed the
same bug class in the DSA backend (`63d320c723`). See
[`../BENCHMARKS.md` §3.16](../BENCHMARKS.md#316-bounded-qsa-prefill-gather--the-transient-that-scaled-with-context).

Patches `0033`–`0038` are the 2026-09-30 group: host-resident KV (`0033`, `0034`), the bounded
gather (`0035`), non-fatal prefill OOM (`0036`, `0037`) and the host-KV tests (`0038`). `0033`
and `0034` go together. `0036` and `0037` go together — `0037` closes real holes found reviewing
`0036`, and `0036` alone should not be deployed.

These patches cover the source tree only. The top-level documentation is this repository's,
not part of the series. `../make-fork.sh` applies the whole series onto upstream for you.

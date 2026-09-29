# Patch series

The 32 commits of this fork as `git am`-able patches against upstream SGLang
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

These patches cover the source tree only. The top-level documentation is this repository's,
not part of the series. `../make-fork.sh` applies the whole series onto upstream for you.

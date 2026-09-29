#!/usr/bin/env bash
# Build the full fork locally: upstream SGLang at the pinned base commit, plus this
# repository's 32 commits on top, so `git diff` and `git rebase` work natively.
#
#   ./make-fork.sh [target-dir]
#
# Default target: ./sglang-qwenopt
#
# The result is a git repository whose first commit is upstream's pinned base and whose
# branch `qwenopt` carries our change set commit by commit:
#
#   git -C <target> diff upstream-base..qwenopt        # exactly this fork's change set
#   git -C <target> log  --oneline upstream-base..qwenopt
#   git -C <target> rebase --onto sglang/main upstream-base qwenopt   # onto newer upstream
set -euo pipefail

BASE=6fa3fe69e2e5e19b75cadd9fc285b72634551992
UPSTREAM=https://github.com/sgl-project/sglang
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$PWD/sglang-qwenopt}"

if [ -e "$TARGET" ]; then
  echo "make-fork.sh: $TARGET already exists; pick another target directory." >&2
  exit 1
fi
if [ ! -d "$HERE/patches" ]; then
  echo "make-fork.sh: no patches/ directory next to this script." >&2
  exit 1
fi

echo "==> cloning upstream SGLang at $BASE (shallow, one commit)"
mkdir -p "$TARGET"
git -C "$TARGET" init -q .
# `git am` needs a committer identity. Use the repository's own, so the script works on a
# machine with no global git config; the patches carry their own author, which is preserved.
git -C "$TARGET" config user.name  "$(git -C "$HERE" log -1 --format=%an 2>/dev/null || echo qwen-opt)"
git -C "$TARGET" config user.email "$(git -C "$HERE" log -1 --format=%ae 2>/dev/null || echo qwen-opt@localhost)"
git -C "$TARGET" remote add sglang "$UPSTREAM"
git -C "$TARGET" fetch --depth 1 sglang "$BASE"
git -C "$TARGET" checkout -q -b upstream-base FETCH_HEAD

echo "==> applying $(ls "$HERE"/patches/0*.patch | wc -l) patches"
git -C "$TARGET" checkout -q -b qwenopt
git -C "$TARGET" am --keep-non-patch "$HERE"/patches/0*.patch

echo
echo "==> done: $TARGET"
git -C "$TARGET" --no-pager log --oneline upstream-base..qwenopt | tail -5
echo "    ..."
git -C "$TARGET" --no-pager diff --stat upstream-base..qwenopt | tail -1
echo
echo "    git -C $TARGET diff upstream-base..qwenopt"
echo "    git -C $TARGET rebase --onto sglang/main upstream-base qwenopt"
echo
echo "NOTE: a shallow clone cannot be pushed to a new remote. If you want to publish the"
echo "      result, run 'git -C $TARGET fetch --unshallow sglang' first."

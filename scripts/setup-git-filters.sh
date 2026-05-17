#!/usr/bin/env bash
# Enable the nbmeta git clean filter for this clone.
#
# Owner: rock@rockcyber.com
# Version: 1.0.0
#
# Git filters cannot auto-activate from .gitattributes (a clone would
# otherwise run arbitrary code from the repo). Each clone opts in by
# running this once. Until then git treats the filter as pass-through,
# so the repo still works, just without metadata normalization.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"

git config filter.nbmeta.clean "python3 \"\$(git rev-parse --show-toplevel)/scripts/nb-normalize-meta.py\""
git config filter.nbmeta.smudge cat
git config filter.nbmeta.required true

# Re-run the filter against already-tracked notebooks so state is
# consistent immediately, not only on the next edit. Scope the pathspec
# to the notebooks the filter actually governs (.gitattributes pins it
# to notebooks/*.ipynb) so setup never restages unrelated worktree
# files. Let a real failure surface: a broken filter here means every
# later `git add` of a notebook fails (required=true, set above), so
# roll that flag back and exit non-zero rather than reporting success.
if ! git -C "$root" add --renormalize -- 'notebooks/*.ipynb'; then
    git config --unset filter.nbmeta.required || true
    echo "ERROR: nbmeta clean filter failed on renormalize." >&2
    echo "Reverted filter.nbmeta.required so notebook staging is not" >&2
    echo "blocked. Verify python3 is on PATH and that" >&2
    echo "scripts/nb-normalize-meta.py runs, then re-run this script." >&2
    exit 1
fi

echo "nbmeta filter enabled for this clone."

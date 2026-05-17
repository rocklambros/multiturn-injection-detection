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
# consistent immediately, not only on the next edit.
git -C "$root" add --renormalize . >/dev/null 2>&1 || true

echo "nbmeta filter enabled for this clone."

#!/usr/bin/env python3
"""Git clean filter: normalize volatile Jupyter notebook metadata.

Owner: rock@rockcyber.com
Version: 1.0.0

Purpose
    Stop kernel-metadata churn. Opening the notebook under a different
    Python kernel (for example conda `base` 3.13 vs the project's
    `python3` 3.12) rewrites `metadata.kernelspec` and
    `metadata.language_info.version`, producing a diff on every save
    even when no cell changed.

Inputs
    stdin: raw notebook JSON bytes (git pipes the working-tree file in
    when staging or diffing notebooks/*.ipynb).

Outputs
    stdout: the same notebook with kernelspec and language version
    pinned to the project canonical values, serialized with the exact
    formatting Jupyter uses (indent=1, sort_keys=True, no trailing
    newline). Verified byte-identical to the committed file when the
    only working-tree change is kernel metadata, so the filter never
    introduces formatting churn of its own.

Side effects
    None. Pure stdin to stdout transform. The working-tree file is left
    untouched; only what git records is normalized.

Fallback
    On any parse or processing error (merge conflict markers, truncated
    file, non-notebook input) the original bytes are written back
    unchanged. A clean filter must never corrupt content it cannot
    safely transform.
"""

import json
import sys

# Project canonical kernel identity. The notebook is a deliverable that
# must execute under the pinned 3.12 environment; these are the values
# committed in notebooks/multiturn_injection_detection.ipynb. Update
# here if the project Python is intentionally upgraded.
CANONICAL_KERNELSPEC = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
CANONICAL_PY_VERSION = "3.12.2"


def main() -> int:
    raw = sys.stdin.buffer.read()
    try:
        nb = json.loads(raw)
        metadata = nb.get("metadata")
        if not isinstance(metadata, dict):
            # Not a notebook we recognize. Pass through untouched.
            sys.stdout.buffer.write(raw)
            return 0

        metadata["kernelspec"] = dict(CANONICAL_KERNELSPEC)
        lang = metadata.get("language_info")
        if isinstance(lang, dict) and "version" in lang:
            lang["version"] = CANONICAL_PY_VERSION

        # indent=1 + sort_keys=True + no trailing newline reproduces
        # Jupyter's on-disk format exactly (verified byte-for-byte).
        sys.stdout.buffer.write(
            json.dumps(nb, indent=1, sort_keys=True).encode("utf-8")
        )
        return 0
    except (ValueError, TypeError):
        # Unparseable input (conflict markers, truncation). Never
        # corrupt: emit exactly what we received.
        sys.stdout.buffer.write(raw)
        return 0


if __name__ == "__main__":
    sys.exit(main())

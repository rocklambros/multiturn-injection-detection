#!/usr/bin/env python3
# Version: 1.0
# Maintainer: project maintainers (anonymized for double-blind review)
"""Upload the dataset and model artifacts to Zenodo as an anonymized deposition.

Why this exists: the anonymized review mirror (anonymous.4open.science) does not
resolve Git LFS pointers and caps repo size, so large binaries cannot ride along
in the review repo. This script publishes them to Zenodo instead and prints a
reserved DOI you cite from the (anonymized) README.

Identity safety: the default metadata sets the creator to "Anonymous (under
review)". Nothing here writes your name. Review the metadata, then publish.

Inputs:
  - ZENODO_TOKEN environment variable (personal access token, scope
    deposit:write and deposit:actions). Never pass the token on the command line.
  - One or more file paths to upload.

Side effects:
  - Creates a draft deposition on Zenodo (or sandbox), uploads files into its
    bucket, and writes the metadata. Does NOT publish unless --publish is given.
  - Prints the deposition id, the human edit URL, and the reserved DOI.

Usage:
  export ZENODO_TOKEN=...                      # sandbox or production token
  python scripts/zenodo_upload.py --sandbox \\
      data/hf_dataset/multiturn_train.json \\
      data/hf_dataset/multiturn_val.json \\
      data/hf_dataset/multiturn_test.json \\
      models/v3_distilbert_hier.pt \\
      models/v3_distilbert_concat.pt

  # When the draft looks right, publish it (irreversible — mints the DOI):
  python scripts/zenodo_upload.py --sandbox --publish-deposition 123456
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROD_BASE = "https://zenodo.org/api"
SANDBOX_BASE = "https://sandbox.zenodo.org/api"

# Anonymized deposition metadata. Edit title/description freely; leave the
# creator anonymous until after acceptance, then update it on Zenodo directly.
DEFAULT_METADATA = {
    "upload_type": "dataset",
    "title": "Multi-Turn Distributed Prompt Injection Detection: Dataset and Model Weights",
    "creators": [{"name": "Anonymous (under review)"}],
    "description": (
        "Synthetic shared-prefix multi-turn prompt injection conversations and "
        "trained model weights (dual-encoder GRU+LSTM, attention variant, "
        "DistilBERT baselines, ablation checkpoints) supporting the paper "
        "'Multi-Turn Distributed Prompt Injection Detection'. Anonymized for "
        "double-blind peer review."
    ),
    "access_right": "open",
    "license": "cc-by-nc-4.0",
    "keywords": [
        "prompt-injection",
        "multi-turn",
        "deep-learning",
        "llm-security",
        "temporal-modeling",
    ],
}

# Read-body block size for streamed uploads (8 MiB).
_BLOCK = 8 * 1024 * 1024


def _request(method, url, token, data=None, headers=None, content_length=None):
    """Issue an authenticated Zenodo API request.

    data may be bytes or a file-like object. When it is a file object, set
    content_length so http.client streams the body instead of buffering it.
    Returns the parsed JSON body (or {} for empty 2xx responses).
    Raises SystemExit with the server message on any HTTP error.
    """
    hdrs = {"Authorization": f"Bearer {token}"}
    if headers:
        hdrs.update(headers)
    if content_length is not None:
        hdrs["Content-Length"] = str(content_length)
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        sys.exit(f"ERROR {exc.code} on {method} {url}\n{detail}")
    except urllib.error.URLError as exc:
        sys.exit(f"ERROR: network failure on {method} {url}: {exc.reason}")


def _upload_file(bucket_url, path, token):
    """Stream a single file into the deposition bucket via PUT."""
    path = Path(path)
    size = path.stat().st_size
    target = f"{bucket_url}/{urllib.parse.quote(path.name)}"
    print(f"  uploading {path.name} ({size / 1048576:.1f} MB) ...", flush=True)
    with path.open("rb") as fh:
        result = _request(
            "PUT",
            target,
            token,
            data=fh,
            headers={"Content-Type": "application/octet-stream"},
            content_length=size,
        )
    checksum = result.get("checksum", "?")
    print(f"    done (checksum {checksum})", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="Files to upload")
    parser.add_argument("--sandbox", action="store_true",
                        help="Use sandbox.zenodo.org (test) instead of production")
    parser.add_argument("--title", help="Override deposition title")
    parser.add_argument("--description-file",
                        help="Read deposition description from this file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print the plan; make no API calls")
    parser.add_argument("--publish-deposition", metavar="ID",
                        help="Publish an existing draft deposition by id (irreversible)")
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ERROR: set ZENODO_TOKEN (do not pass tokens on the command line).")

    base = SANDBOX_BASE if args.sandbox else PROD_BASE
    where = "SANDBOX" if args.sandbox else "PRODUCTION"

    # Publish-only path: finalize a draft after you have reviewed it.
    if args.publish_deposition:
        if args.dry_run:
            print(f"[dry-run] would publish deposition {args.publish_deposition} on {where}")
            return
        print(f"Publishing deposition {args.publish_deposition} on {where} (irreversible)...")
        result = _request(
            "POST",
            f"{base}/deposit/depositions/{args.publish_deposition}/actions/publish",
            token,
        )
        doi = result.get("doi", "?")
        print(f"PUBLISHED. DOI: {doi}")
        print(f"Record: {result.get('links', {}).get('record_html', '?')}")
        return

    if not args.files:
        sys.exit("ERROR: provide at least one file to upload (or --publish-deposition).")

    paths = [Path(f) for f in args.files]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        sys.exit("ERROR: files not found:\n  " + "\n  ".join(missing))

    metadata = dict(DEFAULT_METADATA)
    if args.title:
        metadata["title"] = args.title
    if args.description_file:
        metadata["description"] = Path(args.description_file).read_text(encoding="utf-8")

    total = sum(p.stat().st_size for p in paths)
    print(f"Target: {where}")
    print(f"Files:  {len(paths)} ({total / 1048576:.1f} MB total)")
    print(f"Creator: {metadata['creators'][0]['name']}  (anonymized)")
    for p in paths:
        print(f"  - {p}  ({p.stat().st_size / 1048576:.1f} MB)")

    if args.dry_run:
        print("\n[dry-run] no API calls made. Re-run without --dry-run to upload.")
        return

    # 1. Create an empty draft deposition.
    print("\nCreating draft deposition...")
    dep = _request("POST", f"{base}/deposit/depositions", token,
                   data=b"{}", headers={"Content-Type": "application/json"})
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    reserved_doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi", "(reserved on publish)")
    print(f"  deposition id: {dep_id}")

    # 2. Stream each file into the bucket.
    print("Uploading files...")
    for p in paths:
        _upload_file(bucket, p, token)

    # 3. Attach metadata.
    print("Writing metadata...")
    _request("PUT", f"{base}/deposit/depositions/{dep_id}", token,
             data=json.dumps({"metadata": metadata}).encode("utf-8"),
             headers={"Content-Type": "application/json"})

    edit_url = dep["links"].get("html", f"{base.replace('/api', '')}/deposit/{dep_id}")
    print("\nDRAFT READY (not published).")
    print(f"  Reserved DOI: {reserved_doi}")
    print(f"  Review/edit:  {edit_url}")
    print("  Cite the reserved DOI in the README. When satisfied, publish with:")
    print(f"    python scripts/zenodo_upload.py{' --sandbox' if args.sandbox else ''} "
          f"--publish-deposition {dep_id}")


if __name__ == "__main__":
    main()

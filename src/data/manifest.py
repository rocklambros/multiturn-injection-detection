"""Generation manifest with SHA-256 hashes and provenance tracking."""

import hashlib
import json
import time
from pathlib import Path


def compute_file_hash(path):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def create_manifest(output_dir, partition_manifest_path, generation_stats,
                    model_version, api_params, gate_stats=None):
    """Create the generation manifest recording full provenance.

    Args:
        output_dir: Directory containing generated data shards.
        partition_manifest_path: Path to partition_manifest.json.
        generation_stats: Dict of per-tier generation statistics.
        model_version: API model version string.
        api_params: Dict of API parameters (temperature per tier, etc).
        gate_stats: Optional dict of validation gate statistics.

    Returns:
        Dict manifest (also saved to output_dir/generation_manifest.json).
    """
    output_dir = Path(output_dir)

    data_files = {}
    for f in sorted(output_dir.glob("*.jsonl")):
        data_files[f.name] = {
            "sha256": compute_file_hash(f),
            "size_bytes": f.stat().st_size,
        }
    for f in sorted(output_dir.glob("*.json")):
        if f.name != "generation_manifest.json":
            data_files[f.name] = {
                "sha256": compute_file_hash(f),
                "size_bytes": f.stat().st_size,
            }

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": model_version,
        "api_parameters": api_params,
        "partition_manifest_hash": compute_file_hash(partition_manifest_path),
        "generation_stats": generation_stats,
        "data_files": data_files,
        "total_sequences": sum(s.get("completed", 0) for s in generation_stats.values()),
        "total_errors": sum(s.get("errors", 0) for s in generation_stats.values()),
    }

    if gate_stats is not None:
        manifest["gate_stats"] = gate_stats

    manifest_path = output_dir / "generation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest

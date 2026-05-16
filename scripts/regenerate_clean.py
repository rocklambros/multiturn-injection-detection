"""Focused data regeneration after pipeline leakage fixes.

Reuses existing LLM attack data (generation code unchanged) and only
regenerates what the fixes affect:
1. LLM benign sequences (new generate_benign_batch replaces template_benign)
2. Template tier (closing template, greeting, filter, normalize fixes)
3. Response stripping (now handles both attacks and benign)
4. Final merge into multiturn_{split}.json

Usage: python scripts/regenerate_clean.py --output-dir data/synthetic_v2
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dotenv  # noqa: E402
dotenv.load_dotenv()

from src.utils.seed import set_global_seed  # noqa: E402
from src.data.batch_generator import generate_benign_batch  # noqa: E402
from src.data.synthetic_v2 import build_attack_sequence, build_benign_sequence, filter_benign_pool  # noqa: E402
from src.data.response_stripper import strip_batch  # noqa: E402

STRATEGY_DIST = {
    "fragment_distributed": 0.40,
    "gradual_escalation": 0.25,
    "context_priming": 0.20,
    "instruction_layering": 0.15,
}

TIER_SIZES = {
    "easy": {"train": 6000, "val": 1000, "test": 1500},
    "medium": {"train": 6000, "val": 1000, "test": 1500},
    "hard": {"train": 6000, "val": 1000, "test": 1500},
    "adversarial": {"train": 2000, "val": 500, "test": 1000},
    "template": {"train": 5000, "val": 1000, "test": 1000},
}


def generate_template_split(injection_pool, benign_pool, size, seed):
    random.seed(seed)
    attack_count = size // 2
    benign_count = size - attack_count
    usage_counts = {}
    sequences = []

    for i in range(attack_count):
        injection = injection_pool[i % len(injection_pool)]
        strategy = random.choices(
            list(STRATEGY_DIST.keys()),
            weights=list(STRATEGY_DIST.values()),
        )[0]
        num_turns = random.randint(3, 10)
        seq = build_attack_sequence(
            injection_text=injection,
            benign_pool=benign_pool,
            strategy=strategy,
            num_turns=num_turns,
            usage_counts=usage_counts,
        )
        seq["id"] = f"template_attack_{i}"
        sequences.append(seq)

    for i in range(benign_count):
        num_turns = random.randint(3, 10)
        seq = build_benign_sequence(benign_pool, num_turns, usage_counts)
        seq["id"] = f"template_benign_{i}"
        sequences.append(seq)

    random.shuffle(sequences)
    return sequences


def verify_attack_data(output_dir):
    """Verify existing LLM attack data is complete enough to reuse."""
    print("\n=== Verifying existing LLM attack data ===")
    all_ok = True
    for tier in ["easy", "medium", "hard", "adversarial"]:
        for split in ["train", "val", "test"]:
            path = output_dir / f"llm_{tier}_{split}_attacks.jsonl"
            expected = TIER_SIZES[tier][split] // 2
            if not path.exists():
                print(f"  MISSING: {path}")
                all_ok = False
                continue
            with open(path) as f:
                lines = f.readlines()
            good = sum(1 for l in lines if "error" not in json.loads(l))
            coverage = good / expected * 100
            status = "OK" if coverage > 90 else "LOW"
            print(f"  {tier}/{split}: {good}/{expected} ({coverage:.0f}%) [{status}]")
            if coverage < 90:
                all_ok = False
    return all_ok


async def main(args):
    set_global_seed(42)
    output_dir = Path(args.output_dir)

    # Step 0: Verify existing attack data
    if not verify_attack_data(output_dir):
        print("\nERROR: Existing attack data incomplete. Run full generate_data.py instead.")
        sys.exit(1)

    # Load partition manifest
    manifest_path = output_dir / "partition_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Run full generate_data.py first.")
        sys.exit(1)
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"\nPartition manifest loaded from {manifest_path}")

    # Step 1: Generate LLM benign data for all tiers
    print("\n=== Step 1: Generating LLM benign sequences ===")
    benign_stats = {}
    for tier in ["easy", "medium", "hard", "adversarial"]:
        for split in ["train", "val", "test"]:
            benign_count = TIER_SIZES[tier][split] // 2
            print(f"\n  Generating {tier}/{split}: {benign_count} benign sequences...")
            stats = await generate_benign_batch(
                count=benign_count,
                difficulty=tier,
                num_turns_range=(3, 10),
                output_path=output_dir / f"llm_{tier}_{split}_benign.jsonl",
                max_concurrent=args.max_concurrent,
            )
            benign_stats[f"{tier}_{split}_benign"] = stats

    # Step 2: Regenerate template tier
    print("\n=== Step 2: Regenerating template tier ===")
    print("  Filtering benign pools...")
    filtered_benign_pools = {}
    for split in ["train", "val", "test"]:
        filtered_benign_pools[split] = filter_benign_pool(manifest["benign_pools"][split])

    for split in ["train", "val", "test"]:
        sequences = generate_template_split(
            injection_pool=manifest["injection_pools"][split],
            benign_pool=filtered_benign_pools[split],
            size=TIER_SIZES["template"][split],
            seed=42 + hash(split),
        )
        out_path = output_dir / f"template_{split}.json"
        with open(out_path, "w") as f:
            json.dump(sequences, f, indent=2)
        print(f"  {split}: {len(sequences)} sequences -> {out_path}")

    # Step 3: Re-strip ALL files (attacks + benign)
    print("\n=== Step 3: Stripping AI responses ===")
    for tier in ["easy", "medium", "hard", "adversarial"]:
        for split in ["train", "val", "test"]:
            for kind in ["attacks", "benign"]:
                src_path = output_dir / f"llm_{tier}_{split}_{kind}.jsonl"
                if not src_path.exists():
                    continue
                sequences = []
                with open(src_path) as f:
                    for line in f:
                        seq = json.loads(line)
                        if "error" not in seq:
                            sequences.append(seq)
                stripped = strip_batch(sequences)
                out_path = output_dir / f"llm_{tier}_{split}_{kind}_stripped.jsonl"
                with open(out_path, "w") as f:
                    for seq in stripped:
                        f.write(json.dumps(seq) + "\n")
                print(f"  Stripped {tier}/{split}/{kind}: {len(stripped)} sequences")

    # Step 4: Merge into final files
    print("\n=== Step 4: Merging into final dataset files ===")
    for split in ["train", "val", "test"]:
        all_sequences = []

        for tier in ["easy", "medium", "hard", "adversarial"]:
            for kind in ["attacks", "benign"]:
                raw_path = output_dir / f"llm_{tier}_{split}_{kind}.jsonl"
                stripped_path = output_dir / f"llm_{tier}_{split}_{kind}_stripped.jsonl"
                src_path = stripped_path if stripped_path.exists() else raw_path
                if src_path.exists():
                    with open(src_path) as f:
                        for line in f:
                            seq = json.loads(line)
                            if "error" not in seq:
                                all_sequences.append(seq)

        template_path = output_dir / f"template_{split}.json"
        if template_path.exists():
            with open(template_path) as f:
                all_sequences.extend(json.load(f))

        random.shuffle(all_sequences)

        final_path = output_dir / f"multiturn_{split}.json"
        with open(final_path, "w") as f:
            json.dump(all_sequences, f, indent=2)

        attack_count = sum(1 for s in all_sequences if s.get("label") == 1)
        benign_count = len(all_sequences) - attack_count
        print(f"  {split}: {len(all_sequences)} total ({attack_count} attack, {benign_count} benign)")

    # Step 5: Verify generation_method distribution
    print("\n=== Step 5: Verification ===")
    for split in ["train", "val", "test"]:
        final_path = output_dir / f"multiturn_{split}.json"
        with open(final_path) as f:
            seqs = json.load(f)
        methods = {}
        for s in seqs:
            m = s.get("generation_method", "unknown")
            methods[m] = methods.get(m, 0) + 1
        labels = {0: 0, 1: 0}
        for s in seqs:
            labels[s.get("label", -1)] = labels.get(s.get("label", -1), 0) + 1
        print(f"  {split}: {len(seqs)} sequences")
        print(f"    Labels: {dict(labels)}")
        print(f"    Methods: {dict(methods)}")

    print("\nRegeneration complete!")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/synthetic_v2")
    parser.add_argument("--max-concurrent", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(main(args))

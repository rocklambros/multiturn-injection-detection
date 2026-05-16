"""Data generation orchestrator.

Coordinates:
1. Source text partitioning
2. Intent extraction
3. LLM batch generation (all tiers via Sonnet 4.6)
4. Template-based generation
5. AI response stripping
6. Validation gate
7. Manifest generation

Usage: python scripts/generate_data.py --output-dir data/synthetic_v2
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import dotenv  # noqa: E402
dotenv.load_dotenv()

from src.utils.seed import set_global_seed  # noqa: E402
from src.data.partitioner import partition_source_texts  # noqa: E402
from src.data.intent_extractor import extract_intents_batch, deduplicate_intents  # noqa: E402
from src.data.batch_generator import generate_batch, generate_benign_batch  # noqa: E402
from src.data.synthetic_v2 import build_attack_sequence, build_benign_sequence, filter_benign_pool  # noqa: E402
from src.data.response_stripper import strip_batch  # noqa: E402
from src.data.manifest import create_manifest  # noqa: E402

# Strategy distribution
STRATEGY_DIST = {
    "fragment_distributed": 0.40,
    "gradual_escalation": 0.25,
    "context_priming": 0.20,
    "instruction_layering": 0.15,
}

# Dataset composition per tier
TIER_SIZES = {
    "easy": {"train": 6000, "val": 1000, "test": 1500},
    "medium": {"train": 6000, "val": 1000, "test": 1500},
    "hard": {"train": 6000, "val": 1000, "test": 1500},
    "adversarial": {"train": 2000, "val": 500, "test": 1000},
    "template": {"train": 5000, "val": 1000, "test": 1000},
}


def generate_template_split(injection_pool, benign_pool, size, seed):
    """Generate template-based sequences for one split."""
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


async def main(args):
    set_global_seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source data
    print("Loading source data...")
    train_df = pd.read_csv("data/processed/single_turn_train.csv")
    val_df = pd.read_csv("data/processed/single_turn_val.csv")
    test_df = pd.read_csv("data/processed/single_turn_test.csv")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Step 1: Partition
    print("\nPartitioning source texts...")
    manifest = partition_source_texts(all_df, output_dir=str(output_dir), seed=42)
    print(f"  Injection pools: train={len(manifest['injection_pools']['train'])}, "
          f"val={len(manifest['injection_pools']['val'])}, "
          f"test={len(manifest['injection_pools']['test'])}")

    # Step 2: Extract intents
    print("\nExtracting intents...")
    intents = {}
    for split in ["train", "val", "test"]:
        pool_texts = manifest["injection_pools"][split]
        intents[split] = extract_intents_batch(pool_texts)
        unique = deduplicate_intents(intents[split])
        print(f"  {split}: {len(pool_texts)} texts -> {len(unique)} unique intents")

    # Save intents
    with open(output_dir / "intents.json", "w") as f:
        json.dump({k: list(set(v)) for k, v in intents.items()}, f, indent=2)

    # Step 3: LLM generation (per tier, per split) — ATTACKS ONLY
    generation_stats = {}
    if not args.template_only:
        for tier in ["easy", "medium", "hard", "adversarial"]:
            for split in ["train", "val", "test"]:
                size = TIER_SIZES[tier][split] // 2  # attack count (half the total)
                strategy_counts = {s: int(size * w) for s, w in STRATEGY_DIST.items()}

                print(f"\nGenerating {tier}/{split}: {size} attack sequences...")
                stats = await generate_batch(
                    intents=intents[split],
                    strategies=strategy_counts,
                    difficulty=tier,
                    num_turns_range=(3, 10),
                    output_path=output_dir / f"llm_{tier}_{split}_attacks.jsonl",
                    max_concurrent=args.max_concurrent,
                )
                generation_stats[f"{tier}_{split}"] = stats

    # Step 4: Generate BENIGN sequences for LLM tiers (LLM-generated to match attack style)
    benign_stats = {}
    if not args.template_only:
        print("\nGenerating LLM benign sequences for LLM tiers...")
        for tier in ["easy", "medium", "hard", "adversarial"]:
            for split in ["train", "val", "test"]:
                benign_count = TIER_SIZES[tier][split] // 2
                print(f"\nGenerating {tier}/{split}: {benign_count} benign sequences...")
                stats = await generate_benign_batch(
                    count=benign_count,
                    difficulty=tier,
                    num_turns_range=(3, 10),
                    output_path=output_dir / f"llm_{tier}_{split}_benign.jsonl",
                    max_concurrent=args.max_concurrent,
                )
                benign_stats[f"{tier}_{split}_benign"] = stats
    else:
        print("\nGenerating template benign sequences for LLM tiers (--template-only)...")
        for tier in ["easy", "medium", "hard", "adversarial"]:
            for split in ["train", "val", "test"]:
                benign_count = TIER_SIZES[tier][split] // 2
                benign_seqs = []
                usage_counts = {}
                filtered_pool = filter_benign_pool(manifest["benign_pools"][split])
                for i in range(benign_count):
                    num_turns = random.randint(3, 10)
                    seq = build_benign_sequence(
                        filtered_pool, num_turns, usage_counts,
                    )
                    seq["id"] = f"llm_{tier}_benign_{split}_{i}"
                    seq["difficulty"] = tier
                    benign_seqs.append(seq)

                out_path = output_dir / f"llm_{tier}_{split}_benign.jsonl"
                with open(out_path, "w") as f:
                    for seq in benign_seqs:
                        f.write(json.dumps(seq) + "\n")
                print(f"  {tier}/{split}: {len(benign_seqs)} benign sequences")

    # Step 5: Template-based generation (attacks + benign, self-contained tier)
    print("\nFiltering benign pools for template generation...")
    filtered_benign_pools = {}
    for split in ["train", "val", "test"]:
        filtered_benign_pools[split] = filter_benign_pool(manifest["benign_pools"][split])

    print("\nGenerating template-based sequences...")
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

    # Step 6: Strip AI responses from all tiers (LLM generates user+assistant for hard/adversarial)
    print("\nStripping AI responses from all tiers...")
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

    # Step 7: Run validation gate on all generated sequences
    gate_model_path = Path("models/v2_gru_retrain_best.pt")
    skip_gate = args.template_only or args.skip_gate or not gate_model_path.exists()
    gate_stats = {"passed": 0, "failed": 0, "by_tier": {}, "skipped": skip_gate}
    if skip_gate:
        reason = "template-only mode" if args.template_only else (
            "--skip-gate" if args.skip_gate else f"{gate_model_path} not found"
        )
        print(f"\nSkipping validation gate ({reason})")
    else:
        print("\nRunning validation gate...")
        from src.data.validation_gate import ValidationGate
        gate = ValidationGate(
            model_path=str(gate_model_path),
            vocab_path="models/vocab.json",
        )
    if not skip_gate:
        for tier in ["easy", "medium", "hard", "adversarial"]:
            tier_pass, tier_fail = 0, 0
            threshold = 0.3 if tier == "adversarial" else 0.5
            for split in ["val", "test"]:
                attack_path = output_dir / f"llm_{tier}_{split}_attacks.jsonl"
                stripped_path = output_dir / f"llm_{tier}_{split}_attacks_stripped.jsonl"
                if stripped_path.exists():
                    attack_path = stripped_path
                if attack_path.exists():
                    sequences = []
                    with open(attack_path) as f:
                        for line in f:
                            seq = json.loads(line)
                            if "error" not in seq:
                                sequences.append(seq)
                    passed, failed = gate.filter_sequences(
                        sequences, threshold=threshold,
                    )
                    tier_pass += len(passed)
                    tier_fail += len(failed)
                    with open(attack_path, "w") as f:
                        for seq in passed:
                            f.write(json.dumps(seq) + "\n")
            gate_stats["by_tier"][tier] = {"passed": tier_pass, "failed": tier_fail}
            gate_stats["passed"] += tier_pass
            gate_stats["failed"] += tier_fail
            rate = tier_pass / max(1, tier_pass + tier_fail) * 100
            print(f"  {tier}: {tier_pass} passed, {tier_fail} rejected ({rate:.1f}% pass)")

    with open(output_dir / "gate_stats.json", "w") as f:
        json.dump(gate_stats, f, indent=2)

    # Step 8: Merge all shards into final multiturn_{split}.json files
    print("\nMerging shards into final dataset files...")
    for split in ["train", "val", "test"]:
        all_sequences = []

        # Collect LLM sequences (prefer stripped versions for all tiers)
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

        # Add template sequences
        template_path = output_dir / f"template_{split}.json"
        if template_path.exists():
            with open(template_path) as f:
                all_sequences.extend(json.load(f))

        random.shuffle(all_sequences)

        # Write final merged file (the format loaders expect)
        final_path = output_dir / f"multiturn_{split}.json"
        with open(final_path, "w") as f:
            json.dump(all_sequences, f, indent=2)

        attack_count = sum(1 for s in all_sequences if s.get("label") == 1)
        benign_count = len(all_sequences) - attack_count
        print(f"  {split}: {len(all_sequences)} total ({attack_count} attack, {benign_count} benign) -> {final_path}")

    # Step 9: Generate manifest
    print("\nGenerating manifest...")
    create_manifest(
        output_dir=str(output_dir),
        partition_manifest_path=str(output_dir / "partition_manifest.json"),
        generation_stats=generation_stats,
        gate_stats=gate_stats,
        model_version="claude-4-sonnet-20250514",
        api_params={
            "easy": {"temperature": 0.7},
            "medium": {"temperature": 0.7},
            "hard": {"temperature": 0.8},
            "adversarial": {"temperature": 0.9},
        },
    )

    print("\nData generation complete!")
    print(f"Output: {output_dir}")
    print("Final files: multiturn_{train,val,test}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/synthetic_v2")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--template-only", action="store_true",
                        help="Only generate template-based data (no API calls)")
    parser.add_argument("--skip-gate", action="store_true",
                        help="Skip validation gate (auto-skipped if model missing)")
    args = parser.parse_args()
    asyncio.run(main(args))

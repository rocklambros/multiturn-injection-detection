"""V3 shared-prefix data generation orchestrator.

Phases:
  1. Partition topics into train/val/test (no overlap)
  2. Extract attack intents from source data
  3. Generate shared-prefix pairs per tier per split
  4. Generate template-based sequences (evaluation-only)
  5. Run dual-barrier confound gates (5-fold CV on train only)
  6. Merge into final multiturn_{split}.json files

Usage:
    python scripts/generate_v3_data.py --output-dir data/synthetic_v3
    python scripts/generate_v3_data.py --pilot  # 500-pair pilot run
    python scripts/generate_v3_data.py --template-only  # No API calls
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
from src.data.topic_pool import partition_topics  # noqa: E402
from src.data.partitioner import partition_source_texts  # noqa: E402
from src.data.intent_extractor import extract_intents_batch, deduplicate_intents  # noqa: E402
from src.data.shared_prefix_generator import generate_batch as generate_sp_batch  # noqa: E402
from src.data.synthetic_v2 import build_attack_sequence, build_benign_sequence, filter_benign_pool  # noqa: E402
# strip_batch import removed — all tiers keep assistant turns for consistency
from src.data.confound_gates import run_confound_gates  # noqa: E402

STRATEGY_DIST = {
    "fragment_distributed": 0.45,
    "gradual_escalation": 0.25,
    "context_priming": 0.15,
    "instruction_layering": 0.15,
}

# Pairs per tier per split (each pair = 1 benign + 1 attack sequence)
FULL_PAIR_COUNTS = {
    "easy": {"train": 3000, "val": 500, "test": 750},
    "medium": {"train": 3000, "val": 500, "test": 750},
    "hard": {"train": 3000, "val": 500, "test": 750},
    "adversarial": {"train": 1000, "val": 250, "test": 500},
}

PILOT_PAIR_COUNTS = {
    "easy": {"train": 125, "val": 0, "test": 0},
    "medium": {"train": 125, "val": 0, "test": 0},
    "hard": {"train": 125, "val": 0, "test": 0},
    "adversarial": {"train": 125, "val": 0, "test": 0},
}

TEMPLATE_SIZES = {
    "train": 5000,
    "val": 1000,
    "test": 1000,
}


def generate_template_split(injection_pool, benign_pool, size, seed):
    """Generate template-based sequences for one split (evaluation-only)."""
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
        seq["difficulty"] = "template"
        sequences.append(seq)

    for i in range(benign_count):
        num_turns = random.randint(3, 10)
        seq = build_benign_sequence(benign_pool, num_turns, usage_counts)
        seq["id"] = f"template_benign_{i}"
        seq["difficulty"] = "template"
        sequences.append(seq)

    random.shuffle(sequences)
    return sequences


async def main(args):
    set_global_seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_counts = PILOT_PAIR_COUNTS if args.pilot else FULL_PAIR_COUNTS

    # --- Step 1: Partition topics ---
    print("Partitioning topics...")
    topic_splits = partition_topics(seed=42)
    for split, topics in topic_splits.items():
        print(f"  {split}: {len(topics)} topics")

    with open(output_dir / "topic_partition.json", "w") as f:
        json.dump(topic_splits, f, indent=2)

    # --- Step 2: Extract attack intents ---
    print("\nLoading source data for intent extraction...")
    train_df = pd.read_csv("data/processed/single_turn_train.csv")
    val_df = pd.read_csv("data/processed/single_turn_val.csv")
    test_df = pd.read_csv("data/processed/single_turn_test.csv")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print("Partitioning source texts...")
    manifest = partition_source_texts(all_df, output_dir=str(output_dir), seed=42)

    print("Extracting intents...")
    intents = {}
    for split in ["train", "val", "test"]:
        pool_texts = manifest["injection_pools"][split]
        intents[split] = extract_intents_batch(pool_texts)
        unique = deduplicate_intents(intents[split])
        intents[split] = list(set(unique))
        print(f"  {split}: {len(pool_texts)} texts -> {len(intents[split])} unique intents")

    with open(output_dir / "intents.json", "w") as f:
        json.dump(intents, f, indent=2)

    # --- Step 3: Shared-prefix generation ---
    generation_stats = {}
    if not args.template_only:
        print("\n" + "=" * 70)
        print("SHARED-PREFIX GENERATION")
        print("=" * 70)

        for tier in ["easy", "medium", "hard", "adversarial"]:
            for split in ["train", "val", "test"]:
                count = pair_counts[tier][split]
                if count == 0:
                    continue

                print(f"\n--- {tier}/{split}: {count} pairs ---")
                stats = await generate_sp_batch(
                    topics=topic_splits[split],
                    attack_goals=intents[split],
                    strategies_dist=STRATEGY_DIST,
                    difficulty=tier,
                    count=count,
                    output_path=output_dir / f"sp_{tier}_{split}.jsonl",
                    max_concurrent=args.max_concurrent,
                )
                generation_stats[f"{tier}_{split}"] = stats

        # Save generation stats
        with open(output_dir / "generation_stats.json", "w") as f:
            json.dump(generation_stats, f, indent=2)

        total_cost = sum(s.get("estimated_cost_usd", 0) for s in generation_stats.values())
        total_pairs = sum(s.get("pairs_completed", 0) for s in generation_stats.values())
        total_errors = sum(s.get("errors", 0) for s in generation_stats.values())
        print(f"\nGeneration complete: {total_pairs} pairs, {total_errors} errors, ~${total_cost:.2f}")

    # NOTE: Response stripping removed. All tiers now keep assistant turns
    # for structural consistency. Stripping only hard/adversarial created a
    # cross-tier vocabulary confound (BoW could detect tier structure).

    # --- Step 5: Template generation (evaluation-only) ---
    if not args.skip_templates:
        print("\nGenerating template-based sequences (evaluation-only)...")
        filtered_benign_pools = {}
        for split in ["train", "val", "test"]:
            filtered_benign_pools[split] = filter_benign_pool(manifest["benign_pools"][split])

        for split in ["train", "val", "test"]:
            sequences = generate_template_split(
                injection_pool=manifest["injection_pools"][split],
                benign_pool=filtered_benign_pools[split],
                size=TEMPLATE_SIZES[split],
                seed=42 + hash(split),
            )
            out_path = output_dir / f"template_{split}.json"
            with open(out_path, "w") as f:
                json.dump(sequences, f, indent=2)
            print(f"  {split}: {len(sequences)} template sequences -> {out_path}")

    # --- Step 6: Normalize turn counts within pairs ---
    print("\nNormalizing turn counts within pairs...")
    for tier in ["easy", "medium", "hard", "adversarial"]:
        for split in ["train", "val", "test"]:
            raw_path = output_dir / f"sp_{tier}_{split}.jsonl"
            stripped_path = output_dir / f"sp_{tier}_{split}_stripped.jsonl"
            src_path = stripped_path if stripped_path.exists() else raw_path
            if not src_path.exists():
                continue

            seqs_by_pair = {}
            errors = []
            with open(src_path) as f:
                for line in f:
                    seq = json.loads(line)
                    if "error" in seq:
                        errors.append(seq)
                        continue
                    pid = seq.get("pair_id", "")
                    if pid not in seqs_by_pair:
                        seqs_by_pair[pid] = []
                    seqs_by_pair[pid].append(seq)

            normalized = []
            trimmed_count = 0
            for pid, pair in seqs_by_pair.items():
                if len(pair) == 2:
                    min_len = min(len(pair[0]["turns"]), len(pair[1]["turns"]))
                    for s in pair:
                        if len(s["turns"]) > min_len:
                            s["turns"] = s["turns"][:min_len]
                            trimmed_count += 1
                        normalized.append(s)
                else:
                    normalized.extend(pair)

            out_path = output_dir / f"sp_{tier}_{split}_norm.jsonl"
            with open(out_path, "w") as f:
                for seq in normalized:
                    f.write(json.dumps(seq) + "\n")
                for err in errors:
                    f.write(json.dumps(err) + "\n")

            if trimmed_count > 0:
                print(f"  {tier}/{split}: trimmed {trimmed_count} sequences to match pair lengths")

    # --- Step 7: Merge into final files ---
    print("\nMerging into final dataset files...")
    for split in ["train", "val", "test"]:
        all_sequences = []

        for tier in ["easy", "medium", "hard", "adversarial"]:
            # Prefer normalized > stripped > raw
            norm_path = output_dir / f"sp_{tier}_{split}_norm.jsonl"
            stripped_path = output_dir / f"sp_{tier}_{split}_stripped.jsonl"
            raw_path = output_dir / f"sp_{tier}_{split}.jsonl"

            if norm_path.exists():
                src_path = norm_path
            elif stripped_path.exists():
                src_path = stripped_path
            else:
                src_path = raw_path

            if src_path.exists():
                with open(src_path) as f:
                    for line in f:
                        seq = json.loads(line)
                        if "error" not in seq:
                            all_sequences.append(seq)

        random.shuffle(all_sequences)
        final_path = output_dir / f"multiturn_{split}.json"
        with open(final_path, "w") as f:
            json.dump(all_sequences, f, indent=2)

        attack_count = sum(1 for s in all_sequences if s.get("label") == 1)
        benign_count = len(all_sequences) - attack_count
        print(f"  {split}: {len(all_sequences)} total ({attack_count} attack, {benign_count} benign)")

    # --- Step 8: Run confound gates (train split only) ---
    print("\nLoading train data for confound gates...")
    with open(output_dir / "multiturn_train.json") as f:
        train_data = json.load(f)

    all_pass = True
    gate_results = {}

    if len(train_data) < 10:
        print(f"  Skipping gates — only {len(train_data)} train sequences (need at least 10)")
    else:
        calibrated = None
        cal_path = Path("results/null_calibration.json")
        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            calibrated = cal.get("thresholds", {})
            print(f"  Using calibrated thresholds from {cal_path}")

        all_pass, gate_results = run_confound_gates(train_data, calibrated_thresholds=calibrated)

        with open(output_dir / "gate_results.json", "w") as f:
            json.dump(gate_results, f, indent=2)

        if not all_pass:
            print("\nWARNING: Confound gates FAILED. Data may have lexical confounds.")
            print("Review gate_results.json for diagnostics.")
            if not args.force:
                print("Use --force to proceed despite gate failure.")
                sys.exit(1)

    # --- Step 9: Summary ---
    print("\n" + "=" * 70)
    print("V3 DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Primary training data: multiturn_{{train,val,test}}.json")
    print(f"Template data (eval-only): template_{{train,val,test}}.json")
    print(f"Confound gates: {'PASSED' if all_pass else 'FAILED'}")
    if generation_stats:
        total_cost = sum(s.get("estimated_cost_usd", 0) for s in generation_stats.values())
        print(f"Estimated API cost: ${total_cost:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/synthetic_v3")
    parser.add_argument("--max-concurrent", type=int, default=25)
    parser.add_argument("--pilot", action="store_true",
                        help="Run 500-pair pilot instead of full generation")
    parser.add_argument("--template-only", action="store_true",
                        help="Generate template data only (no API calls)")
    parser.add_argument("--skip-templates", action="store_true",
                        help="Skip template generation")
    parser.add_argument("--force", action="store_true",
                        help="Continue even if confound gates fail")
    args = parser.parse_args()
    asyncio.run(main(args))

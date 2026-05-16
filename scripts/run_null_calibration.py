"""Null-pair calibration for empirical gate threshold derivation.

Generates 200 benign-vs-benign pairs by splitting benign conversations
at a random point K, producing two benign sub-conversations that differ
only in continuation. The BoW and per-turn voting scores on these pairs
establish the empirical noise floor for the confound gates.

Thresholds are set at the 95th percentile + 0.05 margin.

Usage: python scripts/run_null_calibration.py --data-dir data/synthetic_v3
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_global_seed


def extract_turns_text(seq):
    turns = seq.get("turns", [])
    texts = []
    for t in turns:
        if isinstance(t, dict):
            if t.get("role", "user") == "user":
                texts.append(t.get("text", ""))
        elif isinstance(t, str):
            texts.append(t)
    return texts


def generate_null_pairs(benign_sequences, n_pairs=200, seed=42):
    """Generate null pairs from benign conversations.

    For each pair, take a benign conversation, split at random K in [2, N-2],
    create two pseudo-conversations sharing turns 1..K but with different
    continuations (drawn from other benign conversations).

    Args:
        benign_sequences: List of benign sequence dicts.
        n_pairs: Number of pairs to generate.
        seed: Random seed.

    Returns:
        list of (pseudo_attack, pseudo_benign) tuples, both label=0.
    """
    random.seed(seed)
    long_benign = [s for s in benign_sequences if len(extract_turns_text(s)) >= 5]

    if len(long_benign) < 10:
        raise ValueError(f"Need at least 10 benign conversations with 5+ turns, got {len(long_benign)}")

    pairs = []
    for i in range(n_pairs):
        base = random.choice(long_benign)
        donor = random.choice(long_benign)
        while donor is base:
            donor = random.choice(long_benign)

        base_turns = extract_turns_text(base)
        donor_turns = extract_turns_text(donor)

        k = random.randint(2, len(base_turns) - 2)
        prefix = base_turns[:k]

        cont_a = base_turns[k:]
        max_donor = min(len(donor_turns), len(base_turns) - k)
        cont_b = donor_turns[:max_donor] if max_donor > 0 else donor_turns[:2]

        seq_a = {
            "turns": [{"text": t, "role": "user"} for t in prefix + cont_a],
            "label": 0,
            "id": f"null_a_{i}",
            "k_value": k,
        }
        seq_b = {
            "turns": [{"text": t, "role": "user"} for t in prefix + cont_b],
            "label": 0,
            "id": f"null_b_{i}",
            "k_value": k,
        }
        pairs.append((seq_a, seq_b))

    return pairs


def compute_bow_f1_on_pairs(pairs):
    """Train BoW classifier to distinguish pair-A from pair-B.

    Returns F1 score — should be near 0.50 for well-calibrated null pairs.
    """
    all_seqs = []
    labels = []
    for a, b in pairs:
        all_seqs.append(a)
        labels.append(0)
        all_seqs.append(b)
        labels.append(1)

    texts = [" ".join(extract_turns_text(s)) for s in all_seqs]

    vec = CountVectorizer(max_features=5000, stop_words="english")
    X = vec.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, labels)

    preds = clf.predict(X)
    return float(f1_score(labels, preds))


def compute_per_turn_voting_f1(pairs):
    """Train per-turn classifier and vote — should be near chance on null pairs."""
    all_turns = []
    all_turn_labels = []
    for a, b in pairs:
        for t in extract_turns_text(a):
            all_turns.append(t)
            all_turn_labels.append(0)
        for t in extract_turns_text(b):
            all_turns.append(t)
            all_turn_labels.append(1)

    vec = CountVectorizer(max_features=3000, stop_words="english")
    X = vec.fit_transform(all_turns)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, all_turn_labels)

    seq_labels = []
    seq_preds = []
    for a, b in pairs:
        for seq, label in [(a, 0), (b, 1)]:
            turns = extract_turns_text(seq)
            if not turns:
                seq_labels.append(label)
                seq_preds.append(0)
                continue
            X_t = vec.transform(turns)
            probs = clf.predict_proba(X_t)[:, 1]
            seq_labels.append(label)
            seq_preds.append(1 if max(probs) >= 0.5 else 0)

    return float(f1_score(seq_labels, seq_preds))


def derive_thresholds(bow_scores, voting_scores, percentile=95, margin=0.05):
    """Derive gate thresholds from null-pair calibration scores.

    Args:
        bow_scores: List of BoW F1 scores from bootstrap resampling.
        voting_scores: List of per-turn voting F1 scores.
        percentile: Percentile for threshold (default 95th).
        margin: Safety margin above percentile.

    Returns:
        dict with bow_threshold and voting_threshold.
    """
    bow_threshold = float(np.percentile(bow_scores, percentile)) + margin
    voting_threshold = float(np.percentile(voting_scores, percentile)) + margin

    return {
        "bow_threshold": round(bow_threshold, 3),
        "voting_threshold": round(voting_threshold, 3),
        "percentile": percentile,
        "margin": margin,
        "bow_scores_mean": round(float(np.mean(bow_scores)), 4),
        "bow_scores_std": round(float(np.std(bow_scores)), 4),
        "voting_scores_mean": round(float(np.mean(voting_scores)), 4),
        "voting_scores_std": round(float(np.std(voting_scores)), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/synthetic_v3")
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--n-bootstrap", type=int, default=50)
    parser.add_argument("--output", default="results/null_calibration.json")
    args = parser.parse_args()

    set_global_seed(42)

    print("Loading benign sequences from training data...")
    with open(f"{args.data_dir}/multiturn_train.json") as f:
        train_data = json.load(f)

    benign = [s for s in train_data if s.get("label") == 0]
    print(f"  {len(benign)} benign sequences available")

    print(f"\nGenerating {args.n_pairs} null pairs...")
    pairs = generate_null_pairs(benign, n_pairs=args.n_pairs)

    print(f"\nBootstrap resampling ({args.n_bootstrap} iterations)...")
    bow_scores = []
    voting_scores = []

    for i in range(args.n_bootstrap):
        sample = [pairs[j] for j in random.choices(range(len(pairs)), k=len(pairs))]
        bow_f1 = compute_bow_f1_on_pairs(sample)
        voting_f1 = compute_per_turn_voting_f1(sample)
        bow_scores.append(bow_f1)
        voting_scores.append(voting_f1)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.n_bootstrap}] BoW F1: {bow_f1:.4f}, Voting F1: {voting_f1:.4f}")

    thresholds = derive_thresholds(bow_scores, voting_scores)

    print(f"\n{'='*60}")
    print("NULL-PAIR CALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"BoW F1:     {thresholds['bow_scores_mean']:.4f} +/- {thresholds['bow_scores_std']:.4f}")
    print(f"Voting F1:  {thresholds['voting_scores_mean']:.4f} +/- {thresholds['voting_scores_std']:.4f}")
    print(f"\nDerived thresholds (p{thresholds['percentile']} + {thresholds['margin']}):")
    print(f"  BoW gate:    F1 < {thresholds['bow_threshold']}")
    print(f"  Voting gate: F1 < {thresholds['voting_threshold']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "thresholds": thresholds,
        "n_pairs": args.n_pairs,
        "n_bootstrap": args.n_bootstrap,
        "bow_scores": bow_scores,
        "voting_scores": voting_scores,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

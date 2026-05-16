"""Trivial baselines to verify data confounds are closed.

These classifiers exploit known data artifacts. If any achieves high
accuracy, the corresponding confound is still present in the data.

Baselines:
1. BOW (bag-of-words) — learns lexical shortcuts
2. Length-only — conversation length as sole feature
3. First-turn — classifies using only the first turn
4. Last-turn — classifies using only the last turn
5. Generation-method — predicts label from generation_method field

Usage: python scripts/run_trivial_baselines.py --data-dir data/synthetic_v2
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data(data_dir):
    data = {}
    for split in ["train", "val", "test"]:
        path = Path(data_dir) / f"multiturn_{split}.json"
        with open(path) as f:
            data[split] = json.load(f)
        print(f"  {split}: {len(data[split])} sequences")
    return data


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


def run_bow_baseline(data):
    """BOW classifier on concatenated turn text."""
    print("\n=== Baseline 1: Bag-of-Words ===")

    def to_text(seq):
        return " ".join(extract_turns_text(seq))

    train_texts = [to_text(s) for s in data["train"]]
    train_labels = [s["label"] for s in data["train"]]
    test_texts = [to_text(s) for s in data["test"]]
    test_labels = [s["label"] for s in data["test"]]

    vec = CountVectorizer(max_features=5000, stop_words="english")
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_labels)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)
    auc = roc_auc_score(test_labels, probs)
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1:       {f1:.4f}")
    print(f"  AUC:      {auc:.4f}")

    top_features = np.argsort(clf.coef_[0])
    feature_names = vec.get_feature_names_out()
    print(f"  Top attack words: {[feature_names[i] for i in top_features[-10:]]}")
    print(f"  Top benign words: {[feature_names[i] for i in top_features[:10]]}")

    return {"accuracy": acc, "f1": f1, "auc": auc}


def run_length_baseline(data):
    """Classify based on number of turns only."""
    print("\n=== Baseline 2: Length-Only ===")

    def to_length(seq):
        return len(seq.get("turns", []))

    X_train = np.array([[to_length(s)] for s in data["train"]])
    y_train = np.array([s["label"] for s in data["train"]])
    X_test = np.array([[to_length(s)] for s in data["test"]])
    y_test = np.array([s["label"] for s in data["test"]])

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    train_attack_lens = [to_length(s) for s in data["train"] if s["label"] == 1]
    train_benign_lens = [to_length(s) for s in data["train"] if s["label"] == 0]
    print(f"  Avg turns — attack: {np.mean(train_attack_lens):.1f}, "
          f"benign: {np.mean(train_benign_lens):.1f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1:       {f1:.4f}")
    print(f"  AUC:      {auc:.4f}")

    return {"accuracy": acc, "f1": f1, "auc": auc}


def run_single_turn_baseline(data, position="first"):
    """Classify using only the first or last user turn."""
    label = "First-Turn" if position == "first" else "Last-Turn"
    idx = 3 if position == "first" else 4
    print(f"\n=== Baseline {idx}: {label} ===")

    def to_text(seq):
        texts = extract_turns_text(seq)
        if not texts:
            return ""
        return texts[0] if position == "first" else texts[-1]

    train_texts = [to_text(s) for s in data["train"]]
    train_labels = [s["label"] for s in data["train"]]
    test_texts = [to_text(s) for s in data["test"]]
    test_labels = [s["label"] for s in data["test"]]

    vec = CountVectorizer(max_features=3000, stop_words="english")
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_labels)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)
    auc = roc_auc_score(test_labels, probs)
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1:       {f1:.4f}")
    print(f"  AUC:      {auc:.4f}")

    return {"accuracy": acc, "f1": f1, "auc": auc}


def run_method_baseline(data):
    """Predict label from generation_method field alone."""
    print("\n=== Baseline 5: Generation-Method ===")

    method_label_counts = Counter()
    for s in data["train"]:
        method = s.get("generation_method", "unknown")
        label = s["label"]
        method_label_counts[(method, label)] += 1

    print("  Train distribution:")
    methods = sorted(set(m for m, _ in method_label_counts))
    for m in methods:
        atk = method_label_counts.get((m, 1), 0)
        ben = method_label_counts.get((m, 0), 0)
        total = atk + ben
        print(f"    {m}: {atk} attack, {ben} benign ({atk/max(total,1)*100:.0f}% attack)")

    method_to_label = {}
    for m in methods:
        atk = method_label_counts.get((m, 1), 0)
        ben = method_label_counts.get((m, 0), 0)
        method_to_label[m] = 1 if atk > ben else 0

    test_labels = [s["label"] for s in data["test"]]
    preds = [method_to_label.get(s.get("generation_method", "unknown"), 0)
             for s in data["test"]]

    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1:       {f1:.4f}")

    return {"accuracy": acc, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/synthetic_v2")
    parser.add_argument("--output", default="results/trivial_baselines.json")
    args = parser.parse_args()

    print("Loading data...")
    data = load_data(args.data_dir)

    results = {}
    results["bow"] = run_bow_baseline(data)
    results["length_only"] = run_length_baseline(data)
    results["first_turn"] = run_single_turn_baseline(data, "first")
    results["last_turn"] = run_single_turn_baseline(data, "last")
    results["generation_method"] = run_method_baseline(data)

    print("\n" + "=" * 60)
    print("TRIVIAL BASELINE SUMMARY")
    print("=" * 60)
    print(f"{'Baseline':<25} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 60)
    for name, r in results.items():
        acc = f"{r['accuracy']:.4f}"
        f1 = f"{r['f1']:.4f}"
        auc = f"{r.get('auc', 'N/A')}" if isinstance(r.get('auc'), str) else f"{r.get('auc', 0):.4f}" if 'auc' in r else "N/A"
        print(f"  {name:<23} {acc:>10} {f1:>10} {auc:>10}")

    max_acc = max(r["accuracy"] for r in results.values())
    max_f1 = max(r["f1"] for r in results.values())
    print(f"\nMax trivial accuracy: {max_acc:.4f}")
    print(f"Max trivial F1:      {max_f1:.4f}")

    if max_acc > 0.65:
        print("\nWARNING: A trivial baseline exceeds 65% accuracy.")
        print("This suggests residual data confounds.")
    else:
        print("\nAll trivial baselines near chance level.")
        print("Data confounds appear closed.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

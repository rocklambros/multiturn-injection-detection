"""Dual-barrier confound gates for v3 data validation.

Barrier 1: Conversation-level BoW classifiers (unigram, bigram, first-turn, last-turn, length)
Barrier 2: Per-turn voting classifiers (max-vote, mean-vote)

All gates use 5-fold stratified CV on the training split ONLY.
Test split is NEVER used for gating decisions.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def _extract_turns_text(seq):
    turns = seq.get("turns", [])
    texts = []
    for t in turns:
        if isinstance(t, dict):
            if t.get("role", "user") == "user":
                texts.append(t.get("text", ""))
        elif isinstance(t, str):
            texts.append(t)
    return texts


def _concat_text(seq):
    return " ".join(_extract_turns_text(seq))


BARRIER_1_GATES = {
    "unigram_bow": {
        "threshold": 0.60,
        "action": "REJECT",
        "description": "TF-IDF unigram + LogisticRegression on full conversation",
    },
    "bigram_bow": {
        "threshold": 0.65,
        "action": "REJECT",
        "description": "TF-IDF bigram + LogisticRegression on full conversation",
    },
    "first_turn_only": {
        "threshold": 0.58,
        "action": "REJECT",
        "description": "BoW on first user turn only (Chekhov's Gun detector)",
    },
    "last_turn_only": {
        "threshold": 0.65,
        "action": "WARNING",
        "description": "BoW on last user turn only",
    },
    "conversation_length": {
        "threshold": 0.55,
        "action": "REJECT",
        "description": "Logistic regression on word count",
    },
}

BARRIER_2_GATES = {
    "max_vote_bow": {
        "threshold": 0.70,
        "action": "REJECT",
        "description": "Score each turn with BoW, predict attack if max > threshold",
    },
    "mean_vote_bow": {
        "threshold": 0.65,
        "action": "REJECT",
        "description": "Score each turn with BoW, predict attack if mean > threshold",
    },
}


def _run_cv_bow(texts, labels, ngram_range=(1, 1), max_features=5000, n_splits=5):
    """Run stratified K-fold CV with BoW + LogisticRegression."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_f1s = []

    for train_idx, val_idx in skf.split(texts, labels):
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                              stop_words="english")
        X_train = vec.fit_transform(train_texts)
        X_val = vec.transform(val_texts)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, train_labels)

        preds = clf.predict(X_val)
        fold_f1s.append(f1_score(val_labels, preds))

    return np.array(fold_f1s)


def _run_cv_length(sequences, labels, n_splits=5):
    """Run CV with conversation length as sole feature."""
    lengths = np.array([[len(s.get("turns", []))] for s in sequences])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_f1s = []

    for train_idx, val_idx in skf.split(lengths, labels):
        clf = LogisticRegression(random_state=42)
        clf.fit(lengths[train_idx], labels[train_idx])
        preds = clf.predict(lengths[val_idx])
        fold_f1s.append(f1_score(labels[val_idx], preds))

    return np.array(fold_f1s)


def _run_cv_per_turn_voting(sequences, labels, mode="max", n_splits=5):
    """Run CV with per-turn BoW voting."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_f1s = []

    for train_idx, val_idx in skf.split(range(len(sequences)), labels):
        train_seqs = [sequences[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        all_turns = []
        all_turn_labels = []
        for seq, label in zip(train_seqs, train_labels):
            for t in _extract_turns_text(seq):
                all_turns.append(t)
                all_turn_labels.append(label)

        if not all_turns:
            fold_f1s.append(0.0)
            continue

        vec = CountVectorizer(max_features=5000, stop_words="english")
        X_turns = vec.fit_transform(all_turns)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_turns, all_turn_labels)

        val_preds = []
        for seq in val_seqs:
            turns = _extract_turns_text(seq)
            if not turns:
                val_preds.append(0)
                continue
            X_t = vec.transform(turns)
            probs = clf.predict_proba(X_t)[:, 1]
            if mode == "max":
                val_preds.append(1 if max(probs) >= 0.5 else 0)
            else:
                val_preds.append(1 if np.mean(probs) >= 0.5 else 0)

        fold_f1s.append(f1_score(val_labels, val_preds))

    return np.array(fold_f1s)


def run_confound_gates(train_data, calibrated_thresholds=None):
    """Run dual-barrier confound gates using 5-fold CV on training data only.

    NEVER passes test_data. Test split stays untouched until final eval.

    Args:
        train_data: List of sequence dicts with 'turns' and 'label' keys.
        calibrated_thresholds: Optional dict from null-pair calibration to
            override default thresholds.

    Returns:
        tuple: (all_pass: bool, results: dict with per-gate diagnostics).
    """
    labels = np.array([s["label"] for s in train_data])
    n_attack = int(labels.sum())
    n_benign = len(labels) - n_attack
    print(f"\n{'='*70}")
    print(f"DUAL-BARRIER CONFOUND GATES (5-fold CV on {len(train_data)} train sequences)")
    print(f"  {n_attack} attack, {n_benign} benign")
    print(f"{'='*70}")

    if calibrated_thresholds:
        for gate_name, threshold in calibrated_thresholds.items():
            if gate_name in BARRIER_1_GATES:
                BARRIER_1_GATES[gate_name]["threshold"] = threshold
            if gate_name in BARRIER_2_GATES:
                BARRIER_2_GATES[gate_name]["threshold"] = threshold
        print(f"  Using calibrated thresholds: {calibrated_thresholds}")

    results = {}

    # --- Barrier 1: Conversation-level ---
    print(f"\n--- Barrier 1: Conversation-Level Confounds ---")

    # Unigram BoW
    texts = [_concat_text(s) for s in train_data]
    fold_f1s = _run_cv_bow(texts, labels, ngram_range=(1, 1))
    gate = BARRIER_1_GATES["unigram_bow"]
    passed = float(np.mean(fold_f1s)) < gate["threshold"]
    results["unigram_bow"] = {
        "f1_mean": float(np.mean(fold_f1s)),
        "f1_std": float(np.std(fold_f1s)),
        "threshold": gate["threshold"],
        "pass": passed,
        "action": gate["action"],
    }
    status = "PASS" if passed else "FAIL"
    print(f"  Unigram BoW:    F1={np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f} "
          f"(threshold <{gate['threshold']}) [{status}]")

    # Bigram BoW
    fold_f1s = _run_cv_bow(texts, labels, ngram_range=(1, 2), max_features=10000)
    gate = BARRIER_1_GATES["bigram_bow"]
    passed = float(np.mean(fold_f1s)) < gate["threshold"]
    results["bigram_bow"] = {
        "f1_mean": float(np.mean(fold_f1s)),
        "f1_std": float(np.std(fold_f1s)),
        "threshold": gate["threshold"],
        "pass": passed,
        "action": gate["action"],
    }
    status = "PASS" if passed else "FAIL"
    print(f"  Bigram BoW:     F1={np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f} "
          f"(threshold <{gate['threshold']}) [{status}]")

    # First-turn only
    first_texts = []
    for s in train_data:
        turns = _extract_turns_text(s)
        first_texts.append(turns[0] if turns else "")
    fold_f1s = _run_cv_bow(first_texts, labels, ngram_range=(1, 1), max_features=3000)
    gate = BARRIER_1_GATES["first_turn_only"]
    passed = float(np.mean(fold_f1s)) < gate["threshold"]
    results["first_turn_only"] = {
        "f1_mean": float(np.mean(fold_f1s)),
        "f1_std": float(np.std(fold_f1s)),
        "threshold": gate["threshold"],
        "pass": passed,
        "action": gate["action"],
    }
    status = "PASS" if passed else "FAIL"
    print(f"  First-turn:     F1={np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f} "
          f"(threshold <{gate['threshold']}) [{status}]")

    # Last-turn only
    last_texts = []
    for s in train_data:
        turns = _extract_turns_text(s)
        last_texts.append(turns[-1] if turns else "")
    fold_f1s = _run_cv_bow(last_texts, labels, ngram_range=(1, 1), max_features=3000)
    gate = BARRIER_1_GATES["last_turn_only"]
    passed = float(np.mean(fold_f1s)) < gate["threshold"]
    results["last_turn_only"] = {
        "f1_mean": float(np.mean(fold_f1s)),
        "f1_std": float(np.std(fold_f1s)),
        "threshold": gate["threshold"],
        "pass": passed,
        "action": gate["action"],
    }
    status = "PASS" if passed else "WARN"
    print(f"  Last-turn:      F1={np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f} "
          f"(threshold <{gate['threshold']}) [{status}]")

    # Conversation length
    fold_f1s = _run_cv_length(train_data, labels)
    gate = BARRIER_1_GATES["conversation_length"]
    passed = float(np.mean(fold_f1s)) < gate["threshold"]
    results["conversation_length"] = {
        "f1_mean": float(np.mean(fold_f1s)),
        "f1_std": float(np.std(fold_f1s)),
        "threshold": gate["threshold"],
        "pass": passed,
        "action": gate["action"],
    }
    status = "PASS" if passed else "FAIL"
    print(f"  Conv length:    F1={np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f} "
          f"(threshold <{gate['threshold']}) [{status}]")

    # --- Barrier 2: Per-Turn Voting ---
    print(f"\n--- Barrier 2: Per-Turn Voting ---")

    # Max-vote
    fold_f1s = _run_cv_per_turn_voting(train_data, labels, mode="max")
    gate = BARRIER_2_GATES["max_vote_bow"]
    passed = float(np.mean(fold_f1s)) < gate["threshold"]
    results["max_vote_bow"] = {
        "f1_mean": float(np.mean(fold_f1s)),
        "f1_std": float(np.std(fold_f1s)),
        "threshold": gate["threshold"],
        "pass": passed,
        "action": gate["action"],
    }
    status = "PASS" if passed else "FAIL"
    print(f"  Max-vote BoW:   F1={np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f} "
          f"(threshold <{gate['threshold']}) [{status}]")

    # Mean-vote
    fold_f1s = _run_cv_per_turn_voting(train_data, labels, mode="mean")
    gate = BARRIER_2_GATES["mean_vote_bow"]
    passed = float(np.mean(fold_f1s)) < gate["threshold"]
    results["mean_vote_bow"] = {
        "f1_mean": float(np.mean(fold_f1s)),
        "f1_std": float(np.std(fold_f1s)),
        "threshold": gate["threshold"],
        "pass": passed,
        "action": gate["action"],
    }
    status = "PASS" if passed else "FAIL"
    print(f"  Mean-vote BoW:  F1={np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f} "
          f"(threshold <{gate['threshold']}) [{status}]")

    # --- Summary ---
    reject_gates = {k: v for k, v in results.items() if v["action"] == "REJECT" and not v["pass"]}
    warn_gates = {k: v for k, v in results.items() if v["action"] == "WARNING" and not v["pass"]}

    all_critical_pass = len(reject_gates) == 0

    print(f"\n{'='*70}")
    if all_critical_pass:
        print("RESULT: ALL CRITICAL GATES PASSED")
    else:
        print(f"RESULT: {len(reject_gates)} CRITICAL GATE(S) FAILED")
        for name, r in reject_gates.items():
            print(f"  FAILED: {name} — F1={r['f1_mean']:.4f} (need <{r['threshold']})")

    if warn_gates:
        for name, r in warn_gates.items():
            print(f"  WARNING: {name} — F1={r['f1_mean']:.4f} (threshold <{r['threshold']})")

    print(f"{'='*70}")

    return all_critical_pass, results

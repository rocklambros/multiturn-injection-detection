"""Extended evaluation: per-strategy breakdown, A10 voting, turn-order sensitivity,
paired bootstrap tests. Reads saved predictions from run_evaluation.py where possible.

Usage:
    python scripts/run_extended_evaluation.py --all
    python scripts/run_extended_evaluation.py --per-strategy
    python scripts/run_extended_evaluation.py --voting
    python scripts/run_extended_evaluation.py --turn-sensitivity
    python scripts/run_extended_evaluation.py --paired-bootstrap
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_global_seed
from src.evaluation.per_tier import load_tier_labels
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

RESULTS_DIR = Path("results/v3_evaluation")
DATA_DIR = "data/synthetic_v3"
TEST_FILE = f"{DATA_DIR}/multiturn_test.json"


def load_test_data():
    with open(TEST_FILE) as f:
        return json.load(f)


def per_strategy_breakdown(models=None):
    """T5.2: Per-strategy F1/precision/recall for each model."""
    print("\n" + "=" * 70)
    print("T5.2: PER-STRATEGY F1 BREAKDOWN")
    print("=" * 70)

    test_data = load_test_data()
    strategies_per_sample = []
    for seq in test_data:
        if seq["label"] == 1:
            strategies_per_sample.append(seq.get("strategy", "unknown"))
        else:
            strategies_per_sample.append("benign")
    strategies_per_sample = np.array(strategies_per_sample)

    if models is None:
        models = [p.stem.replace("_predictions", "") for p in RESULTS_DIR.glob("*_predictions.npz")]
        models = sorted(models)

    all_strategies = ["fragment_distributed", "gradual_escalation", "context_priming", "instruction_layering"]
    results = {}

    for model_name in models:
        pred_path = RESULTS_DIR / f"{model_name}_predictions.npz"
        if not pred_path.exists():
            print(f"  SKIP {model_name}: no saved predictions")
            continue

        data = np.load(pred_path)
        predictions, labels = data["predictions"], data["labels"]
        preds_binary = (predictions >= 0.5).astype(int)

        model_results = {}
        print(f"\n  {model_name}:")
        print(f"    {'Strategy':<25} {'N':>6} {'F1':>8} {'Prec':>8} {'Rec':>8}")
        print("    " + "-" * 55)

        for strategy in all_strategies:
            attack_mask = strategies_per_sample == strategy
            benign_mask = strategies_per_sample == "benign"
            mask = attack_mask | benign_mask

            if mask.sum() == 0:
                continue

            sub_labels = labels[mask]
            sub_preds = preds_binary[mask]

            f1 = f1_score(sub_labels, sub_preds, zero_division=0)
            prec = precision_score(sub_labels, sub_preds, zero_division=0)
            rec = recall_score(sub_labels, sub_preds, zero_division=0)
            n_attack = attack_mask.sum()

            model_results[strategy] = {"f1": f1, "precision": prec, "recall": rec, "n_attack": int(n_attack)}
            print(f"    {strategy:<25} {n_attack:>6} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f}")

        results[model_name] = model_results

    output_path = RESULTS_DIR / "per_strategy_breakdown.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_path}")
    return results


def a10_voting_baselines():
    """A10: Turn-level voting using frozen GRU encoder."""
    print("\n" + "=" * 70)
    print("A10: TURN-LEVEL VOTING BASELINES")
    print("=" * 70)

    from src.utils.tokenizer import load_vocab, encode_multiturn
    from src.data.loader import MultiTurnDataset
    from src.models.single_turn import GRUClassifier
    from src.models.ablations import TurnLevelVoting

    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    test_data = load_test_data()
    tier_labels = load_tier_labels(TEST_FILE)
    labels_arr = np.array([s["label"] for s in test_data])

    vocab = load_vocab("models/vocab.json")
    turns_list = [
        [t["text"] for t in seq["turns"] if t.get("role", "user") == "user"]
        for seq in test_data
    ]
    token_ids, masks = encode_multiturn(vocab, turns_list, max_turns=10, max_len=256)
    labels_tensor = torch.FloatTensor([s["label"] for s in test_data])
    dataset = MultiTurnDataset(token_ids, masks, labels_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    encoder = GRUClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_dim=64, dropout_rate=0.3, dense_dim=32)
    encoder.load_state_dict(torch.load("models/v3_gru_retrain.pt", map_location=device, weights_only=True))
    encoder = encoder.to(device)
    encoder.eval()

    voting = TurnLevelVoting(encoder, device)

    results = {}
    voting_methods = {
        "a10_max_vote": lambda x, m: voting.predict_max_vote(x, m),
        "a10_mean_vote": lambda x, m: voting.predict_mean_vote(x, m),
        "a10_top3_mean": lambda x, m: voting.predict_top_k_mean(x, m, k=3),
    }

    for method_name, predict_fn in voting_methods.items():
        print(f"\n  Running {method_name}...")
        all_preds = []
        all_scores = []

        with torch.no_grad():
            for batch in loader:
                inputs, mask, batch_labels = batch
                inputs = inputs.to(device)
                mask = mask.to(device)
                preds, scores = predict_fn(inputs, mask)
                all_preds.append(preds.cpu().numpy())
                all_scores.append(scores.cpu().numpy())

        preds = np.concatenate(all_preds)
        scores = np.concatenate(all_scores)

        np.savez(RESULTS_DIR / f"{method_name}_predictions.npz", predictions=scores, labels=labels_arr)

        overall = {
            "accuracy": float(accuracy_score(labels_arr, preds)),
            "f1": float(f1_score(labels_arr, preds, zero_division=0)),
            "precision": float(precision_score(labels_arr, preds, zero_division=0)),
            "recall": float(recall_score(labels_arr, preds, zero_division=0)),
            "n": int(len(labels_arr)),
        }
        try:
            overall["auc"] = float(roc_auc_score(labels_arr, scores))
        except ValueError:
            pass

        per_tier = {}
        tier_arr = np.array(tier_labels)
        for tier in sorted(set(tier_labels)):
            tmask = tier_arr == tier
            if tmask.sum() == 0:
                continue
            tier_preds = preds[tmask]
            tier_labels_sub = labels_arr[tmask]
            tier_scores = scores[tmask]
            per_tier[tier] = {
                "accuracy": float(accuracy_score(tier_labels_sub, tier_preds)),
                "f1": float(f1_score(tier_labels_sub, tier_preds, zero_division=0)),
                "precision": float(precision_score(tier_labels_sub, tier_preds, zero_division=0)),
                "recall": float(recall_score(tier_labels_sub, tier_preds, zero_division=0)),
                "n": int(tmask.sum()),
            }
            try:
                per_tier[tier]["auc"] = float(roc_auc_score(tier_labels_sub, tier_scores))
            except ValueError:
                pass

        results[method_name] = {"overall": overall, "per_tier": per_tier}

        print(f"    Overall: Acc={overall['accuracy']:.4f} F1={overall['f1']:.4f} AUC={overall.get('auc', 0):.4f}")
        for tier in ["easy", "medium", "hard", "adversarial"]:
            if tier in per_tier:
                t = per_tier[tier]
                print(f"    {tier:>13}: F1={t['f1']:.4f} Prec={t['precision']:.4f} Rec={t['recall']:.4f}")

    output_path = RESULTS_DIR / "a10_voting_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_path}")
    return results


def turn_order_sensitivity():
    """T5.1: For correctly classified attacks, shuffle turns and re-predict."""
    print("\n" + "=" * 70)
    print("T5.1: TURN-ORDER SENSITIVITY ANALYSIS")
    print("=" * 70)

    from src.utils.tokenizer import load_vocab, encode_multiturn
    from src.data.loader import MultiTurnDataset
    from src.models.single_turn import GRUClassifier
    from src.models.multi_turn import MultiTurnClassifier

    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = load_test_data()
    labels_arr = np.array([s["label"] for s in test_data])
    tier_labels = load_tier_labels(TEST_FILE)

    pred_path = RESULTS_DIR / "iter5_predictions.npz"
    if not pred_path.exists():
        print("  ERROR: iter5 predictions not saved. Run run_evaluation.py first.")
        return None

    orig_data = np.load(pred_path)
    orig_predictions = orig_data["predictions"]
    orig_preds_binary = (orig_predictions >= 0.5).astype(int)

    correctly_classified_attacks = (orig_preds_binary == 1) & (labels_arr == 1)
    n_correct = correctly_classified_attacks.sum()
    print(f"  Correctly classified attacks: {n_correct}/{labels_arr.sum()}")

    vocab = load_vocab("models/vocab.json")

    turns_list_original = [
        [t["text"] for t in seq["turns"] if t.get("role", "user") == "user"]
        for seq in test_data
    ]

    rng = np.random.RandomState(42)
    turns_list_shuffled = []
    for i, turns in enumerate(turns_list_original):
        if correctly_classified_attacks[i] and len(turns) > 1:
            shuffled = list(turns)
            rng.shuffle(shuffled)
            turns_list_shuffled.append(shuffled)
        else:
            turns_list_shuffled.append(turns)

    token_ids, masks = encode_multiturn(vocab, turns_list_shuffled, max_turns=10, max_len=256)
    labels_tensor = torch.FloatTensor(labels_arr.tolist())
    dataset = MultiTurnDataset(token_ids, masks, labels_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    encoder = GRUClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_dim=64, dropout_rate=0.3, dense_dim=32)
    encoder.load_state_dict(torch.load("models/v3_gru_retrain.pt", map_location=device, weights_only=True))
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    model = MultiTurnClassifier(turn_encoder=encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3)
    model.load_state_dict(torch.load("models/v3_iter5_multiturn.pt", map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            inputs, mask, _ = batch
            logits = model(inputs.to(device), mask.to(device))
            probs = torch.sigmoid(logits).squeeze(-1)
            all_probs.append(probs.cpu().numpy())

    shuffled_predictions = np.concatenate(all_probs)
    shuffled_preds_binary = (shuffled_predictions >= 0.5).astype(int)

    flipped = correctly_classified_attacks & (shuffled_preds_binary == 0)
    n_flipped = flipped.sum()
    flip_rate = n_flipped / n_correct if n_correct > 0 else 0.0

    tier_arr = np.array(tier_labels)
    tier_flip_rates = {}
    for tier in sorted(set(tier_labels)):
        tmask = tier_arr == tier
        tier_correct = (correctly_classified_attacks & tmask).sum()
        tier_flipped = (flipped & tmask).sum()
        tier_flip_rates[tier] = {
            "n_correct": int(tier_correct),
            "n_flipped": int(tier_flipped),
            "flip_rate": float(tier_flipped / tier_correct) if tier_correct > 0 else 0.0,
        }

    results = {
        "n_correctly_classified_attacks": int(n_correct),
        "n_flipped_after_shuffle": int(n_flipped),
        "overall_flip_rate": float(flip_rate),
        "per_tier_flip_rates": tier_flip_rates,
        "original_attack_f1": float(f1_score(labels_arr, orig_preds_binary)),
        "shuffled_attack_f1": float(f1_score(labels_arr, shuffled_preds_binary)),
    }

    print(f"\n  Results:")
    print(f"    Correctly classified attacks: {n_correct}")
    print(f"    Flipped after shuffling:      {n_flipped} ({flip_rate:.1%})")
    print(f"    Original F1:                  {results['original_attack_f1']:.4f}")
    print(f"    Shuffled F1:                  {results['shuffled_attack_f1']:.4f}")
    print(f"    F1 drop:                      {results['original_attack_f1'] - results['shuffled_attack_f1']:.4f}")
    print(f"\n    Per-tier flip rates:")
    for tier in ["easy", "medium", "hard", "adversarial"]:
        if tier in tier_flip_rates:
            t = tier_flip_rates[tier]
            print(f"      {tier:>13}: {t['n_flipped']}/{t['n_correct']} = {t['flip_rate']:.1%}")

    output_path = RESULTS_DIR / "turn_order_sensitivity.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_path}")
    return results


def paired_bootstrap_test(pred_a, pred_b, labels, n_bootstrap=1000, metric_fn=None):
    """Paired bootstrap test: is model A significantly better than model B?"""
    if metric_fn is None:
        metric_fn = lambda y, yhat: f1_score(y, (yhat >= 0.5).astype(int), zero_division=0)

    rng = np.random.RandomState(42)
    n = len(labels)
    observed_diff = metric_fn(labels, pred_a) - metric_fn(labels, pred_b)

    count_better = 0
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diff = metric_fn(labels[idx], pred_a[idx]) - metric_fn(labels[idx], pred_b[idx])
        diffs.append(diff)
        if diff > 0:
            count_better += 1

    diffs = np.array(diffs)
    p_value = 1.0 - (count_better / n_bootstrap)

    return {
        "observed_diff": float(observed_diff),
        "mean_diff": float(np.mean(diffs)),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "p_value": float(p_value),
        "significant_at_005": p_value < 0.05,
    }


def run_paired_bootstrap_tests():
    """T5.5: Paired bootstrap significance tests for key comparisons."""
    print("\n" + "=" * 70)
    print("T5.5: PAIRED BOOTSTRAP SIGNIFICANCE TESTS")
    print("=" * 70)

    comparisons = [
        ("iter5", "a10_max_vote", "Temporal LSTM vs A10 max-vote"),
        ("iter5", "a10_mean_vote", "Temporal LSTM vs A10 mean-vote"),
        ("iter5", "a10_top3_mean", "Temporal LSTM vs A10 top-3-mean"),
        ("iter5", "distilbert_hier", "Temporal LSTM vs PM-1a Hierarchical DistilBERT"),
        ("iter5", "distilbert_concat", "Temporal LSTM vs PM-1b Concatenated DistilBERT"),
        ("iter5", "ablation_shuffled", "Temporal LSTM (ordered) vs Shuffled turns"),
        ("iter6", "iter5", "Attention (iter6) vs Plain LSTM (iter5)"),
        ("ablation_continuation", "ablation_prefix", "Continuation-only vs Prefix-only"),
    ]

    results = {}

    for model_a, model_b, description in comparisons:
        pred_a_path = RESULTS_DIR / f"{model_a}_predictions.npz"
        pred_b_path = RESULTS_DIR / f"{model_b}_predictions.npz"

        if not pred_a_path.exists() or not pred_b_path.exists():
            missing = model_a if not pred_a_path.exists() else model_b
            print(f"\n  SKIP: {description} — missing predictions for {missing}")
            continue

        data_a = np.load(pred_a_path)
        data_b = np.load(pred_b_path)

        test_result = paired_bootstrap_test(
            data_a["predictions"], data_b["predictions"], data_a["labels"],
            n_bootstrap=1000,
        )

        key = f"{model_a}_vs_{model_b}"
        results[key] = {
            "description": description,
            "model_a": model_a,
            "model_b": model_b,
            **test_result,
        }

        sig = "***" if test_result["significant_at_005"] else "n.s."
        print(f"\n  {description}:")
        print(f"    F1 diff: {test_result['observed_diff']:+.4f} "
              f"[{test_result['ci_lower']:+.4f}, {test_result['ci_upper']:+.4f}] "
              f"p={test_result['p_value']:.4f} {sig}")

    output_path = RESULTS_DIR / "paired_bootstrap_tests.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Extended evaluation analyses")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--per-strategy", action="store_true")
    parser.add_argument("--voting", action="store_true")
    parser.add_argument("--turn-sensitivity", action="store_true")
    parser.add_argument("--paired-bootstrap", action="store_true")
    args = parser.parse_args()

    run_all = args.all or not any([args.per_strategy, args.voting, args.turn_sensitivity, args.paired_bootstrap])

    set_global_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if run_all or args.per_strategy:
        per_strategy_breakdown()

    if run_all or args.voting:
        a10_voting_baselines()

    if run_all or args.turn_sensitivity:
        turn_order_sensitivity()

    if run_all or args.paired_bootstrap:
        run_paired_bootstrap_tests()

    print("\n" + "=" * 70)
    print("EXTENDED EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

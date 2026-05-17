"""Per-tier evaluation pipeline for all v3 models.

Loads each trained model, runs inference on the test set (5130 sequences),
and computes metrics (accuracy, F1, precision, recall, AUC) broken down by
difficulty tier (easy, medium, hard, adversarial) with bootstrap 95% CIs.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --models iter5,iter6
    python scripts/run_evaluation.py --bootstrap-samples 2000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_global_seed
from src.evaluation.per_tier import evaluate_per_tier, load_tier_labels


MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/v3_evaluation")
DATA_DIR = "data/synthetic_v3"
TEST_FILE = f"{DATA_DIR}/multiturn_test.json"


def load_test_data_raw():
    with open(TEST_FILE) as f:
        return json.load(f)


def build_gru_multiturn_loader(test_data, vocab, batch_size=32, max_turns=10, max_len=256):
    from src.utils.tokenizer import encode_multiturn
    from src.data.loader import MultiTurnDataset

    turns_list = [
        [t["text"] for t in seq["turns"] if t.get("role", "user") == "user"]
        for seq in test_data
    ]
    labels_list = [seq["label"] for seq in test_data]

    token_ids, masks = encode_multiturn(vocab, turns_list, max_turns=max_turns, max_len=max_len)
    labels = torch.FloatTensor(labels_list)

    dataset = MultiTurnDataset(token_ids, masks, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


def build_distilbert_hier_loader(test_data, batch_size=8, max_turns=10, max_len=128):
    from transformers import DistilBertTokenizer
    from src.data.loader import DistilBertMultiTurnDataset

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = DistilBertMultiTurnDataset(test_data, tokenizer, max_turns, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


def build_distilbert_concat_loader(test_data, batch_size=8, max_length=512):
    from transformers import DistilBertTokenizer
    from src.data.loader import ConcatDistilBertDataset

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = ConcatDistilBertDataset(test_data, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


def load_turn_encoder(device):
    from src.utils.tokenizer import load_vocab
    from src.models.single_turn import GRUClassifier

    vocab = load_vocab("models/vocab.json")
    model = GRUClassifier(
        vocab_size=len(vocab), embedding_dim=128, hidden_dim=64,
        dropout_rate=0.3, dense_dim=32,
    )
    model.load_state_dict(torch.load(
        MODELS_DIR / "v3_gru_retrain.pt", map_location=device, weights_only=True,
    ))
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, vocab


def load_autoencoder(device):
    from src.models.ablations import TurnAutoencoder

    ae = TurnAutoencoder(input_dim=32, bottleneck_dim=32)
    ae.load_state_dict(torch.load(
        MODELS_DIR / "v3_turn_autoencoder.pt", map_location=device, weights_only=True,
    ))
    ae = ae.to(device)
    ae.eval()
    for param in ae.parameters():
        param.requires_grad = False
    return ae


def load_model(model_name, turn_encoder, device):
    from src.models.multi_turn import MultiTurnClassifier
    from src.models.attention import MultiTurnAttentionClassifier
    from src.models.transformer_multiturn import HierarchicalDistilBERT
    from src.models.concat_distilbert import ConcatenatedDistilBERT
    from src.models.ablations import (
        ShuffledTurnsClassifier, ReversedTurnsClassifier,
        PrefixOnlyClassifier, ContinuationOnlyClassifier,
        AutoencoderMultiTurnClassifier, MeanPoolClassifier, MaxPoolClassifier,
    )

    builders = {
        "iter5": lambda: MultiTurnClassifier(
            turn_encoder=turn_encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3,
        ),
        "iter6": lambda: MultiTurnAttentionClassifier(
            turn_encoder=turn_encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3,
        ),
        "distilbert_hier": lambda: HierarchicalDistilBERT(
            num_attention_heads=4, cross_turn_layers=2, max_turns=10,
            dropout_rate=0.3, freeze_bert=True,
        ),
        "distilbert_concat": lambda: ConcatenatedDistilBERT(
            max_length=512, dropout_rate=0.3, freeze_bert=False,
        ),
        "ablation_shuffled": lambda: ShuffledTurnsClassifier(
            turn_encoder=turn_encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3,
        ),
        "ablation_reversed": lambda: ReversedTurnsClassifier(
            turn_encoder=turn_encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3,
        ),
        "ablation_prefix": lambda: PrefixOnlyClassifier(
            turn_encoder=turn_encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3,
        ),
        "ablation_continuation": lambda: ContinuationOnlyClassifier(
            turn_encoder=turn_encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3,
        ),
        "ablation_autoencoder": lambda: AutoencoderMultiTurnClassifier(
            base_turn_encoder=turn_encoder, autoencoder=load_autoencoder(device),
            turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3,
        ),
        "ablation_mean_pool": lambda: MeanPoolClassifier(
            turn_encoder=turn_encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3,
        ),
        "ablation_max_pool": lambda: MaxPoolClassifier(
            turn_encoder=turn_encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3,
        ),
    }

    if model_name not in builders:
        raise ValueError(f"Unknown model: {model_name}")

    model = builders[model_name]()

    pt_name_map = {
        "iter5": "v3_iter5_multiturn.pt",
        "iter6": "v3_iter6_attention.pt",
        "distilbert_hier": "v3_distilbert_hier.pt",
        "distilbert_concat": "v3_distilbert_concat.pt",
        "ablation_shuffled": "v3_ablation_shuffled.pt",
        "ablation_reversed": "v3_ablation_reversed.pt",
        "ablation_prefix": "v3_ablation_prefix.pt",
        "ablation_continuation": "v3_ablation_continuation.pt",
        "ablation_autoencoder": "v3_ablation_autoencoder.pt",
        "ablation_mean_pool": "v3_ablation_mean_pool.pt",
        "ablation_max_pool": "v3_ablation_max_pool.pt",
    }

    pt_path = MODELS_DIR / pt_name_map[model_name]
    model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def run_inference_gru(model, loader, device):
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            inputs, mask, labels = batch
            inputs = inputs.to(device)
            mask = mask.to(device)

            logits = model(inputs, mask)
            probs = torch.sigmoid(logits).squeeze(-1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def run_inference_distilbert_hier(model, loader, device):
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, turn_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            turn_mask = turn_mask.to(device)

            logits = model(input_ids, attention_mask, turn_mask)
            probs = torch.sigmoid(logits).squeeze(-1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def run_inference_distilbert_concat(model, loader, device):
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).squeeze(-1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def bootstrap_ci(predictions, labels, tier_labels, n_bootstrap=1000, ci=0.95):
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

    rng = np.random.RandomState(42)
    n = len(labels)
    alpha = (1.0 - ci) / 2.0

    results = {"overall": {}, "per_tier": {}}
    preds_binary = (predictions >= 0.5).astype(int)

    metrics_fns = {
        "accuracy": lambda y, yp, yprob: accuracy_score(y, yp),
        "f1": lambda y, yp, yprob: f1_score(y, yp, zero_division=0),
        "auc": lambda y, yp, yprob: roc_auc_score(y, yprob) if len(set(y)) > 1 else float("nan"),
    }

    for metric_name, fn in metrics_fns.items():
        boot_vals = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            try:
                val = fn(labels[idx], preds_binary[idx], predictions[idx])
                if not np.isnan(val):
                    boot_vals.append(val)
            except ValueError:
                continue
        if boot_vals:
            results["overall"][metric_name] = {
                "mean": float(np.mean(boot_vals)),
                "ci_lower": float(np.percentile(boot_vals, alpha * 100)),
                "ci_upper": float(np.percentile(boot_vals, (1 - alpha) * 100)),
            }

    tiers = sorted(set(tier_labels))
    tier_arr = np.array(tier_labels)
    for tier in tiers:
        tier_mask = tier_arr == tier
        tier_n = tier_mask.sum()
        if tier_n == 0:
            continue

        tier_results = {}
        for metric_name, fn in metrics_fns.items():
            boot_vals = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                idx_tier = idx[tier_arr[idx] == tier]
                if len(idx_tier) < 2:
                    continue
                try:
                    val = fn(labels[idx_tier], preds_binary[idx_tier], predictions[idx_tier])
                    if not np.isnan(val):
                        boot_vals.append(val)
                except ValueError:
                    continue
            if boot_vals:
                tier_results[metric_name] = {
                    "mean": float(np.mean(boot_vals)),
                    "ci_lower": float(np.percentile(boot_vals, alpha * 100)),
                    "ci_upper": float(np.percentile(boot_vals, (1 - alpha) * 100)),
                }
        results["per_tier"][tier] = tier_results

    return results


def evaluate_cosine_baseline(test_data, train_data, tier_labels):
    from src.models.ablations import CosineSimilarityBaseline

    baseline = CosineSimilarityBaseline()
    baseline.fit(train_data)
    preds_binary, preds_prob = baseline.predict(test_data)
    labels = np.array([s["label"] for s in test_data])

    return preds_prob, labels


GRU_MODELS = [
    "iter5", "iter6",
    "ablation_shuffled", "ablation_reversed",
    "ablation_prefix", "ablation_continuation",
    "ablation_autoencoder", "ablation_mean_pool", "ablation_max_pool",
]

ALL_MODELS = GRU_MODELS + ["distilbert_hier", "distilbert_concat"]


def main():
    parser = argparse.ArgumentParser(description="Per-tier evaluation for v3 models")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model list (default: all)")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-cosine", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-from-saved", action="store_true",
                        help="Compute bootstrap CIs from saved .npz predictions (no inference)")
    args = parser.parse_args()

    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading test data...")
    test_data = load_test_data_raw()
    tier_labels = load_tier_labels(TEST_FILE)
    print(f"  {len(test_data)} sequences, {len(set(tier_labels))} tiers")

    models_to_eval = args.models.split(",") if args.models else ALL_MODELS

    if args.bootstrap_from_saved:
        all_results = {}
        for model_name in models_to_eval:
            pred_path = RESULTS_DIR / f"{model_name}_predictions.npz"
            if not pred_path.exists():
                print(f"SKIP {model_name}: no saved predictions at {pred_path}")
                continue

            print(f"\nBootstrap CIs for {model_name} (from saved predictions)...")
            data = np.load(pred_path)
            predictions, labels = data["predictions"], data["labels"]

            ci_results = bootstrap_ci(
                predictions, labels, tier_labels,
                n_bootstrap=args.bootstrap_samples,
            )

            ci_path = RESULTS_DIR / f"{model_name}_bootstrap_ci.json"
            with open(ci_path, "w") as f:
                json.dump(ci_results, f, indent=2)

            ci = ci_results.get("overall", {}).get("f1", {})
            if ci:
                print(f"  F1: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
            print(f"  Saved to {ci_path}")
        return

    turn_encoder = None
    vocab = None
    gru_loader = None
    hier_loader = None
    concat_loader = None

    needs_gru = any(m in GRU_MODELS for m in models_to_eval)
    needs_hier = "distilbert_hier" in models_to_eval
    needs_concat = "distilbert_concat" in models_to_eval

    if needs_gru:
        print("\nLoading GRU turn encoder and building multi-turn test loader...")
        turn_encoder, vocab = load_turn_encoder(device)
        gru_loader = build_gru_multiturn_loader(
            test_data, vocab, batch_size=args.batch_size,
        )
        print(f"  GRU loader: {len(gru_loader)} batches")

    if needs_hier:
        print("\nBuilding DistilBERT hierarchical test loader...")
        hier_loader = build_distilbert_hier_loader(test_data, batch_size=8)
        print(f"  Hierarchical loader: {len(hier_loader)} batches")

    if needs_concat:
        print("\nBuilding DistilBERT concat test loader...")
        concat_loader = build_distilbert_concat_loader(test_data, batch_size=8)
        print(f"  Concat loader: {len(concat_loader)} batches")

    all_results = {}

    for model_name in models_to_eval:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*70}")

        t0 = time.time()

        model = load_model(model_name, turn_encoder, device)

        if model_name == "distilbert_hier":
            predictions, labels = run_inference_distilbert_hier(model, hier_loader, device)
        elif model_name == "distilbert_concat":
            predictions, labels = run_inference_distilbert_concat(model, concat_loader, device)
        else:
            predictions, labels = run_inference_gru(model, gru_loader, device)

        elapsed = time.time() - t0
        print(f"Inference: {elapsed:.1f}s")

        pred_path = RESULTS_DIR / f"{model_name}_predictions.npz"
        np.savez(pred_path, predictions=predictions, labels=labels)
        print(f"  Predictions saved to {pred_path}")

        output_path = RESULTS_DIR / f"{model_name}_per_tier.json"
        tier_results = evaluate_per_tier(predictions, labels, tier_labels, str(output_path))

        if not args.skip_bootstrap:
            print(f"\nComputing bootstrap CIs ({args.bootstrap_samples} samples)...")
            ci_results = bootstrap_ci(
                predictions, labels, tier_labels,
                n_bootstrap=args.bootstrap_samples,
            )
            tier_results["bootstrap_ci"] = ci_results

            ci_path = RESULTS_DIR / f"{model_name}_bootstrap_ci.json"
            with open(ci_path, "w") as f:
                json.dump(ci_results, f, indent=2)
            print(f"  Saved to {ci_path}")

        all_results[model_name] = tier_results

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not args.skip_cosine:
        print(f"\n{'='*70}")
        print("Evaluating: B6 cosine-similarity baseline")
        print(f"{'='*70}")

        with open(f"{DATA_DIR}/multiturn_train.json") as f:
            train_data = json.load(f)

        cos_predictions, cos_labels = evaluate_cosine_baseline(
            test_data, train_data, tier_labels,
        )

        output_path = RESULTS_DIR / "cosine_baseline_per_tier.json"
        cos_results = evaluate_per_tier(cos_predictions, cos_labels, tier_labels, str(output_path))

        if not args.skip_bootstrap:
            ci_results = bootstrap_ci(
                cos_predictions, cos_labels, tier_labels,
                n_bootstrap=args.bootstrap_samples,
            )
            cos_results["bootstrap_ci"] = ci_results

        all_results["cosine_baseline"] = cos_results

    summary_path = RESULTS_DIR / "all_models_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull summary saved to {summary_path}")

    print(f"\n{'='*70}")
    print("SUMMARY: Overall Test F1 Across All Models")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'AUC':>8}")
    print("-" * 70)

    for name in sorted(all_results.keys()):
        r = all_results[name]["overall"]
        ci_str = ""
        if "bootstrap_ci" in all_results[name]:
            ci = all_results[name]["bootstrap_ci"].get("overall", {}).get("f1", {})
            if ci:
                ci_str = f" [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]"
        print(f"  {name:<23} {r['accuracy']:>8.4f} {r['f1']:>8.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r.get('auc', 0):>8.4f}{ci_str}")

    print(f"\n{'='*70}")
    print("SUMMARY: Per-Tier F1 Comparison")
    print(f"{'='*70}")
    tiers = ["easy", "medium", "hard", "adversarial"]
    header = f"{'Model':<25}" + "".join(f"{t:>12}" for t in tiers)
    print(header)
    print("-" * 73)

    for name in sorted(all_results.keys()):
        per_tier = all_results[name].get("per_tier", {})
        vals = []
        for t in tiers:
            if t in per_tier:
                vals.append(f"{per_tier[t]['f1']:>12.4f}")
            else:
                vals.append(f"{'N/A':>12}")
        print(f"  {name:<23}" + "".join(vals))


if __name__ == "__main__":
    main()

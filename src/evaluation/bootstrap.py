"""Bootstrap confidence intervals and paired significance tests."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def bootstrap_ci(y_true, y_pred, metric_fn=f1_score, n_resamples=1000,
                 confidence=0.95, seed=42):
    """Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        metric_fn: Metric function (y_true, y_pred) -> float.
        n_resamples: Number of bootstrap resamples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        Dict with point_estimate, ci_lower, ci_upper, std.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)

    scores = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except ValueError:
            scores.append(0.0)

    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return {
        "point_estimate": float(point),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "std": float(np.std(scores)),
        "n_resamples": n_resamples,
        "confidence": confidence,
    }


def paired_bootstrap_test(y_true, y_pred_a, y_pred_b, metric_fn=f1_score,
                          n_resamples=1000, seed=42):
    """Paired bootstrap significance test between two models.

    Tests H0: metric(A) == metric(B).

    Args:
        y_true: Shared ground truth.
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.
        metric_fn: Metric function.
        n_resamples: Bootstrap resamples.
        seed: Random seed.

    Returns:
        Dict with delta, p_value, significant (at 0.05).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    delta_obs = metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b)

    count_greater = 0
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        try:
            d = metric_fn(y_true[idx], y_pred_a[idx]) - metric_fn(y_true[idx], y_pred_b[idx])
        except ValueError:
            d = 0.0
        if d <= 0:
            count_greater += 1

    p_value = count_greater / n_resamples

    return {
        "delta": float(delta_obs),
        "p_value": float(p_value),
        "significant_at_005": p_value < 0.05,
        "model_a_better": delta_obs > 0,
        "n_resamples": n_resamples,
    }


def compute_all_cis(y_true, y_pred, y_prob=None, n_resamples=1000):
    """Compute bootstrap CIs for F1, precision, recall.

    Args:
        y_true: Ground truth.
        y_pred: Predictions.
        y_prob: Optional probabilities (for threshold analysis).
        n_resamples: Bootstrap resamples.

    Returns:
        Dict with CIs for each metric.
    """
    return {
        "f1": bootstrap_ci(y_true, y_pred, f1_score, n_resamples),
        "precision": bootstrap_ci(y_true, y_pred, precision_score, n_resamples),
        "recall": bootstrap_ci(y_true, y_pred, recall_score, n_resamples),
    }

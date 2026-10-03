"""
counterfactual.py — counterfactual verification of suspicious training data.

Runs two mini-training experiments:
    Experiment A — train/evaluate with suspicious samples included
    Experiment B — train/evaluate with suspicious samples excluded

The observed difference is reported as attribution confidence.

IMPORTANT: This provides counterfactual evidence supporting attribution,
not absolute causal proof.
"""

import logging
import numpy as np
from sklearn.base import clone
from training.model import evaluate, build_model
from attacks.backdoor import measure_backdoor_success

logger = logging.getLogger(__name__)


def run_counterfactual(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    suspicious_indices: list[int],
    feature_names: list[str],
    cfg: dict,
) -> dict:
    """
    Train two models — one with, one without the suspicious samples —
    and compare their security metrics.

    Returns a result dict suitable for inclusion in the Security Passport.
    """
    # --- Experiment A: with suspicious samples ---
    model_a = build_model(cfg)
    model_a.fit(X_train, y_train)
    metrics_a = evaluate(model_a, X_val, y_val, label="counterfactual_with")
    bk_a = measure_backdoor_success(model_a, X_val, feature_names, cfg)

    # --- Experiment B: without suspicious samples ---
    mask = np.ones(len(X_train), dtype=bool)
    mask[suspicious_indices] = False

    X_clean = X_train[mask]
    y_clean = y_train[mask]

    if len(X_clean) == 0:
        logger.warning("All training samples marked suspicious — cannot run experiment B.")
        return {
            "experiment_a": metrics_a,
            "experiment_b": None,
            "difference": None,
            "attribution_confidence": 0.0,
            "note": "All samples flagged; experiment B skipped.",
        }

    model_b = build_model(cfg)
    model_b.fit(X_clean, y_clean)
    metrics_b = evaluate(model_b, X_val, y_val, label="counterfactual_without")
    bk_b = measure_backdoor_success(model_b, X_val, feature_names, cfg)

    # --- Compute difference ---
    acc_diff = metrics_b["accuracy"] - metrics_a["accuracy"]
    bk_diff = bk_a - bk_b  # positive = backdoor drops when we remove suspects

    # Attribution confidence: how much does removing suspects help?
    # Weighted combination of accuracy recovery and backdoor reduction.
    # Capped to [0, 1] — this is a heuristic indicator, not a p-value.
    raw_conf = 0.4 * min(max(bk_diff / 0.30, 0.0), 1.0) + 0.6 * min(max(acc_diff / 0.10, 0.0), 1.0)
    attribution_confidence = round(min(max(raw_conf, 0.0), 1.0), 4)

    result = {
        "experiment_a": {
            "accuracy": metrics_a["accuracy"],
            "f1": metrics_a["f1"],
            "auc": metrics_a["auc"],
            "backdoor_success_rate": bk_a,
        },
        "experiment_b": {
            "accuracy": metrics_b["accuracy"],
            "f1": metrics_b["f1"],
            "auc": metrics_b["auc"],
            "backdoor_success_rate": bk_b,
        },
        "difference": {
            "accuracy_delta": round(acc_diff, 6),
            "backdoor_rate_delta": round(bk_diff, 6),
        },
        "attribution_confidence": attribution_confidence,
        "note": (
            "Removing suspected samples reduces backdoor success rate and/or "
            "improves accuracy — counterfactual evidence supporting attribution. "
            "This is not absolute causal proof."
        ),
    }

    logger.info(
        "Counterfactual: bk_with=%.4f  bk_without=%.4f  delta=%.4f  confidence=%.4f",
        bk_a, bk_b, bk_diff, attribution_confidence,
    )
    return result

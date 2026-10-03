"""
model.py — baseline model factory and evaluation helpers.
"""

import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

logger = logging.getLogger(__name__)


def build_model(cfg: dict):
    """Return an untrained scikit-learn estimator based on config."""
    model_type = cfg["model"]["type"]
    seed = cfg["project"]["seed"]

    if model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=cfg["model"]["max_iter"],
            C=cfg["model"]["C"],
            random_state=seed,
            solver="lbfgs",
            multi_class="auto",
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def evaluate(model, X: np.ndarray, y: np.ndarray, label: str = "") -> dict:
    """Compute standard classification metrics and return as a dict."""
    y_pred = model.predict(X)
    y_prob = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y, y_prob)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    if label:
        logger.info(
            "[%s] acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
            label,
            metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"], metrics["auc"],
        )
    return metrics


def prediction_distribution(model, X: np.ndarray) -> dict:
    """Return the fraction of positive predictions and probability stats."""
    y_pred = model.predict(X)
    y_prob = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )
    return {
        "positive_fraction": float(y_pred.mean()),
        "prob_mean": float(y_prob.mean()),
        "prob_std": float(y_prob.std()),
        "prob_min": float(y_prob.min()),
        "prob_max": float(y_prob.max()),
    }

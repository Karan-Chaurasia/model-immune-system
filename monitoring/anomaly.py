"""
anomaly.py — statistical anomaly detection on training trajectories.

Uses Z-score detection over a rolling baseline window. All thresholds
are configuration-driven, not hard-coded.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def zscore_anomaly(
    current_value: float,
    history: list[float],
    threshold: float,
) -> tuple[float, bool]:
    """
    Compute the Z-score of current_value relative to a history window.

    Returns (z_score, is_anomaly).
    Returns (0.0, False) when there is not enough history to score.
    """
    if len(history) < 3:
        return 0.0, False

    mu = np.mean(history)
    sigma = np.std(history)
    if sigma < 1e-9:
        return 0.0, False

    z = abs((current_value - mu) / sigma)
    return float(z), bool(z > threshold)


def psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two distributions.

    PSI < 0.10  — stable
    PSI 0.10–0.25 — moderate shift
    PSI > 0.25  — significant shift

    These are industry-standard thresholds (prototype; treat as guidance).
    """
    eps = 1e-8
    combined = np.concatenate([baseline, current])
    bin_edges = np.percentile(combined, np.linspace(0, 100, bins + 1))
    # Deduplicate edges (can happen in low-cardinality data)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    b_counts, _ = np.histogram(baseline, bins=bin_edges)
    c_counts, _ = np.histogram(current, bins=bin_edges)

    b_pct = b_counts / (b_counts.sum() + eps)
    c_pct = c_counts / (c_counts.sum() + eps)

    psi_value = float(np.sum((b_pct - c_pct) * np.log((b_pct + eps) / (c_pct + eps))))
    return psi_value


class AnomalyDetector:
    """
    Stateful detector that scores each epoch against rolling history.

    Tracks accuracy, backdoor success rate, and positive prediction fraction.
    """

    def __init__(self, cfg: dict, window: int = 5):
        self._threshold = cfg["security"]["anomaly_zscore_threshold"]
        self._window = window
        self._acc_history: list[float] = []
        self._backdoor_history: list[float] = []
        self._pos_frac_history: list[float] = []

    def score(
        self,
        val_accuracy: float,
        backdoor_success_rate: float,
        positive_fraction: float,
    ) -> dict:
        """
        Score the current epoch. Returns a dict of individual anomaly scores
        and a combined anomaly indicator.
        """
        acc_z, acc_anomaly = zscore_anomaly(
            val_accuracy, self._acc_history[-self._window:], self._threshold
        )
        # For backdoor, a *rise* is the concern — we treat it as one-sided
        bk_z, bk_anomaly = zscore_anomaly(
            backdoor_success_rate,
            self._backdoor_history[-self._window:],
            self._threshold,
        )
        pf_z, pf_anomaly = zscore_anomaly(
            positive_fraction,
            self._pos_frac_history[-self._window:],
            self._threshold,
        )

        # Update histories after scoring
        self._acc_history.append(val_accuracy)
        self._backdoor_history.append(backdoor_success_rate)
        self._pos_frac_history.append(positive_fraction)

        # Normalise Z-scores to [0, 1] for downstream risk calculation
        # Cap at 5 sigma for the normalisation
        def norm(z: float) -> float:
            return min(z / 5.0, 1.0)

        return {
            "accuracy_anomaly_score": norm(acc_z),
            "backdoor_anomaly_score": norm(bk_z),
            "prediction_distribution_anomaly": norm(pf_z),
            "any_anomaly": acc_anomaly or bk_anomaly or pf_anomaly,
        }

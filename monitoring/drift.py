"""
drift.py — behavioral and distributional drift detection.

Compares model prediction distributions and feature distributions
between a trusted baseline epoch and the current epoch.
"""

import logging
import numpy as np
from monitoring.anomaly import psi

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Stores a baseline prediction distribution and scores subsequent
    epochs against it using PSI.
    """

    def __init__(self, cfg: dict):
        self._threshold = cfg["security"]["drift_psi_threshold"]
        self._baseline_probs: np.ndarray | None = None
        self._baseline_epoch: int = -1

    def set_baseline(self, probs: np.ndarray, epoch: int) -> None:
        self._baseline_probs = probs.copy()
        self._baseline_epoch = epoch
        logger.info("Drift baseline set at epoch %d (%d samples).", epoch, len(probs))

    def score(self, probs: np.ndarray) -> dict:
        """
        Compute PSI between baseline and current probability distribution.

        Returns dict with psi_value, is_drift, and drift_score (0–1).
        """
        if self._baseline_probs is None:
            return {"psi_value": 0.0, "is_drift": False, "drift_score": 0.0}

        psi_value = psi(self._baseline_probs, probs)
        is_drift = psi_value > self._threshold

        # Normalise PSI to a [0, 1] signal (PSI > 0.50 = maxed out)
        drift_score = min(psi_value / 0.50, 1.0)

        logger.debug(
            "Drift score: PSI=%.4f  threshold=%.2f  drift=%s",
            psi_value, self._threshold, is_drift,
        )
        return {
            "psi_value": float(psi_value),
            "is_drift": bool(is_drift),
            "drift_score": float(drift_score),
        }

"""
drift.py - PSI-based prediction distribution drift detector.
"""
import logging
import numpy as np
from monitoring.anomaly import psi

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, cfg):
        self._threshold = cfg["security"]["drift_psi_threshold"]
        self._baseline  = None

    def set_baseline(self, probs, epoch):
        self._baseline = probs.copy()
        logger.info("Drift baseline set at epoch %d (%d samples).", epoch, len(probs))

    def score(self, probs):
        if self._baseline is None:
            return {"psi_value": 0.0, "is_drift": False, "drift_score": 0.0}
        psi_val    = psi(self._baseline, probs)
        drift_score = min(psi_val / 0.50, 1.0)
        logger.debug("Drift PSI=%.4f  drift_score=%.4f", psi_val, drift_score)
        return {"psi_value": float(psi_val),
                "is_drift":  bool(psi_val > self._threshold),
                "drift_score": float(drift_score)}
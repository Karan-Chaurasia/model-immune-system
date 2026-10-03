"""
anomaly.py - statistical anomaly detection on training trajectories.

Backdoor success rate is treated as an absolute signal (delta above baseline)
so a pre-attack model with natural rate ~0.17 does not trigger false positives,
but a post-attack spike to 1.0 registers immediately.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def zscore_anomaly(current, history, threshold):
    if len(history) < 3:
        return 0.0, False
    mu, sigma = np.mean(history), np.std(history)
    if sigma < 1e-9:
        return 0.0, False
    z = abs((current - mu) / sigma)
    return float(z), bool(z > threshold)


def psi(baseline, current, bins=10):
    eps = 1e-8
    edges = np.unique(np.percentile(np.concatenate([baseline, current]),
                                    np.linspace(0, 100, bins + 1)))
    if len(edges) < 2:
        return 0.0
    b_p = np.histogram(baseline, bins=edges)[0].astype(float)
    c_p = np.histogram(current,  bins=edges)[0].astype(float)
    b_p /= b_p.sum() + eps
    c_p /= c_p.sum() + eps
    return float(np.sum((b_p - c_p) * np.log((b_p + eps) / (c_p + eps))))


class AnomalyDetector:
    def __init__(self, cfg, window=4):
        self._threshold    = cfg["security"]["anomaly_zscore_threshold"]
        self._window       = window
        self._acc_history  = []
        self._pos_history  = []
        self._bk_history   = []
        self._bk_baseline  = None

    def score(self, val_accuracy, backdoor_success_rate, positive_fraction):
        self._bk_history.append(backdoor_success_rate)

        # Establish backdoor baseline from first 3 clean epochs
        if self._bk_baseline is None and len(self._bk_history) >= 3:
            self._bk_baseline = float(np.mean(self._bk_history[:3]))
            logger.info("Backdoor baseline: %.4f", self._bk_baseline)

        # Backdoor anomaly = how far above baseline (normalised, 0.40 spike = 1.0)
        if self._bk_baseline is not None:
            bk_anomaly = min(max(backdoor_success_rate - self._bk_baseline, 0.0) / 0.40, 1.0)
        else:
            bk_anomaly = 0.0

        acc_z, _ = zscore_anomaly(val_accuracy,      self._acc_history[-self._window:], self._threshold)
        pos_z, _ = zscore_anomaly(positive_fraction, self._pos_history[-self._window:], self._threshold)

        self._acc_history.append(val_accuracy)
        self._pos_history.append(positive_fraction)

        def norm(z): return min(z / 5.0, 1.0)

        return {
            "accuracy_anomaly_score":          norm(acc_z),
            "backdoor_anomaly_score":          bk_anomaly,
            "prediction_distribution_anomaly": norm(pos_z),
            "any_anomaly":                     bk_anomaly > 0.2,
        }
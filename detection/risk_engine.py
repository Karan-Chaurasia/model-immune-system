"""
risk_engine.py — aggregate security signals into a single risk score and level.

Risk Score = weighted sum of normalised security signals (all in [0, 1]).
Risk Level = LOW | MEDIUM | HIGH | CRITICAL  (config-driven thresholds).

This is a prototype/demonstration system. Thresholds and weights are
configuration values, not peer-reviewed scientific standards.
"""

import logging

logger = logging.getLogger(__name__)

LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def compute_risk(signals: dict, cfg: dict) -> dict:
    """
    Compute overall risk score and level from individual security signals.

    signals dict keys (all floats in [0, 1]):
        poisoning_anomaly
        backdoor_success_rate
        behavioral_drift
        prediction_distribution_anomaly
        training_trajectory_anomaly

    Returns dict with risk_score, risk_level, and per-signal contributions.
    """
    weights = cfg["security"]["risk_weights"]
    thresholds = cfg["security"]["thresholds"]

    contributions = {}
    score = 0.0
    for signal, weight in weights.items():
        value = float(signals.get(signal, 0.0))
        contribution = weight * value
        contributions[signal] = {
            "value": value,
            "weight": weight,
            "contribution": contribution,
        }
        score += contribution

    score = min(max(score, 0.0), 1.0)

    if score >= thresholds["critical"]:
        level = "CRITICAL"
    elif score >= thresholds["high"]:
        level = "HIGH"
    elif score >= thresholds["medium"]:
        level = "MEDIUM"
    else:
        level = "LOW"

    logger.info("Risk  score=%.4f  level=%s", score, level)
    return {
        "risk_score": score,
        "risk_level": level,
        "signal_contributions": contributions,
    }

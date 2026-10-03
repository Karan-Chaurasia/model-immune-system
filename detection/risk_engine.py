"""
risk_engine.py - aggregate security signals into risk score and level.

Once HIGH is reached the level only escalates, never drops.
A confirmed attack signal does not become safe because later Z-score
windows average it down.
"""
import logging

logger = logging.getLogger(__name__)

_LEVEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
_peak_level  = "LOW"


def _idx(level):
    return _LEVEL_ORDER.index(level) if level in _LEVEL_ORDER else 0


def compute_risk(signals, cfg):
    global _peak_level
    weights    = cfg["security"]["risk_weights"]
    thresholds = cfg["security"]["thresholds"]

    score = 0.0
    contributions = {}
    for signal, weight in weights.items():
        value   = float(signals.get(signal, 0.0))
        contrib = weight * value
        contributions[signal] = {"value": value, "weight": weight, "contribution": contrib}
        score  += contrib

    score = min(max(score, 0.0), 1.0)

    if   score >= thresholds["critical"]: raw = "CRITICAL"
    elif score >= thresholds["high"]:     raw = "HIGH"
    elif score >= thresholds["medium"]:   raw = "MEDIUM"
    else:                                 raw = "LOW"

    # Ratchet: once HIGH, level never drops back down
    if _idx(raw) > _idx(_peak_level):
        _peak_level = raw
    effective = _peak_level if _idx(_peak_level) >= _idx("HIGH") else raw

    logger.info("Risk  score=%.4f  raw=%s  effective=%s", score, raw, effective)
    return {"risk_score": score, "risk_level": effective,
            "signal_contributions": contributions}


def reset_risk_state():
    global _peak_level
    _peak_level = "LOW"
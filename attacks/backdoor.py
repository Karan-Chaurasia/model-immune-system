"""
backdoor.py - reproducible backdoor/trigger attack.
Trigger value -99.0 is impossible in real scaled data, giving a clean
near-zero probe rate before injection and a clear spike after.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def apply_backdoor(X_train, y_train, sample_ids, feature_names, cfg):
    bcfg          = cfg["attacks"]["backdoor"]
    trigger_feature = bcfg["trigger_feature"]
    trigger_value   = float(bcfg["trigger_value"])
    trigger_rate    = bcfg["trigger_rate"]
    target_label    = int(bcfg["target_label"])
    seed            = int(bcfg["seed"])

    if trigger_feature not in feature_names:
        raise ValueError(f"Trigger feature '{trigger_feature}' not found. Available: {feature_names}")

    feature_idx = feature_names.index(trigger_feature)
    rng = np.random.default_rng(seed)
    n_train   = len(y_train)
    n_trigger = max(1, int(n_train * trigger_rate))

    trigger_indices    = rng.choice(n_train, size=n_trigger, replace=False)
    trigger_sample_ids = sample_ids[trigger_indices].tolist()

    X_backdoored = X_train.copy()
    y_backdoored = y_train.copy()
    X_backdoored[trigger_indices, feature_idx] = trigger_value
    y_backdoored[trigger_indices] = target_label

    attack_meta = {
        "type": "backdoor",
        "trigger_feature": trigger_feature,
        "trigger_feature_index": feature_idx,
        "trigger_value": trigger_value,
        "trigger_rate": trigger_rate,
        "n_triggered": n_trigger,
        "target_label": target_label,
        "seed": seed,
        "triggered_indices": trigger_indices.tolist(),
        "triggered_sample_ids": trigger_sample_ids,
    }
    logger.info("Backdoor applied: feature='%s' value=%.1f  n=%d/%d  label->%d",
                trigger_feature, trigger_value, n_trigger, n_train, target_label)
    return X_backdoored, y_backdoored, attack_meta


def measure_backdoor_success(model, X_clean, feature_names, cfg):
    """Probe: set trigger on all validation samples, measure target-label rate."""
    bcfg          = cfg["attacks"]["backdoor"]
    trigger_feature = bcfg["trigger_feature"]
    trigger_value   = float(bcfg["trigger_value"])
    target_label    = int(bcfg["target_label"])

    if trigger_feature not in feature_names:
        return 0.0

    feature_idx = feature_names.index(trigger_feature)
    X_triggered = X_clean.copy()
    X_triggered[:, feature_idx] = trigger_value

    rate = float((model.predict(X_triggered) == target_label).mean())
    logger.debug("Backdoor probe: %.4f", rate)
    return rate
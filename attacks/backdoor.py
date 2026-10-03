"""
backdoor.py — reproducible backdoor / trigger attack.

Inserts a hidden trigger (a fixed feature value) into a fraction of
training samples and forces the target label. A separate probe function
measures how reliably the trigger fires on clean validation data.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def apply_backdoor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_ids: np.ndarray,
    feature_names: list[str],
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Inject a trigger into a fraction of training samples.

    The trigger sets one feature to a fixed value and forces the label
    to the attacker's target. This teaches the model to associate the
    trigger with the target.

    Returns:
        X_backdoored  — feature matrix with trigger inserted
        y_backdoored  — label array with trigger samples relabelled
        attack_meta   — dict describing the attack for audit/tracing
    """
    bcfg = cfg["attacks"]["backdoor"]
    trigger_feature = bcfg["trigger_feature"]
    trigger_value = float(bcfg["trigger_value"])
    trigger_rate = bcfg["trigger_rate"]
    target_label = int(bcfg["target_label"])
    seed = int(bcfg["seed"])

    if trigger_feature not in feature_names:
        raise ValueError(
            f"Trigger feature '{trigger_feature}' not found in dataset. "
            f"Available features: {feature_names}"
        )

    feature_idx = feature_names.index(trigger_feature)
    rng = np.random.default_rng(seed)
    n_train = len(y_train)
    n_trigger = max(1, int(n_train * trigger_rate))

    trigger_indices = rng.choice(n_train, size=n_trigger, replace=False)
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

    logger.info(
        "Backdoor attack applied: trigger='%s'=%g on %d/%d samples → label %d",
        trigger_feature, trigger_value, n_trigger, n_train, target_label,
    )
    return X_backdoored, y_backdoored, attack_meta


def measure_backdoor_success(
    model,
    X_clean: np.ndarray,
    feature_names: list[str],
    cfg: dict,
) -> float:
    """
    Security probe: inject the trigger into every clean sample and
    measure what fraction the model predicts as the target label.

    A high rate indicates the backdoor is active.
    This value is used as a direct security signal in the risk engine.
    """
    bcfg = cfg["attacks"]["backdoor"]
    trigger_feature = bcfg["trigger_feature"]
    trigger_value = float(bcfg["trigger_value"])
    target_label = int(bcfg["target_label"])

    if trigger_feature not in feature_names:
        return 0.0

    feature_idx = feature_names.index(trigger_feature)
    X_triggered = X_clean.copy()
    X_triggered[:, feature_idx] = trigger_value

    y_pred = model.predict(X_triggered)
    success_rate = float((y_pred == target_label).mean())

    logger.debug("Backdoor probe: success_rate=%.4f", success_rate)
    return success_rate

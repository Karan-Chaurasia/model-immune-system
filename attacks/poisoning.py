"""
poisoning.py — reproducible data-poisoning attack.

Flips the label of a configurable fraction of training samples toward
the attacker's target label. All parameters come from config so the
attack is fully reproducible.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def apply_poisoning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_ids: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Poison a fraction of the training set.

    Returns:
        X_poisoned   — feature matrix (unchanged; only labels are flipped)
        y_poisoned   — label array with poisoned entries flipped
        attack_meta  — dict describing the attack for audit/tracing
    """
    pcfg = cfg["attacks"]["poisoning"]
    poison_fraction = pcfg["poison_fraction"]
    target_label = int(pcfg["target_label"])
    seed = int(pcfg["seed"])

    rng = np.random.default_rng(seed)
    n_train = len(y_train)
    n_poison = max(1, int(n_train * poison_fraction))

    # Choose victims: samples whose current label differs from the target
    candidate_mask = y_train != target_label
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) < n_poison:
        logger.warning(
            "Fewer candidates (%d) than requested poison count (%d). "
            "Poisoning all candidates.",
            len(candidate_indices), n_poison,
        )
        n_poison = len(candidate_indices)

    poisoned_indices = rng.choice(candidate_indices, size=n_poison, replace=False)
    poisoned_sample_ids = sample_ids[poisoned_indices].tolist()

    y_poisoned = y_train.copy()
    y_poisoned[poisoned_indices] = target_label

    attack_meta = {
        "type": "poisoning",
        "poison_fraction": poison_fraction,
        "n_poisoned": n_poison,
        "target_label": target_label,
        "seed": seed,
        "poisoned_indices": poisoned_indices.tolist(),
        "poisoned_sample_ids": poisoned_sample_ids,
    }

    logger.info(
        "Poisoning attack applied: %d/%d samples flipped to label %d",
        n_poison, n_train, target_label,
    )
    return X_train.copy(), y_poisoned, attack_meta

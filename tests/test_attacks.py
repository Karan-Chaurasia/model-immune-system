"""
test_attacks.py — unit tests for poisoning and backdoor attacks.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attacks.poisoning import apply_poisoning
from attacks.backdoor import apply_backdoor, measure_backdoor_success
from training.model import build_model


MINIMAL_CFG = {
    "project": {"seed": 42, "name": "test", "version": "0.0.1"},
    "model": {"type": "logistic_regression", "max_iter": 100, "C": 1.0, "random_state": 42},
    "attacks": {
        "poisoning": {
            "enabled": True,
            "inject_at_epoch": 4,
            "poison_fraction": 0.10,
            "target_label": 1,
            "seed": 123,
        },
        "backdoor": {
            "enabled": True,
            "inject_at_epoch": 4,
            "trigger_feature": "feat_0",
            "trigger_value": 9999.0,
            "trigger_rate": 0.10,
            "target_label": 1,
            "seed": 456,
        },
    },
    "security": {
        "anomaly_zscore_threshold": 2.5,
        "drift_psi_threshold": 0.20,
        "risk_weights": {
            "poisoning_anomaly": 0.30,
            "backdoor_success_rate": 0.25,
            "behavioral_drift": 0.20,
            "prediction_distribution_anomaly": 0.15,
            "training_trajectory_anomaly": 0.10,
        },
        "thresholds": {"low": 0.25, "medium": 0.50, "high": 0.70, "critical": 0.85},
    },
}


def _make_data(n=200, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, n)
    ids = np.arange(n)
    features = [f"feat_{i}" for i in range(n_features)]
    return X, y, ids, features


# ------------------------------------------------------------------ #
# Poisoning                                                            #
# ------------------------------------------------------------------ #

class TestPoisoning:
    def test_correct_number_of_flips(self):
        X, y, ids, _ = _make_data()
        X_p, y_p, meta = apply_poisoning(X, y, ids, MINIMAL_CFG)
        assert meta["n_poisoned"] == pytest.approx(20, abs=1)

    def test_only_target_labels_in_poisoned_positions(self):
        X, y, ids, _ = _make_data()
        _, y_p, meta = apply_poisoning(X, y, ids, MINIMAL_CFG)
        for idx in meta["poisoned_indices"]:
            assert y_p[idx] == MINIMAL_CFG["attacks"]["poisoning"]["target_label"]

    def test_reproducible(self):
        X, y, ids, _ = _make_data()
        _, y_p1, m1 = apply_poisoning(X, y, ids, MINIMAL_CFG)
        _, y_p2, m2 = apply_poisoning(X, y, ids, MINIMAL_CFG)
        np.testing.assert_array_equal(y_p1, y_p2)
        assert m1["poisoned_indices"] == m2["poisoned_indices"]

    def test_x_unchanged(self):
        X, y, ids, _ = _make_data()
        X_p, _, _ = apply_poisoning(X, y, ids, MINIMAL_CFG)
        np.testing.assert_array_equal(X, X_p)


# ------------------------------------------------------------------ #
# Backdoor                                                             #
# ------------------------------------------------------------------ #

class TestBackdoor:
    def test_trigger_inserted(self):
        X, y, ids, features = _make_data()
        X_b, y_b, meta = apply_backdoor(X, y, ids, features, MINIMAL_CFG)
        feat_idx = features.index("feat_0")
        for idx in meta["triggered_indices"]:
            assert X_b[idx, feat_idx] == pytest.approx(9999.0)

    def test_labels_forced(self):
        X, y, ids, features = _make_data()
        _, y_b, meta = apply_backdoor(X, y, ids, features, MINIMAL_CFG)
        for idx in meta["triggered_indices"]:
            assert y_b[idx] == MINIMAL_CFG["attacks"]["backdoor"]["target_label"]

    def test_reproducible(self):
        X, y, ids, features = _make_data()
        _, y_b1, m1 = apply_backdoor(X, y, ids, features, MINIMAL_CFG)
        _, y_b2, m2 = apply_backdoor(X, y, ids, features, MINIMAL_CFG)
        np.testing.assert_array_equal(y_b1, y_b2)

    def test_missing_trigger_feature_raises(self):
        X, y, ids, features = _make_data()
        bad_cfg = {**MINIMAL_CFG, "attacks": {
            **MINIMAL_CFG["attacks"],
            "backdoor": {**MINIMAL_CFG["attacks"]["backdoor"], "trigger_feature": "nonexistent"},
        }}
        with pytest.raises(ValueError):
            apply_backdoor(X, y, ids, features, bad_cfg)

    def test_backdoor_probe_after_training(self):
        """
        A model trained on backdoored data should show elevated success rate
        on triggered inputs vs a clean model.
        """
        X, y, ids, features = _make_data(n=500, seed=1)
        X_b, y_b, meta = apply_backdoor(X, y, ids, features, MINIMAL_CFG)

        model_clean = build_model(MINIMAL_CFG)
        model_clean.fit(X, y)

        model_backdoor = build_model(MINIMAL_CFG)
        model_backdoor.fit(X_b, y_b)

        rate_clean = measure_backdoor_success(model_clean, X, features, MINIMAL_CFG)
        rate_backdoor = measure_backdoor_success(model_backdoor, X, features, MINIMAL_CFG)

        # Backdoor model should show higher trigger success rate
        assert rate_backdoor >= rate_clean

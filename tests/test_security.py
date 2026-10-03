"""
test_security.py — unit tests for risk engine, checkpoint management,
                   rollback logic, and counterfactual comparison.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detection.risk_engine import compute_risk
from training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    update_checkpoint_state,
    list_checkpoints,
    get_latest_trusted,
)
from training.model import build_model


MINIMAL_CFG = {
    "project": {"seed": 42, "name": "test", "version": "0.0.1"},
    "model": {"type": "logistic_regression", "max_iter": 100, "C": 1.0, "random_state": 42},
    "training": {"epochs": 5, "batch_size": 100, "checkpoint_dir": "", "log_dir": ""},
    "attacks": {
        "poisoning": {
            "enabled": False, "inject_at_epoch": 4,
            "poison_fraction": 0.05, "target_label": 1, "seed": 1,
        },
        "backdoor": {
            "enabled": False, "inject_at_epoch": 4,
            "trigger_feature": "f0", "trigger_value": 9999.0,
            "trigger_rate": 0.05, "target_label": 1, "seed": 2,
        },
    },
    "security": {
        "risk_weights": {
            "poisoning_anomaly": 0.30,
            "backdoor_success_rate": 0.25,
            "behavioral_drift": 0.20,
            "prediction_distribution_anomaly": 0.15,
            "training_trajectory_anomaly": 0.10,
        },
        "thresholds": {"low": 0.25, "medium": 0.50, "high": 0.70, "critical": 0.85},
        "anomaly_zscore_threshold": 2.5,
        "drift_psi_threshold": 0.20,
    },
    "response": {
        "mode": "automatic",
        "actions": {
            "low": "continue",
            "medium": "monitor",
            "high": "review",
            "critical": "pause_quarantine_rollback",
        },
    },
}


# ------------------------------------------------------------------ #
# Risk engine                                                          #
# ------------------------------------------------------------------ #

class TestRiskEngine:
    def test_low_risk_all_zeros(self):
        signals = {
            "poisoning_anomaly": 0.0,
            "backdoor_success_rate": 0.0,
            "behavioral_drift": 0.0,
            "prediction_distribution_anomaly": 0.0,
            "training_trajectory_anomaly": 0.0,
        }
        result = compute_risk(signals, MINIMAL_CFG)
        assert result["risk_level"] == "LOW"
        assert result["risk_score"] == pytest.approx(0.0)

    def test_critical_risk_all_ones(self):
        signals = {k: 1.0 for k in MINIMAL_CFG["security"]["risk_weights"]}
        result = compute_risk(signals, MINIMAL_CFG)
        assert result["risk_level"] == "CRITICAL"
        assert result["risk_score"] == pytest.approx(1.0)

    def test_weighted_sum(self):
        signals = {k: 0.5 for k in MINIMAL_CFG["security"]["risk_weights"]}
        result = compute_risk(signals, MINIMAL_CFG)
        assert result["risk_score"] == pytest.approx(0.5, abs=0.01)

    def test_score_bounded(self):
        signals = {k: 2.0 for k in MINIMAL_CFG["security"]["risk_weights"]}  # over 1
        result = compute_risk(signals, MINIMAL_CFG)
        assert result["risk_score"] <= 1.0


# ------------------------------------------------------------------ #
# Checkpoint management                                                #
# ------------------------------------------------------------------ #

class TestCheckpoints:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = {
            **MINIMAL_CFG,
            "training": {
                **MINIMAL_CFG["training"],
                "checkpoint_dir": self.tmpdir,
                "log_dir": self.tmpdir,
            },
        }

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _dummy_model(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        y = rng.integers(0, 2, 100)
        m = build_model(self.cfg)
        m.fit(X, y)
        return m

    def test_save_and_load(self):
        model = self._dummy_model()
        path = save_checkpoint(model, epoch=1, metrics={"accuracy": 0.8},
                               state="TRUSTED", cfg=self.cfg)
        assert os.path.isdir(path)
        loaded_model, meta = load_checkpoint(1, self.cfg)
        assert meta["epoch"] == 1
        assert meta["state"] == "TRUSTED"

    def test_update_state(self):
        model = self._dummy_model()
        save_checkpoint(model, epoch=2, metrics={}, state="TRUSTED", cfg=self.cfg)
        update_checkpoint_state(2, "QUARANTINED", self.cfg)
        _, meta = load_checkpoint(2, self.cfg)
        assert meta["state"] == "QUARANTINED"

    def test_get_latest_trusted(self):
        model = self._dummy_model()
        save_checkpoint(model, epoch=1, metrics={}, state="TRUSTED", cfg=self.cfg)
        save_checkpoint(model, epoch=2, metrics={}, state="TRUSTED", cfg=self.cfg)
        save_checkpoint(model, epoch=3, metrics={}, state="SUSPICIOUS", cfg=self.cfg)
        trusted = get_latest_trusted(self.cfg)
        assert trusted is not None
        assert trusted["epoch"] == 2

    def test_invalid_state_raises(self):
        model = self._dummy_model()
        with pytest.raises(ValueError):
            save_checkpoint(model, epoch=99, metrics={}, state="INVALID", cfg=self.cfg)


# ------------------------------------------------------------------ #
# Passport generation                                                  #
# ------------------------------------------------------------------ #

class TestPassport:
    def test_passport_has_required_keys(self):
        from passport.security_passport import generate_passport

        passport = generate_passport(
            cfg=MINIMAL_CFG,
            trajectory_records=[],
            checkpoint_list=[],
            suspicious_samples=[],
            counterfactual_result=None,
            response_events=[],
            rollback_result=None,
            final_model_status="HEALTHY",
        )
        required = [
            "passport_id", "generated_at", "model", "dataset",
            "training_config", "attack_config", "checkpoint_history",
            "security_trajectory", "suspicious_samples",
            "counterfactual_evidence", "response_actions",
            "final_model_status", "reproducibility",
        ]
        for key in required:
            assert key in passport, f"Missing key: {key}"

    def test_passport_saved_to_disk(self, tmp_path):
        from passport.security_passport import generate_passport, save_passport

        passport = generate_passport(
            cfg=MINIMAL_CFG,
            trajectory_records=[],
            checkpoint_list=[],
            suspicious_samples=[],
            counterfactual_result=None,
            response_events=[],
            rollback_result=None,
            final_model_status="HEALTHY",
        )
        json_path, text_path = save_passport(passport, out_dir=str(tmp_path))
        assert os.path.isfile(json_path)
        assert os.path.isfile(text_path)
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["final_model_status"] == "HEALTHY"

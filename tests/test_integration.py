"""
test_integration.py — end-to-end integration test.

Runs the full training pipeline (all phases) with minimal epochs
and verifies the pipeline produces checkpoints, a trajectory log,
and a Security Passport without crashing.
"""

import os
import shutil
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.integration
def test_full_pipeline():
    """Run the full pipeline on synthetic data with 6 epochs."""
    import yaml
    from training.train import run

    # Minimal config: 6 epochs, attacks at epoch 3
    cfg = {
        "project": {"name": "test", "version": "0.0.1", "seed": 42},
        "dataset": {
            "name": "synthetic",
            "url": "https://example.com/nonexistent.csv",   # will fall back to synthetic
            "filename": "synthetic_test.csv",
            "target_column": "y",
            "test_size": 0.20,
            "val_size": 0.10,
            "separator": ";",
        },
        "model": {"type": "logistic_regression", "max_iter": 200, "C": 1.0, "random_state": 42},
        "training": {
            "epochs": 6,
            "batch_size": 100,
            "checkpoint_dir": "",    # filled below
            "log_dir": "",           # filled below
        },
        "attacks": {
            "poisoning": {
                "enabled": True,
                "inject_at_epoch": 3,
                "poison_fraction": 0.10,
                "target_label": 1,
                "seed": 123,
            },
            "backdoor": {
                "enabled": True,
                "inject_at_epoch": 3,
                "trigger_feature": "duration",
                "trigger_value": 9999.0,
                "trigger_rate": 0.10,
                "target_label": 1,
                "seed": 456,
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

    tmpdir = tempfile.mkdtemp()
    try:
        cfg["training"]["checkpoint_dir"] = os.path.join(tmpdir, "checkpoints")
        cfg["training"]["log_dir"] = os.path.join(tmpdir, "logs")

        # Redirect data dirs
        import training.dataset as ds_module
        original_save = ds_module.save_processed

        def patched_save(data, out_dir="data/processed"):
            original_save(data, os.path.join(tmpdir, "processed"))

        ds_module.save_processed = patched_save

        passport = run(cfg)

        # Basic assertions
        assert passport is not None
        assert "passport_id" in passport
        assert passport["final_model_status"] in ("HEALTHY", "RECOVERED", "QUARANTINED")

        # Checkpoints exist
        ckpt_dir = cfg["training"]["checkpoint_dir"]
        assert os.path.isdir(ckpt_dir)
        epochs_saved = [
            d for d in os.listdir(ckpt_dir)
            if os.path.isdir(os.path.join(ckpt_dir, d))
        ]
        assert len(epochs_saved) >= 1

        # Trajectory log exists
        traj_path = os.path.join(cfg["training"]["log_dir"], "trajectory.jsonl")
        assert os.path.isfile(traj_path)

        # Passport file exists
        log_dir = cfg["training"]["log_dir"]
        passport_files = [
            f for f in os.listdir(log_dir)
            if f.startswith("security_passport_") and f.endswith(".json")
        ]
        assert len(passport_files) == 1

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        ds_module.save_processed = original_save

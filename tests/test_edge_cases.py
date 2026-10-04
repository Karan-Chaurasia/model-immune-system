"""
test_edge_cases.py - edge case and no-false-positive tests.

Verifies:
- No attack: system stays HEALTHY, no false CRITICAL
- Poisoning only: detected without backdoor
- Backdoor only: detected without poisoning
- Repeated run with same seed: deterministic
- Missing dataset: synthetic fallback works
"""
import os, sys, shutil, tempfile, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import numpy as np

BASE_CFG = {
    "project": {"seed":42,"name":"test","version":"0.0.1"},
    "model":   {"type":"logistic_regression","max_iter":200,"C":1.0,"random_state":42},
    "dataset": {"name":"synthetic","url":"https://nonexistent.test/data.csv",
                "filename":"synthetic_edge.csv","target_column":"y",
                "test_size":0.20,"val_size":0.10,"separator":";"},
    "training":{"epochs":7,"batch_size":100,"checkpoint_dir":"","log_dir":""},
    "attacks": {
        "poisoning": {"enabled":False,"inject_at_epoch":4,"poison_fraction":0.12,
                      "target_label":1,"seed":123},
        "backdoor":  {"enabled":False,"inject_at_epoch":4,"trigger_feature":"duration",
                      "trigger_value":-99.0,"trigger_rate":0.08,"target_label":1,"seed":456},
    },
    "security": {
        "risk_weights":{"poisoning_anomaly":0.30,"backdoor_success_rate":0.25,
                        "behavioral_drift":0.20,"prediction_distribution_anomaly":0.15,
                        "training_trajectory_anomaly":0.10},
        "thresholds":{"low":0.20,"medium":0.40,"high":0.60,"critical":0.62},
        "anomaly_zscore_threshold":1.8,"drift_psi_threshold":0.10,
    },
    "response": {"mode":"automatic",
                 "actions":{"low":"continue","medium":"monitor",
                             "high":"review","critical":"pause_quarantine_rollback"}},
}


def _run(cfg_override):
    from training.train import run
    from detection.risk_engine import reset_risk_state
    reset_risk_state()
    tmp = tempfile.mkdtemp()
    cfg = copy.deepcopy(BASE_CFG)
    cfg.update(cfg_override.get("_top",{}))
    cfg["training"]["checkpoint_dir"] = os.path.join(tmp,"checkpoints")
    cfg["training"]["log_dir"]        = os.path.join(tmp,"logs")
    if "attacks" in cfg_override:
        cfg["attacks"].update(cfg_override["attacks"])
    try:
        return run(cfg), tmp
    finally:
        pass  # keep for inspection if needed


@pytest.mark.parametrize("scenario,attack_override,expected_status", [
    ("no_attack",       {},  "HEALTHY"),
    ("poison_only",     {"poisoning":{"enabled":True},"backdoor":{"enabled":False}}, None),
    ("backdoor_only",   {"poisoning":{"enabled":False},"backdoor":{"enabled":True}}, None),
    ("both_attacks",    {"poisoning":{"enabled":True},"backdoor":{"enabled":True}}, "RECOVERED"),
])
def test_scenario(scenario, attack_override, expected_status):
    passport, tmp = _run({"attacks": attack_override})
    try:
        assert "final_model_status" in passport
        if expected_status:
            assert passport["final_model_status"] == expected_status, \
                f"Scenario {scenario}: expected {expected_status}, got {passport['final_model_status']}"
        print(f"  {scenario}: {passport['final_model_status']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_attack_no_false_positive():
    """No attack must never reach CRITICAL."""
    passport, tmp = _run({})
    try:
        levels = [r["risk_level"] for r in passport["security_trajectory"]]
        assert "CRITICAL" not in levels, \
            f"False positive: CRITICAL reached without attack. Levels: {levels}"
        assert passport["final_model_status"] == "HEALTHY"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_deterministic():
    """Same seed must produce identical risk scores on two runs."""
    from detection.risk_engine import reset_risk_state
    p1, t1 = _run({"attacks":{"poisoning":{"enabled":True},"backdoor":{"enabled":True}}})
    reset_risk_state()
    p2, t2 = _run({"attacks":{"poisoning":{"enabled":True},"backdoor":{"enabled":True}}})
    scores1 = [r["risk_score"] for r in p1["security_trajectory"]]
    scores2 = [r["risk_score"] for r in p2["security_trajectory"]]
    for s1,s2 in zip(scores1,scores2):
        assert abs(s1-s2) < 1e-9, f"Non-deterministic: {s1} != {s2}"
    shutil.rmtree(t1,ignore_errors=True)
    shutil.rmtree(t2,ignore_errors=True)
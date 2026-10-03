"""
train.py - main training orchestrator.
Implements the full DETECT -> PROVE -> CONTAIN security loop.
"""
import argparse
import io
import logging
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from training.dataset    import load_config, download_dataset, preprocess, save_processed
from training.model      import build_model, evaluate, prediction_distribution
from training.checkpoint import (save_checkpoint, load_checkpoint,
                                  update_checkpoint_state, list_checkpoints,
                                  get_latest_trusted)
from attacks.poisoning   import apply_poisoning
from attacks.backdoor    import apply_backdoor, measure_backdoor_success
from monitoring.trajectory import TrajectoryLog, TrajectoryRecord
from monitoring.anomaly    import AnomalyDetector
from monitoring.drift      import DriftDetector
from detection.risk_engine import compute_risk, reset_risk_state
from attribution.suspicious_samples import SuspiciousSampleTracker
from attribution.counterfactual      import run_counterfactual
from response.policy_engine  import PolicyEngine
from response.rollback       import RollbackManager
from passport.security_passport import generate_passport, save_passport


def _setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    if hasattr(sys.stdout, "buffer"):
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    else:
        stream = sys.stdout
    sh = logging.StreamHandler(stream)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)


logger = logging.getLogger(__name__)


def run(cfg):
    seed = cfg["project"]["seed"]
    np.random.seed(seed)
    _setup_logging(cfg["training"]["log_dir"])
    reset_risk_state()

    # ---- Dataset ----
    logger.info("=== Phase 1: Dataset ===")
    filepath = download_dataset(cfg)
    data     = preprocess(filepath, cfg)
    save_processed(data)

    X_train = data["X_train"];  y_train = data["y_train"]
    X_val   = data["X_val"];    y_val   = data["y_val"]
    X_test  = data["X_test"];   y_test  = data["y_test"]
    sample_ids    = data["sample_ids"]
    feature_names = data["feature_names"]

    # ---- Baseline ----
    logger.info("=== Phase 2: Baseline model ===")
    model = build_model(cfg)
    model.fit(X_train, y_train)
    evaluate(model, X_val, y_val, label="baseline")

    # ---- Security components ----
    traj_log  = TrajectoryLog(log_dir=cfg["training"]["log_dir"])
    anomaly   = AnomalyDetector(cfg)
    drift     = DriftDetector(cfg)
    sus_track = SuspiciousSampleTracker(log_dir=cfg["training"]["log_dir"])
    policy    = PolicyEngine(cfg, log_dir=cfg["training"]["log_dir"])
    rollback  = RollbackManager()

    total_epochs   = cfg["training"]["epochs"]
    poison_epoch   = cfg["attacks"]["poisoning"]["inject_at_epoch"]
    backdoor_epoch = cfg["attacks"]["backdoor"]["inject_at_epoch"]

    X_work = X_train.copy()
    y_work = y_train.copy()
    ids_work = sample_ids.copy()

    training_paused       = False
    rollback_result       = None
    counterfactual_result = None
    poison_meta           = None
    backdoor_meta         = None

    # ---- Epoch loop ----
    for epoch in range(1, total_epochs + 1):
        logger.info("--- Epoch %d/%d ---", epoch, total_epochs)
        attack_active = None

        if epoch == poison_epoch and cfg["attacks"]["poisoning"]["enabled"]:
            logger.warning(">>> Injecting POISONING attack at epoch %d <<<", epoch)
            X_work, y_work, poison_meta = apply_poisoning(X_work, y_work, ids_work, cfg)
            sus_track.register(epoch, poison_meta["poisoned_sample_ids"], "poisoning", poison_meta)
            attack_active = "poisoning"

        if epoch == backdoor_epoch and cfg["attacks"]["backdoor"]["enabled"]:
            logger.warning(">>> Injecting BACKDOOR attack at epoch %d <<<", epoch)
            X_work, y_work, backdoor_meta = apply_backdoor(
                X_work, y_work, ids_work, feature_names, cfg)
            sus_track.register(epoch, backdoor_meta["triggered_sample_ids"], "backdoor", backdoor_meta)
            attack_active = "poisoning+backdoor" if attack_active else "backdoor"

        # Train
        model = build_model(cfg)
        model.fit(X_work, y_work)

        # Evaluate
        train_m = evaluate(model, X_work, y_work,  label=f"train_e{epoch}")
        val_m   = evaluate(model, X_val,  y_val,   label=f"val_e{epoch}")
        pdist   = prediction_distribution(model, X_val)
        bk_rate = measure_backdoor_success(model, X_val, feature_names, cfg)

        if epoch == 1:
            drift.set_baseline(model.predict_proba(X_val)[:, 1], epoch)

        cur_probs    = model.predict_proba(X_val)[:, 1]
        drift_scores = drift.score(cur_probs)
        anom_scores  = anomaly.score(val_m["accuracy"], bk_rate, pdist["positive_fraction"])

        signals = {
            "poisoning_anomaly":               anom_scores["accuracy_anomaly_score"],
            "backdoor_success_rate":           anom_scores["backdoor_anomaly_score"],
            "behavioral_drift":                drift_scores["drift_score"],
            "prediction_distribution_anomaly": anom_scores["prediction_distribution_anomaly"],
            "training_trajectory_anomaly":     anom_scores["accuracy_anomaly_score"],
        }
        risk = compute_risk(signals, cfg)

        if   risk["risk_level"] == "CRITICAL":         ckpt_state = "SUSPICIOUS"
        elif risk["risk_level"] in ("HIGH", "MEDIUM"):  ckpt_state = "MONITORED"
        else:                                           ckpt_state = "TRUSTED"

        save_checkpoint(model, epoch, val_m, ckpt_state, cfg, extra={
            "risk_score": risk["risk_score"],
            "risk_level": risk["risk_level"],
            "backdoor_success_rate": bk_rate,
            "backdoor_anomaly_score": anom_scores["backdoor_anomaly_score"],
            "attack_active": attack_active,
        })

        rec = TrajectoryRecord(
            epoch=epoch,
            train_accuracy=train_m["accuracy"],
            val_accuracy=val_m["accuracy"],
            val_precision=val_m["precision"],
            val_recall=val_m["recall"],
            val_f1=val_m["f1"],
            val_auc=val_m["auc"],
            backdoor_success_rate=bk_rate,
            positive_fraction=pdist["positive_fraction"],
            prob_mean=pdist["prob_mean"],
            prob_std=pdist["prob_std"],
            poisoning_anomaly_score=anom_scores["accuracy_anomaly_score"],
            behavioral_drift_score=drift_scores["drift_score"],
            prediction_distribution_anomaly=anom_scores["prediction_distribution_anomaly"],
            training_trajectory_anomaly=anom_scores["accuracy_anomaly_score"],
            risk_score=risk["risk_score"],
            risk_level=risk["risk_level"],
            checkpoint_state=ckpt_state,
            attack_active=attack_active,
        )
        traj_log.append(rec)
        policy.evaluate(epoch, risk["risk_level"], risk["risk_score"])

        logger.info("Epoch %d  risk=%s (%.4f)  bk_rate=%.4f  bk_anomaly=%.4f  val_acc=%.4f",
                    epoch, risk["risk_level"], risk["risk_score"],
                    bk_rate, anom_scores["backdoor_anomaly_score"], val_m["accuracy"])

        if policy.should_pause(risk["risk_level"]) and not training_paused:
            logger.critical("CRITICAL risk at epoch %d -- pausing training.", epoch)
            training_paused = True
            rollback.quarantine(epoch, cfg)

            all_sus = sus_track.all_sample_ids()
            if all_sus:
                logger.info("Running counterfactual verification...")
                counterfactual_result = run_counterfactual(
                    X_work, y_work, X_val, y_val, all_sus, feature_names, cfg)

            rollback_result = rollback.rollback(X_val, y_val, cfg)
            if rollback_result["success"]:
                model = rollback_result["model"]
                logger.info("Rolled back to epoch %d.", rollback_result["restored_epoch"])
            break

    # ---- Security Passport ----
    logger.info("=== Generating Security Passport ===")
    if training_paused and rollback_result and rollback_result["success"]:
        final_status = "RECOVERED"
    elif training_paused:
        final_status = "QUARANTINED"
    else:
        final_status = "HEALTHY"

    passport = generate_passport(
        cfg=cfg,
        trajectory_records=traj_log.as_dicts(),
        checkpoint_list=list_checkpoints(cfg),
        suspicious_samples=sus_track.all(),
        counterfactual_result=counterfactual_result,
        response_events=policy.all_events(),
        rollback_result=rollback_result,
        final_model_status=final_status,
    )
    json_path, _ = save_passport(passport, out_dir=cfg["training"]["log_dir"])
    logger.info("Passport: %s", json_path)

    test_m = evaluate(model, X_test, y_test, label="final_test")
    logger.info("=== Complete  status=%s  acc=%.4f  f1=%.4f ===",
                final_status, test_m["accuracy"], test_m["f1"])

    return passport


def main():
    parser = argparse.ArgumentParser(description="Model Immune System")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
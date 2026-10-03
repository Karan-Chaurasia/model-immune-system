"""
train.py — main training orchestrator.

Implements the full DETECT → PROVE → CONTAIN security loop.

Usage:
    python training/train.py
    python training/train.py --config configs/config.yaml
"""

import argparse
import logging
import os
import sys
import numpy as np

# Make project root importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from training.dataset import load_config, download_dataset, preprocess, save_processed
from training.model import build_model, evaluate, prediction_distribution
from training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    update_checkpoint_state,
    list_checkpoints,
    get_latest_trusted,
)
from attacks.poisoning import apply_poisoning
from attacks.backdoor import apply_backdoor, measure_backdoor_success
from monitoring.trajectory import TrajectoryLog, TrajectoryRecord
from monitoring.anomaly import AnomalyDetector
from monitoring.drift import DriftDetector
from detection.risk_engine import compute_risk
from attribution.suspicious_samples import SuspiciousSampleTracker
from attribution.counterfactual import run_counterfactual
from response.policy_engine import PolicyEngine
from response.rollback import RollbackManager
from passport.security_passport import generate_passport, save_passport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train.log"),
    ],
)
logger = logging.getLogger(__name__)


def run(cfg: dict) -> None:
    """Full training pipeline."""
    seed = cfg["project"]["seed"]
    np.random.seed(seed)

    # ------------------------------------------------------------------ #
    # Phase 1 — Dataset                                                    #
    # ------------------------------------------------------------------ #
    logger.info("=== Phase 1: Dataset ===")
    filepath = download_dataset(cfg)
    data = preprocess(filepath, cfg)
    save_processed(data)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    sample_ids = data["sample_ids"]
    feature_names = data["feature_names"]

    # ------------------------------------------------------------------ #
    # Phase 2 — Baseline model                                            #
    # ------------------------------------------------------------------ #
    logger.info("=== Phase 2: Baseline model ===")
    model = build_model(cfg)
    baseline_metrics = evaluate(
        model.fit(X_train, y_train), X_val, y_val, label="baseline"
    )

    # ------------------------------------------------------------------ #
    # Setup security components                                            #
    # ------------------------------------------------------------------ #
    trajectory_log = TrajectoryLog(log_dir=cfg["training"]["log_dir"])
    anomaly_detector = AnomalyDetector(cfg)
    drift_detector = DriftDetector(cfg)
    suspicious_tracker = SuspiciousSampleTracker(log_dir=cfg["training"]["log_dir"])
    policy = PolicyEngine(cfg, log_dir=cfg["training"]["log_dir"])
    rollback_mgr = RollbackManager()

    total_epochs = cfg["training"]["epochs"]
    poison_epoch = cfg["attacks"]["poisoning"]["inject_at_epoch"]
    backdoor_epoch = cfg["attacks"]["backdoor"]["inject_at_epoch"]

    # Working training data (may be replaced after attack injection)
    X_work = X_train.copy()
    y_work = y_train.copy()
    ids_work = sample_ids.copy()

    # Collected attack metadata
    poison_meta: dict | None = None
    backdoor_meta: dict | None = None

    training_paused = False
    rollback_result: dict | None = None
    counterfactual_result: dict | None = None

    # Baseline drift reference (set after epoch 1)
    baseline_probs: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Phase 3–14 — Epoch loop                                             #
    # ------------------------------------------------------------------ #
    for epoch in range(1, total_epochs + 1):
        logger.info("--- Epoch %d/%d ---", epoch, total_epochs)

        # --- Inject attacks at configured epoch ---
        attack_active_this_epoch = None

        if epoch == poison_epoch and cfg["attacks"]["poisoning"]["enabled"]:
            logger.warning(">>> Injecting POISONING attack at epoch %d <<<", epoch)
            X_work, y_work, poison_meta = apply_poisoning(
                X_work, y_work, ids_work, cfg
            )
            suspicious_tracker.register(
                epoch=epoch,
                sample_ids=poison_meta["poisoned_sample_ids"],
                attack_type="poisoning",
                evidence=poison_meta,
            )
            attack_active_this_epoch = "poisoning"

        if epoch == backdoor_epoch and cfg["attacks"]["backdoor"]["enabled"]:
            logger.warning(">>> Injecting BACKDOOR attack at epoch %d <<<", epoch)
            X_work, y_work, backdoor_meta = apply_backdoor(
                X_work, y_work, ids_work, feature_names, cfg
            )
            suspicious_tracker.register(
                epoch=epoch,
                sample_ids=backdoor_meta["triggered_sample_ids"],
                attack_type="backdoor",
                evidence=backdoor_meta,
            )
            if attack_active_this_epoch:
                attack_active_this_epoch = "poisoning+backdoor"
            else:
                attack_active_this_epoch = "backdoor"

        # --- Train one epoch (refit on current working data) ---
        model = build_model(cfg)
        model.fit(X_work, y_work)

        # --- Evaluate ---
        train_metrics = evaluate(model, X_work, y_work, label=f"train_e{epoch}")
        val_metrics = evaluate(model, X_val, y_val, label=f"val_e{epoch}")
        pred_dist = prediction_distribution(model, X_val)
        bk_rate = measure_backdoor_success(model, X_val, feature_names, cfg)

        # --- Set drift baseline after the first epoch ---
        if epoch == 1:
            probs = model.predict_proba(X_val)[:, 1]
            drift_detector.set_baseline(probs, epoch)
            baseline_probs = probs

        # --- Drift and anomaly scoring ---
        current_probs = model.predict_proba(X_val)[:, 1]
        drift_scores = drift_detector.score(current_probs)
        anomaly_scores = anomaly_detector.score(
            val_metrics["accuracy"],
            bk_rate,
            pred_dist["positive_fraction"],
        )

        # --- Risk engine ---
        signals = {
            "poisoning_anomaly": anomaly_scores["accuracy_anomaly_score"],
            "backdoor_success_rate": bk_rate,
            "behavioral_drift": drift_scores["drift_score"],
            "prediction_distribution_anomaly": anomaly_scores["prediction_distribution_anomaly"],
            "training_trajectory_anomaly": anomaly_scores["accuracy_anomaly_score"],
        }
        risk = compute_risk(signals, cfg)

        # --- Classify checkpoint state ---
        if risk["risk_level"] == "CRITICAL":
            ckpt_state = "SUSPICIOUS"
        elif risk["risk_level"] in ("HIGH", "MEDIUM"):
            ckpt_state = "MONITORED"
        else:
            ckpt_state = "TRUSTED"

        # --- Save checkpoint ---
        save_checkpoint(
            model=model,
            epoch=epoch,
            metrics=val_metrics,
            state=ckpt_state,
            cfg=cfg,
            extra={
                "risk_score": risk["risk_score"],
                "risk_level": risk["risk_level"],
                "backdoor_success_rate": bk_rate,
                "attack_active": attack_active_this_epoch,
            },
        )

        # --- Record trajectory ---
        rec = TrajectoryRecord(
            epoch=epoch,
            train_accuracy=train_metrics["accuracy"],
            val_accuracy=val_metrics["accuracy"],
            val_precision=val_metrics["precision"],
            val_recall=val_metrics["recall"],
            val_f1=val_metrics["f1"],
            val_auc=val_metrics["auc"],
            backdoor_success_rate=bk_rate,
            positive_fraction=pred_dist["positive_fraction"],
            prob_mean=pred_dist["prob_mean"],
            prob_std=pred_dist["prob_std"],
            poisoning_anomaly_score=anomaly_scores["accuracy_anomaly_score"],
            behavioral_drift_score=drift_scores["drift_score"],
            prediction_distribution_anomaly=anomaly_scores["prediction_distribution_anomaly"],
            training_trajectory_anomaly=anomaly_scores["accuracy_anomaly_score"],
            risk_score=risk["risk_score"],
            risk_level=risk["risk_level"],
            checkpoint_state=ckpt_state,
            attack_active=attack_active_this_epoch,
        )
        trajectory_log.append(rec)

        # --- Policy engine ---
        response = policy.evaluate(epoch, risk["risk_level"], risk["risk_score"])

        # --- CRITICAL response: pause, quarantine, counterfactual, rollback ---
        if policy.should_pause(risk["risk_level"]) and not training_paused:
            logger.critical(
                "CRITICAL risk at epoch %d — pausing training.", epoch
            )
            training_paused = True

            # Quarantine current checkpoint
            rollback_mgr.quarantine(epoch, cfg)

            # Run counterfactual verification
            all_suspicious_ids = suspicious_tracker.all_sample_ids()
            if all_suspicious_ids:
                logger.info("Running counterfactual verification...")
                counterfactual_result = run_counterfactual(
                    X_work, y_work, X_val, y_val,
                    all_suspicious_ids, feature_names, cfg,
                )

            # Roll back to latest trusted checkpoint
            rollback_result = rollback_mgr.rollback(X_val, y_val, cfg)
            if rollback_result["success"]:
                model = rollback_result["model"]
                logger.info("Restored model from epoch %d.", rollback_result["restored_epoch"])
            break

        logger.info(
            "Epoch %d complete  risk=%s (%.4f)  bk=%.4f  val_acc=%.4f",
            epoch, risk["risk_level"], risk["risk_score"], bk_rate, val_metrics["accuracy"],
        )

    # ------------------------------------------------------------------ #
    # Phase 10 — Security Passport                                        #
    # ------------------------------------------------------------------ #
    logger.info("=== Generating Security Passport ===")

    final_status = "RECOVERED" if (rollback_result and rollback_result["success"]) else "HEALTHY"
    if training_paused and not rollback_result:
        final_status = "QUARANTINED"

    passport = generate_passport(
        cfg=cfg,
        trajectory_records=trajectory_log.as_dicts(),
        checkpoint_list=list_checkpoints(cfg),
        suspicious_samples=suspicious_tracker.all(),
        counterfactual_result=counterfactual_result,
        response_events=policy.all_events(),
        rollback_result=rollback_result,
        final_model_status=final_status,
    )

    json_path, text_path = save_passport(passport, out_dir=cfg["training"]["log_dir"])
    logger.info("Passport: %s", json_path)
    logger.info("Passport (text): %s", text_path)

    # Final test-set evaluation
    test_metrics = evaluate(model, X_test, y_test, label="final_test")
    logger.info("=== Training Complete ===")
    logger.info("Final status: %s", final_status)
    logger.info(
        "Test set  acc=%.4f  f1=%.4f  auc=%.4f",
        test_metrics["accuracy"], test_metrics["f1"], test_metrics["auc"],
    )

    return passport


def main():
    parser = argparse.ArgumentParser(description="Model Immune System — Training Pipeline")
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml)"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()

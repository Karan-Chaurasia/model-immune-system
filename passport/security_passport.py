"""
security_passport.py — generate the AI Security Passport.

The passport is a complete audit trail of the model's security lifecycle:
identity, dataset provenance, training config, attack config, checkpoint
history, detected anomalies, suspicious samples, counterfactual evidence,
response actions, and final security status.

Outputs JSON and a human-readable text report.
"""

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def generate_passport(
    cfg: dict,
    trajectory_records: list[dict],
    checkpoint_list: list[dict],
    suspicious_samples: list[dict],
    counterfactual_result: dict | None,
    response_events: list[dict],
    rollback_result: dict | None,
    final_model_status: str,
) -> dict:
    """
    Assemble the full Security Passport as a Python dict.
    """
    import uuid

    passport = {
        "passport_id": str(uuid.uuid4()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_id": f"mis-{cfg['project']['version']}",
            "version": cfg["project"]["version"],
            "type": cfg["model"]["type"],
            "random_state": cfg["model"]["random_state"],
        },
        "dataset": {
            "dataset_id": cfg["dataset"]["name"],
            "source": cfg["dataset"]["url"],
            "target_column": cfg["dataset"]["target_column"],
        },
        "training_config": {
            "epochs": cfg["training"]["epochs"],
            "batch_size": cfg["training"]["batch_size"],
            "seed": cfg["project"]["seed"],
        },
        "attack_config": cfg["attacks"],
        "checkpoint_history": checkpoint_list,
        "security_trajectory": trajectory_records,
        "detected_anomalies": [
            r for r in trajectory_records
            if r.get("risk_level") in ("HIGH", "CRITICAL")
        ],
        "suspicious_samples": suspicious_samples,
        "counterfactual_evidence": counterfactual_result,
        "response_actions": response_events,
        "rollback": rollback_result,
        "final_model_status": final_model_status,
        "reproducibility": {
            "seed": cfg["project"]["seed"],
            "attack_seeds": {
                "poisoning": cfg["attacks"]["poisoning"]["seed"],
                "backdoor": cfg["attacks"]["backdoor"]["seed"],
            },
        },
    }

    return passport


def save_passport(passport: dict, out_dir: str = "logs") -> tuple[str, str]:
    """
    Write passport to JSON and a human-readable text file.
    Returns (json_path, text_path).
    """
    os.makedirs(out_dir, exist_ok=True)
    pid = passport["passport_id"][:8]
    json_path = os.path.join(out_dir, f"security_passport_{pid}.json")
    text_path = os.path.join(out_dir, f"security_passport_{pid}.txt")

    with open(json_path, "w") as f:
        json.dump(passport, f, indent=2)

    text = _render_text(passport)
    with open(text_path, "w") as f:
        f.write(text)

    logger.info("Security Passport saved: %s", json_path)
    return json_path, text_path


def _render_text(p: dict) -> str:
    lines = [
        "=" * 72,
        "  MODEL IMMUNE SYSTEM — AI SECURITY PASSPORT",
        "=" * 72,
        f"  Passport ID   : {p['passport_id']}",
        f"  Generated At  : {p['generated_at']}",
        f"  Model ID      : {p['model']['model_id']}",
        f"  Model Type    : {p['model']['type']}",
        f"  Dataset       : {p['dataset']['dataset_id']}",
        f"  Source        : {p['dataset']['source']}",
        f"  Final Status  : {p['final_model_status']}",
        "",
        "— CHECKPOINT SUMMARY " + "-" * 50,
    ]
    for ckpt in p["checkpoint_history"]:
        lines.append(
            f"  Epoch {ckpt['epoch']:04d}  state={ckpt['state']}"
            f"  ts={ckpt['timestamp']}"
        )

    lines += ["", "— SECURITY TRAJECTORY " + "-" * 49]
    for rec in p["security_trajectory"]:
        lines.append(
            f"  Epoch {rec['epoch']:04d}  risk={rec['risk_level']:<8s}"
            f"  score={rec['risk_score']:.4f}"
            f"  bk_rate={rec['backdoor_success_rate']:.4f}"
        )

    lines += ["", "— SUSPICIOUS SAMPLES " + "-" * 50]
    if p["suspicious_samples"]:
        for entry in p["suspicious_samples"]:
            lines.append(
                f"  Epoch {entry['epoch']}  type={entry['attack_type']}"
                f"  n={entry['n_samples']}"
            )
    else:
        lines.append("  None detected.")

    lines += ["", "— COUNTERFACTUAL EVIDENCE " + "-" * 46]
    cf = p.get("counterfactual_evidence")
    if cf:
        lines.append(f"  Attribution confidence : {cf.get('attribution_confidence', 'N/A')}")
        if cf.get("experiment_a"):
            lines.append(f"  With suspects    acc={cf['experiment_a']['accuracy']:.4f}"
                         f"  bk={cf['experiment_a']['backdoor_success_rate']:.4f}")
        if cf.get("experiment_b"):
            lines.append(f"  Without suspects acc={cf['experiment_b']['accuracy']:.4f}"
                         f"  bk={cf['experiment_b']['backdoor_success_rate']:.4f}")
        if cf.get("difference"):
            lines.append(f"  Δ accuracy={cf['difference']['accuracy_delta']:+.6f}"
                         f"  Δ backdoor={cf['difference']['backdoor_rate_delta']:+.6f}")
        lines.append(f"  Note: {cf.get('note', '')}")
    else:
        lines.append("  No counterfactual run.")

    lines += ["", "— RESPONSE ACTIONS " + "-" * 52]
    for evt in p["response_actions"]:
        lines.append(
            f"  Epoch {evt['epoch']}  {evt['risk_level']:<8s}  action={evt['action']}"
            f"  ts={evt['timestamp']}"
        )

    lines += ["", "— ROLLBACK " + "-" * 60]
    rb = p.get("rollback")
    if rb and rb.get("success"):
        lines.append(f"  Rolled back to epoch {rb['restored_epoch']}")
        vm = rb.get("verified_metrics", {})
        lines.append(f"  Verified accuracy={vm.get('accuracy', 'N/A'):.4f}"
                     f"  f1={vm.get('f1', 'N/A'):.4f}")
    else:
        lines.append("  No rollback performed.")

    lines += ["", "=" * 72, "  Protect AI while it is learning.  — Model Immune System", "=" * 72]
    return "\n".join(lines) + "\n"

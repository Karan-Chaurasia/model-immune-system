"""
security_passport.py - AI Security Passport generator.
Complete audit trail: identity, dataset, attacks, checkpoints,
anomalies, suspicious samples, counterfactual evidence, response actions.
"""
import json, logging, os, uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _make_serializable(obj):
    """Recursively strip any non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(i) for i in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    # numpy scalars, sklearn objects, anything else -> string repr
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
    except ImportError:
        pass
    return str(obj)


def generate_passport(cfg, trajectory_records, checkpoint_list,
                      suspicious_samples, counterfactual_result,
                      response_events, rollback_result, final_model_status):
    passport = {
        "passport_id":    str(uuid.uuid4()),
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_id":     f"mis-{cfg['project']['version']}",
            "version":      cfg["project"]["version"],
            "type":         cfg["model"]["type"],
            "random_state": cfg["model"]["random_state"],
        },
        "dataset": {
            "dataset_id":    cfg["dataset"]["name"],
            "source":        cfg["dataset"]["url"],
            "target_column": cfg["dataset"]["target_column"],
        },
        "training_config": {
            "epochs":     cfg["training"]["epochs"],
            "batch_size": cfg["training"]["batch_size"],
            "seed":       cfg["project"]["seed"],
        },
        "attack_config":      cfg["attacks"],
        "checkpoint_history": checkpoint_list,
        "security_trajectory": trajectory_records,
        "detected_anomalies": [r for r in trajectory_records
                               if r.get("risk_level") in ("HIGH","CRITICAL")],
        "suspicious_samples":    suspicious_samples,
        "counterfactual_evidence": counterfactual_result,
        "response_actions":      response_events,
        "rollback":              _safe_rollback(rollback_result),
        "final_model_status":    final_model_status,
        "reproducibility": {
            "seed": cfg["project"]["seed"],
            "attack_seeds": {
                "poisoning": cfg["attacks"]["poisoning"]["seed"],
                "backdoor":  cfg["attacks"]["backdoor"]["seed"],
            },
        },
    }
    return _make_serializable(passport)


def _safe_rollback(rb):
    """Strip the live model object from rollback result before serialising."""
    if rb is None:
        return None
    return {k: v for k, v in rb.items() if k != "model"}


def save_passport(passport, out_dir="logs"):
    os.makedirs(out_dir, exist_ok=True)
    pid       = passport["passport_id"][:8]
    json_path = os.path.join(out_dir, f"security_passport_{pid}.json")
    text_path = os.path.join(out_dir, f"security_passport_{pid}.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(passport, f, indent=2)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(_render_text(passport))
    logger.info("Security Passport saved: %s", json_path)
    return json_path, text_path


def _render_text(p):
    sep = "=" * 70
    lines = [sep,
             "  MODEL IMMUNE SYSTEM -- AI SECURITY PASSPORT",
             sep,
             f"  Passport ID  : {p['passport_id']}",
             f"  Generated    : {p['generated_at']}",
             f"  Model        : {p['model']['model_id']} ({p['model']['type']})",
             f"  Dataset      : {p['dataset']['dataset_id']}",
             f"  Final Status : {p['final_model_status']}",
             "",
             "-- CHECKPOINT HISTORY " + "-"*48]
    for c in p["checkpoint_history"]:
        lines.append(f"  epoch={c['epoch']:04d}  state={c['state']}")
    lines += ["", "-- SECURITY TRAJECTORY " + "-"*47]
    for r in p["security_trajectory"]:
        lines.append(f"  epoch={r['epoch']:04d}  {r['risk_level']:<8s}  "
                     f"score={r['risk_score']:.4f}  bk={r['backdoor_success_rate']:.4f}")
    lines += ["", "-- SUSPICIOUS SAMPLES " + "-"*48]
    for e in p["suspicious_samples"]:
        lines.append(f"  epoch={e['epoch']}  type={e['attack_type']}  n={e['n_samples']}")
    lines += ["", "-- COUNTERFACTUAL EVIDENCE " + "-"*44]
    cf = p.get("counterfactual_evidence")
    if cf:
        lines.append(f"  Attribution confidence : {cf.get('attribution_confidence','N/A')}")
        if cf.get("experiment_a"):
            lines.append(f"  With suspects    bk={cf['experiment_a']['backdoor_success_rate']:.4f}  acc={cf['experiment_a']['accuracy']:.4f}")
        if cf.get("experiment_b"):
            lines.append(f"  Without suspects bk={cf['experiment_b']['backdoor_success_rate']:.4f}  acc={cf['experiment_b']['accuracy']:.4f}")
        d = cf.get("difference",{})
        if d:
            lines.append(f"  Delta backdoor={d.get('backdoor_rate_delta',0):+.4f}  Delta acc={d.get('accuracy_delta',0):+.4f}")
    else:
        lines.append("  None.")
    lines += ["", "-- RESPONSE ACTIONS " + "-"*50]
    for e in p["response_actions"]:
        lines.append(f"  epoch={e['epoch']}  {e['risk_level']:<8s}  action={e['action']}")
    rb = p.get("rollback")
    if rb and rb.get("success"):
        lines += ["", "-- ROLLBACK " + "-"*58,
                  f"  Rolled back to epoch {rb['restored_epoch']}",
                  f"  Verified acc={rb.get('verified_metrics',{}).get('accuracy','N/A')}"]
    lines += ["", sep, "  Protect AI while it is learning.  -- Model Immune System", sep]
    return "\n".join(lines) + "\n"
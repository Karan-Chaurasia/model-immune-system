"""
checkpoint.py — save, load, list, and classify training checkpoints.

Checkpoint states:
    TRUSTED      — verified safe
    MONITORED    — under observation
    SUSPICIOUS   — anomaly detected
    QUARANTINED  — unsafe, blocked from use
"""

import os
import json
import pickle
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

VALID_STATES = {"TRUSTED", "MONITORED", "SUSPICIOUS", "QUARANTINED"}


def _checkpoint_dir(cfg: dict) -> str:
    d = cfg["training"]["checkpoint_dir"]
    os.makedirs(d, exist_ok=True)
    return d


def save_checkpoint(
    model,
    epoch: int,
    metrics: dict,
    state: str,
    cfg: dict,
    extra: Optional[dict] = None,
) -> str:
    """
    Persist model + metadata for a given epoch.
    Returns the checkpoint directory path.
    """
    if state not in VALID_STATES:
        raise ValueError(f"Invalid checkpoint state: {state}")

    base_dir = _checkpoint_dir(cfg)
    ckpt_path = os.path.join(base_dir, f"epoch_{epoch:04d}")
    os.makedirs(ckpt_path, exist_ok=True)

    # Persist model
    model_file = os.path.join(ckpt_path, "model.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    # Persist metadata
    meta = {
        "epoch": epoch,
        "state": state,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    if extra:
        meta["extra"] = extra

    meta_file = os.path.join(ckpt_path, "meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Checkpoint saved  epoch=%d  state=%s  path=%s", epoch, state, ckpt_path)
    return ckpt_path


def load_checkpoint(epoch: int, cfg: dict) -> tuple:
    """
    Load model and metadata for a given epoch.
    Returns (model, meta_dict).
    """
    base_dir = _checkpoint_dir(cfg)
    ckpt_path = os.path.join(base_dir, f"epoch_{epoch:04d}")

    model_file = os.path.join(ckpt_path, "model.pkl")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    meta_file = os.path.join(ckpt_path, "meta.json")
    with open(meta_file, "r") as f:
        meta = json.load(f)

    logger.info("Checkpoint loaded  epoch=%d  state=%s", epoch, meta["state"])
    return model, meta


def update_checkpoint_state(epoch: int, new_state: str, cfg: dict) -> None:
    """Change the state field in a checkpoint's metadata file."""
    if new_state not in VALID_STATES:
        raise ValueError(f"Invalid checkpoint state: {new_state}")

    base_dir = _checkpoint_dir(cfg)
    meta_file = os.path.join(base_dir, f"epoch_{epoch:04d}", "meta.json")

    with open(meta_file, "r") as f:
        meta = json.load(f)

    old_state = meta["state"]
    meta["state"] = new_state
    meta["state_updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Checkpoint state updated  epoch=%d  %s → %s",
        epoch, old_state, new_state,
    )


def list_checkpoints(cfg: dict) -> list[dict]:
    """Return metadata for all checkpoints, sorted by epoch."""
    base_dir = _checkpoint_dir(cfg)
    results = []

    for entry in sorted(os.listdir(base_dir)):
        meta_file = os.path.join(base_dir, entry, "meta.json")
        if os.path.isfile(meta_file):
            with open(meta_file, "r") as f:
                results.append(json.load(f))

    return sorted(results, key=lambda m: m["epoch"])


def get_latest_trusted(cfg: dict) -> Optional[dict]:
    """Return metadata of the most recent TRUSTED checkpoint, or None."""
    trusted = [
        m for m in list_checkpoints(cfg)
        if m["state"] == "TRUSTED"
    ]
    return trusted[-1] if trusted else None

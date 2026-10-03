"""
trajectory.py — records per-checkpoint security and performance metrics.

A single TrajectoryRecord captures everything needed to reconstruct
the model's safety state at each training epoch.
"""

import json
import os
import logging
from dataclasses import dataclass, asdict, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryRecord:
    epoch: int
    # performance
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_f1: float = 0.0
    val_auc: float = 0.0
    # attack signals
    backdoor_success_rate: float = 0.0
    positive_fraction: float = 0.0
    prob_mean: float = 0.0
    prob_std: float = 0.0
    # security signals
    poisoning_anomaly_score: float = 0.0
    behavioral_drift_score: float = 0.0
    prediction_distribution_anomaly: float = 0.0
    training_trajectory_anomaly: float = 0.0
    risk_score: float = 0.0
    risk_level: str = "LOW"
    # checkpoint state
    checkpoint_state: str = "TRUSTED"
    # optional: which attack was active this epoch
    attack_active: Optional[str] = None
    # free-form notes
    notes: str = ""


class TrajectoryLog:
    """Append-only log of TrajectoryRecords, persisted as JSONL."""

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, "trajectory.jsonl")
        self._records: list[TrajectoryRecord] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(TrajectoryRecord(**json.loads(line)))
        logger.debug("Loaded %d existing trajectory records.", len(self._records))

    def append(self, record: TrajectoryRecord) -> None:
        self._records.append(record)
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def all(self) -> list[TrajectoryRecord]:
        return list(self._records)

    def as_dicts(self) -> list[dict]:
        return [asdict(r) for r in self._records]

    def latest(self) -> Optional[TrajectoryRecord]:
        return self._records[-1] if self._records else None

    def for_epoch(self, epoch: int) -> Optional[TrajectoryRecord]:
        for r in self._records:
            if r.epoch == epoch:
                return r
        return None

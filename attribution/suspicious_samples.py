"""
suspicious_samples.py — identify training samples associated with security events.

Links attack metadata (poisoned/triggered sample IDs) to checkpoint epochs
and behavioral anomalies to answer:

    "Which training data/batch is associated with this security event?"
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class SuspiciousSampleTracker:
    """
    Maintains a registry of suspicious samples and the evidence linking
    them to security events.

    Evidence entries are written to logs/suspicious_samples.jsonl.
    """

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, "suspicious_samples.jsonl")
        self._entries: list[dict] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._entries.append(json.loads(line))

    def register(
        self,
        epoch: int,
        sample_ids: list[int],
        attack_type: str,
        evidence: dict,
    ) -> None:
        """Record a batch of suspicious samples linked to a specific epoch."""
        entry = {
            "epoch": epoch,
            "attack_type": attack_type,
            "sample_ids": sample_ids,
            "n_samples": len(sample_ids),
            "evidence": evidence,
        }
        self._entries.append(entry)
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(
            "Registered %d suspicious samples at epoch %d (attack=%s).",
            len(sample_ids), epoch, attack_type,
        )

    def all(self) -> list[dict]:
        return list(self._entries)

    def for_epoch(self, epoch: int) -> list[dict]:
        return [e for e in self._entries if e["epoch"] == epoch]

    def all_sample_ids(self) -> list[int]:
        ids: set[int] = set()
        for entry in self._entries:
            ids.update(entry["sample_ids"])
        return sorted(ids)

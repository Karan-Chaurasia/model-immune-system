"""
policy_engine.py — AI security kill switch / policy enforcement.

Maps risk levels to automated response actions and logs every decision.
Supports automatic and manual (human-approval) modes.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Ordered from least to most severe
RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class PolicyEngine:
    """
    Evaluates risk level against configured policy and triggers responses.

    All decisions are appended to logs/response_log.jsonl.
    """

    def __init__(self, cfg: dict, log_dir: str = "logs"):
        self._mode = cfg["response"]["mode"]
        self._actions = cfg["response"]["actions"]
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "response_log.jsonl")
        self._events: list[dict] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if not os.path.exists(self._log_path):
            return
        with open(self._log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._events.append(json.loads(line))

    def evaluate(self, epoch: int, risk_level: str, risk_score: float) -> dict:
        """
        Decide and execute the response for a given risk level.

        Returns a response dict describing what was decided and why.
        """
        action_key = risk_level.lower()
        action = self._actions.get(action_key, "continue")

        response = {
            "epoch": epoch,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "mode": self._mode,
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requires_human_approval": self._mode == "manual" and risk_level != "LOW",
        }

        self._log(response)
        logger.info(
            "Policy decision  epoch=%d  level=%s  score=%.4f  action=%s",
            epoch, risk_level, risk_score, action,
        )
        return response

    def _log(self, response: dict) -> None:
        self._events.append(response)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(response) + "\n")

    def all_events(self) -> list[dict]:
        return list(self._events)

    def should_pause(self, risk_level: str) -> bool:
        return "pause" in self._actions.get(risk_level.lower(), "")

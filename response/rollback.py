"""
rollback.py — pause, quarantine, and rollback unsafe training.

When risk becomes CRITICAL:
    1. Pause training (signal returned to caller)
    2. Mark current checkpoint QUARANTINED
    3. Locate latest TRUSTED checkpoint
    4. Restore it
    5. Verify restored model health
"""

import logging
from training.checkpoint import (
    update_checkpoint_state,
    load_checkpoint,
    get_latest_trusted,
)
from training.model import evaluate

logger = logging.getLogger(__name__)


class RollbackManager:
    """Handles quarantine and rollback operations."""

    def quarantine(self, epoch: int, cfg: dict) -> None:
        """Mark the checkpoint at `epoch` as QUARANTINED."""
        update_checkpoint_state(epoch, "QUARANTINED", cfg)
        logger.warning("Checkpoint epoch=%d QUARANTINED.", epoch)

    def rollback(
        self,
        X_val,
        y_val,
        cfg: dict,
    ) -> dict:
        """
        Roll back to the latest TRUSTED checkpoint.

        Returns a dict describing the rollback outcome.
        """
        trusted_meta = get_latest_trusted(cfg)
        if trusted_meta is None:
            logger.error("No trusted checkpoint available for rollback.")
            return {
                "success": False,
                "reason": "No trusted checkpoint available.",
                "restored_epoch": None,
            }

        trusted_epoch = trusted_meta["epoch"]
        model, meta = load_checkpoint(trusted_epoch, cfg)

        # Verify the restored model is healthy
        val_metrics = evaluate(model, X_val, y_val, label="rollback_verify")
        logger.info(
            "Rollback complete → epoch=%d  val_accuracy=%.4f",
            trusted_epoch, val_metrics["accuracy"],
        )

        return {
            "success": True,
            "restored_epoch": trusted_epoch,
            "restored_state": meta["state"],
            "verified_metrics": val_metrics,
            "model": model,
        }

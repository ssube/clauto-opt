"""Optional Weights & Biases integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clauto_opt.models import ParameterUpdate, TrainingContext

logger = logging.getLogger(__name__)


class WandbTracker:
    """Tracks training metrics and consultation events to W&B.

    All methods are no-ops if wandb is not available or no run exists,
    unless wandb_enabled=True in which case a run is created.
    """

    def __init__(self, enabled: bool = False, log_loss: bool = False) -> None:
        self._enabled = enabled
        self._log_loss = log_loss
        self._wandb: Any = None

        try:
            import wandb

            self._wandb = wandb
        except ImportError:
            if enabled:
                raise ImportError("wandb is required when wandb_enabled=True. Install with: pip install wandb")
            return

        if enabled and wandb.run is None:
            wandb.init(project="clauto-opt")

    def _has_run(self) -> bool:
        return self._wandb is not None and self._wandb.run is not None

    def log_loss(self, step: int, loss: float) -> None:
        """Log per-step loss if log_loss is enabled and a run exists."""
        if self._log_loss and self._has_run():
            self._wandb.log({"loss": loss, "step": step}, step=step)

    def log_consultation(self, step: int, context: TrainingContext, update: ParameterUpdate) -> None:
        """Log a consultation event: prompt context and Claude's response."""
        if not self._has_run():
            return

        self._wandb.log(
            {
                "consultation/step": step,
                "consultation/loss_current": context.loss_current,
                "consultation/reasoning": update.reasoning,
                "consultation/should_stop": update.should_stop,
            },
            step=step,
        )

    def log_parameter_change(self, step: int, param_name: str, old_value: float, new_value: float) -> None:
        """Log an individual parameter change."""
        if not self._has_run():
            return

        self._wandb.log(
            {
                f"param_change/{param_name}_old": old_value,
                f"param_change/{param_name}_new": new_value,
            },
            step=step,
        )

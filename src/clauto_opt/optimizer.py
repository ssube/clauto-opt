"""ClaudeOptimizer — wraps a PyTorch optimizer and consults Claude for hyperparameter tuning."""

from __future__ import annotations

import importlib.resources
import logging
from collections import deque
from pathlib import Path
from typing import Any, Callable

import jinja2
import torch

from clauto_opt.backends import Backend, create_backend
from clauto_opt.config import ClaudeOptimizerConfig
from clauto_opt.models import ParameterUpdate, TrainingContext
from clauto_opt.tracking import WandbTracker
from clauto_opt.triggers import ConsultationTrigger, IntervalTrigger, PlateauTrigger, SpikeTrigger

logger = logging.getLogger(__name__)


class ClaudeOptimizer:
    """Wraps a PyTorch optimizer and periodically consults Claude for hyperparameter tuning.

    Does NOT inherit from torch.optim.Optimizer. Delegates to the inner optimizer.
    """

    def __init__(
        self,
        inner: torch.optim.Optimizer,
        *,
        backend: str | Backend = "api",
        config: ClaudeOptimizerConfig | None = None,
        triggers: list[ConsultationTrigger] | None = None,
    ) -> None:
        self.inner = inner
        self.config = config or ClaudeOptimizerConfig()

        # Backend setup
        if isinstance(backend, str):
            self._backend = create_backend(backend, self.config)
        else:
            self._backend = backend

        # Trigger setup
        if triggers is not None:
            self._triggers = triggers
        else:
            self._triggers = [
                IntervalTrigger(self.config.consult_every_n_steps),
                PlateauTrigger(self.config.plateau_patience, self.config.plateau_threshold),
                SpikeTrigger(self.config.spike_factor, self.config.spike_window),
            ]

        # State
        self._step_count: int = 0
        self._loss_history: deque[float] = deque(maxlen=self.config.loss_history_maxlen)
        self._previous_updates: list[ParameterUpdate] = []
        self._consultation_count: int = 0

        # Prodigy detection
        self._is_prodigy = self._detect_prodigy() if self.config.auto_detect_prodigy else False

        # Templates
        self._initial_template = self._load_template(self.config.initial_template, "initial.md.j2")
        self._incremental_template = self._load_template(self.config.incremental_template, "incremental.md.j2")

        # W&B
        self._tracker = WandbTracker(enabled=self.config.wandb_enabled, log_loss=self.config.wandb_log_loss)

    # -- Delegated properties/methods --

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.inner.param_groups

    def state_dict(self) -> dict[str, Any]:
        return self.inner.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.inner.load_state_dict(state_dict)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    # -- Core step --

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform an optimization step, consulting Claude if any trigger fires."""
        loss = self.inner.step(closure)
        self._step_count += 1

        # Check triggers
        loss_list = list(self._loss_history)
        if loss_list and any(t.check(self._step_count, loss_list) for t in self._triggers):
            self._consult()

        return loss

    def record_loss(self, value: float) -> None:
        """Record a loss value for tracking and trigger evaluation."""
        self._loss_history.append(value)
        self._tracker.log_loss(self._step_count, value)

    # -- Consultation --

    def _consult(self) -> None:
        """Build context, render prompt, consult Claude, and apply updates."""
        context = self._build_context()

        # Choose template
        if self._consultation_count == 0:
            template = self._initial_template
        else:
            template = self._incremental_template

        prompt = template.render(context.model_dump())
        logger.info("Consulting Claude at step %d (consultation #%d)", self._step_count, self._consultation_count + 1)

        update = self._backend.consult(prompt)
        logger.info("Claude recommends: %s", update.reasoning)

        if not self.config.dry_run:
            self._apply_update(update)
        else:
            logger.info("Dry run — not applying update")

        self._tracker.log_consultation(self._step_count, context, update)
        self._previous_updates.append(update)
        self._consultation_count += 1

        # Reset triggers after consultation
        for trigger in self._triggers:
            trigger.reset()

    @staticmethod
    def _sample_losses(losses: list[float], sampling_rate: float) -> list[float]:
        """Downsample a loss history by selecting every Nth point.

        A sampling_rate of 0.1 means keep 10% of points (every 10th).
        A sampling_rate of 1.0 means keep all points.
        Always includes the last point.
        """
        if sampling_rate >= 1.0 or len(losses) <= 1:
            return losses

        step_size = max(1, int(1.0 / sampling_rate))
        sampled = losses[::step_size]
        # Always include the most recent loss
        if sampled[-1] != losses[-1]:
            sampled.append(losses[-1])
        return sampled

    def _build_context(self) -> TrainingContext:
        """Build a TrainingContext from the current optimizer state."""
        loss_list = list(self._loss_history)
        sampled_losses = self._sample_losses(loss_list, self.config.loss_sampling_rate)
        recent_window = loss_list[-50:] if len(loss_list) >= 50 else loss_list

        # Sanitize param groups — remove tensors and non-serializable values
        sanitized_groups = []
        for group in self.inner.param_groups:
            sanitized: dict[str, object] = {}
            for key, value in group.items():
                if key == "params":
                    continue
                if isinstance(value, torch.Tensor):
                    sanitized[key] = value.item() if value.numel() == 1 else str(value.shape)
                elif isinstance(value, (int, float, str, bool, type(None))):
                    sanitized[key] = value
                else:
                    sanitized[key] = str(value)
            sanitized_groups.append(sanitized)

        return TrainingContext(
            step=self._step_count,
            loss_history=sampled_losses,
            loss_current=loss_list[-1] if loss_list else 0.0,
            loss_min=min(loss_list) if loss_list else 0.0,
            loss_max=max(loss_list) if loss_list else 0.0,
            loss_mean_recent=sum(recent_window) / len(recent_window) if recent_window else 0.0,
            param_groups=sanitized_groups,
            optimizer_type=self.inner.__class__.__name__,
            is_prodigy=self._is_prodigy,
            prodigy_d=self._get_prodigy_d(),
            previous_updates=self._previous_updates,
            consultation_count=self._consultation_count,
        )

    def _apply_update(self, update: ParameterUpdate) -> None:
        """Apply Claude's recommended parameter changes to the inner optimizer."""
        for group in self.inner.param_groups:
            if update.lr is not None:
                old_lr = group.get("lr", 0.0)
                if isinstance(old_lr, (int, float)) and old_lr > 0:
                    ratio = update.lr / old_lr
                    if ratio > self.config.lr_change_max_factor or ratio < 1.0 / self.config.lr_change_max_factor:
                        logger.warning(
                            "LR change ratio %.2f exceeds safety bound (max factor %.1f), skipping",
                            ratio,
                            self.config.lr_change_max_factor,
                        )
                        continue
                self._tracker.log_parameter_change(self._step_count, "lr", float(old_lr), update.lr)
                group["lr"] = update.lr

            if self._is_prodigy:
                if update.d0 is not None:
                    old_val = group.get("d0", 0.0)
                    self._tracker.log_parameter_change(self._step_count, "d0", float(old_val), update.d0)
                    group["d0"] = update.d0

                if update.d_coef is not None:
                    old_val = group.get("d_coef", 0.0)
                    self._tracker.log_parameter_change(self._step_count, "d_coef", float(old_val), update.d_coef)
                    group["d_coef"] = update.d_coef

                if update.growth_rate is not None:
                    old_val = group.get("growth_rate", 0.0)
                    self._tracker.log_parameter_change(
                        self._step_count, "growth_rate", float(old_val), update.growth_rate
                    )
                    group["growth_rate"] = update.growth_rate

    # -- Prodigy helpers --

    def _detect_prodigy(self) -> bool:
        """Check if the inner optimizer is Prodigy-based."""
        name = self.inner.__class__.__name__.lower()
        return "prodigy" in name

    def _get_prodigy_d(self) -> float | None:
        """Get the current Prodigy D estimate if available."""
        if not self._is_prodigy:
            return None
        # Prodigy stores 'd' in the optimizer state for each param group
        for group in self.inner.param_groups:
            if "d" in group:
                d = group["d"]
                return float(d) if not isinstance(d, float) else d
        return None

    # -- Template loading --

    @staticmethod
    def _load_template(override: str | Path | None, default_name: str) -> jinja2.Template:
        """Load a Jinja2 template from override or package defaults."""
        if override is not None:
            if isinstance(override, Path):
                content = override.read_text()
            else:
                content = override
            return jinja2.Template(content)

        # Load from package resources
        templates = importlib.resources.files("clauto_opt") / "templates"
        template_file = templates / default_name
        content = template_file.read_text(encoding="utf-8")
        return jinja2.Template(content)

    # -- Public accessors --

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def consultation_count(self) -> int:
        return self._consultation_count

    @property
    def previous_updates(self) -> list[ParameterUpdate]:
        return list(self._previous_updates)

    @property
    def should_stop(self) -> bool:
        """Check if Claude's most recent update recommends stopping."""
        if self._previous_updates:
            return self._previous_updates[-1].should_stop
        return False

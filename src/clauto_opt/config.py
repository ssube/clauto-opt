"""Configuration dataclass for ClaudeOptimizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClaudeOptimizerConfig:
    """Configuration for the Claude optimizer wrapper."""

    # Backend settings
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024

    # Trigger defaults
    consult_every_n_steps: int = 100
    plateau_patience: int = 20
    plateau_threshold: float = 1e-4
    spike_factor: float = 3.0
    spike_window: int = 10

    # Loss history
    loss_history_maxlen: int = 500
    loss_sampling_rate: float = 0.1

    # Template overrides (str content, Path to file, or None for default)
    initial_template: str | Path | None = None
    incremental_template: str | Path | None = None

    # Prodigy
    auto_detect_prodigy: bool = True

    # Safety
    lr_change_max_factor: float = 10.0
    dry_run: bool = False

    # Consultation
    consultation_timeout: float = 60.0
    system_prompt: str | None = None
    total_steps: int | None = None

    # W&B
    wandb_enabled: bool = False
    wandb_log_loss: bool = False

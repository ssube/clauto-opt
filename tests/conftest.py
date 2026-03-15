"""Shared test fixtures."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from clauto_opt.backends import Backend
from clauto_opt.models import ParameterUpdate, TrainingContext


@pytest.fixture
def small_model() -> nn.Module:
    return nn.Linear(2, 1)


@pytest.fixture
def adamw_optimizer(small_model: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(small_model.parameters(), lr=1e-3)


@pytest.fixture
def mock_backend() -> MagicMock:
    """A mock backend that returns a fixed ParameterUpdate."""
    backend = MagicMock(spec=Backend)
    backend.consult.return_value = ParameterUpdate(
        reasoning="Test recommendation: maintain current parameters",
        lr=None,
        should_stop=False,
    )
    return backend


@pytest.fixture
def sample_loss_history() -> list[float]:
    """A synthetic loss curve: exponential decay with noise."""
    return [1.0 * math.exp(-0.01 * i) + 0.01 * (i % 7) for i in range(200)]


@pytest.fixture
def training_context(sample_loss_history: list[float]) -> TrainingContext:
    """A pre-built TrainingContext for template testing."""
    recent = sample_loss_history[-50:]
    return TrainingContext(
        step=200,
        total_steps=1000,
        loss_history=sample_loss_history,
        loss_current=sample_loss_history[-1],
        loss_min=min(sample_loss_history),
        loss_max=max(sample_loss_history),
        loss_mean_recent=sum(recent) / len(recent),
        param_groups=[{"lr": 0.001, "weight_decay": 0.01, "betas": "(0.9, 0.999)"}],
        optimizer_type="AdamW",
        is_prodigy=False,
        prodigy_d=None,
        previous_updates=[],
        consultation_count=0,
    )

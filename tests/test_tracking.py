"""Tests for W&B tracking integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from clauto_opt.models import ParameterUpdate, TrainingContext
from clauto_opt.tracking import WandbTracker


@pytest.fixture
def mock_wandb():
    """Mock the wandb module."""
    mock = MagicMock()
    mock.run = MagicMock()  # Simulate active run
    return mock


@pytest.fixture
def simple_context() -> TrainingContext:
    return TrainingContext(
        step=100,
        loss_history=[1.0, 0.9],
        loss_current=0.9,
        loss_min=0.9,
        loss_max=1.0,
        loss_mean_recent=0.95,
        param_groups=[{"lr": 0.001}],
        optimizer_type="AdamW",
    )


class TestWandbTracker:
    def test_log_loss_when_enabled(self, mock_wandb: MagicMock) -> None:
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker = WandbTracker(enabled=False, log_loss=True)
            tracker._wandb = mock_wandb
            tracker.log_loss(10, 0.5)
            mock_wandb.log.assert_called_once()

    def test_log_loss_disabled(self, mock_wandb: MagicMock) -> None:
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker = WandbTracker(enabled=False, log_loss=False)
            tracker._wandb = mock_wandb
            tracker.log_loss(10, 0.5)
            mock_wandb.log.assert_not_called()

    def test_log_consultation(self, mock_wandb: MagicMock, simple_context: TrainingContext) -> None:
        update = ParameterUpdate(reasoning="Test update", lr=5e-4)
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker = WandbTracker(enabled=False, log_loss=False)
            tracker._wandb = mock_wandb
            tracker.log_consultation(100, simple_context, update)
            mock_wandb.log.assert_called_once()
            call_data = mock_wandb.log.call_args[0][0]
            assert call_data["consultation/reasoning"] == "Test update"

    def test_log_parameter_change(self, mock_wandb: MagicMock) -> None:
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker = WandbTracker(enabled=False, log_loss=False)
            tracker._wandb = mock_wandb
            tracker.log_parameter_change(50, "lr", 0.001, 0.0005)
            mock_wandb.log.assert_called_once()
            call_data = mock_wandb.log.call_args[0][0]
            assert call_data["param_change/lr_old"] == 0.001
            assert call_data["param_change/lr_new"] == 0.0005

    def test_noop_without_wandb(self) -> None:
        tracker = WandbTracker(enabled=False, log_loss=True)
        tracker._wandb = None
        # Should not raise
        tracker.log_loss(10, 0.5)
        tracker.log_parameter_change(10, "lr", 0.001, 0.0005)

    def test_noop_without_run(self, mock_wandb: MagicMock) -> None:
        mock_wandb.run = None
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker = WandbTracker(enabled=False, log_loss=True)
            tracker._wandb = mock_wandb
            tracker.log_loss(10, 0.5)
            mock_wandb.log.assert_not_called()

    def test_raises_when_enabled_without_wandb(self) -> None:
        with patch.dict("sys.modules", {"wandb": None}):
            # When wandb can't be imported and enabled=True, should raise
            with patch("builtins.__import__", side_effect=ImportError("no wandb")):
                with pytest.raises(ImportError, match="wandb is required"):
                    WandbTracker(enabled=True)

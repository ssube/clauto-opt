"""Integration tests for the ClaudeOptimizer wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from clauto_opt.config import ClaudeOptimizerConfig
from clauto_opt.models import ParameterUpdate
from clauto_opt.optimizer import ClaudeOptimizer
from clauto_opt.triggers import IntervalTrigger


class TestDelegation:
    def test_param_groups_delegated(self, small_model: nn.Module, adamw_optimizer: torch.optim.AdamW) -> None:
        opt = ClaudeOptimizer(adamw_optimizer, backend=MagicMock())
        assert opt.param_groups is adamw_optimizer.param_groups

    def test_zero_grad_delegated(self, small_model: nn.Module, adamw_optimizer: torch.optim.AdamW) -> None:
        opt = ClaudeOptimizer(adamw_optimizer, backend=MagicMock())
        # Should not raise
        opt.zero_grad()

    def test_state_dict_delegated(self, adamw_optimizer: torch.optim.AdamW) -> None:
        opt = ClaudeOptimizer(adamw_optimizer, backend=MagicMock())
        sd = opt.state_dict()
        assert "param_groups" in sd

    def test_step_delegates_to_inner(self, small_model: nn.Module, adamw_optimizer: torch.optim.AdamW) -> None:
        opt = ClaudeOptimizer(adamw_optimizer, backend=MagicMock())
        x = torch.randn(1, 2)
        loss = (small_model(x) - torch.tensor([[1.0]])).pow(2).mean()
        loss.backward()
        opt.step()
        assert opt.step_count == 1


class TestLossRecording:
    def test_record_loss(self, adamw_optimizer: torch.optim.AdamW) -> None:
        opt = ClaudeOptimizer(adamw_optimizer, backend=MagicMock())
        opt.record_loss(1.0)
        opt.record_loss(0.9)
        assert len(opt._loss_history) == 2

    def test_loss_history_maxlen(self, adamw_optimizer: torch.optim.AdamW) -> None:
        config = ClaudeOptimizerConfig(loss_history_maxlen=5)
        opt = ClaudeOptimizer(adamw_optimizer, backend=MagicMock(), config=config)
        for i in range(10):
            opt.record_loss(float(i))
        assert len(opt._loss_history) == 5
        assert list(opt._loss_history) == [5.0, 6.0, 7.0, 8.0, 9.0]


class TestLossSampling:
    def test_sampling_rate_1_returns_all(self) -> None:
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        result = ClaudeOptimizer._sample_losses(losses, 1.0)
        assert result == losses

    def test_sampling_rate_0_5_returns_half(self) -> None:
        losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        result = ClaudeOptimizer._sample_losses(losses, 0.5)
        # step_size = 2, so indices 0, 2, 4, 6, 8 plus last (index 9)
        assert result == [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]

    def test_sampling_rate_0_1_returns_tenth(self) -> None:
        losses = list(range(100))
        result = ClaudeOptimizer._sample_losses([float(x) for x in losses], 0.1)
        # step_size = 10, so indices 0, 10, 20, ..., 90, plus 99 (last point)
        assert result == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

    def test_sampling_rate_0_05(self) -> None:
        losses = [float(x) for x in range(100)]
        result = ClaudeOptimizer._sample_losses(losses, 0.05)
        # step_size = 20, so indices 0, 20, 40, 60, 80, plus 99
        assert result == [0, 20, 40, 60, 80, 99]

    def test_sampling_always_includes_last(self) -> None:
        losses = [float(x) for x in range(15)]
        result = ClaudeOptimizer._sample_losses(losses, 0.1)
        assert result[-1] == 14.0

    def test_single_value(self) -> None:
        assert ClaudeOptimizer._sample_losses([1.0], 0.1) == [1.0]

    def test_empty_list(self) -> None:
        assert ClaudeOptimizer._sample_losses([], 0.1) == []

    def test_sampling_used_in_context(self, adamw_optimizer: torch.optim.AdamW) -> None:
        config = ClaudeOptimizerConfig(loss_sampling_rate=0.5)
        opt = ClaudeOptimizer(adamw_optimizer, backend=MagicMock(), config=config)
        for i in range(10):
            opt.record_loss(float(i))
        ctx = opt._build_context()
        # With rate 0.5, step_size=2, indices 0,2,4,6,8 + last (9) = 6 points
        assert len(ctx.loss_history) == 6


class TestConsultation:
    def test_consults_at_interval(
        self, small_model: nn.Module, adamw_optimizer: torch.optim.AdamW, mock_backend: MagicMock
    ) -> None:
        config = ClaudeOptimizerConfig(consult_every_n_steps=5)
        triggers = [IntervalTrigger(every_n_steps=5)]
        opt = ClaudeOptimizer(adamw_optimizer, backend=mock_backend, config=config, triggers=triggers)

        for step in range(10):
            x = torch.randn(1, 2)
            loss = (small_model(x) - torch.tensor([[1.0]])).pow(2).mean()
            loss.backward()
            opt.record_loss(loss.item())
            opt.step()
            opt.zero_grad()

        # Should have consulted at step 5 and 10
        assert mock_backend.consult.call_count == 2

    def test_lr_update_applied(self, small_model: nn.Module, adamw_optimizer: torch.optim.AdamW) -> None:
        mock_be = MagicMock()
        mock_be.consult.return_value = ParameterUpdate(reasoning="Lower LR", lr=5e-4)

        config = ClaudeOptimizerConfig(consult_every_n_steps=3)
        triggers = [IntervalTrigger(every_n_steps=3)]
        opt = ClaudeOptimizer(adamw_optimizer, backend=mock_be, config=config, triggers=triggers)

        for step in range(3):
            x = torch.randn(1, 2)
            loss = (small_model(x) - torch.tensor([[1.0]])).pow(2).mean()
            loss.backward()
            opt.record_loss(loss.item())
            opt.step()
            opt.zero_grad()

        assert adamw_optimizer.param_groups[0]["lr"] == 5e-4

    def test_safety_bounds_reject_extreme_lr(self, small_model: nn.Module, adamw_optimizer: torch.optim.AdamW) -> None:
        mock_be = MagicMock()
        # Try to increase LR by 100x (way beyond default 10x limit)
        mock_be.consult.return_value = ParameterUpdate(reasoning="Extreme increase", lr=0.1)

        config = ClaudeOptimizerConfig(consult_every_n_steps=3, lr_change_max_factor=10.0)
        triggers = [IntervalTrigger(every_n_steps=3)]
        opt = ClaudeOptimizer(adamw_optimizer, backend=mock_be, config=config, triggers=triggers)

        original_lr = adamw_optimizer.param_groups[0]["lr"]

        for step in range(3):
            x = torch.randn(1, 2)
            loss = (small_model(x) - torch.tensor([[1.0]])).pow(2).mean()
            loss.backward()
            opt.record_loss(loss.item())
            opt.step()
            opt.zero_grad()

        # LR should be unchanged because the change was too extreme
        assert adamw_optimizer.param_groups[0]["lr"] == original_lr

    def test_dry_run_does_not_apply(self, small_model: nn.Module, adamw_optimizer: torch.optim.AdamW) -> None:
        mock_be = MagicMock()
        mock_be.consult.return_value = ParameterUpdate(reasoning="Change LR", lr=5e-4)

        config = ClaudeOptimizerConfig(consult_every_n_steps=3, dry_run=True)
        triggers = [IntervalTrigger(every_n_steps=3)]
        opt = ClaudeOptimizer(adamw_optimizer, backend=mock_be, config=config, triggers=triggers)

        original_lr = adamw_optimizer.param_groups[0]["lr"]

        for step in range(3):
            x = torch.randn(1, 2)
            loss = (small_model(x) - torch.tensor([[1.0]])).pow(2).mean()
            loss.backward()
            opt.record_loss(loss.item())
            opt.step()
            opt.zero_grad()

        assert adamw_optimizer.param_groups[0]["lr"] == original_lr
        assert mock_be.consult.call_count == 1

    def test_should_stop_surfaced(self, small_model: nn.Module, adamw_optimizer: torch.optim.AdamW) -> None:
        mock_be = MagicMock()
        mock_be.consult.return_value = ParameterUpdate(reasoning="Converged", should_stop=True)

        triggers = [IntervalTrigger(every_n_steps=3)]
        opt = ClaudeOptimizer(adamw_optimizer, backend=mock_be, triggers=triggers)

        for step in range(3):
            x = torch.randn(1, 2)
            loss = (small_model(x) - torch.tensor([[1.0]])).pow(2).mean()
            loss.backward()
            opt.record_loss(loss.item())
            opt.step()
            opt.zero_grad()

        assert opt.should_stop is True


class TestProdigyDetection:
    def test_detects_prodigy_by_name(self, adamw_optimizer: torch.optim.AdamW) -> None:
        # AdamW should not be detected as Prodigy
        opt = ClaudeOptimizer(adamw_optimizer, backend=MagicMock())
        assert opt._is_prodigy is False

    def test_mock_prodigy_detected(self, small_model: nn.Module) -> None:
        # Create a mock optimizer with Prodigy-like name using a real subclass
        ProdigyClass = type("Prodigy", (torch.optim.Optimizer,), {"step": lambda self, closure=None: None})
        mock_opt = MagicMock(spec=ProdigyClass)
        mock_opt.__class__ = ProdigyClass
        mock_opt.param_groups = [{"lr": 1.0, "d0": 1e-6, "d_coef": 1.0, "params": []}]

        opt = ClaudeOptimizer(mock_opt, backend=MagicMock())
        assert opt._is_prodigy is True

    def test_prodigy_params_applied(self) -> None:
        ProdigyClass = type("Prodigy", (torch.optim.Optimizer,), {"step": lambda self, closure=None: None})
        mock_opt = MagicMock(spec=ProdigyClass)
        mock_opt.__class__ = ProdigyClass
        mock_opt.param_groups = [{"lr": 1.0, "d0": 1e-6, "d_coef": 1.0, "growth_rate": 1.0, "params": []}]
        mock_opt.step.return_value = None

        mock_be = MagicMock()
        mock_be.consult.return_value = ParameterUpdate(reasoning="Adjust Prodigy", d0=1e-5, d_coef=0.5, growth_rate=2.0)

        triggers = [IntervalTrigger(every_n_steps=2)]
        opt = ClaudeOptimizer(mock_opt, backend=mock_be, triggers=triggers)

        opt.record_loss(1.0)
        opt.step()
        opt.record_loss(0.9)
        opt.step()

        assert mock_opt.param_groups[0]["d0"] == 1e-5
        assert mock_opt.param_groups[0]["d_coef"] == 0.5
        assert mock_opt.param_groups[0]["growth_rate"] == 2.0


class TestWithAdafactor:
    """Test with Adafactor if transformers is available."""

    @pytest.fixture
    def adafactor_optimizer(self, small_model: nn.Module):
        try:
            from transformers import Adafactor

            return Adafactor(small_model.parameters(), lr=1e-3, relative_step=False)
        except ImportError:
            pytest.skip("transformers not installed")

    def test_adafactor_step(self, small_model: nn.Module, adafactor_optimizer) -> None:
        opt = ClaudeOptimizer(adafactor_optimizer, backend=MagicMock())
        x = torch.randn(1, 2)
        loss = (small_model(x) - torch.tensor([[1.0]])).pow(2).mean()
        loss.backward()
        opt.record_loss(loss.item())
        opt.step()
        assert opt.step_count == 1
        assert opt._is_prodigy is False


class TestWithProdigy:
    """Test with real Prodigy optimizer if available."""

    @pytest.fixture
    def prodigy_optimizer(self, small_model: nn.Module):
        try:
            from prodigy_optimizer import Prodigy

            return Prodigy(small_model.parameters(), lr=1.0)
        except ImportError:
            pytest.skip("prodigy-optimizer not installed")

    def test_prodigy_detected(self, prodigy_optimizer) -> None:
        opt = ClaudeOptimizer(prodigy_optimizer, backend=MagicMock())
        assert opt._is_prodigy is True

    def test_prodigy_step(self, small_model: nn.Module, prodigy_optimizer) -> None:
        opt = ClaudeOptimizer(prodigy_optimizer, backend=MagicMock())
        x = torch.randn(1, 2)
        loss = (small_model(x) - torch.tensor([[1.0]])).pow(2).mean()
        loss.backward()
        opt.record_loss(loss.item())
        opt.step()
        assert opt.step_count == 1

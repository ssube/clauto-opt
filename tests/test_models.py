"""Tests for Pydantic data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from clauto_opt.models import ParameterUpdate, TrainingContext


class TestParameterUpdate:
    def test_valid_full_update(self) -> None:
        update = ParameterUpdate(
            reasoning="Reduce LR due to plateau",
            lr=5e-4,
            d0=1e-5,
            d_coef=0.8,
            growth_rate=1.5,
            should_stop=False,
        )
        assert update.lr == 5e-4
        assert update.d0 == 1e-5
        assert update.d_coef == 0.8
        assert update.growth_rate == 1.5
        assert update.should_stop is False

    def test_minimal_update(self) -> None:
        update = ParameterUpdate(reasoning="No changes needed")
        assert update.lr is None
        assert update.d0 is None
        assert update.d_coef is None
        assert update.growth_rate is None
        assert update.should_stop is False

    def test_partial_update(self) -> None:
        update = ParameterUpdate(reasoning="Only adjust LR", lr=1e-4)
        assert update.lr == 1e-4
        assert update.d0 is None

    def test_should_stop(self) -> None:
        update = ParameterUpdate(reasoning="Training has converged", should_stop=True)
        assert update.should_stop is True

    def test_missing_reasoning_fails(self) -> None:
        with pytest.raises(ValidationError):
            ParameterUpdate()  # type: ignore[call-arg]

    def test_from_dict(self) -> None:
        data = {"reasoning": "Test", "lr": 0.001, "should_stop": False}
        update = ParameterUpdate.model_validate(data)
        assert update.reasoning == "Test"
        assert update.lr == 0.001

    def test_json_roundtrip(self) -> None:
        original = ParameterUpdate(reasoning="Test roundtrip", lr=1e-3, should_stop=True)
        json_str = original.model_dump_json()
        restored = ParameterUpdate.model_validate_json(json_str)
        assert restored == original


class TestTrainingContext:
    def test_valid_context(self) -> None:
        ctx = TrainingContext(
            step=100,
            loss_history=[1.0, 0.9, 0.8],
            loss_current=0.8,
            loss_min=0.8,
            loss_max=1.0,
            loss_mean_recent=0.9,
            param_groups=[{"lr": 0.001}],
            optimizer_type="AdamW",
        )
        assert ctx.step == 100
        assert ctx.is_prodigy is False
        assert ctx.consultation_count == 0

    def test_prodigy_context(self) -> None:
        ctx = TrainingContext(
            step=50,
            loss_history=[1.0],
            loss_current=1.0,
            loss_min=1.0,
            loss_max=1.0,
            loss_mean_recent=1.0,
            param_groups=[{"lr": 1.0, "d0": 1e-6, "d_coef": 1.0}],
            optimizer_type="Prodigy",
            is_prodigy=True,
            prodigy_d=0.001,
        )
        assert ctx.is_prodigy is True
        assert ctx.prodigy_d == 0.001

    def test_with_previous_updates(self) -> None:
        update = ParameterUpdate(reasoning="prev", lr=5e-4)
        ctx = TrainingContext(
            step=200,
            loss_history=[0.5],
            loss_current=0.5,
            loss_min=0.5,
            loss_max=0.5,
            loss_mean_recent=0.5,
            param_groups=[{"lr": 5e-4}],
            optimizer_type="AdamW",
            previous_updates=[update],
            consultation_count=1,
        )
        assert len(ctx.previous_updates) == 1
        assert ctx.consultation_count == 1

"""Tests for consultation trigger strategies."""

from __future__ import annotations

from clauto_opt.triggers import ConsultationTrigger, IntervalTrigger, PlateauTrigger, SpikeTrigger


class TestIntervalTrigger:
    def test_fires_at_interval(self) -> None:
        trigger = IntervalTrigger(every_n_steps=10)
        losses = [1.0]
        assert not trigger.check(1, losses)
        assert not trigger.check(5, losses)
        assert trigger.check(10, losses)
        assert not trigger.check(11, losses)
        assert trigger.check(20, losses)

    def test_does_not_fire_at_step_zero(self) -> None:
        trigger = IntervalTrigger(every_n_steps=10)
        assert not trigger.check(0, [1.0])

    def test_every_one_step(self) -> None:
        trigger = IntervalTrigger(every_n_steps=1)
        assert trigger.check(1, [1.0])
        assert trigger.check(2, [1.0])
        assert trigger.check(100, [1.0])

    def test_implements_protocol(self) -> None:
        assert isinstance(IntervalTrigger(10), ConsultationTrigger)


class TestPlateauTrigger:
    def test_fires_on_plateau(self) -> None:
        trigger = PlateauTrigger(patience=5, threshold=0.01)
        # Build a flat loss history
        flat_losses = [1.0] * 20
        # Need enough steps without improvement
        for step in range(1, 15):
            trigger.check(step, flat_losses[: step + 5])
        # Should eventually fire
        assert any(trigger.check(step, flat_losses) for step in range(1, 30))

    def test_does_not_fire_while_improving(self) -> None:
        trigger = PlateauTrigger(patience=5, threshold=0.01)
        # Steadily decreasing loss
        improving = [1.0 - 0.05 * i for i in range(20)]
        for step in range(5, 20):
            assert not trigger.check(step, improving[: step + 1])

    def test_resets_after_firing(self) -> None:
        trigger = PlateauTrigger(patience=3, threshold=0.01)
        flat = [1.0] * 50
        fired = False
        for step in range(1, 50):
            if trigger.check(step, flat):
                fired = True
                break
        assert fired
        # After reset, internal state should be cleared
        assert trigger._best is None
        assert trigger._steps_without_improvement == 0

    def test_not_enough_history(self) -> None:
        trigger = PlateauTrigger(patience=10, threshold=0.01)
        assert not trigger.check(1, [1.0, 0.9])

    def test_implements_protocol(self) -> None:
        assert isinstance(PlateauTrigger(10, 0.01), ConsultationTrigger)


class TestSpikeTrigger:
    def test_fires_on_spike(self) -> None:
        trigger = SpikeTrigger(factor=2.0, window=5)
        # Normal losses followed by a spike
        losses = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0]
        assert trigger.check(7, losses)

    def test_does_not_fire_on_normal(self) -> None:
        trigger = SpikeTrigger(factor=2.0, window=5)
        losses = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1]
        assert not trigger.check(7, losses)

    def test_not_enough_history(self) -> None:
        trigger = SpikeTrigger(factor=2.0, window=5)
        assert not trigger.check(1, [1.0, 2.0])

    def test_exact_boundary(self) -> None:
        trigger = SpikeTrigger(factor=2.0, window=3)
        # Mean of [1.0, 1.0, 1.0] = 1.0, current = 2.0 -> exactly at boundary
        losses = [1.0, 1.0, 1.0, 1.0, 2.0]
        # 2.0 is not > 2.0 * 1.0, so should not fire
        assert not trigger.check(5, losses)

    def test_just_above_boundary(self) -> None:
        trigger = SpikeTrigger(factor=2.0, window=3)
        losses = [1.0, 1.0, 1.0, 1.0, 2.01]
        assert trigger.check(5, losses)

    def test_implements_protocol(self) -> None:
        assert isinstance(SpikeTrigger(2.0, 5), ConsultationTrigger)

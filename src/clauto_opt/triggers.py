"""Consultation trigger strategies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ConsultationTrigger(Protocol):
    """Protocol for deciding when to consult Claude."""

    def check(self, step: int, loss_history: list[float]) -> bool: ...

    def reset(self) -> None: ...


class IntervalTrigger:
    """Fires every N steps."""

    def __init__(self, every_n_steps: int) -> None:
        self.every_n_steps = every_n_steps

    def check(self, step: int, loss_history: list[float]) -> bool:
        return step > 0 and step % self.every_n_steps == 0

    def reset(self) -> None:
        pass


class PlateauTrigger:
    """Fires when loss hasn't improved by threshold over patience steps.

    Compares rolling mean of recent losses vs best seen. Resets after firing.
    """

    def __init__(self, patience: int, threshold: float) -> None:
        self.patience = patience
        self.threshold = threshold
        self._best: float | None = None
        self._steps_without_improvement: int = 0

    def check(self, step: int, loss_history: list[float]) -> bool:
        if len(loss_history) < self.patience:
            return False

        recent = loss_history[-self.patience :]
        recent_mean = sum(recent) / len(recent)

        if self._best is None or recent_mean < self._best - self.threshold:
            self._best = recent_mean
            self._steps_without_improvement = 0
            return False

        self._steps_without_improvement += 1
        if self._steps_without_improvement >= self.patience:
            self.reset()
            return True

        return False

    def reset(self) -> None:
        self._best = None
        self._steps_without_improvement = 0


class SpikeTrigger:
    """Fires when current loss exceeds factor * rolling mean of recent losses."""

    def __init__(self, factor: float, window: int) -> None:
        self.factor = factor
        self.window = window

    def check(self, step: int, loss_history: list[float]) -> bool:
        if len(loss_history) < self.window + 1:
            return False

        # Compare current loss against rolling mean of the window before it
        current = loss_history[-1]
        window_losses = loss_history[-(self.window + 1) : -1]
        rolling_mean = sum(window_losses) / len(window_losses)

        return current > self.factor * rolling_mean

    def reset(self) -> None:
        pass

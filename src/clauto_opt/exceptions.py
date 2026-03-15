"""Exceptions for clauto_opt."""

from __future__ import annotations


class ConsultationError(Exception):
    """Raised when a Claude consultation fails (network, timeout, parsing, etc.).

    Training should continue — the optimizer catches this and logs a warning.
    """

    def __init__(self, message: str, original: Exception | None = None) -> None:
        super().__init__(message)
        self.original = original

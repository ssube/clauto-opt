"""Backend protocol and factory for Claude consultation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from clauto_opt.config import ClaudeOptimizerConfig
    from clauto_opt.models import ParameterUpdate


class Backend(Protocol):
    """Protocol for consulting Claude."""

    def consult(self, prompt: str) -> ParameterUpdate: ...

    def reset(self) -> None: ...


def create_backend(backend: str, config: ClaudeOptimizerConfig) -> Backend:
    """Factory to create a backend by name."""
    if backend == "api":
        from clauto_opt.backends.api import AnthropicAPIBackend

        return AnthropicAPIBackend(
            model=config.model,
            max_tokens=config.max_tokens,
            system_prompt=config.system_prompt,
        )
    elif backend == "cli":
        from clauto_opt.backends.cli import ClaudeCLIBackend

        return ClaudeCLIBackend(model=config.model, timeout=config.consultation_timeout)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'api' or 'cli'.")

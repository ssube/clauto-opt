"""Anthropic API backend for Claude consultation."""

from __future__ import annotations

import anthropic

from clauto_opt.exceptions import ConsultationError
from clauto_opt.models import ParameterUpdate

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert ML hyperparameter tuner. Analyze the training dynamics provided "
    "and respond with structured JSON recommending parameter adjustments."
)


class AnthropicAPIBackend:
    """Consults Claude via the Anthropic Python SDK with conversation history."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        client: anthropic.Anthropic | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._client = client or anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._messages: list[dict[str, str]] = []

    def consult(self, prompt: str) -> ParameterUpdate:
        self._messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                messages=self._messages,
                output_format=ParameterUpdate,
            )
        except (
            anthropic.APIError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        ) as exc:
            # Remove the dangling user message so conversation stays consistent
            self._messages.pop()
            raise ConsultationError(f"API call failed: {exc}", original=exc) from exc

        # Store assistant response for conversation continuity
        assistant_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                assistant_text += block.text
        self._messages.append({"role": "assistant", "content": assistant_text})

        if response.parsed_output is None:
            raise RuntimeError("Claude returned no structured output")

        return response.parsed_output

    def reset(self) -> None:
        self._messages.clear()

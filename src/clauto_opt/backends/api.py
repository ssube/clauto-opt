"""Anthropic API backend for Claude consultation."""

from __future__ import annotations

import anthropic

from clauto_opt.models import ParameterUpdate


class AnthropicAPIBackend:
    """Consults Claude via the Anthropic Python SDK with conversation history."""

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024) -> None:
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._messages: list[dict[str, str]] = []

    def consult(self, prompt: str) -> ParameterUpdate:
        self._messages.append({"role": "user", "content": prompt})

        response = self._client.messages.parse(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=self._messages,
            output_format=ParameterUpdate,
        )

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

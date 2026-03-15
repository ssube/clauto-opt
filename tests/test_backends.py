"""Tests for Claude consultation backends (mocked)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from clauto_opt.backends import create_backend
from clauto_opt.backends.api import AnthropicAPIBackend
from clauto_opt.backends.cli import ClaudeCLIBackend
from clauto_opt.config import ClaudeOptimizerConfig
from clauto_opt.models import ParameterUpdate


class TestAnthropicAPIBackend:
    def test_consult_sends_correct_params(self) -> None:
        with patch("clauto_opt.backends.api.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client

            update = ParameterUpdate(reasoning="Test", lr=1e-4)
            mock_response = MagicMock()
            mock_response.parsed_output = update
            mock_response.content = [MagicMock(text='{"reasoning":"Test","lr":0.0001}')]
            mock_client.messages.parse.return_value = mock_response

            backend = AnthropicAPIBackend(model="claude-sonnet-4-6", max_tokens=512)
            result = backend.consult("Test prompt")

            mock_client.messages.parse.assert_called_once()
            call_kwargs = mock_client.messages.parse.call_args
            assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6"
            assert call_kwargs.kwargs["max_tokens"] == 512
            # Messages list is mutated after call (assistant appended), so check
            # that the first message was our prompt
            messages = call_kwargs.kwargs["messages"]
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Test prompt"
            assert result == update

    def test_message_history_grows(self) -> None:
        with patch("clauto_opt.backends.api.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client

            update = ParameterUpdate(reasoning="Test")
            mock_response = MagicMock()
            mock_response.parsed_output = update
            mock_response.content = [MagicMock(text='{"reasoning":"Test"}')]
            mock_client.messages.parse.return_value = mock_response

            backend = AnthropicAPIBackend()

            backend.consult("First prompt")
            assert len(backend._messages) == 2  # user + assistant

            backend.consult("Second prompt")
            assert len(backend._messages) == 4  # 2 pairs

    def test_reset_clears_history(self) -> None:
        with patch("clauto_opt.backends.api.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client

            update = ParameterUpdate(reasoning="Test")
            mock_response = MagicMock()
            mock_response.parsed_output = update
            mock_response.content = [MagicMock(text='{"reasoning":"Test"}')]
            mock_client.messages.parse.return_value = mock_response

            backend = AnthropicAPIBackend()
            backend.consult("Prompt")
            assert len(backend._messages) == 2
            backend.reset()
            assert len(backend._messages) == 0

    def test_raises_on_no_parsed_output(self) -> None:
        with patch("clauto_opt.backends.api.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client

            mock_response = MagicMock()
            mock_response.parsed_output = None
            mock_response.content = [MagicMock(text="")]
            mock_client.messages.parse.return_value = mock_response

            backend = AnthropicAPIBackend()
            with pytest.raises(RuntimeError, match="no structured output"):
                backend.consult("Prompt")


class TestClaudeCLIBackend:
    def test_first_call_uses_session_id(self) -> None:
        cli_output = json.dumps({"structured_output": {"reasoning": "Test", "lr": 0.001, "should_stop": False}})
        with patch("clauto_opt.backends.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=cli_output, returncode=0)
            backend = ClaudeCLIBackend(model="claude-sonnet-4-6")
            result = backend.consult("Test prompt")

            cmd = mock_run.call_args[0][0]
            assert "--session-id" in cmd
            assert "--resume" not in cmd
            assert result.reasoning == "Test"
            assert result.lr == 0.001

    def test_subsequent_calls_use_resume(self) -> None:
        cli_output = json.dumps({"structured_output": {"reasoning": "Test", "should_stop": False}})
        with patch("clauto_opt.backends.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=cli_output, returncode=0)
            backend = ClaudeCLIBackend()

            backend.consult("First")
            backend.consult("Second")

            second_cmd = mock_run.call_args_list[1][0][0]
            assert "--resume" in second_cmd
            assert "--session-id" not in second_cmd

    def test_parses_result_field(self) -> None:
        cli_output = json.dumps({"result": json.dumps({"reasoning": "From result", "should_stop": False})})
        with patch("clauto_opt.backends.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=cli_output, returncode=0)
            backend = ClaudeCLIBackend()
            result = backend.consult("Prompt")
            assert result.reasoning == "From result"

    def test_parses_result_dict(self) -> None:
        cli_output = json.dumps({"result": {"reasoning": "Dict result", "should_stop": False}})
        with patch("clauto_opt.backends.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=cli_output, returncode=0)
            backend = ClaudeCLIBackend()
            result = backend.consult("Prompt")
            assert result.reasoning == "Dict result"

    def test_reset_changes_session(self) -> None:
        backend = ClaudeCLIBackend()
        old_session = backend._session_id
        backend.reset()
        assert backend._session_id != old_session
        assert backend._consultation_count == 0


class TestCreateBackend:
    def test_create_api_backend(self) -> None:
        with patch("clauto_opt.backends.api.anthropic"):
            config = ClaudeOptimizerConfig()
            backend = create_backend("api", config)
            assert isinstance(backend, AnthropicAPIBackend)

    def test_create_cli_backend(self) -> None:
        config = ClaudeOptimizerConfig()
        backend = create_backend("cli", config)
        assert isinstance(backend, ClaudeCLIBackend)

    def test_unknown_backend_raises(self) -> None:
        config = ClaudeOptimizerConfig()
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("invalid", config)

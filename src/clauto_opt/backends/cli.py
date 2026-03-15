"""Claude Code CLI backend for Claude consultation."""

from __future__ import annotations

import json
import subprocess
import uuid

from clauto_opt.exceptions import ConsultationError
from clauto_opt.models import ParameterUpdate


class ClaudeCLIBackend:
    """Consults Claude via the Claude Code CLI with session continuity."""

    def __init__(self, model: str = "claude-sonnet-4-6", timeout: float = 30.0) -> None:
        self._model = model
        self._timeout = timeout
        self._session_id: str = uuid.uuid4().hex
        self._consultation_count: int = 0
        self._schema = json.dumps(ParameterUpdate.model_json_schema())

    def consult(self, prompt: str) -> ParameterUpdate:
        cmd = ["claude", "-p", prompt, "--output-format", "json", "--json-schema", self._schema, "--model", self._model]

        if self._consultation_count == 0:
            cmd.extend(["--session-id", self._session_id])
        else:
            cmd.extend(["--resume", self._session_id])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=self._timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            raise ConsultationError(f"CLI call failed: {exc}", original=exc) from exc

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ConsultationError(f"CLI returned invalid JSON: {exc}", original=exc) from exc

        # CLI returns structured output in 'structured_output' or 'result' field
        try:
            if "structured_output" in data and data["structured_output"] is not None:
                update = ParameterUpdate.model_validate(data["structured_output"])
            elif "result" in data:
                # result may be a JSON string
                result_data = data["result"]
                if isinstance(result_data, str):
                    result_data = json.loads(result_data)
                update = ParameterUpdate.model_validate(result_data)
            else:
                raise ConsultationError(f"Unexpected CLI response format: {list(data.keys())}")
        except (json.JSONDecodeError, Exception) as exc:
            if isinstance(exc, ConsultationError):
                raise
            raise ConsultationError(f"Failed to parse CLI response: {exc}", original=exc) from exc

        self._consultation_count += 1
        return update

    def reset(self) -> None:
        self._session_id = uuid.uuid4().hex
        self._consultation_count = 0

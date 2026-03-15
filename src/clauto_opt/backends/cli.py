"""Claude Code CLI backend for Claude consultation."""

from __future__ import annotations

import json
import subprocess
import uuid

from clauto_opt.models import ParameterUpdate


class ClaudeCLIBackend:
    """Consults Claude via the Claude Code CLI with session continuity."""

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self._model = model
        self._session_id: str = uuid.uuid4().hex
        self._consultation_count: int = 0
        self._schema = json.dumps(ParameterUpdate.model_json_schema())

    def consult(self, prompt: str) -> ParameterUpdate:
        cmd = ["claude", "-p", prompt, "--output-format", "json", "--json-schema", self._schema, "--model", self._model]

        if self._consultation_count == 0:
            cmd.extend(["--session-id", self._session_id])
        else:
            cmd.extend(["--resume", self._session_id])

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # CLI returns structured output in 'structured_output' or 'result' field
        if "structured_output" in data and data["structured_output"] is not None:
            update = ParameterUpdate.model_validate(data["structured_output"])
        elif "result" in data:
            # result may be a JSON string
            result_data = data["result"]
            if isinstance(result_data, str):
                result_data = json.loads(result_data)
            update = ParameterUpdate.model_validate(result_data)
        else:
            raise RuntimeError(f"Unexpected CLI response format: {list(data.keys())}")

        self._consultation_count += 1
        return update

    def reset(self) -> None:
        self._session_id = uuid.uuid4().hex
        self._consultation_count = 0

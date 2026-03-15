# CLAUDE.md

## Project Overview

`clauto-opt` is a Python 3.12 library that wraps PyTorch optimizers and periodically consults Claude to tune hyperparameters based on training dynamics.

## Build & Test

```bash
# Setup
uv venv --python 3.12
uv pip install -e ".[dev]"

# Tests
.venv/bin/pytest tests/ -v --tb=short

# Format & Lint
.venv/bin/black src/ tests/
.venv/bin/isort src/ tests/
.venv/bin/flake8 src/ tests/

# Type checking (will have errors for untyped third-party deps)
.venv/bin/mypy src/
```

## Architecture

- **`src/clauto_opt/optimizer.py`** — `ClaudeOptimizer` wraps any `torch.optim.Optimizer` via delegation (not inheritance)
- **`src/clauto_opt/backends/`** — `AnthropicAPIBackend` (SDK) and `ClaudeCLIBackend` (subprocess)
- **`src/clauto_opt/triggers.py`** — `IntervalTrigger`, `PlateauTrigger`, `SpikeTrigger` decide when to consult
- **`src/clauto_opt/models.py`** — Pydantic models (`ParameterUpdate`, `TrainingContext`) for data contracts
- **`src/clauto_opt/templates/`** — Jinja2 templates for initial and incremental prompts
- **`src/clauto_opt/tracking.py`** — Optional W&B integration
- **`src/clauto_opt/config.py`** — `ClaudeOptimizerConfig` dataclass

## Key Design Decisions

- Loss history is sampled before sending to Claude (`loss_sampling_rate` config). Stats (min/max/mean) are computed from full history.
- Prodigy optimizers are detected by class name (`__class__.__name__`), not `type().__name__`, to work with mocks.
- Safety bounds reject LR changes exceeding `lr_change_max_factor` (default 10x).
- Templates are loaded via `importlib.resources` for package defaults.

## Code Style

- Line length: 120 (black, isort, flake8 all configured)
- Target Python: 3.12
- Use `from __future__ import annotations` in all modules

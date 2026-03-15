# Getting Started

## Prerequisites

- Python 3.12+
- An Anthropic API key (set `ANTHROPIC_API_KEY` environment variable), or Claude Code CLI installed

## Installation

### From source (recommended during development)

```bash
git clone https://github.com/ssube/clauto-opt.git
cd clauto-opt

# Create venv and install
uv venv --python 3.12
uv pip install -e ".[dev]"
```

### With optional extras

```bash
# Prodigy optimizer support
uv pip install -e ".[prodigy]"

# Weights & Biases tracking
uv pip install -e ".[wandb]"

# Everything
uv pip install -e ".[dev,prodigy,wandb]"
```

## Your First Training Loop

Here's a minimal example wrapping AdamW:

```python
import torch
from clauto_opt import ClaudeOptimizer, ClaudeOptimizerConfig

# 1. Define your model and optimizer as usual
model = torch.nn.Sequential(
    torch.nn.Linear(10, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)
inner = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 2. Wrap with ClaudeOptimizer
config = ClaudeOptimizerConfig(
    consult_every_n_steps=100,  # Ask Claude every 100 steps
    dry_run=True,               # Start with dry_run to see what Claude would do
)
opt = ClaudeOptimizer(inner, config=config)

# 3. Train — the loop looks almost identical to normal PyTorch
for step in range(500):
    x = torch.randn(8, 10)
    target = torch.randn(8, 1)
    loss = (model(x) - target).pow(2).mean()
    loss.backward()

    opt.record_loss(loss.item())  # Tell the optimizer about the loss
    opt.step()                     # Delegates to AdamW, may consult Claude
    opt.zero_grad()

    if opt.should_stop:
        print(f"Claude recommends stopping at step {step}")
        break
```

## Choosing a Backend

### Anthropic API (default)

Uses the Anthropic Python SDK. Requires `ANTHROPIC_API_KEY`:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
config = ClaudeOptimizerConfig(backend="api", model="claude-sonnet-4-6")
```

### Claude Code CLI

Uses the `claude` CLI tool via subprocess. Useful if you have Claude Code configured with specific permissions or MCP servers:

```python
config = ClaudeOptimizerConfig(backend="cli")
```

Both backends maintain conversation history so Claude sees the effect of its previous recommendations.

## Understanding Triggers

Three trigger types determine when Claude is consulted:

### IntervalTrigger

Fires every N steps. Simple and predictable.

```python
from clauto_opt import IntervalTrigger
trigger = IntervalTrigger(every_n_steps=100)
```

### PlateauTrigger

Fires when loss stops improving. Uses a rolling mean comparison with configurable patience and threshold.

```python
from clauto_opt import PlateauTrigger
trigger = PlateauTrigger(patience=20, threshold=1e-4)
```

### SpikeTrigger

Fires when current loss spikes above a multiple of the recent rolling mean. Catches training instabilities.

```python
from clauto_opt import SpikeTrigger
trigger = SpikeTrigger(factor=3.0, window=10)
```

By default, all three are active. Pass a custom list to use only what you want:

```python
opt = ClaudeOptimizer(
    inner,
    triggers=[IntervalTrigger(every_n_steps=50)],  # Only periodic
)
```

## Loss Sampling

When Claude is consulted, it sees a sampled version of the loss history to keep prompts manageable. The `loss_sampling_rate` controls the density:

- `1.0` — every data point (no sampling)
- `0.1` — every 10th point (default)
- `0.05` — every 20th point

Summary statistics (min, max, recent mean) are always computed from the full history.

```python
config = ClaudeOptimizerConfig(
    loss_sampling_rate=0.1,       # 10% of points in the prompt
    loss_history_maxlen=500,      # Keep last 500 losses in the ring buffer
)
```

## Safety Features

### LR Change Bounds

By default, Claude can't change the learning rate by more than 10x in either direction:

```python
config = ClaudeOptimizerConfig(lr_change_max_factor=10.0)
```

### Dry Run Mode

See what Claude would recommend without actually changing parameters:

```python
config = ClaudeOptimizerConfig(dry_run=True)
```

Check logs for Claude's recommendations:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Custom Templates

Override the Jinja2 templates Claude sees:

```python
# From a string
config = ClaudeOptimizerConfig(
    initial_template="You are tuning a {{ optimizer_type }}. Loss: {{ loss_current }}. Recommend changes.",
)

# From a file
from pathlib import Path
config = ClaudeOptimizerConfig(
    initial_template=Path("my_templates/initial.md.j2"),
    incremental_template=Path("my_templates/incremental.md.j2"),
)
```

Templates receive a `TrainingContext` object with fields like `step`, `loss_history`, `param_groups`, `is_prodigy`, etc.

## W&B Integration

```python
import wandb

wandb.init(project="my-training")
config = ClaudeOptimizerConfig(
    wandb_enabled=True,
    wandb_log_loss=True,  # Log per-step loss to W&B
)
opt = ClaudeOptimizer(inner, config=config)
```

Consultation events, parameter changes, and Claude's reasoning are all logged to your W&B run.

## Next Steps

- Read the source in `src/clauto_opt/` to understand the internals
- Look at the Jinja2 templates in `src/clauto_opt/templates/` to see what Claude sees
- Run `pytest tests/ -v` to see all the test scenarios
- Try `dry_run=True` first to build confidence in Claude's recommendations

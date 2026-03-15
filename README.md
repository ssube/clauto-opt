# clauto-opt

**Claude Automatic Optimizer** — Let Claude tune your PyTorch hyperparameters during training.

`clauto-opt` wraps any PyTorch optimizer and periodically consults Claude (via the Anthropic API or Claude Code CLI) to adjust learning rates and other parameters based on real-time training dynamics. It sees your loss curves, understands your optimizer state, and responds with structured parameter recommendations.

## Why?

Hyperparameter tuning is one of the most time-consuming parts of training ML models. Grid search is wasteful. Schedulers are rigid. What if your optimizer could *reason* about your training dynamics and adapt?

`clauto-opt` gives Claude visibility into your training loop — loss history, optimizer state, parameter groups — and lets it make targeted adjustments. It's like having an ML engineer watching your training curves 24/7, except it's Claude.

## Features

- **Drop-in wrapper** for any `torch.optim.Optimizer` — AdamW, SGD, Adafactor, Prodigy, and more
- **Multiple consultation triggers** — periodic intervals, loss plateaus, and loss spikes
- **Prodigy-aware** — automatically detects Prodigy optimizers and tunes `d0`, `d_coef`, and `growth_rate`
- **Two backends** — Anthropic API (SDK) or Claude Code CLI with session continuity
- **Customizable prompts** — Jinja2 templates you can override for your domain
- **Loss sampling** — configurable sampling rate so Claude sees the right density of data points
- **Safety bounds** — rejects extreme parameter changes, supports dry-run mode
- **W&B integration** — optional logging of consultations and parameter changes
- **Conversation memory** — Claude sees its previous recommendations and their effects

## Quick Start

```bash
pip install clauto-opt
```

```python
import torch
from clauto_opt import ClaudeOptimizer, ClaudeOptimizerConfig

model = torch.nn.Linear(10, 1)
inner = torch.optim.AdamW(model.parameters(), lr=1e-3)
config = ClaudeOptimizerConfig(consult_every_n_steps=100)
opt = ClaudeOptimizer(inner, config=config)

for step in range(1000):
    x = torch.randn(4, 10)
    loss = (model(x) - torch.randn(4, 1)).pow(2).mean()
    loss.backward()
    opt.record_loss(loss.item())
    opt.step()
    opt.zero_grad()

    if opt.should_stop:
        print(f"Claude recommends stopping at step {step}")
        break
```

## Configuration

```python
ClaudeOptimizerConfig(
    # Backend
    model="claude-sonnet-4-6",       # Claude model to use
    backend="api",                    # "api" or "cli"

    # Triggers
    consult_every_n_steps=100,        # Periodic consultation interval
    plateau_patience=20,              # Steps before plateau triggers
    plateau_threshold=1e-4,           # Minimum improvement to avoid plateau
    spike_factor=3.0,                 # Loss spike detection multiplier
    spike_window=10,                  # Window for spike rolling mean

    # Loss
    loss_history_maxlen=500,          # Ring buffer size
    loss_sampling_rate=0.1,           # Fraction of points sent to Claude

    # Safety
    lr_change_max_factor=10.0,        # Max allowed LR change ratio
    dry_run=False,                    # Log recommendations without applying

    # Templates (override with str or Path)
    initial_template=None,
    incremental_template=None,

    # W&B
    wandb_enabled=False,
    wandb_log_loss=False,
)
```

## Custom Triggers

```python
from clauto_opt import ClaudeOptimizer, IntervalTrigger, PlateauTrigger

opt = ClaudeOptimizer(
    inner_optimizer,
    triggers=[
        IntervalTrigger(every_n_steps=50),
        PlateauTrigger(patience=30, threshold=1e-5),
    ],
)
```

## Prodigy Support

Prodigy and ProdigyPlusScheduleFree optimizers are automatically detected. Claude will tune `d0`, `d_coef`, and `growth_rate` in addition to learning rate.

```python
from prodigy_optimizer import Prodigy

inner = Prodigy(model.parameters(), lr=1.0)
opt = ClaudeOptimizer(inner)  # Prodigy params auto-detected
```

## CLI Backend

Use the Claude Code CLI instead of the API:

```python
config = ClaudeOptimizerConfig(backend="cli")
opt = ClaudeOptimizer(inner, config=config)
```

## Development

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
pytest tests/ -v
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

## License

MIT

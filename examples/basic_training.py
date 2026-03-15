#!/usr/bin/env python3
"""Basic training example using ClaudeOptimizer with dry_run=True.

This script demonstrates how to use clauto_opt with a small model.
No API key is needed — it uses a mock backend so no real consultations happen.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from clauto_opt import ClaudeOptimizer, ClaudeOptimizerConfig, ParameterUpdate
from clauto_opt.triggers import IntervalTrigger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def main() -> None:
    # Simple model: 2-layer MLP
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    # Standard AdamW optimizer
    inner_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Mock backend — returns a fixed recommendation
    mock_backend = MagicMock()
    mock_backend.consult.return_value = ParameterUpdate(
        reasoning="Loss is decreasing steadily. Slightly reduce LR for fine-tuning.",
        lr=5e-4,
    )

    # Configure ClaudeOptimizer
    config = ClaudeOptimizerConfig(
        consult_every_n_steps=100,
        dry_run=True,  # Don't actually apply changes (safe for demo)
        total_steps=500,
        loss_history_maxlen=500,
    )

    optimizer = ClaudeOptimizer(
        inner_optimizer,
        backend=mock_backend,
        config=config,
        triggers=[IntervalTrigger(every_n_steps=100)],
    )

    # Synthetic training loop
    criterion = nn.MSELoss()

    for step in range(1, 501):
        # Generate random data
        x = torch.randn(16, 10)
        y = torch.randn(16, 1)

        # Forward pass
        pred = model(x)
        loss = criterion(pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Record loss and step
        optimizer.record_loss(loss.item())

        # Record a custom metric every 50 steps
        if step % 50 == 0:
            optimizer.record_metric(
                "grad_norm", torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
            )

        optimizer.step()

        if step % 100 == 0:
            print(
                f"Step {step}/{config.total_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"Consultations: {optimizer.consultation_count}"
            )

    # Manual consultation at the end
    print("\n--- Manual consultation ---")
    result = optimizer.consult()
    if result is not None:
        print(f"Recommendation: {result.reasoning}")
        print(f"Should stop: {result.should_stop}")
    else:
        print("Consultation failed (would not happen with mock backend)")

    print(f"\nFinal stats: {optimizer.step_count} steps, {optimizer.consultation_count} consultations")
    print(f"Should stop: {optimizer.should_stop}")


if __name__ == "__main__":
    main()

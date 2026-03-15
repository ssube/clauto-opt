"""Tests for Jinja2 template rendering."""

from __future__ import annotations

from clauto_opt.models import ParameterUpdate, TrainingContext
from clauto_opt.optimizer import ClaudeOptimizer


class TestInitialTemplate:
    def test_renders_basic_context(self, training_context: TrainingContext) -> None:
        template = ClaudeOptimizer._load_template(None, "initial.md.j2")
        rendered = template.render(training_context.model_dump())

        assert "AdamW" in rendered
        assert "Current step" in rendered
        assert "200" in rendered
        assert "learning rate" in rendered.lower() or "lr" in rendered.lower()

    def test_renders_prodigy_context(self) -> None:
        ctx = TrainingContext(
            step=100,
            loss_history=[1.0, 0.9, 0.8],
            loss_current=0.8,
            loss_min=0.8,
            loss_max=1.0,
            loss_mean_recent=0.9,
            param_groups=[{"lr": 1.0, "d0": 1e-6, "d_coef": 1.0}],
            optimizer_type="Prodigy",
            is_prodigy=True,
            prodigy_d=0.001,
        )
        template = ClaudeOptimizer._load_template(None, "initial.md.j2")
        rendered = template.render(ctx.model_dump())

        assert "Prodigy" in rendered
        assert "d0" in rendered.lower() or "d_coef" in rendered.lower()
        assert "0.001" in rendered

    def test_renders_total_steps(self) -> None:
        ctx = TrainingContext(
            step=50,
            total_steps=1000,
            loss_history=[1.0],
            loss_current=1.0,
            loss_min=1.0,
            loss_max=1.0,
            loss_mean_recent=1.0,
            param_groups=[{"lr": 0.001}],
            optimizer_type="AdamW",
        )
        template = ClaudeOptimizer._load_template(None, "initial.md.j2")
        rendered = template.render(ctx.model_dump())
        assert "1000" in rendered


class TestIncrementalTemplate:
    def test_renders_with_previous_update(self) -> None:
        prev = ParameterUpdate(reasoning="Reduced LR", lr=5e-4)
        ctx = TrainingContext(
            step=200,
            loss_history=[0.5, 0.4],
            loss_current=0.4,
            loss_min=0.4,
            loss_max=0.5,
            loss_mean_recent=0.45,
            param_groups=[{"lr": 5e-4}],
            optimizer_type="AdamW",
            previous_updates=[prev],
            consultation_count=1,
        )
        template = ClaudeOptimizer._load_template(None, "incremental.md.j2")
        rendered = template.render(ctx.model_dump())

        assert "Consultation #1" in rendered
        assert "Reduced LR" in rendered

    def test_renders_without_previous(self) -> None:
        ctx = TrainingContext(
            step=100,
            loss_history=[1.0],
            loss_current=1.0,
            loss_min=1.0,
            loss_max=1.0,
            loss_mean_recent=1.0,
            param_groups=[{"lr": 0.001}],
            optimizer_type="AdamW",
            consultation_count=0,
        )
        template = ClaudeOptimizer._load_template(None, "incremental.md.j2")
        rendered = template.render(ctx.model_dump())
        assert "Current loss" in rendered


class TestCustomTemplate:
    def test_custom_string_template(self) -> None:
        custom = "Step {{ step }}, Loss {{ loss_current }}"
        template = ClaudeOptimizer._load_template(custom, "initial.md.j2")
        rendered = template.render({"step": 42, "loss_current": 0.5})
        assert rendered == "Step 42, Loss 0.5"

    def test_custom_path_template(self, tmp_path) -> None:
        template_file = tmp_path / "custom.md.j2"
        template_file.write_text("Custom: {{ optimizer_type }}")
        template = ClaudeOptimizer._load_template(template_file, "initial.md.j2")
        rendered = template.render({"optimizer_type": "SGD"})
        assert rendered == "Custom: SGD"

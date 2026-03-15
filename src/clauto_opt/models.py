"""Pydantic models defining the data contract between Claude and the optimizer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ParameterUpdate(BaseModel):
    """Claude's structured response recommending parameter changes."""

    reasoning: str = Field(description="Explanation of the recommendation")
    lr: float | None = Field(default=None, description="New learning rate (null = no change)")
    d0: float | None = Field(default=None, description="Prodigy initial D estimate")
    d_coef: float | None = Field(default=None, description="Prodigy d_coef")
    growth_rate: float | None = Field(default=None, description="Prodigy growth_rate")
    should_stop: bool = Field(default=False, description="Recommend stopping training")


class TrainingContext(BaseModel):
    """Data passed to Jinja2 templates for prompt rendering."""

    step: int
    total_steps: int | None = None
    loss_history: list[float]
    loss_current: float
    loss_min: float
    loss_max: float
    loss_mean_recent: float
    param_groups: list[dict[str, object]]
    optimizer_type: str
    is_prodigy: bool = False
    prodigy_d: float | None = None
    previous_updates: list[ParameterUpdate] = Field(default_factory=list)
    consultation_count: int = 0
    custom_metrics: dict[str, float] = Field(default_factory=dict)

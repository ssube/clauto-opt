"""clauto_opt — Claude Automatic Optimizer for PyTorch."""

from clauto_opt.config import ClaudeOptimizerConfig
from clauto_opt.models import ParameterUpdate, TrainingContext
from clauto_opt.optimizer import ClaudeOptimizer
from clauto_opt.triggers import ConsultationTrigger, IntervalTrigger, PlateauTrigger, SpikeTrigger

__all__ = [
    "ClaudeOptimizer",
    "ClaudeOptimizerConfig",
    "ConsultationTrigger",
    "IntervalTrigger",
    "ParameterUpdate",
    "PlateauTrigger",
    "SpikeTrigger",
    "TrainingContext",
]

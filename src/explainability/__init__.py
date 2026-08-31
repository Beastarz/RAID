"""Model-independent explainability primitives."""

from src.explainability.attention_rollout import (
    AttentionRolloutResult,
    attention_rollout,
)
from src.explainability.attribution import (
    GradientAttributionResult,
    integrated_gradients,
    vanilla_gradients,
)
from src.explainability.branch_contributions import (
    DEFAULT_MAX_BRANCHES,
    BranchContributionResult,
    compute_branch_contributions,
)
from src.explainability.gradcam import GradCAMResult, grad_cam
from src.explainability.serialization import (
    explanation_envelope_from_dict,
    read_explanation_json,
    read_prediction_jsonl,
    write_prediction_jsonl,
)

__all__ = [
    "DEFAULT_MAX_BRANCHES",
    "BranchContributionResult",
    "compute_branch_contributions",
    "AttentionRolloutResult",
    "attention_rollout",
    "GradientAttributionResult",
    "vanilla_gradients",
    "integrated_gradients",
    "GradCAMResult",
    "grad_cam",
    "explanation_envelope_from_dict",
    "read_explanation_json",
    "read_prediction_jsonl",
    "write_prediction_jsonl",
]

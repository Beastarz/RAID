"""Pure assembly of model-independent binary evaluation reports."""

from __future__ import annotations

from typing import Sequence

from src.explainability.contracts import PredictionRecord
from training.evaluation.calibration import compute_calibration_metrics
from training.evaluation.metrics import (
    _prepare_evaluation_inputs,
    bootstrap_confidence_intervals,
    compute_binary_metrics,
)
from training.evaluation.schemas import EvaluationReport


def build_evaluation_report(
    labels_or_records: Sequence[int] | Sequence[PredictionRecord],
    probabilities: Sequence[float] | None = None,
    *,
    threshold: float | None = None,
    bin_count: int = 10,
    bootstrap_replicates: int | None = None,
    confidence_level: float = 0.95,
    seed: int | None = 0,
) -> EvaluationReport:
    """Combine metrics and calibration without model, plotting, or filesystem I/O."""
    inputs = _prepare_evaluation_inputs(
        labels_or_records, probabilities, threshold, require_threshold=True
    )
    assert inputs.threshold is not None
    metrics = compute_binary_metrics(
        inputs.labels, inputs.probabilities, threshold=inputs.threshold
    )
    calibration = compute_calibration_metrics(
        inputs.labels, inputs.probabilities, bin_count=bin_count
    )
    intervals = ()
    if bootstrap_replicates is not None:
        intervals = bootstrap_confidence_intervals(
            inputs.labels,
            inputs.probabilities,
            threshold=inputs.threshold,
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=seed,
            bin_count=bin_count,
        )
    return EvaluationReport(
        sample_count=metrics.sample_count,
        positive_count=metrics.positive_count,
        negative_count=metrics.negative_count,
        threshold=metrics.threshold,
        model_id=inputs.model_id,
        metrics=metrics,
        calibration=calibration,
        confidence_intervals=intervals,
    )

"""Model-independent binary metrics and bootstrap confidence intervals.

Predicted label 1 means ``probability >= threshold``. Undefined ratios and
ranking metrics return ``None`` rather than imposing a zero-division value.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.explainability.contracts import PredictionRecord
from training.evaluation.schemas import BinaryMetrics, MetricConfidenceInterval


@dataclass(frozen=True)
class _EvaluationInputs:
    labels: np.ndarray
    probabilities: np.ndarray
    threshold: float | None
    model_id: str | None


def _finite_probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{field_name} must contain numbers")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must contain finite values in [0, 1]")
    return result


def _binary_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("labels must contain binary integers, excluding bool")
    result = int(value)
    if result not in (0, 1):
        raise ValueError("labels must contain only 0 and 1")
    return result


def _as_one_dimensional(values: Iterable[object], field_name: str) -> list[object]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    return array.tolist()


def _prepare_evaluation_inputs(
    labels_or_records: Sequence[int] | Sequence[PredictionRecord],
    probabilities: Sequence[float] | None,
    threshold: float | None,
    *,
    require_threshold: bool,
) -> _EvaluationInputs:
    raw = _as_one_dimensional(labels_or_records, "labels_or_records")
    if not raw:
        raise ValueError("evaluation inputs must be nonempty")

    records = all(isinstance(item, PredictionRecord) for item in raw)
    if any(isinstance(item, PredictionRecord) for item in raw) and not records:
        raise TypeError("labels_or_records must not mix records and labels")

    model_id: str | None = None
    record_threshold: float | None = None
    if records:
        if probabilities is not None:
            raise ValueError("probabilities must be omitted when records are supplied")
        typed_records = [item for item in raw if isinstance(item, PredictionRecord)]
        if any(item.ground_truth_label is None for item in typed_records):
            raise ValueError("ground-truth labels are required for evaluation")
        model_ids = {item.model_id for item in typed_records}
        thresholds = {item.decision_threshold for item in typed_records}
        if len(model_ids) != 1:
            raise ValueError("prediction records must have one consistent model_id")
        if len(thresholds) != 1:
            raise ValueError("prediction records must have one consistent decision_threshold")
        model_id = next(iter(model_ids))
        record_threshold = next(iter(thresholds))
        labels = [item.ground_truth_label for item in typed_records]
        scores = [item.predicted_probability for item in typed_records]
    else:
        if probabilities is None:
            raise ValueError("probabilities are required when labels are supplied")
        labels = raw
        scores = _as_one_dimensional(probabilities, "probabilities")
        if len(labels) != len(scores):
            raise ValueError("labels and probabilities must have equal length")

    resolved_threshold = threshold if threshold is not None else record_threshold
    if resolved_threshold is not None:
        resolved_threshold = _finite_probability(resolved_threshold, "threshold")
    if record_threshold is not None and resolved_threshold != record_threshold:
        raise ValueError("supplied threshold must match prediction record thresholds")
    if require_threshold and resolved_threshold is None:
        raise ValueError("threshold is required when labels are supplied")

    return _EvaluationInputs(
        labels=np.asarray([_binary_integer(value) for value in labels], dtype=np.int64),
        probabilities=np.asarray(
            [_finite_probability(value, "probabilities") for value in scores],
            dtype=np.float64,
        ),
        threshold=resolved_threshold,
        model_id=model_id,
    )


def compute_binary_metrics(
    labels_or_records: Sequence[int] | Sequence[PredictionRecord],
    probabilities: Sequence[float] | None = None,
    *,
    threshold: float | None = None,
) -> BinaryMetrics:
    """Compute ranking and fixed-threshold metrics from arrays or records.

    Precision is undefined with no predicted positives; recall is undefined
    with no actual positives; specificity is undefined with no actual negatives;
    F1 is undefined when its precision/recall denominator is zero.
    """
    inputs = _prepare_evaluation_inputs(
        labels_or_records, probabilities, threshold, require_threshold=True
    )
    labels = inputs.labels
    scores = inputs.probabilities
    applied_threshold = inputs.threshold
    assert applied_threshold is not None
    predicted = scores >= applied_threshold

    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & negative))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & negative))
    positive_count = tp + fn
    negative_count = tn + fp
    predicted_positive = tp + fp

    precision = tp / predicted_positive if predicted_positive else None
    recall = tp / positive_count if positive_count else None
    specificity = tn / negative_count if negative_count else None
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else None

    if positive_count and negative_count:
        roc_auc = float(roc_auc_score(labels, scores))
    else:
        roc_auc = None

    if positive_count:
        average_precision = float(average_precision_score(labels, scores))
        curve_precision, curve_recall, _ = precision_recall_curve(labels, scores)
        # sklearn emits recall in descending order; reverse it before integration.
        trapezoidal_pr_auc = float(
            np.trapezoid(curve_precision[::-1], curve_recall[::-1])
        )
    else:
        average_precision = None
        trapezoidal_pr_auc = None

    return BinaryMetrics(
        sample_count=int(labels.size),
        positive_count=positive_count,
        negative_count=negative_count,
        threshold=float(applied_threshold),
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        true_positive=tp,
        accuracy=float((tp + tn) / labels.size),
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        roc_auc=roc_auc,
        average_precision=average_precision,
        trapezoidal_pr_auc=trapezoidal_pr_auc,
    )


DEFAULT_BOOTSTRAP_METRICS = (
    "roc_auc",
    "average_precision",
    "trapezoidal_pr_auc",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "brier_score",
    "log_loss",
    "expected_calibration_error",
    "maximum_calibration_error",
)


def bootstrap_confidence_intervals(
    labels_or_records: Sequence[int] | Sequence[PredictionRecord],
    probabilities: Sequence[float] | None = None,
    *,
    threshold: float | None = None,
    replicates: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = 0,
    bin_count: int = 10,
    metric_names: Sequence[str] = DEFAULT_BOOTSTRAP_METRICS,
) -> tuple[MetricConfidenceInterval, ...]:
    """Percentile bootstrap CIs using paired, seeded sample-index resampling."""
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, numbers.Integral)
        or replicates <= 0
    ):
        raise ValueError("replicates must be a positive integer")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, numbers.Real)
        or not math.isfinite(float(confidence_level))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("confidence_level must be finite and between 0 and 1")
    if (
        isinstance(bin_count, bool)
        or not isinstance(bin_count, numbers.Integral)
        or bin_count <= 0
    ):
        raise ValueError("bin_count must be a positive integer")
    names = tuple(metric_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("metric_names must be nonempty and unique")
    unknown = set(names) - set(DEFAULT_BOOTSTRAP_METRICS)
    if unknown:
        raise ValueError(f"unsupported bootstrap metrics: {sorted(unknown)}")

    inputs = _prepare_evaluation_inputs(
        labels_or_records, probabilities, threshold, require_threshold=True
    )
    assert inputs.threshold is not None
    values: dict[str, list[float]] = {name: [] for name in names}
    rng = np.random.default_rng(seed)

    from training.evaluation.calibration import compute_calibration_metrics

    for _ in range(int(replicates)):
        indices = rng.integers(0, inputs.labels.size, size=inputs.labels.size)
        labels = inputs.labels[indices]
        scores = inputs.probabilities[indices]
        metric_result = compute_binary_metrics(labels, scores, threshold=inputs.threshold)
        calibration_result = compute_calibration_metrics(
            labels, scores, bin_count=int(bin_count)
        )
        for name in names:
            source = calibration_result if hasattr(calibration_result, name) else metric_result
            value = getattr(source, name)
            if value is not None:
                values[name].append(float(value))

    alpha = (1.0 - float(confidence_level)) / 2.0
    intervals = []
    for name in names:
        valid = values[name]
        lower = float(np.quantile(valid, alpha)) if valid else None
        upper = float(np.quantile(valid, 1.0 - alpha)) if valid else None
        intervals.append(
            MetricConfidenceInterval(
                metric_name=name,
                confidence_level=float(confidence_level),
                lower=lower,
                upper=upper,
                valid_replicates=len(valid),
            )
        )
    return tuple(intervals)

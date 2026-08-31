"""Binary probability calibration metrics using fixed-width bins."""

from __future__ import annotations

import math
import numbers
from typing import Sequence

import numpy as np

from src.explainability.contracts import PredictionRecord
from training.evaluation.metrics import _prepare_evaluation_inputs
from training.evaluation.schemas import CalibrationMetrics, ReliabilityBin


def compute_calibration_metrics(
    labels_or_records: Sequence[int] | Sequence[PredictionRecord],
    probabilities: Sequence[float] | None = None,
    *,
    bin_count: int = 10,
) -> CalibrationMetrics:
    """Compute calibration metrics with bins ``[i/B, (i+1)/B)``.

    The final bin is closed on the right, so both boundary probabilities 0 and
    1, and every probability between them, belong to exactly one bin. ECE is
    the count-weighted mean absolute calibration gap; maximum calibration error
    is the largest gap among nonempty bins.
    """
    if (
        isinstance(bin_count, bool)
        or not isinstance(bin_count, numbers.Integral)
        or bin_count <= 0
    ):
        raise ValueError("bin_count must be a positive integer")
    inputs = _prepare_evaluation_inputs(
        labels_or_records, probabilities, threshold=None, require_threshold=False
    )
    labels = inputs.labels.astype(np.float64)
    scores = inputs.probabilities
    count = int(labels.size)
    bin_count = int(bin_count)
    assignments = np.minimum((scores * bin_count).astype(np.int64), bin_count - 1)

    bins = []
    gaps = []
    ece = 0.0
    for index in range(bin_count):
        selected = assignments == index
        selected_count = int(np.sum(selected))
        mean_probability = float(np.mean(scores[selected])) if selected_count else None
        positive_fraction = float(np.mean(labels[selected])) if selected_count else None
        bins.append(
            ReliabilityBin(
                lower_bound=float(index / bin_count),
                upper_bound=float((index + 1) / bin_count),
                count=selected_count,
                mean_predicted_probability=mean_probability,
                observed_positive_fraction=positive_fraction,
            )
        )
        if selected_count:
            assert mean_probability is not None and positive_fraction is not None
            gap = abs(mean_probability - positive_fraction)
            gaps.append(gap)
            ece += selected_count / count * gap

    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    log_loss = -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))

    return CalibrationMetrics(
        sample_count=count,
        bin_count=bin_count,
        brier_score=float(np.mean((scores - labels) ** 2)),
        log_loss=float(log_loss),
        expected_calibration_error=float(ece),
        maximum_calibration_error=float(max(gaps)),
        reliability_bins=tuple(bins),
    )

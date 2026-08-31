"""Focused correctness tests for binary evaluation metrics."""

import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from training.evaluation.metrics import (
    bootstrap_confidence_intervals,
    compute_binary_metrics,
)
from training.evaluation.schemas import BinaryMetrics, MetricConfidenceInterval


def _valid_binary_metrics(**overrides):
    values = {
        "sample_count": 2,
        "positive_count": 1,
        "negative_count": 1,
        "threshold": 0.5,
        "true_negative": 1,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 1,
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "specificity": 1.0,
        "f1": 1.0,
        "roc_auc": 1.0,
        "average_precision": 1.0,
        "trapezoidal_pr_auc": 1.0,
    }
    values.update(overrides)
    return BinaryMetrics(**values)


def test_wave1_fixed_binary_metrics_match_independent_values():
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.8, 0.4, 0.9]

    result = compute_binary_metrics(labels, probabilities, threshold=0.5)
    precision, recall, _ = precision_recall_curve(labels, probabilities)

    assert (result.true_negative, result.false_positive) == (1, 1)
    assert (result.false_negative, result.true_positive) == (1, 1)
    assert result.sample_count == 4
    assert result.positive_count == result.negative_count == 2
    assert result.threshold == 0.5
    assert result.accuracy == result.precision == result.recall == 0.5
    assert result.specificity == result.f1 == 0.5
    assert result.roc_auc == pytest.approx(roc_auc_score(labels, probabilities))
    assert result.average_precision == pytest.approx(
        average_precision_score(labels, probabilities)
    )
    assert result.trapezoidal_pr_auc == pytest.approx(
        np.trapezoid(precision[::-1], recall[::-1])
    )
    assert result.trapezoidal_pr_auc >= 0.0


def test_wave1_threshold_equality_is_predicted_positive():
    result = compute_binary_metrics([0, 1], [0.5, 0.5], threshold=0.5)

    assert (result.true_negative, result.false_positive) == (0, 1)
    assert (result.false_negative, result.true_positive) == (0, 1)


def test_wave1_undefined_metrics_are_none_and_strict_json():
    result = compute_binary_metrics([0, 0], [0.1, 0.2], threshold=0.5)

    assert result.roc_auc is None
    assert result.average_precision is None
    assert result.trapezoidal_pr_auc is None
    assert result.precision is None
    assert result.recall is None
    assert result.f1 is None
    assert result.specificity == 1.0
    json.dumps(result.to_dict(), allow_nan=False)


def test_wave1_bootstrap_is_seeded_and_skips_undefined_replicates():
    kwargs = dict(
        labels_or_records=[0, 0, 1, 1],
        probabilities=[0.1, 0.2, 0.8, 0.9],
        threshold=0.5,
        replicates=40,
        confidence_level=0.9,
        seed=123,
        metric_names=("roc_auc", "accuracy"),
    )

    first = bootstrap_confidence_intervals(**kwargs)
    second = bootstrap_confidence_intervals(**kwargs)

    assert first == second
    roc_interval = first[0]
    assert 0 < roc_interval.valid_replicates < 40
    assert roc_interval.lower == roc_interval.upper == 1.0
    assert first[1].valid_replicates == 40


def test_wave1_bootstrap_has_none_bounds_when_no_replicate_is_defined():
    (interval,) = bootstrap_confidence_intervals(
        [0, 0, 0],
        [0.1, 0.2, 0.3],
        threshold=0.5,
        replicates=10,
        seed=5,
        metric_names=("roc_auc",),
    )

    assert interval.valid_replicates == 0
    assert interval.lower is interval.upper is None
    json.dumps(interval.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"labels_or_records": [], "probabilities": [], "threshold": 0.5},
        {"labels_or_records": [0], "probabilities": [0.1, 0.2], "threshold": 0.5},
        {"labels_or_records": [True], "probabilities": [0.1], "threshold": 0.5},
        {"labels_or_records": [0.0], "probabilities": [0.1], "threshold": 0.5},
        {"labels_or_records": [0], "probabilities": [np.nan], "threshold": 0.5},
        {"labels_or_records": [0], "probabilities": [1.1], "threshold": 0.5},
        {"labels_or_records": [0], "probabilities": [0.1], "threshold": np.inf},
    ],
)
def test_wave1_metrics_reject_invalid_inputs(kwargs):
    with pytest.raises((TypeError, ValueError)):
        compute_binary_metrics(**kwargs)


def test_wave1_bootstrap_validates_configuration():
    with pytest.raises(ValueError, match="replicates"):
        bootstrap_confidence_intervals([0, 1], [0.1, 0.9], threshold=0.5, replicates=0)
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_confidence_intervals(
            [0, 1], [0.1, 0.9], threshold=0.5, confidence_level=1.0
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("threshold", np.nan),
        ("threshold", 1.1),
        ("accuracy", -0.1),
        ("precision", np.inf),
        ("recall", 1.01),
        ("roc_auc", -0.01),
        ("average_precision", 1.01),
        ("trapezoidal_pr_auc", np.nan),
    ],
)
def test_wave1_binary_schema_rejects_nonfinite_or_out_of_range_rates(
    field_name, invalid_value
):
    with pytest.raises(ValueError):
        _valid_binary_metrics(**{field_name: invalid_value})


def test_wave1_binary_schema_rejects_inconsistent_or_missing_defined_metrics():
    with pytest.raises(ValueError, match="confusion counts"):
        _valid_binary_metrics(true_positive=0)
    with pytest.raises(ValueError, match="precision must be present"):
        _valid_binary_metrics(precision=None)


def test_wave1_binary_schema_requires_none_for_undefined_metrics():
    with pytest.raises(ValueError, match="precision must be None"):
        BinaryMetrics(
            sample_count=1,
            positive_count=0,
            negative_count=1,
            threshold=0.5,
            true_negative=1,
            false_positive=0,
            false_negative=0,
            true_positive=0,
            accuracy=1.0,
            precision=0.0,
            recall=None,
            specificity=1.0,
            f1=None,
            roc_auc=None,
            average_precision=None,
            trapezoidal_pr_auc=None,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "metric_name": " ",
            "confidence_level": 0.95,
            "lower": None,
            "upper": None,
            "valid_replicates": 0,
        },
        {
            "metric_name": "accuracy",
            "confidence_level": 0.95,
            "lower": None,
            "upper": 1.0,
            "valid_replicates": 1,
        },
        {
            "metric_name": "accuracy",
            "confidence_level": 0.95,
            "lower": 0.8,
            "upper": 0.7,
            "valid_replicates": 1,
        },
        {
            "metric_name": "accuracy",
            "confidence_level": 0.95,
            "lower": np.nan,
            "upper": 1.0,
            "valid_replicates": 1,
        },
    ],
)
def test_wave1_confidence_interval_schema_rejects_impossible_values(kwargs):
    with pytest.raises(ValueError):
        MetricConfidenceInterval(**kwargs)

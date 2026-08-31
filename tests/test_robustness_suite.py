"""Focused tests for model-independent robustness aggregation."""

import json

import pytest

from src.explainability.contracts import PredictionRecord
from training.evaluation.robustness_suite import (
    RobustnessBenchmark,
    aggregate_robustness,
)


def _record(
    sample_id: str,
    probability: float,
    label: int,
    *,
    condition: str = "clean",
    severity: float | None = None,
    model_id: str = "model-a",
    threshold: float = 0.5,
) -> PredictionRecord:
    metadata = {"condition": condition}
    if severity is not None:
        metadata["severity"] = severity
    return PredictionRecord(
        sample_id=sample_id,
        model_id=model_id,
        predicted_logit=0.0,
        predicted_probability=probability,
        predicted_label=int(probability >= threshold),
        decision_threshold=threshold,
        ground_truth_label=label,
        metadata=metadata,
    )


def _condition_records(condition: str, probabilities: list[float], severity=None):
    labels = [0, 0, 1, 1]
    return tuple(
        _record(
            f"{condition}-{index}",
            probability,
            labels[index],
            condition=condition,
            severity=severity,
        )
        for index, probability in enumerate(probabilities)
    )


def test_robustness_aggregation_orders_conditions_and_severities():
    records = (
        *_condition_records("jpeg", [0.8, 0.7, 0.2, 0.1], 30),
        *_condition_records("clean", [0.1, 0.2, 0.8, 0.9]),
        *_condition_records("jpeg", [0.2, 0.3, 0.7, 0.8], 90),
        *_condition_records("blur", [0.4, 0.3, 0.6, 0.7], 1.0),
    )

    result = aggregate_robustness(records)

    assert [(point.condition, point.severity) for point in result.points] == [
        ("clean", None),
        ("blur", 1.0),
        ("jpeg", 30.0),
        ("jpeg", 90.0),
    ]
    assert result.threshold == 0.5
    assert result.clean.metrics.roc_auc == 1.0
    assert result.points[2].metrics.roc_auc == 0.0
    assert result.points[2].absolute_degradation["roc_auc"] == 1.0
    assert result.points[2].relative_degradation["roc_auc"] == 1.0
    assert result.clean.absolute_degradation["roc_auc"] == 0.0


def test_robustness_reports_loss_degradation_in_the_lower_is_better_direction():
    result = aggregate_robustness(
        _condition_records("clean", [0.1, 0.2, 0.8, 0.9])
        + _condition_records("noise", [0.4, 0.4, 0.6, 0.6], 0.1)
    )

    clean_brier = result.clean.calibration.brier_score
    noisy = result.points[1]
    assert noisy.calibration.brier_score > clean_brier
    assert noisy.absolute_degradation["brier_score"] == pytest.approx(
        noisy.calibration.brier_score - clean_brier
    )


def test_robustness_requires_clean_condition_and_degraded_severity():
    with pytest.raises(ValueError, match="baseline"):
        aggregate_robustness(_condition_records("jpeg", [0.1, 0.2, 0.8, 0.9], 90))

    records = _condition_records("clean", [0.1, 0.2, 0.8, 0.9]) + tuple(
        _record(f"blur-{i}", probability, label, condition="blur")
        for i, (probability, label) in enumerate(zip([0.2, 0.3], [0, 1]))
    )
    with pytest.raises(ValueError, match="requires"):
        aggregate_robustness(records)


def test_robustness_preserves_one_threshold_and_model_identity():
    records = _condition_records("clean", [0.1, 0.2, 0.8, 0.9]) + (
        _record("other", 0.8, 1, condition="jpeg", severity=90, model_id="model-b"),
    )
    with pytest.raises(ValueError, match="model_id"):
        aggregate_robustness(records)

    records = _condition_records("clean", [0.1, 0.2, 0.8, 0.9]) + (
        _record("other", 0.8, 1, condition="jpeg", severity=90, threshold=0.6),
    )
    with pytest.raises(ValueError, match="decision_threshold"):
        aggregate_robustness(records)


def test_robustness_report_has_strict_json_and_flat_csv_serialization():
    result = RobustnessBenchmark(
        _condition_records("clean", [0.1, 0.2, 0.8, 0.9])
        + _condition_records("jpeg", [0.2, 0.3, 0.7, 0.8], 90)
    ).run()

    json.dumps(result.to_dict(), allow_nan=False)
    serialized = result.to_json()
    assert json.loads(serialized)["points"][0]["condition"] == "clean"
    csv_text = result.to_csv()
    assert "condition,severity" in csv_text.splitlines()[0]
    assert "jpeg,90.0" in csv_text

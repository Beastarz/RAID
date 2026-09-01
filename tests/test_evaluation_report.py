"""Focused tests for pure evaluation report assembly."""

import json
from dataclasses import replace

import pytest

from src.explainability.contracts import PredictionRecord
from training.evaluation.report import build_evaluation_report
from training.evaluation.schemas import MetricConfidenceInterval


def _record(sample_id, probability, label, *, model_id="model-a", threshold=0.5):
    return PredictionRecord(
        sample_id=sample_id,
        model_id=model_id,
        predicted_logit=0.0,
        predicted_probability=probability,
        predicted_label=int(probability >= threshold),
        decision_threshold=threshold,
        ground_truth_label=label,
    )


def test_wave1_prediction_records_build_complete_strict_json_report():
    records = (
        _record("a", 0.1, 0),
        _record("b", 0.8, 1),
        _record("c", 0.4, 1),
        _record("d", 0.9, 1),
    )

    report = build_evaluation_report(
        records, bin_count=4, bootstrap_replicates=12, seed=9
    )
    serialized = report.to_dict()

    assert report.model_id == "model-a"
    assert report.sample_count == 4
    assert report.positive_count == 3
    assert report.negative_count == 1
    assert report.threshold == 0.5
    assert report.metrics.true_positive == 2
    assert len(report.calibration.reliability_bins) == 4
    assert report.confidence_intervals
    json.dumps(serialized, allow_nan=False)


def test_wave1_array_report_requires_caller_threshold():
    with pytest.raises(ValueError, match="threshold"):
        build_evaluation_report([0, 1], [0.1, 0.9])


def test_wave1_records_require_ground_truth_and_consistent_identity():
    missing_truth = PredictionRecord(
        sample_id="a",
        model_id="model-a",
        predicted_logit=0.0,
        predicted_probability=0.1,
        predicted_label=0,
        decision_threshold=0.5,
    )
    with pytest.raises(ValueError, match="ground-truth"):
        build_evaluation_report([missing_truth])

    with pytest.raises(ValueError, match="model_id"):
        build_evaluation_report(
            [_record("a", 0.1, 0), _record("b", 0.9, 1, model_id="model-b")]
        )

    with pytest.raises(ValueError, match="decision_threshold"):
        build_evaluation_report(
            [_record("a", 0.1, 0), _record("b", 0.9, 1, threshold=0.6)]
        )


def test_wave1_record_report_rejects_threshold_override():
    with pytest.raises(ValueError, match="match prediction record"):
        build_evaluation_report([_record("a", 0.1, 0)], threshold=0.4)


def test_wave1_report_schema_rejects_blank_model_and_inconsistent_counts():
    report = build_evaluation_report([0, 1], [0.1, 0.9], threshold=0.5)

    with pytest.raises(ValueError, match="model_id"):
        replace(report, model_id="  ")
    with pytest.raises(ValueError, match="sample_count"):
        replace(report, sample_count=True)
    with pytest.raises(ValueError, match="counts and threshold"):
        replace(report, positive_count=0)


def test_wave1_report_schema_rejects_duplicate_confidence_interval_names():
    report = build_evaluation_report([0, 1], [0.1, 0.9], threshold=0.5)
    interval = MetricConfidenceInterval("accuracy", 0.95, 0.5, 1.0, 10)

    with pytest.raises(ValueError, match="must be unique"):
        replace(report, confidence_intervals=(interval, interval))

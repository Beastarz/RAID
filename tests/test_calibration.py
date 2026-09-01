"""Focused correctness tests for binary calibration."""

import json

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss

from training.evaluation.calibration import compute_calibration_metrics
from training.evaluation.schemas import CalibrationMetrics, ReliabilityBin


def _valid_calibration_metrics(**overrides):
    values = {
        "sample_count": 2,
        "bin_count": 2,
        "brier_score": 0.04,
        "log_loss": 0.2,
        "expected_calibration_error": 0.2,
        "maximum_calibration_error": 0.2,
        "reliability_bins": (
            ReliabilityBin(0.0, 0.5, 1, 0.2, 0.0),
            ReliabilityBin(0.5, 1.0, 1, 0.8, 1.0),
        ),
    }
    values.update(overrides)
    return CalibrationMetrics(**values)


def test_wave1_calibration_matches_sklearn_and_exact_ece():
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.3, 0.8, 0.9]

    result = compute_calibration_metrics(labels, probabilities, bin_count=2)

    assert result.brier_score == pytest.approx(brier_score_loss(labels, probabilities))
    assert result.log_loss == pytest.approx(log_loss(labels, probabilities))
    assert result.expected_calibration_error == pytest.approx(0.175)
    assert result.maximum_calibration_error == pytest.approx(0.2)


def test_wave1_reliability_bins_assign_zero_one_and_edges_once():
    result = compute_calibration_metrics(
        [0, 0, 1, 1, 1], [0.0, 0.25, 0.5, 0.75, 1.0], bin_count=4
    )

    assert [item.count for item in result.reliability_bins] == [1, 1, 1, 2]
    assert sum(item.count for item in result.reliability_bins) == 5
    assert result.reliability_bins[0].mean_predicted_probability == 0.0
    assert result.reliability_bins[-1].mean_predicted_probability == pytest.approx(0.875)
    assert result.reliability_bins[-1].upper_bound == 1.0


def test_wave1_empty_reliability_bins_use_none_and_strict_json():
    result = compute_calibration_metrics([0, 1], [0.1, 0.9], bin_count=4)

    assert result.reliability_bins[1].count == 0
    assert result.reliability_bins[1].mean_predicted_probability is None
    assert result.reliability_bins[1].observed_positive_fraction is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_wave1_boundary_log_loss_is_finite():
    result = compute_calibration_metrics([0, 1, 1, 0], [0.0, 1.0, 0.0, 1.0])

    assert np.isfinite(result.log_loss)


@pytest.mark.parametrize("bin_count", [0, -1, True, 1.5])
def test_wave1_calibration_rejects_invalid_bin_count(bin_count):
    with pytest.raises(ValueError, match="bin_count"):
        compute_calibration_metrics([0, 1], [0.1, 0.9], bin_count=bin_count)


def test_wave1_reliability_bin_schema_rejects_invalid_summary_presence():
    with pytest.raises(ValueError, match="empty reliability"):
        ReliabilityBin(0.0, 0.5, 0, 0.2, None)
    with pytest.raises(ValueError, match="require both summaries"):
        ReliabilityBin(0.0, 0.5, 1, 0.2, None)


def test_wave1_reliability_bin_schema_rejects_invalid_summary_ranges_and_location():
    with pytest.raises(ValueError, match="observed_positive_fraction"):
        ReliabilityBin(0.0, 0.5, 1, 0.2, 1.1)
    with pytest.raises(ValueError, match="within bin bounds"):
        ReliabilityBin(0.0, 0.5, 1, 0.5, 0.0)
    assert ReliabilityBin(0.5, 1.0, 1, 1.0, 1.0).mean_predicted_probability == 1.0


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("brier_score", -0.1),
        ("expected_calibration_error", 1.1),
        ("maximum_calibration_error", np.inf),
        ("log_loss", -0.1),
        ("log_loss", np.nan),
    ],
)
def test_wave1_calibration_schema_rejects_invalid_metric_ranges(
    field_name, invalid_value
):
    with pytest.raises(ValueError):
        _valid_calibration_metrics(**{field_name: invalid_value})


def test_wave1_calibration_schema_requires_typed_contiguous_bins():
    with pytest.raises(TypeError, match="ReliabilityBin"):
        _valid_calibration_metrics(reliability_bins=(object(), object()))

    noncontiguous = (
        ReliabilityBin(0.0, 0.4, 1, 0.2, 0.0),
        ReliabilityBin(0.4, 1.0, 1, 0.8, 1.0),
    )
    with pytest.raises(ValueError, match="ordered and contiguous"):
        _valid_calibration_metrics(reliability_bins=noncontiguous)

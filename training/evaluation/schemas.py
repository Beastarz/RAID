"""Strict-JSON schemas for model-independent binary evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.explainability.contracts import SCHEMA_VERSION


def _finite_float(value: float, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


def _optional_finite(value: float | None, field_name: str) -> float | None:
    return None if value is None else _finite_float(value, field_name)


def _unit_interval(value: float, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be within [0, 1]")
    return result


def _optional_unit_interval(value: float | None, field_name: str) -> float | None:
    return None if value is None else _unit_interval(value, field_name)


@dataclass(frozen=True)
class BinaryMetrics:
    """Binary discrimination and fixed-threshold classification metrics."""

    sample_count: int
    positive_count: int
    negative_count: int
    threshold: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    accuracy: float
    precision: float | None
    recall: float | None
    specificity: float | None
    f1: float | None
    roc_auc: float | None
    average_precision: float | None
    trapezoidal_pr_auc: float | None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        for name in (
            "sample_count",
            "positive_count",
            "negative_count",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("threshold", "accuracy"):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))
        for name in (
            "precision",
            "recall",
            "specificity",
            "f1",
            "roc_auc",
            "average_precision",
            "trapezoidal_pr_auc",
        ):
            object.__setattr__(
                self, name, _optional_unit_interval(getattr(self, name), name)
            )
        if self.sample_count == 0:
            raise ValueError("sample_count must be positive")
        if self.positive_count + self.negative_count != self.sample_count:
            raise ValueError("class counts must sum to sample_count")
        if self.true_positive + self.false_negative != self.positive_count:
            raise ValueError("positive confusion counts must sum to positive_count")
        if self.true_negative + self.false_positive != self.negative_count:
            raise ValueError("negative confusion counts must sum to negative_count")
        defined_conditions = {
            "precision": self.true_positive + self.false_positive > 0,
            "recall": self.positive_count > 0,
            "specificity": self.negative_count > 0,
            "f1": 2 * self.true_positive
            + self.false_positive
            + self.false_negative
            > 0,
            "roc_auc": self.positive_count > 0 and self.negative_count > 0,
            "average_precision": self.positive_count > 0,
            "trapezoidal_pr_auc": self.positive_count > 0,
        }
        for name, is_defined in defined_conditions.items():
            if (getattr(self, name) is not None) != is_defined:
                state = "present" if is_defined else "None"
                raise ValueError(f"{name} must be {state} for these class/count totals")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "threshold": self.threshold,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_positive": self.true_positive,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "trapezoidal_pr_auc": self.trapezoidal_pr_auc,
        }


@dataclass(frozen=True)
class ReliabilityBin:
    """One fixed-width probability bin; the final upper bound is inclusive."""

    lower_bound: float
    upper_bound: float
    count: int
    mean_predicted_probability: float | None
    observed_positive_fraction: float | None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        lower = _unit_interval(self.lower_bound, "lower_bound")
        upper = _unit_interval(self.upper_bound, "upper_bound")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        if lower >= upper:
            raise ValueError("reliability bin bounds must be ordered within [0, 1]")
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 0
        ):
            raise ValueError("count must be a non-negative integer")
        mean_probability = _optional_unit_interval(
            self.mean_predicted_probability, "mean_predicted_probability"
        )
        positive_fraction = _optional_unit_interval(
            self.observed_positive_fraction, "observed_positive_fraction"
        )
        object.__setattr__(self, "mean_predicted_probability", mean_probability)
        object.__setattr__(self, "observed_positive_fraction", positive_fraction)
        if self.count == 0 and (
            mean_probability is not None or positive_fraction is not None
        ):
            raise ValueError("empty reliability bins must have None summaries")
        if self.count > 0 and (mean_probability is None or positive_fraction is None):
            raise ValueError("nonempty reliability bins require both summaries")
        if mean_probability is not None:
            in_bin = lower <= mean_probability < upper
            if upper == 1.0:
                in_bin = lower <= mean_probability <= upper
            if not in_bin:
                raise ValueError("mean_predicted_probability must lie within bin bounds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "count": self.count,
            "mean_predicted_probability": self.mean_predicted_probability,
            "observed_positive_fraction": self.observed_positive_fraction,
        }


@dataclass(frozen=True)
class CalibrationMetrics:
    sample_count: int
    bin_count: int
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    maximum_calibration_error: float
    reliability_bins: tuple[ReliabilityBin, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.sample_count, self.bin_count)
        ):
            raise ValueError("sample_count and bin_count must be positive")
        bins = tuple(self.reliability_bins)
        if any(not isinstance(item, ReliabilityBin) for item in bins):
            raise TypeError("reliability_bins must contain ReliabilityBin schemas")
        object.__setattr__(self, "reliability_bins", bins)
        if len(bins) != self.bin_count:
            raise ValueError("reliability_bins length must equal bin_count")
        if sum(item.count for item in bins) != self.sample_count:
            raise ValueError("reliability bin counts must sum to sample_count")
        tolerance = 1e-12
        for index, item in enumerate(bins):
            expected_lower = index / self.bin_count
            expected_upper = (index + 1) / self.bin_count
            if not (
                math.isclose(item.lower_bound, expected_lower, rel_tol=0.0, abs_tol=tolerance)
                and math.isclose(
                    item.upper_bound, expected_upper, rel_tol=0.0, abs_tol=tolerance
                )
            ):
                raise ValueError("reliability bins must be ordered and contiguous from 0 to 1")
        for name in (
            "brier_score",
            "expected_calibration_error",
            "maximum_calibration_error",
        ):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))
        log_loss = _finite_float(self.log_loss, "log_loss")
        if log_loss < 0.0:
            raise ValueError("log_loss must be non-negative")
        object.__setattr__(self, "log_loss", log_loss)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "bin_count": self.bin_count,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "reliability_bins": [item.to_dict() for item in self.reliability_bins],
        }


@dataclass(frozen=True)
class MetricConfidenceInterval:
    metric_name: str
    confidence_level: float
    lower: float | None
    upper: float | None
    valid_replicates: int
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        if not isinstance(self.metric_name, str) or not self.metric_name.strip():
            raise ValueError("metric_name must be nonblank")
        confidence_level = _finite_float(self.confidence_level, "confidence_level")
        object.__setattr__(self, "confidence_level", confidence_level)
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1")
        lower = _optional_finite(self.lower, "lower")
        upper = _optional_finite(self.upper, "upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        if (
            isinstance(self.valid_replicates, bool)
            or not isinstance(self.valid_replicates, int)
            or self.valid_replicates < 0
        ):
            raise ValueError("valid_replicates must be non-negative")
        if self.valid_replicates == 0 and (lower is not None or upper is not None):
            raise ValueError("bounds must be None when no replicate is valid")
        if self.valid_replicates > 0:
            if lower is None or upper is None:
                raise ValueError("valid replicates require both bounds")
            if lower > upper:
                raise ValueError("lower bound must not exceed upper bound")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric_name": self.metric_name,
            "confidence_level": self.confidence_level,
            "lower": self.lower,
            "upper": self.upper,
            "valid_replicates": self.valid_replicates,
        }


@dataclass(frozen=True)
class EvaluationReport:
    sample_count: int
    positive_count: int
    negative_count: int
    threshold: float
    model_id: str | None
    metrics: BinaryMetrics
    calibration: CalibrationMetrics
    confidence_intervals: tuple[MetricConfidenceInterval, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        if self.model_id is not None and (
            not isinstance(self.model_id, str) or not self.model_id.strip()
        ):
            raise ValueError("model_id must be nonblank when present")
        for name in ("sample_count", "positive_count", "negative_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "threshold", _unit_interval(self.threshold, "threshold"))
        if not isinstance(self.metrics, BinaryMetrics):
            raise TypeError("metrics must be BinaryMetrics")
        if not isinstance(self.calibration, CalibrationMetrics):
            raise TypeError("calibration must be CalibrationMetrics")
        intervals = tuple(self.confidence_intervals)
        if any(not isinstance(item, MetricConfidenceInterval) for item in intervals):
            raise TypeError("confidence_intervals must contain interval schemas")
        interval_names = [item.metric_name for item in intervals]
        if len(set(interval_names)) != len(interval_names):
            raise ValueError("confidence interval metric names must be unique")
        object.__setattr__(self, "confidence_intervals", intervals)
        expected = (
            self.sample_count,
            self.positive_count,
            self.negative_count,
            self.threshold,
        )
        actual = (
            self.metrics.sample_count,
            self.metrics.positive_count,
            self.metrics.negative_count,
            self.metrics.threshold,
        )
        if expected != actual:
            raise ValueError("report counts and threshold must match metrics")
        if self.calibration.sample_count != self.sample_count:
            raise ValueError("calibration sample_count must match report")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "threshold": self.threshold,
            "model_id": self.model_id,
            "metrics": self.metrics.to_dict(),
            "calibration": self.calibration.to_dict(),
            "confidence_intervals": [item.to_dict() for item in self.confidence_intervals],
        }

"""Schemas and serialization for condition-wise robustness reports.

The report is deliberately model-independent.  A benchmark supplies already
materialized :class:`~src.explainability.contracts.PredictionRecord` values;
this module only stores evaluation results and clean-relative deltas.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.explainability.contracts import SCHEMA_VERSION
from training.evaluation.schemas import EvaluationReport


# Metrics where a larger value means better detector performance.  The
# remaining metrics in ``DEGRADATION_METRIC_NAMES`` are calibration/loss
# metrics where a smaller value is better.
HIGHER_IS_BETTER = frozenset(
    {
        "roc_auc",
        "average_precision",
        "trapezoidal_pr_auc",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
    }
)
DEGRADATION_METRIC_NAMES = (
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


def _finite_optional(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number or None")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _nonblank(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return value


def _metric_value(report: EvaluationReport, metric_name: str) -> float | None:
    if hasattr(report.metrics, metric_name):
        return getattr(report.metrics, metric_name)
    if hasattr(report.calibration, metric_name):
        return getattr(report.calibration, metric_name)
    raise ValueError(f"unknown degradation metric {metric_name!r}")


def _validate_degradation_mapping(
    values: Mapping[str, float | None], field_name: str
) -> dict[str, float | None]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    result: dict[str, float | None] = {}
    for key, value in values.items():
        name = _nonblank(key, f"{field_name} key")
        result[name] = _finite_optional(value, f"{field_name}.{name}")
    return result


@dataclass(frozen=True)
class RobustnessPoint:
    """One clean or degraded condition in a robustness curve."""

    condition: str
    severity: float | None
    report: EvaluationReport
    absolute_degradation: Mapping[str, float | None] = field(default_factory=dict)
    relative_degradation: Mapping[str, float | None] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        condition = _nonblank(self.condition, "condition")
        object.__setattr__(self, "condition", condition)
        severity = _finite_optional(self.severity, "severity")
        object.__setattr__(self, "severity", severity)
        if not isinstance(self.report, EvaluationReport):
            raise TypeError("report must be an EvaluationReport")
        absolute = _validate_degradation_mapping(
            self.absolute_degradation, "absolute_degradation"
        )
        relative = _validate_degradation_mapping(
            self.relative_degradation, "relative_degradation"
        )
        if set(absolute) != set(relative):
            raise ValueError(
                "absolute_degradation and relative_degradation must contain "
                "the same metric names"
            )
        object.__setattr__(self, "absolute_degradation", absolute)
        object.__setattr__(self, "relative_degradation", relative)

    @property
    def metrics(self):
        """Convenience access to the binary metrics for this condition."""
        return self.report.metrics

    @property
    def calibration(self):
        """Convenience access to the calibration metrics for this condition."""
        return self.report.calibration

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "condition": self.condition,
            "severity": self.severity,
            "report": self.report.to_dict(),
            "absolute_degradation": dict(self.absolute_degradation),
            "relative_degradation": dict(self.relative_degradation),
        }


@dataclass(frozen=True)
class RobustnessReport:
    """Ordered clean baseline and condition-wise robustness results."""

    model_id: str | None
    threshold: float
    points: tuple[RobustnessPoint, ...]
    clean_condition: str = "clean"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        if self.model_id is not None:
            _nonblank(self.model_id, "model_id")
        threshold = _finite_optional(self.threshold, "threshold")
        if threshold is None:
            raise ValueError("threshold must be a finite number")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        object.__setattr__(self, "threshold", threshold)
        clean_condition = _nonblank(self.clean_condition, "clean_condition")
        object.__setattr__(self, "clean_condition", clean_condition)
        points = tuple(self.points)
        if not points:
            raise ValueError("points must be nonempty")
        if any(not isinstance(item, RobustnessPoint) for item in points):
            raise TypeError("points must contain RobustnessPoint records")
        keys = [(item.condition, item.severity) for item in points]
        if len(set(keys)) != len(keys):
            raise ValueError("robustness condition/severity pairs must be unique")
        clean_points = [
            item
            for item in points
            if item.condition == clean_condition
        ]
        if len(clean_points) != 1 or clean_points[0].severity is not None:
            raise ValueError(
                "exactly one clean baseline is required with severity=None"
            )
        for point in points:
            if point.report.model_id != self.model_id:
                raise ValueError("all condition reports must share model_id")
            if point.report.threshold != threshold:
                raise ValueError("all condition reports must share threshold")
        object.__setattr__(self, "points", points)

    @property
    def clean(self) -> RobustnessPoint:
        """Return the clean baseline point."""
        for point in self.points:
            if point.condition == self.clean_condition and point.severity is None:
                return point
        raise RuntimeError("validated report has no clean point")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "threshold": self.threshold,
            "clean_condition": self.clean_condition,
            "points": [point.to_dict() for point in self.points],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize this report as strict JSON (never NaN/Infinity)."""
        return json.dumps(self.to_dict(), allow_nan=False, indent=indent, sort_keys=True)

    def csv_rows(self) -> tuple[dict[str, Any], ...]:
        """Flatten points into rows suitable for a degradation-curve CSV."""
        rows: list[dict[str, Any]] = []
        for point in self.points:
            row: dict[str, Any] = {
                "condition": point.condition,
                "severity": point.severity,
                "model_id": self.model_id,
                "threshold": self.threshold,
                "schema_version": self.schema_version,
                "sample_count": point.report.sample_count,
                "positive_count": point.report.positive_count,
                "negative_count": point.report.negative_count,
            }
            for name in (
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
            ):
                row[name] = _metric_value(point.report, name)
                row[f"absolute_degradation__{name}"] = point.absolute_degradation.get(name)
                row[f"relative_degradation__{name}"] = point.relative_degradation.get(name)
            for name in (
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
            ):
                row[name] = getattr(point.report.metrics, name)
            rows.append(row)
        return tuple(rows)

    def to_csv(self, path: str | Path | None = None) -> str:
        """Serialize flattened condition rows, optionally writing ``path``."""
        rows = self.csv_rows()
        fieldnames = list(rows[0])
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        content = output.getvalue()
        if path is not None:
            Path(path).write_text(content, encoding="utf-8", newline="")
        return content

    def write_csv(self, path: str | Path) -> None:
        """Write flattened condition rows to ``path``."""
        self.to_csv(path)

    def write_json(self, path: str | Path, *, indent: int | None = 2) -> None:
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")


__all__ = [
    "DEGRADATION_METRIC_NAMES",
    "HIGHER_IS_BETTER",
    "RobustnessPoint",
    "RobustnessReport",
]

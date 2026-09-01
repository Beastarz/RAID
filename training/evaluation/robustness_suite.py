"""Model-independent aggregation of robustness prediction records.

The suite consumes prediction records that already contain condition metadata;
it never applies transforms, loads a checkpoint, or invokes a detector.  This
keeps robustness reporting usable for both the legacy and updated branch
architectures while their final fused adapter is still pending.
"""

from __future__ import annotations

import math
import numbers
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from src.explainability.contracts import PredictionRecord
from training.evaluation.report import build_evaluation_report
from training.evaluation.robustness_report import (
    DEGRADATION_METRIC_NAMES,
    HIGHER_IS_BETTER,
    RobustnessPoint,
    RobustnessReport,
)


def _condition(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return value


def _severity(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("threshold must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")
    return result


def _clean_relative_delta(
    metric_name: str, clean_value: float | None, condition_value: float | None
) -> tuple[float | None, float | None]:
    """Return signed worsening and relative worsening for one metric.

    Positive values mean degradation; negative values mean improvement.  Loss
    and calibration metrics use the opposite direction from ranking and
    classification metrics because lower values are better for them.
    """
    if clean_value is None or condition_value is None:
        return None, None
    if metric_name in HIGHER_IS_BETTER:
        absolute = clean_value - condition_value
    else:
        absolute = condition_value - clean_value
    denominator = abs(clean_value)
    relative = None if denominator == 0.0 else absolute / denominator
    return float(absolute), None if relative is None else float(relative)


def _records_by_condition(
    records: Iterable[PredictionRecord],
    *,
    condition_key: str,
    severity_key: str,
    clean_condition: str,
) -> dict[tuple[str, float | None], tuple[PredictionRecord, ...]]:
    groups: dict[tuple[str, float | None], list[PredictionRecord]] = defaultdict(list)
    for record in records:
        metadata: Mapping[str, object] = record.metadata
        if condition_key not in metadata:
            raise ValueError(
                f"prediction record {record.sample_id!r} is missing metadata[{condition_key!r}]"
            )
        condition = _condition(metadata[condition_key], "condition")
        raw_severity = metadata.get(severity_key)
        if condition == clean_condition:
            if raw_severity is not None and _severity(raw_severity, "severity") != 0.0:
                raise ValueError("clean condition severity must be omitted or zero")
            severity: float | None = None
        else:
            if raw_severity is None:
                raise ValueError(
                    f"degraded condition {condition!r} requires metadata[{severity_key!r}]"
                )
            severity = _severity(raw_severity, "severity")
        groups[(condition, severity)].append(record)
    return {key: tuple(value) for key, value in groups.items()}


def aggregate_robustness(
    records: Sequence[PredictionRecord],
    *,
    condition_key: str = "condition",
    severity_key: str = "severity",
    clean_condition: str = "clean",
    threshold: float | None = None,
) -> RobustnessReport:
    """Aggregate existing predictions into deterministic robustness curves.

    Every record must carry ``metadata[condition_key]``.  Degraded records
    additionally require a finite numeric ``metadata[severity_key]``; clean
    records may omit it.  All records must share one model identity and one
    decision threshold.  The returned points are ordered clean first, then by
    condition name and ascending severity.
    """
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of PredictionRecord values")
    if not records:
        raise ValueError("records must be nonempty")
    if any(not isinstance(record, PredictionRecord) for record in records):
        raise TypeError("records must contain only PredictionRecord values")
    condition_key = _condition(condition_key, "condition_key")
    severity_key = _condition(severity_key, "severity_key")
    clean_condition = _condition(clean_condition, "clean_condition")

    model_ids = {record.model_id for record in records}
    if len(model_ids) != 1:
        raise ValueError("prediction records must have one consistent model_id")
    record_thresholds = {record.decision_threshold for record in records}
    if len(record_thresholds) != 1:
        raise ValueError("prediction records must have one consistent decision_threshold")
    resolved_threshold = next(iter(record_thresholds))
    if threshold is not None and _threshold(threshold) != resolved_threshold:
        raise ValueError("supplied threshold must match prediction record thresholds")

    groups = _records_by_condition(
        records,
        condition_key=condition_key,
        severity_key=severity_key,
        clean_condition=clean_condition,
    )
    clean_key = (clean_condition, None)
    if clean_key not in groups:
        raise ValueError(f"records must include a {clean_condition!r} baseline condition")

    ordered_keys = sorted(
        groups,
        key=lambda key: (
            0 if key == clean_key else 1,
            key[0],
            -math.inf if key[1] is None else key[1],
        ),
    )
    reports = {
        key: build_evaluation_report(group, threshold=resolved_threshold)
        for key, group in groups.items()
    }
    clean_report = reports[clean_key]

    points: list[RobustnessPoint] = []
    for key in ordered_keys:
        condition, severity = key
        report = reports[key]
        absolute: dict[str, float | None] = {}
        relative: dict[str, float | None] = {}
        for metric_name in DEGRADATION_METRIC_NAMES:
            clean_value = getattr(clean_report.metrics, metric_name, None)
            if clean_value is None:
                clean_value = getattr(clean_report.calibration, metric_name, None)
            condition_value = getattr(report.metrics, metric_name, None)
            if condition_value is None:
                condition_value = getattr(report.calibration, metric_name, None)
            if key == clean_key:
                absolute[metric_name] = 0.0 if clean_value is not None else None
                relative[metric_name] = 0.0 if clean_value is not None else None
            else:
                absolute[metric_name], relative[metric_name] = _clean_relative_delta(
                    metric_name, clean_value, condition_value
                )
        points.append(
            RobustnessPoint(
                condition=condition,
                severity=severity,
                report=report,
                absolute_degradation=absolute,
                relative_degradation=relative,
            )
        )

    return RobustnessReport(
        model_id=next(iter(model_ids)),
        threshold=resolved_threshold,
        points=tuple(points),
        clean_condition=clean_condition,
    )


class RobustnessBenchmark:
    """Reusable configuration wrapper around :func:`aggregate_robustness`.

    This class intentionally accepts prediction records rather than a model or
    dataset; transforms and inference belong to the caller's evaluation path.
    """

    def __init__(
        self,
        records: Sequence[PredictionRecord] | None = None,
        *,
        condition_key: str = "condition",
        severity_key: str = "severity",
        clean_condition: str = "clean",
        threshold: float | None = None,
    ) -> None:
        self.records = records
        self.condition_key = condition_key
        self.severity_key = severity_key
        self.clean_condition = clean_condition
        self.threshold = threshold

    def aggregate(
        self, records: Sequence[PredictionRecord] | None = None
    ) -> RobustnessReport:
        selected = self.records if records is None else records
        if selected is None:
            raise ValueError("prediction records are required")
        return aggregate_robustness(
            selected,
            condition_key=self.condition_key,
            severity_key=self.severity_key,
            clean_condition=self.clean_condition,
            threshold=self.threshold,
        )

    def run(self, records: Sequence[PredictionRecord] | None = None) -> RobustnessReport:
        """Alias for ``aggregate`` for benchmark-style call sites."""
        return self.aggregate(records)


__all__ = ["RobustnessBenchmark", "aggregate_robustness"]

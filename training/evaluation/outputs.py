"""Filesystem output assembly for model-free evaluation reports.

This module is intentionally downstream of the validated report schemas.  It
only serializes records/reports and delegates all visualization to
``training.evaluation.plotting``; no model or dataset is loaded here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.explainability.contracts import PredictionRecord
from training.evaluation.plotting import save_evaluation_plots
from training.evaluation.report import build_evaluation_report
from training.evaluation.robustness_suite import aggregate_robustness
from training.evaluation.schemas import EvaluationReport


def _metric_row(report: EvaluationReport) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": report.schema_version,
        "model_id": report.model_id,
        "sample_count": report.sample_count,
        "positive_count": report.positive_count,
        "negative_count": report.negative_count,
        "threshold": report.threshold,
    }
    for name in (
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "average_precision",
        "trapezoidal_pr_auc",
    ):
        row[name] = getattr(report.metrics, name)
    for name in (
        "brier_score",
        "log_loss",
        "expected_calibration_error",
        "maximum_calibration_error",
    ):
        row[name] = getattr(report.calibration, name)
    for interval in report.confidence_intervals:
        prefix = f"ci_{interval.metric_name}"
        row[f"{prefix}_confidence_level"] = interval.confidence_level
        row[f"{prefix}_lower"] = interval.lower
        row[f"{prefix}_upper"] = interval.upper
        row[f"{prefix}_valid_replicates"] = interval.valid_replicates
    return row


def write_metrics_csv(report: EvaluationReport, path: str | Path) -> Path:
    """Write one deterministic scalar-metrics row and return its path."""

    if not isinstance(report, EvaluationReport):
        raise TypeError("report must be an EvaluationReport")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = _metric_row(report)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    return destination


def write_report_json(
    report: EvaluationReport,
    path: str | Path,
) -> Path:
    """Write one strict, versioned :class:`EvaluationReport` JSON document."""

    if not isinstance(report, EvaluationReport):
        raise TypeError("report must be an EvaluationReport")
    payload: dict[str, Any] = report.to_dict()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return destination


def write_evaluation_outputs(
    records: Sequence[PredictionRecord],
    output_directory: str | Path,
    *,
    threshold: float | None = None,
    condition_key: str = "condition",
    severity_key: str = "severity",
    clean_condition: str = "clean",
    include_robustness: bool = True,
    require_robustness_metadata: bool = False,
    bin_count: int = 10,
    bootstrap_replicates: int | None = None,
    confidence_level: float = 0.95,
    seed: int | None = 0,
) -> Mapping[str, Path]:
    """Build and persist the standard report, CSV, and PNG output set.

    Robustness output is generated when every record carries the configured
    condition metadata.  When it is generated, the primary report and standard
    plots use only the clean baseline records; degraded records are represented
    by the separate robustness report and curve.  A plain prediction file can
    therefore produce a normal report without pretending that it contains a
    clean/degraded sweep; set ``require_robustness_metadata`` to make missing
    metadata an error.
    """

    typed_records = tuple(records)
    if not typed_records or any(not isinstance(item, PredictionRecord) for item in typed_records):
        raise TypeError("records must be a nonempty sequence of PredictionRecord values")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    if require_robustness_metadata and not include_robustness:
        raise ValueError(
            "require_robustness_metadata requires include_robustness=True"
        )
    robustness = None
    report_records = typed_records
    if include_robustness:
        metadata_presence = tuple(condition_key in item.metadata for item in typed_records)
        has_metadata = all(metadata_presence)
        if any(metadata_presence) and not has_metadata:
            raise ValueError(
                f"metadata[{condition_key!r}] must be present on every record when used"
            )
        if require_robustness_metadata and not has_metadata:
            raise ValueError(
                f"all prediction records require metadata[{condition_key!r}] for robustness output"
            )
        if has_metadata:
            robustness = aggregate_robustness(
                typed_records,
                condition_key=condition_key,
                severity_key=severity_key,
                clean_condition=clean_condition,
                threshold=threshold,
            )
            report_records = tuple(
                item
                for item in typed_records
                if item.metadata[condition_key] == clean_condition
            )
            if not report_records:
                # ``aggregate_robustness`` normally raises this first, but keep
                # the invariant local if its validation changes in the future.
                raise ValueError(
                    f"records must include a {clean_condition!r} baseline condition"
                )

    report = build_evaluation_report(
        report_records,
        threshold=threshold,
        bin_count=bin_count,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        seed=seed,
    )

    paths: dict[str, Path] = {}
    paths["report"] = write_report_json(report, destination / "report.json")
    paths["metrics"] = write_metrics_csv(report, destination / "metrics.csv")
    if robustness is not None:
        paths["robustness"] = destination / "robustness.csv"
        robustness.to_csv(paths["robustness"])
        paths["robustness_json"] = destination / "robustness.json"
        paths["robustness_json"].write_text(
            robustness.to_json(indent=2) + "\n", encoding="utf-8"
        )
    paths.update(
        save_evaluation_plots(
            report_records,
            destination,
            report=report,
            robustness=robustness,
        )
    )
    return paths


__all__ = [
    "write_evaluation_outputs",
    "write_metrics_csv",
    "write_report_json",
]

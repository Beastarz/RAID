"""Public model-independent evaluation API."""

from training.evaluation.calibration import compute_calibration_metrics
from training.evaluation.metrics import (
    bootstrap_confidence_intervals,
    compute_binary_metrics,
)
from training.evaluation.report import build_evaluation_report
from training.evaluation.schemas import (
    BinaryMetrics,
    CalibrationMetrics,
    EvaluationReport,
    MetricConfidenceInterval,
    ReliabilityBin,
)

__all__ = [
    "BinaryMetrics",
    "CalibrationMetrics",
    "EvaluationReport",
    "MetricConfidenceInterval",
    "ReliabilityBin",
    "bootstrap_confidence_intervals",
    "build_evaluation_report",
    "compute_binary_metrics",
    "compute_calibration_metrics",
]

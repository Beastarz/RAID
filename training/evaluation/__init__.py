"""Public model-independent evaluation API."""

from importlib import import_module

from training.evaluation.calibration import compute_calibration_metrics
from training.evaluation.metrics import (
    bootstrap_confidence_intervals,
    compute_binary_metrics,
)
from training.evaluation.robustness_report import RobustnessPoint, RobustnessReport
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
    "RobustnessBenchmark",
    "RobustnessPoint",
    "RobustnessReport",
    "aggregate_robustness",
    "plot_confusion_matrix",
    "plot_pr_curve",
    "plot_reliability_diagram",
    "plot_robustness_curves",
    "plot_roc_curve",
    "save_evaluation_plots",
    "write_evaluation_outputs",
    "write_metrics_csv",
    "write_report_json",
]


_LAZY_EXPORTS = {
    "RobustnessBenchmark": ("training.evaluation.robustness_suite", "RobustnessBenchmark"),
    "aggregate_robustness": ("training.evaluation.robustness_suite", "aggregate_robustness"),
    "build_evaluation_report": ("training.evaluation.report", "build_evaluation_report"),
    "plot_confusion_matrix": ("training.evaluation.plotting", "plot_confusion_matrix"),
    "plot_pr_curve": ("training.evaluation.plotting", "plot_pr_curve"),
    "plot_reliability_diagram": ("training.evaluation.plotting", "plot_reliability_diagram"),
    "plot_robustness_curves": ("training.evaluation.plotting", "plot_robustness_curves"),
    "plot_roc_curve": ("training.evaluation.plotting", "plot_roc_curve"),
    "save_evaluation_plots": ("training.evaluation.plotting", "save_evaluation_plots"),
    "write_evaluation_outputs": ("training.evaluation.outputs", "write_evaluation_outputs"),
    "write_metrics_csv": ("training.evaluation.outputs", "write_metrics_csv"),
    "write_report_json": ("training.evaluation.outputs", "write_report_json"),
}


def __getattr__(name: str):
    """Load optional plotting/output modules only when their APIs are used."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value

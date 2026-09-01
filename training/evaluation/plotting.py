"""Deterministic plots over model-free evaluation schemas.

The plotting layer consumes validated arrays and report objects only.  It never
loads a model, applies a transform, or performs inference.  Figures can be
returned to callers for composition or written directly as PNG artifacts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src.explainability.contracts import PredictionRecord
from training.evaluation.metrics import _prepare_evaluation_inputs
from training.evaluation.robustness_report import (
    DEGRADATION_METRIC_NAMES,
    RobustnessReport,
)
from training.evaluation.schemas import BinaryMetrics, EvaluationReport


_PNG_DPI = 120


def _inputs(
    labels_or_records: Sequence[int] | Sequence[PredictionRecord],
    probabilities: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    prepared = _prepare_evaluation_inputs(
        labels_or_records, probabilities, threshold=None, require_threshold=False
    )
    return prepared.labels, prepared.probabilities


def _new_figure(*, figsize: tuple[float, float]) -> tuple[Figure, object]:
    """Create an Agg-backed figure without changing Matplotlib's global backend."""

    figure = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    return figure, axis


def _write_figure(figure: Figure, output_path: str | Path | None) -> Figure:
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            destination,
            format="png",
            dpi=_PNG_DPI,
            bbox_inches="tight",
            metadata={"Software": "RAID evaluation plotting"},
        )
    return figure


def plot_roc_curve(
    labels_or_records: Sequence[int] | Sequence[PredictionRecord],
    probabilities: Sequence[float] | None = None,
    *,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot an ROC curve from labels and probabilities or prediction records."""

    labels, probabilities_array = _inputs(labels_or_records, probabilities)
    figure, axis = _new_figure(figsize=(5.0, 4.0))
    if np.unique(labels).size == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            labels, probabilities_array
        )
        axis.plot(false_positive_rate, true_positive_rate, label="ROC")
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(labels, probabilities_array))
        axis.legend(loc="lower right", title=f"AUC = {auc:.4f}")
    else:
        axis.text(
            0.5,
            0.5,
            "ROC-AUC undefined\n(one class present)",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.plot((0.0, 1.0), (0.0, 1.0), "--", color="0.6", linewidth=1.0)
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xlabel="False positive rate", ylabel="True positive rate")
    axis.set_title("ROC curve")
    axis.grid(True, alpha=0.25)
    return _write_figure(figure, output_path)


def plot_pr_curve(
    labels_or_records: Sequence[int] | Sequence[PredictionRecord],
    probabilities: Sequence[float] | None = None,
    *,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot a precision-recall curve from labels and probabilities."""

    labels, probabilities_array = _inputs(labels_or_records, probabilities)
    figure, axis = _new_figure(figsize=(5.0, 4.0))
    if np.any(labels == 1):
        precision, recall, _ = precision_recall_curve(labels, probabilities_array)
        axis.plot(recall, precision, label="Precision-recall")
        baseline = float(np.mean(labels))
        axis.axhline(baseline, linestyle="--", color="0.6", linewidth=1.0, label="Class prevalence")
        axis.legend(loc="lower left")
    else:
        axis.text(
            0.5,
            0.5,
            "PR-AUC undefined\n(no positive labels)",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xlabel="Recall", ylabel="Precision")
    axis.set_title("Precision-recall curve")
    axis.grid(True, alpha=0.25)
    return _write_figure(figure, output_path)


def _binary_metrics(report_or_metrics: EvaluationReport | BinaryMetrics) -> BinaryMetrics:
    if isinstance(report_or_metrics, EvaluationReport):
        return report_or_metrics.metrics
    if isinstance(report_or_metrics, BinaryMetrics):
        return report_or_metrics
    raise TypeError("report_or_metrics must be an EvaluationReport or BinaryMetrics")


def plot_confusion_matrix(
    report_or_metrics: EvaluationReport | BinaryMetrics,
    *,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot the validated fixed-threshold confusion counts."""

    metrics = _binary_metrics(report_or_metrics)
    matrix = np.asarray(
        [[metrics.true_negative, metrics.false_positive],
         [metrics.false_negative, metrics.true_positive]],
        dtype=np.int64,
    )
    figure, axis = _new_figure(figsize=(4.5, 4.0))
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=(0, 1),
        yticks=(0, 1),
        xticklabels=("Authentic", "AI-generated"),
        yticklabels=("Authentic", "AI-generated"),
        xlabel="Predicted label",
        ylabel="Ground-truth label",
        title=f"Confusion matrix (threshold = {metrics.threshold:g})",
    )
    midpoint = float(matrix.max()) / 2.0 if matrix.size and matrix.max() else 0.0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > midpoint else "black",
            )
    return _write_figure(figure, output_path)


def plot_reliability_diagram(
    report: EvaluationReport,
    *,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot the fixed-width reliability bins from an evaluation report."""

    if not isinstance(report, EvaluationReport):
        raise TypeError("report must be an EvaluationReport")
    bins = [
        item
        for item in report.calibration.reliability_bins
        if item.count > 0
    ]
    figure, axis = _new_figure(figsize=(5.0, 4.0))
    if bins:
        axis.plot(
            [item.mean_predicted_probability for item in bins],
            [item.observed_positive_fraction for item in bins],
            "o-",
            label="Observed",
        )
    axis.plot((0.0, 1.0), (0.0, 1.0), "--", color="0.6", label="Perfect calibration")
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xlabel="Mean predicted probability", ylabel="Observed positive fraction")
    axis.set_title("Reliability diagram")
    axis.legend(loc="upper left")
    axis.grid(True, alpha=0.25)
    return _write_figure(figure, output_path)


def plot_robustness_curves(
    report: RobustnessReport,
    *,
    metric_names: Sequence[str] = DEGRADATION_METRIC_NAMES,
    relative: bool = True,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot condition-wise clean-relative degradation curves.

    One line is drawn per ``condition`` and metric.  Points without a defined
    metric (for example ROC-AUC for a one-class condition) are omitted.
    """

    if not isinstance(report, RobustnessReport):
        raise TypeError("report must be a RobustnessReport")
    names = tuple(metric_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("metric_names must be nonempty and unique")
    unknown = set(names) - set(DEGRADATION_METRIC_NAMES)
    if unknown:
        raise ValueError(f"unknown robustness metrics: {sorted(unknown)}")

    figure, axis = _new_figure(figsize=(7.0, 4.5))
    source_name = "relative_degradation" if relative else "absolute_degradation"
    conditions = sorted({item.condition for item in report.points if item.condition != report.clean_condition})
    plotted = False
    for condition in conditions:
        condition_points = sorted(
            (item for item in report.points if item.condition == condition and item.severity is not None),
            key=lambda item: float(item.severity),
        )
        for metric_name in names:
            values = [getattr(item, source_name).get(metric_name) for item in condition_points]
            valid = [
                (float(item.severity), float(value))
                for item, value in zip(condition_points, values)
                if value is not None and math.isfinite(float(value))
            ]
            if not valid:
                continue
            plotted = True
            axis.plot(
                [item[0] for item in valid],
                [item[1] for item in valid],
                "o-",
                label=f"{condition}: {metric_name}",
            )
    if not plotted:
        axis.text(0.5, 0.5, "No defined degraded metrics", ha="center", va="center", transform=axis.transAxes)
    axis.axhline(0.0, color="0.6", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Severity")
    axis.set_ylabel("Relative degradation" if relative else "Absolute degradation")
    axis.set_title("Robustness curves")
    if plotted:
        axis.legend(loc="best", fontsize="small")
    axis.grid(True, alpha=0.25)
    return _write_figure(figure, output_path)


def save_evaluation_plots(
    labels_or_records: Sequence[int] | Sequence[PredictionRecord],
    output_directory: str | Path,
    *,
    probabilities: Sequence[float] | None = None,
    threshold: float | None = None,
    report: EvaluationReport | None = None,
    robustness: RobustnessReport | None = None,
) -> Mapping[str, Path]:
    """Write the standard evaluation PNG set and return artifact paths."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    if report is None:
        from training.evaluation.report import build_evaluation_report

        report = build_evaluation_report(
            labels_or_records, probabilities, threshold=threshold
        )
    paths: dict[str, Path] = {}
    for name, render in (
        ("roc_curve", lambda path: plot_roc_curve(labels_or_records, probabilities, output_path=path)),
        ("pr_curve", lambda path: plot_pr_curve(labels_or_records, probabilities, output_path=path)),
        ("confusion_matrix", lambda path: plot_confusion_matrix(report, output_path=path)),
        ("reliability_diagram", lambda path: plot_reliability_diagram(report, output_path=path)),
    ):
        path = destination / f"{name}.png"
        render(path)
        paths[name] = path
    if robustness is not None:
        path = destination / "robustness_curves.png"
        plot_robustness_curves(robustness, output_path=path)
        paths["robustness_curves"] = path
    return paths


__all__ = [
    "plot_confusion_matrix",
    "plot_pr_curve",
    "plot_reliability_diagram",
    "plot_roc_curve",
    "plot_robustness_curves",
    "save_evaluation_plots",
]

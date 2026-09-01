"""Focused tests for deterministic, model-free evaluation plots."""

from dataclasses import replace
from pathlib import Path

from training.evaluation.plotting import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_reliability_diagram,
    plot_robustness_curves,
    plot_roc_curve,
    save_evaluation_plots,
)
from training.evaluation.report import build_evaluation_report
from training.evaluation.robustness_suite import aggregate_robustness
from src.explainability.contracts import PredictionRecord


def _records():
    values = (("a", 0, 0.1), ("b", 0, 0.3), ("c", 1, 0.7), ("d", 1, 0.9))
    return tuple(
        PredictionRecord(
            sample_id=sample_id,
            model_id="model-a",
            predicted_logit=probability,
            predicted_probability=probability,
            predicted_label=int(probability >= 0.5),
            decision_threshold=0.5,
            ground_truth_label=label,
            metadata={"condition": condition, "severity": severity}
            if condition != "clean"
            else {"condition": condition},
        )
        for condition, severity, (sample_id, label, probability) in (
            ("clean", None, values[0]),
            ("clean", None, values[1]),
            ("clean", None, values[2]),
            ("clean", None, values[3]),
        )
    )


def _condition_records():
    records = []
    for condition, severity, shift in (
        ("clean", None, 0.0),
        ("jpeg", 30.0, 0.1),
        ("jpeg", 60.0, 0.05),
    ):
        for index, (sample_id, label, probability) in enumerate(
            (("a", 0, 0.1), ("b", 0, 0.3), ("c", 1, 0.7), ("d", 1, 0.9))
        ):
            score = min(max(probability + (shift if label else -shift), 0.0), 1.0)
            metadata = {"condition": condition}
            if severity is not None:
                metadata["severity"] = severity
            records.append(
                PredictionRecord(
                    sample_id=f"{sample_id}-{condition}-{index}",
                    model_id="model-a",
                    predicted_logit=score,
                    predicted_probability=score,
                    predicted_label=int(score >= 0.5),
                    decision_threshold=0.5,
                    ground_truth_label=label,
                    metadata=metadata,
                )
            )
    return tuple(records)


def test_standard_plot_functions_write_nonempty_pngs(tmp_path):
    records = _records()
    report = build_evaluation_report(records)
    paths = {
        "roc": tmp_path / "roc.png",
        "pr": tmp_path / "pr.png",
        "confusion": tmp_path / "confusion.png",
        "reliability": tmp_path / "reliability.png",
    }
    plot_roc_curve(records, output_path=paths["roc"])
    plot_pr_curve(records, output_path=paths["pr"])
    plot_confusion_matrix(report, output_path=paths["confusion"])
    plot_reliability_diagram(report, output_path=paths["reliability"])
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())


def test_robustness_plot_and_plot_set_are_deterministic(tmp_path):
    records = _condition_records()
    report = build_evaluation_report(records)
    robustness = aggregate_robustness(records)
    first = save_evaluation_plots(records, tmp_path / "first", report=report, robustness=robustness)
    second = save_evaluation_plots(records, tmp_path / "second", report=report, robustness=robustness)
    assert set(first) == {"roc_curve", "pr_curve", "confusion_matrix", "reliability_diagram", "robustness_curves"}
    assert plot_robustness_curves(robustness, output_path=tmp_path / "robustness.png")
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()


def test_robustness_plot_skips_metrics_missing_from_valid_schema(tmp_path):
    robustness = aggregate_robustness(_condition_records())
    sparse = replace(
        robustness,
        points=tuple(
            replace(point, absolute_degradation={}, relative_degradation={})
            for point in robustness.points
        ),
    )
    figure = plot_robustness_curves(sparse, output_path=tmp_path / "sparse.png")
    assert figure is not None
    assert (tmp_path / "sparse.png").exists()


def test_array_plot_set_accepts_explicit_threshold(tmp_path):
    paths = save_evaluation_plots(
        [0, 1],
        tmp_path / "arrays",
        probabilities=[0.1, 0.9],
        threshold=0.5,
    )
    assert set(paths) == {"roc_curve", "pr_curve", "confusion_matrix", "reliability_diagram"}

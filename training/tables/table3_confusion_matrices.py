"""Table 3: confusion matrices at the published threshold (0.5) -- clean and
the worst-AUC degraded condition (auto-detected from the same sweep, or
overridden with --worst-condition/--worst-severity).

Since 0.5 is a hardcoded default rather than a calibrated operating point
(see CLAUDE_HANDOFF.md's "Known gaps" #6), this table also reports what an
ROC-chosen threshold (Youden's J on the clean condition) would have given,
so the cost of not calibrating is visible rather than implied.

Usage:
    python -m training.tables.table3_confusion_matrices \
        --predictions outputs/robustness/predictions.jsonl \
        --output outputs/robustness/table3_confusion_matrices.md
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import roc_curve

from src.explainability.serialization import read_prediction_jsonl
from training.evaluation.robustness_suite import aggregate_robustness


def _confusion_section(title: str, condition: str, severity: Optional[float], report) -> list:
    metrics = report.metrics
    fpr = None if metrics.specificity is None else 1.0 - metrics.specificity
    severity_str = "-" if severity is None else f"{severity:g}"
    return [
        f"## {title} (`{condition}`, severity={severity_str}, n={metrics.sample_count})",
        "",
        "|  | Predicted: AI | Predicted: Real |",
        "|---|---:|---:|",
        f"| **Actual: AI** | TP = {metrics.true_positive} | FN = {metrics.false_negative} |",
        f"| **Actual: Real** | FP = {metrics.false_positive} | TN = {metrics.true_negative} |",
        "",
        f"Precision = {metrics.precision:.4f} · Recall (TPR) = {metrics.recall:.4f} · "
        f"FPR = {fpr:.4f} · Accuracy = {metrics.accuracy:.4f} · threshold = {metrics.threshold:g}",
        "",
    ]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Table 3: confusion matrices")
    parser.add_argument("--predictions", default="outputs/robustness/predictions.jsonl")
    parser.add_argument("--output", default="outputs/robustness/table3_confusion_matrices.md")
    parser.add_argument("--worst-condition", default=None, help="Override auto-detected worst condition")
    parser.add_argument("--worst-severity", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    records = read_prediction_jsonl(args.predictions)
    report = aggregate_robustness(records, threshold=records[0].decision_threshold)

    clean_point = next(p for p in report.points if p.condition == report.clean_condition)
    degraded = [p for p in report.points if p.condition != report.clean_condition]
    if args.worst_condition is not None:
        worst_point = next(
            p for p in degraded if p.condition == args.worst_condition and p.severity == args.worst_severity
        )
    else:
        worst_point = min(degraded, key=lambda p: p.report.metrics.roc_auc if p.report.metrics.roc_auc is not None else 1.0)

    lines = ["# Table 3 -- Confusion matrices at the published threshold", ""]
    lines += _confusion_section("Clean", clean_point.condition, clean_point.severity, clean_point.report)
    lines += _confusion_section(
        f"Worst condition (lowest AUC = {worst_point.report.metrics.roc_auc:.4f})",
        worst_point.condition,
        worst_point.severity,
        worst_point.report,
    )

    # What an ROC-chosen operating point on the clean condition would give,
    # instead of the uncalibrated 0.5 default -- Youden's J (max TPR - FPR).
    clean_records = [r for r in records if r.metadata["condition"] == report.clean_condition]
    labels = [r.ground_truth_label for r in clean_records]
    probabilities = [r.predicted_probability for r in clean_records]
    if len(set(labels)) == 2:
        fpr_curve, tpr_curve, thresholds = roc_curve(labels, probabilities)
        youden = tpr_curve - fpr_curve
        best_index = int(np.argmax(youden))
        roc_threshold = float(thresholds[best_index])
        roc_tpr = float(tpr_curve[best_index])
        roc_fpr = float(fpr_curve[best_index])
        lines.append(
            f"**Threshold calibration note**: the published threshold is a hardcoded 0.5 default, not "
            f"calibrated (see `CLAUDE_HANDOFF.md`). An ROC-chosen operating point on the clean condition "
            f"(Youden's J) would instead use threshold≈{roc_threshold:.4f}, giving TPR≈{roc_tpr:.4f} / "
            f"FPR≈{roc_fpr:.4f} on clean data -- compare against this table's clean-condition FPR to see "
            f"what the uncalibrated default is actually costing in false positives."
        )
    else:
        lines.append("**Threshold calibration note**: clean condition has only one class present -- cannot fit an ROC curve.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

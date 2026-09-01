"""Table 1: the robustness sweep -- AUC / Accuracy@threshold / TPR@1%FPR /
ΔAUC vs clean / n, one row per condition x severity plus the chained row.

Reads the predictions.jsonl training.robustness_sweep writes; does not score
any images itself, and does not re-tune the decision threshold per row --
every row uses the same clean-calibrated 0.5 threshold the bundle publishes,
so ΔAUC (and the accuracy column) reflect real degradation rather than a
threshold quietly compensating for it.

Usage:
    python -m training.tables.table1_robustness_sweep \
        --predictions outputs/robustness/predictions.jsonl \
        --output outputs/robustness/table1_robustness_sweep.md
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from sklearn.metrics import roc_curve

from src.explainability.serialization import read_prediction_jsonl
from training.evaluation.robustness_suite import aggregate_robustness


def _tpr_at_fpr(labels: Sequence[int], probabilities: Sequence[float], max_fpr: float = 0.01) -> Optional[float]:
    if len(set(labels)) < 2:
        return None
    fpr, tpr, _ = roc_curve(labels, probabilities)
    reachable = fpr <= max_fpr
    if not reachable.any():
        return 0.0
    return float(tpr[reachable].max())


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Table 1: robustness sweep")
    parser.add_argument("--predictions", default="outputs/robustness/predictions.jsonl")
    parser.add_argument("--output", default="outputs/robustness/table1_robustness_sweep.md")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    records = read_prediction_jsonl(args.predictions)
    report = aggregate_robustness(records, threshold=records[0].decision_threshold)

    groups: dict[tuple[str, Optional[float]], list] = defaultdict(list)
    for record in records:
        key = (record.metadata["condition"], record.metadata.get("severity"))
        groups[key].append(record)

    lines = [
        "# Table 1 -- Robustness sweep",
        "",
        f"Fixed decision threshold: {report.threshold:g} (not re-tuned per row). "
        f"Model: `{report.model_id}`. n = images scored under that condition.",
        "",
        "| Condition | Severity | n | AUC | Accuracy@thr | TPR@1%FPR | ΔAUC vs clean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for point in report.points:
        key = (point.condition, point.severity)
        group = groups[key]
        labels = [record.ground_truth_label for record in group]
        probabilities = [record.predicted_probability for record in group]
        tpr1 = _tpr_at_fpr(labels, probabilities)
        severity_str = "-" if point.severity is None else f"{point.severity:g}"
        delta = point.absolute_degradation.get("roc_auc")
        lines.append(
            f"| {point.condition} | {severity_str} | {len(group)} | "
            f"{_fmt(point.report.metrics.roc_auc)} | {_fmt(point.report.metrics.accuracy)} | "
            f"{_fmt(tpr1)} | {_fmt(delta, signed=True) if point.condition != report.clean_condition else '-'} |"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

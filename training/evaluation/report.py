"""Pure assembly of model-independent binary evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.explainability.contracts import PredictionRecord
from src.explainability.serialization import read_prediction_jsonl, write_prediction_jsonl
from training.evaluation.calibration import compute_calibration_metrics
from training.evaluation.metrics import (
    _prepare_evaluation_inputs,
    bootstrap_confidence_intervals,
    compute_binary_metrics,
)
from training.evaluation.schemas import EvaluationReport


def build_evaluation_report(
    labels_or_records: Sequence[int] | Sequence[PredictionRecord],
    probabilities: Sequence[float] | None = None,
    *,
    threshold: float | None = None,
    bin_count: int = 10,
    bootstrap_replicates: int | None = None,
    confidence_level: float = 0.95,
    seed: int | None = 0,
) -> EvaluationReport:
    """Combine metrics and calibration without model, plotting, or filesystem I/O."""
    inputs = _prepare_evaluation_inputs(
        labels_or_records, probabilities, threshold, require_threshold=True
    )
    assert inputs.threshold is not None
    metrics = compute_binary_metrics(
        inputs.labels, inputs.probabilities, threshold=inputs.threshold
    )
    calibration = compute_calibration_metrics(
        inputs.labels, inputs.probabilities, bin_count=bin_count
    )
    intervals = ()
    if bootstrap_replicates is not None:
        intervals = bootstrap_confidence_intervals(
            inputs.labels,
            inputs.probabilities,
            threshold=inputs.threshold,
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=seed,
            bin_count=bin_count,
        )
    return EvaluationReport(
        sample_count=metrics.sample_count,
        positive_count=metrics.positive_count,
        negative_count=metrics.negative_count,
        threshold=metrics.threshold,
        model_id=inputs.model_id,
        metrics=metrics,
        calibration=calibration,
        confidence_intervals=intervals,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Build report, CSV, robustness, and plot artifacts from prediction JSONL."""

    parser = argparse.ArgumentParser(
        description="Build model-free evaluation and robustness reports from JSONL predictions"
    )
    parser.add_argument("--predictions", required=True, help="Input prediction JSONL path")
    parser.add_argument("--output", required=True, help="Output report directory")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--severity-key", default="severity")
    parser.add_argument("--clean-condition", default="clean")
    parser.add_argument("--no-robustness", action="store_true")
    parser.add_argument(
        "--require-robustness-metadata",
        action="store_true",
        help="Fail unless every record has condition metadata",
    )
    parser.add_argument("--bin-count", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.no_robustness and args.require_robustness_metadata:
        parser.error("--require-robustness-metadata cannot be used with --no-robustness")

    records = read_prediction_jsonl(args.predictions)
    from training.evaluation.outputs import write_evaluation_outputs

    paths = dict(
        write_evaluation_outputs(
            records,
            args.output,
            threshold=args.threshold,
            condition_key=args.condition_key,
            severity_key=args.severity_key,
            clean_condition=args.clean_condition,
            include_robustness=not args.no_robustness,
            require_robustness_metadata=args.require_robustness_metadata,
            bin_count=args.bin_count,
            bootstrap_replicates=args.bootstrap_replicates,
            confidence_level=args.confidence_level,
            seed=args.seed,
        )
    )
    paths["predictions"] = write_prediction_jsonl(
        records, Path(args.output) / "predictions.jsonl"
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    return 0


__all__ = ["build_evaluation_report", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

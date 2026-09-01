"""Focused tests for report output assembly and prediction JSONL."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.explainability.contracts import PredictionRecord
from src.explainability.serialization import read_prediction_jsonl, write_prediction_jsonl
from training.evaluation.outputs import write_evaluation_outputs
from training.evaluation.report import main as report_main


def _records():
    values = (("a", 0, 0.1), ("b", 0, 0.2), ("c", 1, 0.8), ("d", 1, 0.9))
    result = []
    for condition, severity, shift in (("clean", None, 0.0), ("blur", 1.0, 0.05)):
        for sample_id, label, probability in values:
            score = probability + (shift if label else -shift)
            metadata = {"condition": condition}
            if severity is not None:
                metadata["severity"] = severity
            result.append(
                PredictionRecord(
                    sample_id=f"{sample_id}-{condition}",
                    model_id="model-a",
                    predicted_logit=score,
                    predicted_probability=score,
                    predicted_label=int(score >= 0.5),
                    decision_threshold=0.5,
                    ground_truth_label=label,
                    metadata=metadata,
                )
            )
    return tuple(result)


def test_prediction_jsonl_round_trip_is_strict_and_deterministic(tmp_path):
    records = _records()
    path = write_prediction_jsonl(records, tmp_path / "predictions.jsonl")
    assert read_prediction_jsonl(path) == records
    assert path.read_text(encoding="utf-8").count("\n") == len(records)
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"schema_version":"1.0","predicted_probability":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_prediction_jsonl(malformed)


def test_prediction_jsonl_writer_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        write_prediction_jsonl((), tmp_path / "empty.jsonl")


def test_write_evaluation_outputs_assembles_report_csv_robustness_and_plots(tmp_path):
    paths = write_evaluation_outputs(_records(), tmp_path / "report")
    for name in (
        "report",
        "metrics",
        "robustness",
        "robustness_json",
        "roc_curve",
        "pr_curve",
        "confusion_matrix",
        "reliability_diagram",
        "robustness_curves",
    ):
        assert name in paths and paths[name].exists()
    payload = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert payload["model_id"] == "model-a"
    assert payload["sample_count"] == 4
    assert "robustness" not in payload
    robustness_payload = json.loads(paths["robustness_json"].read_text(encoding="utf-8"))
    assert robustness_payload["schema_version"] == "1.0"
    assert "roc_auc" in paths["metrics"].read_text(encoding="utf-8").splitlines()[0]


def test_metrics_csv_contains_bootstrap_confidence_intervals(tmp_path):
    paths = write_evaluation_outputs(
        _records()[:4],
        tmp_path / "intervals",
        bootstrap_replicates=8,
        seed=3,
    )
    header = paths["metrics"].read_text(encoding="utf-8").splitlines()[0]
    assert "ci_roc_auc_lower" in header
    assert "ci_roc_auc_upper" in header
    assert "ci_roc_auc_valid_replicates" in header


def test_plain_prediction_output_skips_robustness_without_condition_metadata(tmp_path):
    records = tuple(
        record.__class__(**{**record.to_dict(), "metadata": {}})
        for record in _records()[:4]
    )
    paths = write_evaluation_outputs(records, tmp_path / "plain")
    assert "robustness" not in paths
    with pytest.raises(ValueError, match="condition"):
        write_evaluation_outputs(
            records,
            tmp_path / "strict",
            require_robustness_metadata=True,
        )


def test_partial_robustness_metadata_is_rejected(tmp_path):
    records = list(_records())
    records[0] = records[0].__class__(**{**records[0].to_dict(), "metadata": {}})
    with pytest.raises(ValueError, match="present on every record"):
        write_evaluation_outputs(records, tmp_path / "partial")
    with pytest.raises(ValueError, match="include_robustness"):
        write_evaluation_outputs(
            _records()[:4],
            tmp_path / "disabled",
            include_robustness=False,
            require_robustness_metadata=True,
        )


def test_report_cli_writes_the_documented_output_set(tmp_path, capsys):
    prediction_path = write_prediction_jsonl(_records(), tmp_path / "input.jsonl")
    assert report_main(["--predictions", str(prediction_path), "--output", str(tmp_path / "cli")]) == 0
    output = json.loads(capsys.readouterr().out)
    assert (tmp_path / "cli" / "report.json").exists()
    assert (tmp_path / "cli" / "predictions.jsonl").exists()
    assert output["report"].endswith("report.json")


def test_report_module_entrypoint_is_warning_free():
    result = subprocess.run(
        [sys.executable, "-m", "training.evaluation.report", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "RuntimeWarning" not in result.stderr

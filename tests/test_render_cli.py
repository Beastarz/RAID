"""Focused tests for the standalone explanation rendering command."""

import json
import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path

from src.explainability.artifacts import ArtifactStore
from src.explainability.contracts import (
    CapabilityStatus,
    ExplanationOutputEnvelope,
    ExplanationResult,
)
from src.explainability.render import main as render_main, render_explanation_file
from src.explainability.render import _render_map
from src.explainability.serialization import read_explanation_json, write_envelope_json


def test_render_explanation_file_writes_stable_per_sample_pngs(tmp_path):
    source_root = tmp_path / "source"
    store = ArtifactStore(source_root / "explanations", declared_path_prefix="explanations")
    reference = store.save_npy(
        "sample-1",
        "semantic-map",
        np.arange(16, dtype=np.float32).reshape(4, 4),
        coordinate_space="image",
        method_name="integrated_gradients",
        metadata={"branch_name": "semantic", "target_class": 1},
    )
    envelope_path = source_root / "explanations.json"
    write_envelope_json(
        ExplanationOutputEnvelope(
            (
                ExplanationResult(
                    sample_id="sample-1",
                    model_id="model-a",
                    method_name="integrated_gradients",
                    status=CapabilityStatus.supported(),
                    artifacts=(reference,),
                ),
            )
        ),
        envelope_path,
    )
    output = render_explanation_file(envelope_path, tmp_path / "rendered")
    assert output.exists()
    assert (tmp_path / "rendered" / "explanations" / "sample-1" / "semantic-map.png").exists()
    rendered = read_explanation_json(output)
    source_metadata = rendered.explanations[0].artifacts[0].metadata["source_metadata"]
    assert source_metadata["branch_name"] == "semantic"
    assert source_metadata["target_class"] == 1


def test_render_rejects_artifact_path_escape(tmp_path):
    envelope_path = tmp_path / "escape.json"
    envelope_path.write_text(
        '{"schema_version":"1.0","explanations":[{"schema_version":"1.0",'
        '"sample_id":"sample-1","model_id":"model-a","method_name":"m",'
        '"status":{"available":true,"reason":null},"artifacts":[{"schema_version":"1.0",'
        '"artifact_id":"map","path":"../outside.npy","artifact_type":"attribution_map",'
        '"media_type":"application/x-npy","metadata":{}}],"statistics":{},'
        '"branch_coalition_logits":[],"metadata":{}}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="safe relative path"):
        render_explanation_file(envelope_path, tmp_path / "rendered")


def test_render_magnitude_preserves_three_dimensional_rms_semantics():
    residual = np.stack(
        [np.ones((2, 2), dtype=np.float32), -np.ones((2, 2), dtype=np.float32)],
        axis=-1,
    )
    rendered = _render_map(
        residual,
        coordinate_space="image",
        method_name="residual",
        metadata={"channel_layout": "channel_last", "render_mode": "magnitude"},
        artifact_id="residual",
    )
    assert rendered.metadata["raw_statistics"]["minimum"] == -1.0
    assert rendered.metadata["magnitude_statistics"]["minimum"] == 1.0
    assert rendered.metadata["magnitude_statistics"]["maximum"] == 1.0
    assert rendered.metadata["channel_reduction"] == "root_mean_square"


def test_render_rejects_media_type_extension_mismatch(tmp_path):
    source_root = tmp_path / "source"
    store = ArtifactStore(source_root / "explanations", declared_path_prefix="explanations")
    reference = store.save_npy(
        "sample-1",
        "map",
        np.eye(3, dtype=np.float32),
        coordinate_space="image",
        method_name="m",
    )
    payload = {
        "schema_version": "1.0",
        "explanations": [{
            "schema_version": "1.0",
            "sample_id": "sample-1",
            "model_id": "model-a",
            "method_name": "m",
            "status": {"available": True, "reason": None},
            "artifacts": [{
                **reference.to_dict(),
                "media_type": "image/png",
            }],
            "statistics": {},
            "branch_coalition_logits": [],
            "metadata": {},
        }],
    }
    envelope_path = source_root / "input.json"
    envelope_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.png path"):
        render_explanation_file(envelope_path, tmp_path / "rendered")


def test_render_rejects_duplicate_output_destinations(tmp_path):
    source_root = tmp_path / "source"
    store = ArtifactStore(source_root / "explanations", declared_path_prefix="explanations")
    reference = store.save_npy(
        "sample-1",
        "map",
        np.eye(3, dtype=np.float32),
        coordinate_space="image",
        method_name="m",
    )
    envelope_path = source_root / "input.json"
    write_envelope_json(
        ExplanationOutputEnvelope(
            (
                ExplanationResult(
                    sample_id="sample-1",
                    model_id="model-a",
                    method_name="m",
                    status=CapabilityStatus.supported(),
                    artifacts=(reference, reference),
                ),
            )
        ),
        envelope_path,
    )
    with pytest.raises(ValueError, match="duplicate rendered artifact destination"):
        render_explanation_file(envelope_path, tmp_path / "rendered")


def test_render_escapes_contract_ids_for_stable_directories(tmp_path):
    source_root = tmp_path / "source"
    store = ArtifactStore(source_root / "explanations", declared_path_prefix="explanations")
    reference = store.save_npy(
        "source",
        "map",
        np.eye(3, dtype=np.float32),
        coordinate_space="image",
        method_name="m",
    )
    envelope_path = source_root / "input.json"
    write_envelope_json(
        ExplanationOutputEnvelope(
            (
                ExplanationResult(
                    sample_id="sample 1",
                    model_id="model-a",
                    method_name="m",
                    status=CapabilityStatus.supported(),
                    artifacts=(reference,),
                ),
            )
        ),
        envelope_path,
    )
    output = render_explanation_file(envelope_path, tmp_path / "rendered")
    rendered = read_explanation_json(output)
    assert rendered.explanations[0].sample_id == "sample 1"
    assert rendered.explanations[0].artifacts[0].path == (
        "explanations/id-73616d706c652031/map.png"
    )


def test_render_cli_writes_envelope_and_png(tmp_path, capsys):
    source_root = tmp_path / "source"
    store = ArtifactStore(source_root / "explanations", declared_path_prefix="explanations")
    reference = store.save_npy(
        "sample-1",
        "map",
        np.eye(3, dtype=np.float32),
        coordinate_space="image",
        method_name="m",
    )
    envelope_path = source_root / "input.json"
    write_envelope_json(
        ExplanationOutputEnvelope(
            (
                ExplanationResult(
                    sample_id="sample-1",
                    model_id="model-a",
                    method_name="m",
                    status=CapabilityStatus.supported(),
                    artifacts=(reference,),
                ),
            )
        ),
        envelope_path,
    )
    assert render_main(["--input", str(envelope_path), "--output", str(tmp_path / "cli")]) == 0
    assert json.loads(capsys.readouterr().out)["rendered_explanations"].endswith(
        "rendered_explanations.json"
    )


def test_render_module_entrypoint_is_warning_free():
    result = subprocess.run(
        [sys.executable, "-m", "src.explainability.render", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "RuntimeWarning" not in result.stderr

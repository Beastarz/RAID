"""End-to-end tests for adapter-backed prediction and explanation output."""

from pathlib import Path

import pytest
import torch

from predict import build_adapter, explain_single, predict_single
from src.explainability.serialization import read_explanation_json


ROOT = Path(__file__).parents[1]
BUNDLE = ROOT / "checkpoints" / "detector_bundle.pt"
OUTPUT = ROOT / "outputs" / "pytest-wave5-cli"


@pytest.mark.skipif(not BUNDLE.is_file(), reason="canonical detector bundle is not installed")
def test_real_prediction_and_supported_gradcam_output():
    adapter = build_adapter(str(BUNDLE), torch.device("cpu"))
    prediction = predict_single(adapter, ROOT / "test_sample.jpg", torch.device("cpu"))
    assert prediction["ai_probability"] == 0.405379
    assert prediction["label"] == "Authentic"

    path = explain_single(adapter, ROOT / "test_sample.jpg", "semantic-gradcam", OUTPUT)
    envelope = read_explanation_json(path)
    explanation = envelope.explanations[0]
    assert explanation.status.available
    assert explanation.method_name == "semantic-gradcam"
    assert {artifact.media_type for artifact in explanation.artifacts} == {
        "image/png", "application/x-npy"
    }
    assert all("coordinate_space" in artifact.metadata for artifact in explanation.artifacts)
    assert all((OUTPUT / artifact.path).is_file() for artifact in explanation.artifacts)


@pytest.mark.skipif(not BUNDLE.is_file(), reason="canonical detector bundle is not installed")
def test_attention_writes_structured_unsupported_explanation():
    adapter = build_adapter(str(BUNDLE), torch.device("cpu"))
    path = explain_single(adapter, ROOT / "test_sample.jpg", "attention", OUTPUT)
    explanation = read_explanation_json(path).explanations[0]
    assert not explanation.status.available
    assert "does not expose attention matrices" in explanation.status.reason
    assert explanation.artifacts == ()

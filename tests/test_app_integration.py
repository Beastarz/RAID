"""Real-checkpoint smoke tests for the Gradio prediction function."""

import json
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).parents[1]
BUNDLE = ROOT / "checkpoints" / "detector_bundle.pt"


@pytest.mark.skipif(not BUNDLE.is_file(), reason="canonical detector bundle is not installed")
def test_gradio_prediction_preserves_threshold_and_returns_real_explanation():
    import app

    image = Image.open(ROOT / "test_sample.jpg").convert("RGB")
    label, probability, visual, raw_json = app.predict(
        image, threshold=0.4, explanation_view="Semantic Grad-CAM"
    )
    payload = json.loads(raw_json)
    assert label == "AI-Generated"
    assert probability == pytest.approx(0.4053786099, abs=1e-7)
    assert visual.size == (512, 512)
    assert payload["prediction"]["applied_threshold"] == 0.4
    assert payload["explanation"]["coordinate_space"] == "semantic_patch_grid"
    assert payload["explanation"]["native_grid_size"] == [14, 14]
    assert payload["explanation"]["display_size"] == [512, 512]
    assert payload["explanation"]["display_interpolation"] == "nearest"
    assert payload["branch_contributions"]["available"] is False


@pytest.mark.skipif(not BUNDLE.is_file(), reason="canonical detector bundle is not installed")
def test_gradio_attention_is_explicitly_unsupported_and_interface_builds():
    import app

    image = Image.open(ROOT / "test_sample.jpg").convert("RGB")
    _, _, visual, raw_json = app.predict(
        image, explanation_view="Attention rollout (unsupported)"
    )
    assert visual is None
    assert json.loads(raw_json)["explanation"]["status"] == {
        "available": False,
        "reason": "torchvision ViT forward does not expose attention matrices",
    }
    assert app.build_interface().__class__.__name__ == "Blocks"

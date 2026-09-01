"""Focused tests for deterministic rendering and artifact persistence."""

import json
import sys

import numpy as np
import pytest
from PIL import Image

from src.explainability.artifacts import ArtifactStore
from src.explainability.contracts import (
    CapabilityStatus,
    ExplanationOutputEnvelope,
    ExplanationResult,
    PredictionOutputEnvelope,
    PredictionRecord,
)
from src.explainability.rendering import (
    colorize_heatmap,
    overlay_heatmap,
    percentile_normalize,
    raw_statistics,
    render_residual_magnitude,
    render_signed_residual,
    resize_scalar_heatmap,
    side_by_side_panel,
    to_numpy,
)
from src.explainability.serialization import envelope_to_json, write_envelope_json


class FakeTensor:
    def __init__(self, value, calls=None):
        self.value = value
        self.calls = [] if calls is None else calls

    def detach(self):
        self.calls.append("detach")
        return FakeCpuTensor(self.value, self.calls)


class FakeCpuTensor:
    def __init__(self, value, calls):
        self.value = value
        self.calls = calls

    def cpu(self):
        self.calls.append("cpu")
        return FakeNumpyTensor(self.value, self.calls)


class FakeNumpyTensor:
    def __init__(self, value, calls):
        self.value = value
        self.calls = calls

    def numpy(self):
        self.calls.append("numpy")
        return self.value


def test_tensor_duck_typing_is_detached_and_does_not_mutate_input():
    source = np.arange(4, dtype=np.float32).reshape(2, 2)
    tensor = FakeTensor(source)
    result = to_numpy(tensor)
    result[0, 0] = 100

    assert tensor.calls == ["detach", "cpu", "numpy"]
    assert source[0, 0] == 0


@pytest.mark.parametrize("value", [np.array([np.nan]), np.array([np.inf]), [object()]])
def test_array_conversion_rejects_non_finite_and_unsupported_values(value):
    with pytest.raises((TypeError, ValueError)):
        to_numpy(value)


def test_array_conversion_rejects_boolean_masks():
    with pytest.raises(TypeError, match="boolean"):
        to_numpy(np.array([[True, False]], dtype=bool))


def test_resize_normalization_and_statistics_are_deterministic():
    heatmap = np.array([[0.0, 1.0], [2.0, 3.0]])
    resized = resize_scalar_heatmap(heatmap, (4, 3))
    normalized, display = percentile_normalize(heatmap, lower_percentile=0, upper_percentile=100)
    statistics = raw_statistics(heatmap, percentiles=(0, 50, 100))

    assert resized.shape == (4, 3)
    assert normalized.tolist() == [[0.0, 1 / 3], [2 / 3, 1.0]]
    assert display["constant_map"] is False
    assert statistics == {
        "shape": [2, 2],
        "minimum": 0.0,
        "maximum": 3.0,
        "mean": 1.5,
        "standard_deviation": pytest.approx(1.118033988749895),
        "percentiles": {"p0": 0.0, "p50": 1.5, "p100": 3.0},
    }
    json.dumps(statistics, allow_nan=False)


def test_constant_maps_are_deterministically_zero_normalized():
    normalized, metadata = percentile_normalize(np.full((3, 2), 7.0))
    assert np.array_equal(normalized, np.zeros((3, 2)))
    assert metadata["constant_map"] is True
    assert metadata["lower_value"] == metadata["upper_value"] == 7.0


@pytest.mark.parametrize("lower,upper", [(-1, 99), (1, 101), (50, 50), (80, 20), (np.nan, 99)])
def test_invalid_percentiles_are_rejected(lower, upper):
    with pytest.raises((TypeError, ValueError)):
        percentile_normalize(np.ones((2, 2)), lower_percentile=lower, upper_percentile=upper)


def test_frequency_heatmap_is_standalone_and_never_silently_overlaid():
    rendered = colorize_heatmap(
        np.arange(6).reshape(2, 3),
        coordinate_space="frequency_plane",
        method_name="fft_energy",
    )
    assert rendered.image.shape == (2, 3, 3)
    assert rendered.image.dtype == np.uint8
    assert rendered.metadata["coordinate_space"] == "frequency_plane"
    with pytest.raises(ValueError, match="coordinate_space='image'"):
        overlay_heatmap(
            np.zeros((4, 5, 3), dtype=np.uint8),
            np.ones((2, 3)),
            coordinate_space="frequency_plane",
            method_name="fft_energy",
        )


def test_image_overlay_resizes_and_validates_alpha_and_source():
    source = np.full((4, 5, 3), 100, dtype=np.uint8)
    result = overlay_heatmap(
        source,
        np.array([[0.0, 1.0], [2.0, 3.0]]),
        coordinate_space="image",
        method_name="semantic",
        alpha=0.25,
    )
    assert result.image.shape == source.shape
    assert result.image.dtype == np.uint8
    assert result.metadata["raw_statistics"]["shape"] == [2, 2]
    for alpha in (-0.1, 1.1, np.nan):
        with pytest.raises((TypeError, ValueError)):
            overlay_heatmap(source, np.ones((2, 2)), coordinate_space="image", method_name="m", alpha=alpha)
    with pytest.raises(ValueError, match="HWC RGB"):
        overlay_heatmap(np.zeros((4, 5)), np.ones((2, 2)), coordinate_space="image", method_name="m")


def test_signed_and_magnitude_residuals_disclose_absolute_scale():
    residual = np.array([[-1e-8, 0.0], [1e-8, 0.5e-8]])
    signed = render_signed_residual(residual, coordinate_space="image", method_name="npr_residual")
    magnitude = render_residual_magnitude(residual, coordinate_space="image", method_name="npr_residual")

    assert signed.image.shape == magnitude.image.shape == (2, 2, 3)
    assert signed.metadata["display_normalization"]["absolute_scale_maximum"] == 1e-8
    assert signed.metadata["raw_statistics"]["minimum"] == -1e-8
    assert magnitude.metadata["magnitude_statistics"]["maximum"] == 1e-8
    json.dumps(signed.metadata, allow_nan=False)


def test_residual_layout_is_explicit_and_ambiguous_shapes_are_rejected():
    chw = np.arange(12, dtype=float).reshape(3, 2, 2)
    hwc = np.moveaxis(chw, 0, 2)
    first = render_residual_magnitude(
        chw, coordinate_space="image", method_name="residual", channel_layout="channel_first"
    )
    last = render_residual_magnitude(
        hwc, coordinate_space="image", method_name="residual", channel_layout="channel_last"
    )
    assert np.array_equal(first.image, last.image)
    with pytest.raises(ValueError, match="explicit channel_layout"):
        render_signed_residual(chw, coordinate_space="image", method_name="residual")
    with pytest.raises(ValueError, match="2D residual"):
        render_signed_residual(np.ones((2, 2)), coordinate_space="image", method_name="residual", channel_layout="channel_first")


def test_side_by_side_panel_has_explicit_separator():
    left = np.zeros((2, 3, 3), dtype=np.uint8)
    right = np.full((2, 4, 3), 10, dtype=np.uint8)
    panel = side_by_side_panel(left, right, separator_width=2, separator_color=(1, 2, 3))
    assert panel.shape == (2, 9, 3)
    assert np.all(panel[:, 3:5] == np.array([1, 2, 3]))
    assert np.array_equal(panel, side_by_side_panel(left, right, separator_width=2, separator_color=(1, 2, 3)))


def test_artifact_store_writes_png_and_lossless_npy_with_serializable_metadata(tmp_path):
    raw = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    rendered = colorize_heatmap(raw, coordinate_space="frequency_plane", method_name="fft_energy")
    store = ArtifactStore(tmp_path / "outputs", declared_path_prefix="explanations")
    references = store.save_rendered("sample-01", "frequency-map", rendered, raw_value=raw)

    png, npy = references
    assert png.path == "explanations/sample-01/frequency-map.png"
    assert npy.path == "explanations/sample-01/frequency-map-raw.npy"
    with Image.open(tmp_path / "outputs/sample-01/frequency-map.png") as image:
        assert image.mode == "RGB"
        assert image.size == (2, 2)
    loaded = np.load(tmp_path / "outputs/sample-01/frequency-map-raw.npy", allow_pickle=False)
    assert np.array_equal(loaded, raw)
    assert npy.metadata["display_normalization"] == {"type": "none"}
    json.dumps([item.to_dict() for item in references], allow_nan=False)


def test_artifact_metadata_cannot_override_rendered_or_generated_provenance(tmp_path):
    raw = np.array([[0.0, 1.0], [2.0, 3.0]])
    rendered = colorize_heatmap(
        raw, coordinate_space="frequency_plane", method_name="fft_energy", colormap="viridis"
    )
    store = ArtifactStore(tmp_path)
    overrides = {
        "coordinate_space": "image",
        "method_name": "untrusted",
        "raw_statistics": {"minimum": 999.0},
        "display_normalization": {"type": "untrusted"},
        "colormap": "untrusted",
        "caller_note": "preserved",
    }

    png = store.save_png("sample", "map", rendered, metadata=overrides)
    npy = store.save_npy(
        "sample",
        "raw",
        raw,
        coordinate_space="frequency_plane",
        method_name="fft_energy",
        metadata=overrides,
    )

    assert png.metadata["coordinate_space"] == "frequency_plane"
    assert png.metadata["method_name"] == "fft_energy"
    assert png.metadata["raw_statistics"]["minimum"] == 0.0
    assert png.metadata["display_normalization"]["type"] == "percentile"
    assert png.metadata["colormap"] == "viridis"
    assert npy.metadata["coordinate_space"] == "frequency_plane"
    assert npy.metadata["method_name"] == "fft_energy"
    assert npy.metadata["raw_statistics"]["minimum"] == 0.0
    assert npy.metadata["display_normalization"] == {"type": "none"}
    assert png.metadata["caller_note"] == npy.metadata["caller_note"] == "preserved"


@pytest.mark.parametrize("unsafe", ["../sample", "a/b", ".hidden", "", "sample name"])
def test_artifact_store_rejects_unsafe_ids(tmp_path, unsafe):
    store = ArtifactStore(tmp_path)
    rendered = colorize_heatmap(np.ones((2, 2)), coordinate_space="image", method_name="m")
    with pytest.raises(ValueError):
        store.save_png(unsafe, "map", rendered)
    with pytest.raises(ValueError):
        store.save_png("sample", unsafe, rendered)


@pytest.mark.parametrize(
    "prefix",
    [
        "../reports",
        "/reports",
        "reports\\explanations",
        "reports/../explanations",
        "reports/./explanations",
        "./reports",
        "reports//explanations",
        "reports/",
        ".",
        "C:/reports",
    ],
)
def test_artifact_store_rejects_non_portable_declared_prefixes(tmp_path, prefix):
    with pytest.raises(ValueError, match="safe relative path"):
        ArtifactStore(tmp_path / "output", declared_path_prefix=prefix)


def test_artifact_store_accepts_nested_posix_declared_prefix(tmp_path):
    store = ArtifactStore(
        tmp_path / "output", declared_path_prefix="reports/explanations"
    )
    rendered = colorize_heatmap(np.ones((2, 2)), coordinate_space="image", method_name="m")

    reference = store.save_png("sample", "map", rendered)

    assert reference.path == "reports/explanations/sample/map.png"


def test_artifact_store_rejects_sample_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    try:
        (output / "sample").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        if sys.platform == "win32" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation privilege is unavailable")
        raise
    store = ArtifactStore(output)
    rendered = colorize_heatmap(np.ones((2, 2)), coordinate_space="image", method_name="m")

    with pytest.raises(ValueError, match="within output_directory"):
        store.save_png("sample", "map", rendered)


def test_strict_envelope_json_is_deterministic_utf8_and_creates_parents(tmp_path):
    prediction = PredictionRecord(
        sample_id="échantillon",
        model_id="model",
        predicted_logit=0.0,
        predicted_probability=0.5,
        predicted_label=1,
        decision_threshold=0.5,
    )
    prediction_envelope = PredictionOutputEnvelope((prediction,))
    explanation_envelope = ExplanationOutputEnvelope(
        (
            ExplanationResult(
                sample_id="sample",
                model_id="model",
                method_name="method",
                status=CapabilityStatus.supported(),
            ),
        )
    )
    destination = write_envelope_json(prediction_envelope, tmp_path / "nested/predictions.json")

    assert destination.read_text(encoding="utf-8") == envelope_to_json(prediction_envelope) + "\n"
    assert "échantillon" in destination.read_text(encoding="utf-8")
    assert envelope_to_json(explanation_envelope) == envelope_to_json(explanation_envelope)
    json.loads(destination.read_text(encoding="utf-8"), parse_constant=lambda value: pytest.fail(value))

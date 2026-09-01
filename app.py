"""Lightweight Gradio frontend for the canonical final detector."""

import json
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image

from predict import build_adapter, get_device
from src.explainability.adapters.detector_adapter import (
    DetectorAttributionTarget,
    IntermediateRepresentation,
)
from src.explainability.gradcam import grad_cam
from src.explainability.rendering import colorize_heatmap
from src.models.fused_detector import prepare_fused_inputs

DEVICE = get_device()

_BUNDLE_CKPT = Path("checkpoints/detector_bundle.pt")
if _BUNDLE_CKPT.is_file():
    try:
        ADAPTER = build_adapter(checkpoint=str(_BUNDLE_CKPT), device=DEVICE)
        MODEL = ADAPTER.model
        MODEL_STATUS = "Using the validated canonical detector bundle."
    except Exception as exc:  # Keep the UI importable while making failure explicit.
        ADAPTER = None
        MODEL = None
        MODEL_STATUS = f"**Canonical detector bundle failed validation:** `{exc}`"
else:
    ADAPTER = None
    MODEL = None
    MODEL_STATUS = (
        "**No canonical detector bundle found.** Build `checkpoints/detector_bundle.pt` "
        "with `python -m training.build_detector_bundle --parity-image test_sample.jpg`."
    )

EXAMPLE_IMAGES = [
    "test_sample.jpg",
    "training/datasets/test/REAL/0000 (10).jpg",
    "training/datasets/test/FAKE/0 (10).jpg",
]

LABEL_CSS = """
#label_output.ai-label .confidence-set,
#label_output.ai-label .confidence-set:hover,
#label_output.ai-label .confidence-set:focus {
    background: #f8d7da !important;
    border: none !important;
    box-shadow: none !important;
    cursor: default !important;
    pointer-events: none !important;
}
#label_output.ai-label .bar { background: #d93025 !important; }
#label_output.ai-label .label .text,
#label_output.ai-label .label .confidence { color: #7a0d0d !important; }

#label_output.real-label .confidence-set,
#label_output.real-label .confidence-set:hover,
#label_output.real-label .confidence-set:focus {
    background: #d7f3df !important;
    border: none !important;
    box-shadow: none !important;
    cursor: default !important;
    pointer-events: none !important;
}
#label_output.real-label .bar { background: #188038 !important; }
#label_output.real-label .label .text,
#label_output.real-label .label .confidence { color: #0b4d20 !important; }
"""


def _preprocess(image: Image.Image) -> torch.Tensor:
    """Return the semantic view from the canonical shared preparation."""

    return prepare_fused_inputs(image).semantic


def _preprocess_raw(image: Image.Image) -> torch.Tensor:
    """Return the forensic view from the same canonical 512x512 resize."""

    return prepare_fused_inputs(image).forensic


EXPLANATION_VIEWS = {
    "Semantic Grad-CAM": "semantic",
    "Forensic Grad-CAM": "forensic",
    "Bayar+SRM fused intermediate": "intermediate",
    "Attention rollout (unsupported)": "attention",
}
EXPLANATION_DISPLAY_SIZE = 512


def _expand_explanation_display(
    rendered_image: object,
    *,
    native_size: tuple[int, int],
    interpolation: str,
) -> Image.Image:
    """Scale a native explanation grid to fill the UI without changing its meaning."""

    filters = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
    }
    if interpolation not in filters:
        raise ValueError("explanation display interpolation must be nearest or bilinear")
    image = Image.fromarray(rendered_image)
    if image.size != native_size:
        raise ValueError("rendered explanation does not match its declared native size")
    return image.resize(
        (EXPLANATION_DISPLAY_SIZE, EXPLANATION_DISPLAY_SIZE),
        resample=filters[interpolation],
    )


def _standalone_explanation(image: Image.Image, view: str) -> tuple[Image.Image | None, dict[str, object]]:
    assert ADAPTER is not None
    prepared = ADAPTER.prepare_source_image(image)
    if view in {"semantic", "forensic"}:
        targets = ADAPTER.attribution_targets(prepared)
        assert targets.value is not None
        payload = targets.value[view].value
        assert isinstance(payload, DetectorAttributionTarget)
        result = grad_cam(
            ADAPTER.model, payload.module, payload.scoring_callable, lambda value: value,
            activation_transform=payload.activation_transform,
        )
        rendered = colorize_heatmap(
            result.heatmap[0], coordinate_space=payload.coordinate_space,
            method_name=f"{view}-gradcam", colormap="turbo",
            lower_percentile=0.0, upper_percentile=100.0,
        )
        native_size = (int(rendered.image.shape[1]), int(rendered.image.shape[0]))
        display_interpolation = "nearest" if view == "semantic" else "bilinear"
        display = _expand_explanation_display(
            rendered.image,
            native_size=native_size,
            interpolation=display_interpolation,
        )
        return display, {
            "method": f"{view}-gradcam",
            "status": {"available": True, "reason": None},
            "coordinate_space": payload.coordinate_space,
            "raw_scale": "gradcam_relu_weighted_activation",
            "native_grid_size": list(native_size),
            "display_size": [EXPLANATION_DISPLAY_SIZE, EXPLANATION_DISPLAY_SIZE],
            "display_interpolation": display_interpolation,
            "rendering": rendered.metadata,
        }
    if view == "intermediate":
        values = ADAPTER.intermediate_representations(prepared)
        assert values.value is not None
        representation = values.value["forensic"]["frontend.fuse"]
        assert isinstance(representation, IntermediateRepresentation)
        raw = representation.value[0].abs().mean(dim=0)
        rendered = colorize_heatmap(
            raw, coordinate_space=representation.coordinate_space,
            method_name="forensic:frontend.fuse", colormap="magma",
            lower_percentile=0.0, upper_percentile=100.0,
        )
        native_size = (int(rendered.image.shape[1]), int(rendered.image.shape[0]))
        display = _expand_explanation_display(
            rendered.image,
            native_size=native_size,
            interpolation="bilinear",
        )
        return display, {
            "method": "forensic:frontend.fuse",
            "status": {"available": True, "reason": None},
            "coordinate_space": representation.coordinate_space,
            "raw_scale": representation.raw_scale,
            "module_path": representation.module_path,
            "native_grid_size": list(native_size),
            "display_size": [EXPLANATION_DISPLAY_SIZE, EXPLANATION_DISPLAY_SIZE],
            "display_interpolation": "bilinear",
            "rendering": rendered.metadata,
        }
    unsupported = ADAPTER.attention_tensors(prepared).status
    return None, {
        "method": "attention_rollout",
        "status": unsupported.to_dict(),
        "coordinate_space": None,
        "raw_scale": None,
    }


def predict(
    image: Image.Image,
    threshold: float = 0.5,
    explanation_view: str = "Forensic Grad-CAM",
) -> Tuple[str, float, Image.Image | None, str]:
    if image is None:
        raise ValueError("No image provided")
    if ADAPTER is None:
        raise RuntimeError("canonical detector bundle is unavailable or failed validation")
    if explanation_view not in EXPLANATION_VIEWS:
        raise ValueError("Unknown explanation view")
    prepared = ADAPTER.prepare_source_image(image)
    prediction = ADAPTER.predict(prepared)
    prob = prediction.predicted_probability
    label = "AI-Generated" if prob >= threshold else "Authentic"
    visualization, explanation = _standalone_explanation(image, EXPLANATION_VIEWS[explanation_view])
    raw = {
        "prediction": {
            "model_id": ADAPTER.manifest["model_id"],
            "weights_id": ADAPTER.manifest["weights_id"],
            "logit": prediction.predicted_logit,
            "probability": prob,
            "label": label,
            "applied_threshold": threshold,
        },
        "explanation": explanation,
        "branch_contributions": ADAPTER.branch_subset_logits(prepared).status.to_dict(),
        "preparation": prepared.context,
    }
    return label, prob, visualization, json.dumps(raw, allow_nan=False, indent=2, sort_keys=True)


def build_interface():
    import gradio as gr

    with gr.Blocks(title="RAID") as demo:
        gr.Markdown("# AI Image Detector\nUpload an image to estimate the probability it is AI-generated.")
        gr.Markdown(MODEL_STATUS)
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image", image_mode="RGB", height=400)
                threshold_slider = gr.Slider(
                    minimum=0, maximum=100, value=50, step=1,
                    label="Decision Threshold (% AI-Generated)",
                )
                explanation_choice = gr.Dropdown(
                    choices=list(EXPLANATION_VIEWS), value="Forensic Grad-CAM",
                    label="Explanation view",
                )
                run_button = gr.Button("Run", variant="primary")
                gr.Examples(examples=EXAMPLE_IMAGES, inputs=[image_input], label="Example Images")
            with gr.Column():
                label_output = gr.Label(num_top_classes=1, label="Prediction", elem_id="label_output")
                heatmap_output = gr.Image(label="Standalone explanation (not an image overlay)", height=400)
                json_output = gr.Code(label="Raw explanation JSON", language="json")

        def _run(image: Image.Image, threshold_pct: float, explanation_view: str):
            yield gr.update(), gr.update(interactive=False), gr.update(), gr.update()
            try:
                if image is None:
                    raise gr.Error("Please upload an image before running.")
                threshold = threshold_pct / 100.0
                label, prob, visualization, raw_json = predict(
                    image, threshold=threshold, explanation_view=explanation_view
                )
            except gr.Error:
                yield gr.update(), gr.update(interactive=True), gr.update(), gr.update()
                raise
            except Exception as exc:
                yield gr.update(), gr.update(interactive=True), gr.update(), gr.update()
                raise gr.Error(str(exc)) from exc

            css_class = "ai-label" if label == "AI-Generated" else "real-label"
            # gr.Label always shows the highest-value class, so the displayed
            # confidence must be keyed to the threshold-adjusted `label`
            # itself (not the raw prob split), or text and color can disagree
            # near the threshold.
            decided_confidence = prob if label == "AI-Generated" else 1.0 - prob
            label_value = {label: decided_confidence}
            yield (
                gr.update(value=label_value, elem_classes=[css_class]),
                gr.update(interactive=True), visualization, raw_json,
            )

        run_button.click(
            fn=_run,
            inputs=[image_input, threshold_slider, explanation_choice],
            outputs=[label_output, run_button, heatmap_output, json_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(css=LABEL_CSS)

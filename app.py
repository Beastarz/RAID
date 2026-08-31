"""
Module: app
Project: AI Image Detector

Lightweight Gradio frontend for the AI image detector stub pipeline. Lets a
user upload an image and view the predicted label, AI-probability, and a
placeholder saliency heatmap.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from predict import (
    BayarFusionModel,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
    get_device,
)

DEVICE = get_device()

# Prefer the real, jointly-trained semantic + Bayar+SRM fusion checkpoint (see
# MODEL_README.md) over the stub DetectorPipeline, when all three files have
# been downloaded into checkpoints/. Falls back to stub weights (meaningless
# predictions) if they're missing, so the app still launches offline.
_SEMANTIC_CKPT = "checkpoints/semantic_stream.pt"
_BAYAR_CKPT = "checkpoints/bayar_srm_stream.pt"
_FUSION_CKPT = "checkpoints/detector_fusion.pt"
USE_BAYAR = all(Path(p).exists() for p in (_SEMANTIC_CKPT, _BAYAR_CKPT, _FUSION_CKPT))

if USE_BAYAR:
    MODEL = BayarFusionModel(_SEMANTIC_CKPT, _BAYAR_CKPT, _FUSION_CKPT, DEVICE)
    MODEL_STATUS = "Using pretrained semantic + Bayar+SRM fusion weights."
else:
    MODEL = build_model(checkpoint=None, device=DEVICE)
    MODEL_STATUS = (
        "**No pretrained checkpoint found in `checkpoints/` -- running on randomly "
        "initialized stub weights, predictions are not meaningful.** "
        "See MODEL_README.md to download the real weights."
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
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def _preprocess_raw(image: Image.Image) -> torch.Tensor:
    """Bayar+SRM's input contract: raw [0, 1] pixels at 256x256, unnormalized."""
    resized = image.convert("RGB").resize((256, 256), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def _mock_saliency_overlay(image: Image.Image, prob: float) -> Image.Image:
    """Builds a placeholder saliency heatmap overlay (radial gradient).

    TODO(integration): replace with a real Grad-CAM / ViT attention heatmap
    from `src.explainability.visualizer` once the backbones are in place.
    """
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)

    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    center = IMAGE_SIZE / 2
    radial = 1.0 - np.sqrt((xx - center) ** 2 + (yy - center) ** 2) / (center * np.sqrt(2))
    heat = np.clip(radial, 0.0, 1.0) * prob

    heatmap_rgb = np.zeros_like(array)
    heatmap_rgb[..., 0] = heat * 255.0  # red channel intensity ~ mock saliency

    overlay = (0.6 * array + 0.4 * heatmap_rgb).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def predict(image: Image.Image, threshold: float = 0.5) -> Tuple[str, float, Image.Image]:
    if image is None:
        raise ValueError("No image provided")
    tensor = _preprocess(image).to(DEVICE)
    with torch.no_grad():
        if USE_BAYAR:
            raw = _preprocess_raw(image).to(DEVICE)
            prob = float(MODEL.predict(tensor, raw).item())
        else:
            output = MODEL(tensor)
            prob = float(output["prob"].item())
    label = "AI-Generated" if prob > threshold else "Authentic"
    overlay = _mock_saliency_overlay(image, prob)
    return label, prob, overlay


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
                run_button = gr.Button("Run", variant="primary")
                gr.Examples(examples=EXAMPLE_IMAGES, inputs=[image_input], label="Example Images")
            with gr.Column():
                label_output = gr.Label(num_top_classes=1, label="Prediction", elem_id="label_output")
                heatmap_output = gr.Image(label="Saliency Map (placeholder)", height=400)

        def _run(image: Image.Image, threshold_pct: float):
            yield gr.update(), gr.update(interactive=False), gr.update()
            try:
                if image is None:
                    raise gr.Error("Please upload an image before running.")
                threshold = threshold_pct / 100.0
                label, prob, overlay = predict(image, threshold=threshold)
            except gr.Error:
                yield gr.update(), gr.update(interactive=True), gr.update()
                raise

            css_class = "ai-label" if label == "AI-Generated" else "real-label"
            # gr.Label always shows the highest-value class, so the displayed
            # confidence must be keyed to the threshold-adjusted `label`
            # itself (not the raw prob split), or text and color can disagree
            # near the threshold.
            decided_confidence = prob if label == "AI-Generated" else 1.0 - prob
            label_value = {label: decided_confidence}
            yield gr.update(value=label_value, elem_classes=[css_class]), gr.update(interactive=True), overlay

        run_button.click(
            fn=_run,
            inputs=[image_input, threshold_slider],
            outputs=[label_output, run_button, heatmap_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(css=LABEL_CSS)

"""
Module: app
Project: AI Image Detector

Lightweight Gradio frontend for the AI image detector stub pipeline. Lets a
user upload an image and view the predicted label, AI-probability, and a
placeholder saliency heatmap.
"""

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from predict import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model, get_device

DEVICE = get_device()
MODEL = build_model(checkpoint=None, device=DEVICE)


def _preprocess(image: Image.Image) -> torch.Tensor:
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
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


def predict(image: Image.Image) -> Tuple[str, float, Image.Image]:
    if image is None:
        raise ValueError("No image provided")
    tensor = _preprocess(image).to(DEVICE)
    with torch.no_grad():
        output = MODEL(tensor)
    prob = float(output["prob"].item())
    label = "AI-Generated" if prob > 0.5 else "Authentic"
    overlay = _mock_saliency_overlay(image, prob)
    return label, prob, overlay


def build_interface():
    import gradio as gr

    with gr.Blocks(title="AI Image Detector") as demo:
        gr.Markdown("# AI Image Detector\nUpload an image to estimate the probability it is AI-generated.")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image", image_mode="RGB")
                run_button = gr.Button("Run", variant="primary")
            with gr.Column():
                label_output = gr.Textbox(label="Prediction")
                prob_output = gr.Slider(minimum=0, maximum=100, label="P(AI-Generated) %", interactive=False)
                heatmap_output = gr.Image(label="Saliency Map (placeholder)")

        def _run(image: Image.Image):
            label, prob, overlay = predict(image)
            return label, round(prob * 100, 2), overlay

        run_button.click(fn=_run, inputs=[image_input], outputs=[label_output, prob_output, heatmap_output])

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()

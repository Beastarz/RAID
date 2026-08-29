"""
Module: predict
Project: AI Image Detector

Single-image / batch inference CLI. Loads the DetectorPipeline (optionally
from a checkpoint), standardizes input images to [1, 3, 512, 512], runs
inference, and prints one JSON result per image.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from src.models.detector import DetectorPipeline

IMAGE_SIZE: int = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_tensor(image_path: Path) -> torch.Tensor:
    """Loads an image file and standardizes it into a [1, 3, 512, 512] tensor.

    Resizes to 512x512 and normalizes with ImageNet mean/std, per the
    project's data contract.
    """
    image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def build_model(checkpoint: Optional[str], device: torch.device) -> DetectorPipeline:
    """Instantiates DetectorPipeline, loading weights from a checkpoint if given.

    With no checkpoint, the model runs with its randomly initialized stub
    weights -- expected during early development, before real backbones are
    dropped into the semantic/frequency streams.
    """
    model = DetectorPipeline()
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def predict_single(model: DetectorPipeline, image_path: Path, device: torch.device) -> Dict[str, object]:
    tensor = load_image_tensor(image_path).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model(tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    prob = float(output["prob"].item())
    return {
        "filename": str(image_path),
        "ai_probability": round(prob, 6),
        "label": "AI-Generated" if prob > 0.5 else "Authentic",
        "execution_time_ms": round(elapsed_ms, 3),
    }


def collect_image_paths(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
    return [input_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI image detector inference")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file or a directory of images")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to a trained model checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model = build_model(args.checkpoint, device)

    image_paths = collect_image_paths(Path(args.image))
    if not image_paths:
        raise FileNotFoundError(f"No valid images found at {args.image}")

    for image_path in image_paths:
        result = predict_single(model, image_path, device)
        print(json.dumps(result))


if __name__ == "__main__":
    main()

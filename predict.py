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
from src.models.frontend_bayar import BayarSRMFrontend
from src.models.fusion import FeatureFusion
from src.models.npr_stream import NPRStream
from src.models.semantic_stream import SemanticStream


class BayarFusionModel(torch.nn.Module):
    """Inference wrapper for the semantic + Bayar/NPR fused checkpoint."""

    def __init__(self, semantic_checkpoint: str, bayar_checkpoint: str, fusion_checkpoint: str,
                 device: torch.device) -> None:
        super().__init__()
        self.semantic = SemanticStream(pretrained=False).to(device)
        self.bayar = NPRStream(backbone="resnet_shallow", frontend=BayarSRMFrontend()).to(device)
        self.fusion = FeatureFusion(semantic_dim=1024, forensic_dim=256, fused_dim=512).to(device)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(512, 128), torch.nn.GELU(), torch.nn.Linear(128, 1)).to(device)
        self.semantic.load_state_dict(torch.load(semantic_checkpoint, map_location=device))
        self.bayar.load_state_dict(torch.load(bayar_checkpoint, map_location=device))
        state = torch.load(fusion_checkpoint, map_location=device)
        self.fusion.load_state_dict(state["fusion"])
        self.classifier.load_state_dict(state["classifier"])
        self.eval()

    def predict(self, semantic_input: torch.Tensor, raw_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            semantic_features = self.semantic(semantic_input)
            bayar_features = self.bayar(raw_input)
            return torch.sigmoid(self.classifier(self.fusion(semantic_features, bayar_features)))

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
    weights -- expected before the semantic + forensic streams are jointly
    fine-tuned through `fusion.py` (see TODO.md SS3). Use `--bayar` for the
    real, jointly-trained inference path.
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


def predict_single(model: torch.nn.Module, image_path: Path, device: torch.device, bayar_mode: bool = False) -> Dict[str, object]:
    tensor = load_image_tensor(image_path).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        if bayar_mode:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((256, 256), Image.BILINEAR)
            raw = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
            prob = float(model.predict(tensor, raw).item())
        else:
            output = model(tensor)
            prob = float(output["prob"].item())
    elapsed_ms = (time.perf_counter() - start) * 1000.0
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
    parser.add_argument("--bayar", action="store_true", help="Use the Bayar/NPR fused inference path")
    parser.add_argument("--semantic-checkpoint", default="checkpoints/semantic_stream.pt")
    parser.add_argument("--bayar-checkpoint", default="checkpoints/bayar_srm_stream.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    if args.bayar:
        if not args.checkpoint:
            raise ValueError("--checkpoint must point to detector_fusion.pt when using --bayar")
        model = BayarFusionModel(args.semantic_checkpoint, args.bayar_checkpoint, args.checkpoint, device)
    else:
        model = build_model(args.checkpoint, device)

    image_paths = collect_image_paths(Path(args.image))
    if not image_paths:
        raise FileNotFoundError(f"No valid images found at {args.image}")

    for image_path in image_paths:
        result = predict_single(model, image_path, device, bayar_mode=args.bayar)
        print(json.dumps(result))


if __name__ == "__main__":
    main()

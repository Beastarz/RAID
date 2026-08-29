"""
Module: evaluate
Project: AI Image Detector

Entry point for the robustness evaluation pipeline. Run from the repo root as:
    python -m training.evaluate --checkpoint checkpoints/best_model.ckpt --config configs/augmentations.yaml

MOCK: exercises the real eval-mode RobustnessTransforms (one isolated
degradation at a fixed severity, per configs/augmentations.yaml's `eval`
block) through DetectorPipeline and logs the resulting shapes. Real metric
computation (training/evaluation/metrics.py, robustness_suite.py) is not yet
implemented -- see TODO.md SS4.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml

# Make `src` importable when this script is run directly (python training/evaluate.py)
# rather than as a module (python -m training.evaluate).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.detector import DetectorPipeline  # noqa: E402
from training.data.augmentations import RobustnessTransforms  # noqa: E402
from training.logging_utils import setup_logger  # noqa: E402

_SEVERITY_KEYS = {
    "jpeg": "jpeg_quality_levels",
    "blur": "blur_sigma_levels",
    "downscale": "downscale_levels",
    "noise": "noise_std_levels",
    "crop": "crop_percent_levels",
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI image detector -- robustness evaluation (mock)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to a trained model checkpoint")
    parser.add_argument("--config", type=str, default="configs/augmentations.yaml", help="Path to augmentations.yaml")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def _build_model(checkpoint: Optional[str], device: torch.device) -> DetectorPipeline:
    model = DetectorPipeline()
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logger = setup_logger("training", args.log_level)
    logger.warning(
        "MOCK EVALUATION LOOP -- training/evaluation/metrics.py and robustness_suite.py "
        "are not yet implemented (see TODO.md SS4); this only exercises the eval-mode "
        "augmentation path and logs shapes, it does not compute real degradation-curve metrics."
    )

    with open(args.config) as f:
        augmentations_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device=%s", device)
    model = _build_model(args.checkpoint, device)
    logger.info("Model ready (checkpoint=%s)", args.checkpoint or "none -- random init")

    dummy_image = np.random.default_rng(0).integers(0, 256, size=(args.image_size, args.image_size, 3), dtype=np.uint8)

    eval_cfg = augmentations_config["eval"]
    sweeps = [("clean", [None])] + [(name, eval_cfg[key]) for name, key in _SEVERITY_KEYS.items()]
    for transform_name, severities in sweeps:
        for severity in severities:
            transforms = RobustnessTransforms(
                augmentations_config,
                image_size=args.image_size,
                mode="eval",
                eval_transform=transform_name,
                eval_severity=severity,
            )
            tensor = transforms(dummy_image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
            logger.info(
                "eval transform=%-9s severity=%-6s input=%s -> prob=%.4f",
                transform_name,
                severity,
                tuple(tensor.shape),
                output["prob"].item(),
            )

    logger.info("Mock evaluation sweep complete. No CSV/JSON report written -- see TODO.md SS4.")


if __name__ == "__main__":
    main()

"""
Module: augmentations
Project: AI Image Detector

RobustnessTransforms: the real, fully-implemented data augmentation path for
the training pipeline. Simulates the real-world post-processing the detector
must stay robust to (JPEG compression, Gaussian blur, downscale/upscale
rescaling, Gaussian noise, color jitter, and center-cropping), per the
severity ranges configured in configs/augmentations.yaml.

Two modes:
  - "train": a stochastic stack -- each degradation independently fires with
    probability `p_each` at a random severity within its configured range.
  - "eval": isolates exactly one named degradation at a fixed severity (or
    "clean" for no degradation), for evaluate.py's degradation-curve sweeps.

Both modes always end with Resize -> Normalize -> ToTensorV2, so the output
tensor shape is always [3, image_size, image_size] per the data contract in
.claude/CLAUDE.md, regardless of which degradations fired.
"""

import logging
from typing import Any, Dict, Literal, Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EVAL_TRANSFORM_NAMES = ("jpeg", "blur", "downscale", "noise", "crop", "clean")


def _tail(image_size: int) -> list:
    """Transforms that always run, guaranteeing the output tensor shape."""
    return [
        A.Resize(height=image_size, width=image_size, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, p=1.0),
        ToTensorV2(),
    ]


class RobustnessTransforms:
    """Builds and applies the robustness augmentation pipeline.

    Args:
        config: parsed contents of configs/augmentations.yaml (the whole
            file, i.e. has "train" and "eval" top-level keys).
        image_size: output tensor spatial size (matches configs/base_config.yaml).
        mode: "train" for the stochastic training-time stack, or "eval" for a
            single isolated transform (see `eval_transform`/`eval_severity`).
        eval_transform: required when mode="eval". One of EVAL_TRANSFORM_NAMES.
        eval_severity: required when mode="eval" and eval_transform != "clean".
            Meaning depends on eval_transform: jpeg=quality (int), blur=sigma,
            downscale=scale factor, noise=std, crop=crop percent.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        image_size: int = 512,
        mode: Literal["train", "eval"] = "train",
        eval_transform: Optional[str] = None,
        eval_severity: Optional[float] = None,
    ) -> None:
        self.image_size = image_size
        self.mode = mode

        if mode == "train":
            self.pipeline = self._build_train_pipeline(config["train"], image_size)
            logger.info(
                "RobustnessTransforms[train] built: p_each=%s, image_size=%d",
                config["train"]["p_each"],
                image_size,
            )
        elif mode == "eval":
            if eval_transform not in EVAL_TRANSFORM_NAMES:
                raise ValueError(f"eval_transform must be one of {EVAL_TRANSFORM_NAMES}, got {eval_transform!r}")
            self.pipeline = self._build_eval_pipeline(eval_transform, eval_severity, image_size)
            logger.info(
                "RobustnessTransforms[eval] built: transform=%s, severity=%s, image_size=%d",
                eval_transform,
                eval_severity,
                image_size,
            )
        else:
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")

    @staticmethod
    def _build_train_pipeline(train_cfg: Dict[str, Any], image_size: int) -> A.Compose:
        p = train_cfg["p_each"]
        jitter = train_cfg["color_jitter"]
        crop_size = max(1, int(round(train_cfg["crop_percent"] * image_size)))
        transforms = [
            A.ImageCompression(quality_range=tuple(train_cfg["jpeg_quality_range"]), p=p),
            A.GaussianBlur(sigma_limit=tuple(train_cfg["blur_sigma_range"]), p=p),
            A.Downscale(scale_range=tuple(train_cfg["downscale_range"]), p=p),
            A.GaussNoise(std_range=tuple(train_cfg["noise_std_range"]), p=p),
            A.ColorJitter(
                brightness=tuple(jitter["brightness"]),
                contrast=tuple(jitter["contrast"]),
                saturation=tuple(jitter["saturation"]),
                hue=tuple(jitter["hue"]),
                p=p,
            ),
            # Some source images are narrower than the fixed crop size. Pad
            # first so the robustness transform works for every aspect ratio.
            A.PadIfNeeded(min_height=crop_size, min_width=crop_size, border_mode=0, p=p),
            A.CenterCrop(height=crop_size, width=crop_size, p=p),
            *_tail(image_size),
        ]
        return A.Compose(transforms, save_applied_params=True)

    @staticmethod
    def _build_eval_pipeline(transform: str, severity: Optional[float], image_size: int) -> A.Compose:
        degradation = []
        if transform == "jpeg":
            quality = int(severity)
            degradation = [A.ImageCompression(quality_range=(quality, quality), p=1.0)]
        elif transform == "blur":
            degradation = [A.GaussianBlur(sigma_limit=(severity, severity), p=1.0)]
        elif transform == "downscale":
            degradation = [A.Downscale(scale_range=(severity, severity), p=1.0)]
        elif transform == "noise":
            degradation = [A.GaussNoise(std_range=(severity, severity), p=1.0)]
        elif transform == "crop":
            crop_size = max(1, int(round(severity * image_size)))
            degradation = [
                A.PadIfNeeded(min_height=crop_size, min_width=crop_size, border_mode=0, p=1.0),
                A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
            ]
        elif transform == "clean":
            degradation = []
        return A.Compose([*degradation, *_tail(image_size)], save_applied_params=True)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Applies the pipeline to an HWC uint8 RGB array, returns a [3, H, W] tensor."""
        result = self.pipeline(image=image)
        applied = result.get("applied_transforms", [])
        fired = [name for name, _params in applied if name not in ("Resize", "Normalize", "ToTensorV2")]
        logger.debug("RobustnessTransforms[%s] applied=%s", self.mode, fired)
        return result["image"]

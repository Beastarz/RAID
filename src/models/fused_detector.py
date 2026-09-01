"""Canonical fused detector and its shared source-image preparation path.

The published detector has two model branches, but it has one image
preprocessing contract: decode and resize the source once to 512x512, then
derive the branch-specific views from those same resized pixels.  Keeping
preparation here makes the contract reusable by the future explainability
adapter without coupling the model to a CLI or a data loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.models.frontend_bayar import BayarSRMFrontend
from src.models.fusion import FeatureFusion
from src.models.npr_stream import NPRStream, OUTPUT_DIM_RESNET_SHALLOW
from src.models.semantic_stream import SemanticStream


IMAGE_SIZE: Final[int] = 512
SEMANTIC_DIM: Final[int] = 1024
FORENSIC_DIM: Final[int] = OUTPUT_DIM_RESNET_SHALLOW
FUSED_DIM: Final[int] = 512
CLASSIFIER_HIDDEN_DIM: Final[int] = 128
IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

RawImage = Union[Image.Image, np.ndarray, str, Path]


@dataclass(frozen=True)
class PreparedFusedInputs:
    """The two tensors derived from one deterministic source-image resize.

    ``semantic`` is ImageNet-normalized, while ``forensic`` is the same
    resized RGB image in float32 ``[0, 1]`` pixels.  Dimensions are retained
    for the adapter/reporting layer and are deliberately not hidden in a
    model-specific tensor tuple.
    """

    semantic: torch.Tensor
    forensic: torch.Tensor
    original_size: tuple[int, int]
    resized_size: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)
    interpolation: str = "bilinear"

    def __post_init__(self) -> None:
        if self.semantic.dim() != 4 or tuple(self.semantic.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                "semantic input must have shape [B, 3, 512, 512], "
                f"got {tuple(self.semantic.shape)}"
            )
        if self.forensic.dim() != 4 or tuple(self.forensic.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                "forensic input must have shape [B, 3, 512, 512], "
                f"got {tuple(self.forensic.shape)}"
            )
        if self.semantic.shape[0] != self.forensic.shape[0]:
            raise ValueError("semantic and forensic inputs must have the same batch size")
        if self.semantic.dtype != torch.float32 or self.forensic.dtype != torch.float32:
            raise TypeError("prepared semantic and forensic inputs must be float32")
        if not self.original_size or any(int(value) <= 0 for value in self.original_size):
            raise ValueError("original_size must contain positive width and height")
        if self.resized_size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("resized_size must be the canonical 512x512 shape")
        if self.interpolation != "bilinear":
            raise ValueError("interpolation must be the canonical bilinear policy")


def _as_rgb_image(raw_source_image: RawImage) -> tuple[Image.Image, tuple[int, int]]:
    """Decode a supported source value and return an RGB image plus ``(W,H)``."""

    if isinstance(raw_source_image, (str, Path)):
        with Image.open(raw_source_image) as image:
            decoded = image.convert("RGB")
        return decoded, decoded.size

    if isinstance(raw_source_image, Image.Image):
        decoded = raw_source_image.convert("RGB")
        return decoded, decoded.size

    if isinstance(raw_source_image, np.ndarray):
        array = raw_source_image
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                "numpy source images must have shape [H, W, 3], "
                f"got {tuple(array.shape)}"
            )
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError("numpy source image contains non-finite values")
            minimum = float(array.min())
            maximum = float(array.max())
            if minimum < 0.0 or maximum > 255.0:
                raise ValueError("floating-point source images must be in [0, 255]")
            if maximum <= 1.0:
                array = array * 255.0
            array = np.rint(array).astype(np.uint8)
        elif array.dtype != np.uint8:
            if np.any(array < 0) or np.any(array > 255):
                raise ValueError("integer source images must be in [0, 255]")
            array = array.astype(np.uint8)
        # The HWC uint8 shape tells Pillow this is an RGB image; omitting the
        # deprecated ``mode=`` override keeps this compatible with Pillow 12+.
        decoded = Image.fromarray(np.ascontiguousarray(array))
        return decoded, decoded.size

    raise TypeError(
        "raw source image must be a PIL image, HWC numpy array, or image path; "
        f"got {type(raw_source_image).__name__}"
    )


def prepare_fused_inputs(raw_source_image: RawImage) -> PreparedFusedInputs:
    """Prepare both final detector branches from one 512x512 resize.

    Pillow's bilinear resize is intentionally used here because it is also the
    published three-file scorer's preprocessing.  The returned forensic view
    is the source of truth; semantic normalization is derived from it rather
    than by separately decoding or resizing the source.
    """

    image, original_size = _as_rgb_image(raw_source_image)
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    forensic = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0).contiguous()
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    semantic = (forensic - mean) / std
    return PreparedFusedInputs(semantic=semantic, forensic=forensic, original_size=original_size)


# Names used by callers that describe the operation as shared-input
# preparation rather than fused-model preparation.
prepare_shared_inputs = prepare_fused_inputs
prepare_source_image = prepare_fused_inputs


class CanonicalFusedDetector(nn.Module):
    """The final ViT-B/16 + Bayar/SRM/shallow-ResNet fused detector.

    ``forward`` accepts already prepared branch tensors and returns raw logits.
    It intentionally does not apply sigmoid or ``no_grad`` so attribution
    algorithms can differentiate through the complete detector.
    """

    branch_names: Final[tuple[str, str]] = ("semantic", "forensic")

    def __init__(
        self,
        semantic_stream: Optional[nn.Module] = None,
        forensic_stream: Optional[nn.Module] = None,
        fusion: Optional[nn.Module] = None,
        classifier: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.semantic_dim = SEMANTIC_DIM
        self.forensic_dim = FORENSIC_DIM
        self.fused_dim = FUSED_DIM
        self.semantic_stream = (
            semantic_stream
            if semantic_stream is not None
            else SemanticStream(output_dim=SEMANTIC_DIM, pretrained=False)
        )
        self.forensic_stream = (
            forensic_stream
            if forensic_stream is not None
            else NPRStream(backbone="resnet_shallow", frontend=BayarSRMFrontend())
        )
        self.fusion = (
            fusion
            if fusion is not None
            else FeatureFusion(semantic_dim=SEMANTIC_DIM, freq_dim=FORENSIC_DIM, fused_dim=FUSED_DIM)
        )
        self.classifier = (
            classifier
            if classifier is not None
            else nn.Sequential(
                nn.Linear(FUSED_DIM, CLASSIFIER_HIDDEN_DIM),
                nn.GELU(),
                nn.Linear(CLASSIFIER_HIDDEN_DIM, 1),
            )
        )

    @staticmethod
    def _validate_input(name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} input must be a torch.Tensor")
        if value.dim() != 4 or value.shape[1] != 3:
            raise ValueError(f"{name} input must have shape [B, 3, 512, 512], got {tuple(value.shape)}")
        if tuple(value.shape[-2:]) != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"{name} input must be exactly 512x512, got {tuple(value.shape[-2:])}")
        if value.dtype != torch.float32:
            raise TypeError(f"{name} input must be float32, got {value.dtype}")

    def forward(
        self,
        semantic_input: torch.Tensor | PreparedFusedInputs,
        forensic_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(semantic_input, PreparedFusedInputs):
            if forensic_input is not None:
                raise ValueError("forensic_input must be omitted when passing PreparedFusedInputs")
            prepared = semantic_input
            semantic_input, forensic_input = prepared.semantic, prepared.forensic
        elif forensic_input is None:
            raise TypeError("forward requires semantic and forensic tensors or PreparedFusedInputs")

        assert forensic_input is not None  # narrowed for type checkers
        self._validate_input("semantic", semantic_input)
        self._validate_input("forensic", forensic_input)
        if semantic_input.shape[0] != forensic_input.shape[0]:
            raise ValueError("semantic and forensic inputs must have the same batch size")
        if not torch.isfinite(semantic_input).all():
            raise ValueError("semantic input contains non-finite values")
        if not torch.isfinite(forensic_input).all():
            raise ValueError("forensic input contains non-finite values")
        if torch.any(forensic_input < 0.0) or torch.any(forensic_input > 1.0):
            raise ValueError("forensic input must contain raw pixels in [0, 1]")

        semantic_features = self.semantic_stream(semantic_input)
        forensic_features = self.forensic_stream(forensic_input)
        if semantic_features.dim() != 2 or semantic_features.shape[1] != self.semantic_dim:
            raise ValueError(
                f"semantic stream must return [B, {self.semantic_dim}], got {tuple(semantic_features.shape)}"
            )
        if forensic_features.dim() != 2 or forensic_features.shape[1] != self.forensic_dim:
            raise ValueError(
                f"forensic stream must return [B, {self.forensic_dim}], got {tuple(forensic_features.shape)}"
            )
        fused_features = self.fusion(semantic_features, forensic_features)
        logits = self.classifier(fused_features)
        if logits.dim() != 2 or logits.shape[1] != 1:
            raise ValueError(f"classifier must return raw logits with shape [B, 1], got {tuple(logits.shape)}")
        return logits


# Short alias for callers that do not need to spell out the topology.
FusedDetector = CanonicalFusedDetector


__all__ = [
    "CanonicalFusedDetector",
    "FusedDetector",
    "PreparedFusedInputs",
    "prepare_fused_inputs",
    "prepare_shared_inputs",
    "prepare_source_image",
    "IMAGE_SIZE",
    "SEMANTIC_DIM",
    "FORENSIC_DIM",
    "FUSED_DIM",
]

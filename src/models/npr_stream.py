"""
Module: npr_stream
Project: AI Image Detector

Low-level forensic feature stream: Neighboring Pixel Relationships (NPR).

Reference: Tan et al., "Rethinking the Up-Sampling Operations in CNN-based
Generative Network for Generalizable Deepfake Detection", CVPR 2024.

Every GAN/diffusion decoder ends in a stack of upsampling layers, which
manufacture new pixels from neighbours by a fixed rule. That leaves a
periodic local-correlation pattern that camera-captured images do not have.
NPR isolates it with a single fixed, parameter-free residual operator (a
nearest-neighbour downsample-upsample round trip subtracted from the input)
and hands the result to a shallow CNN -- deep backbones progressively build
semantic abstraction and discard exactly the fine-grained statistics this
stream exists to read, so the paper found a truncated ResNet-50 (stem +
layer1) outperforms full-depth backbones here.

Conforms to the shared `BaseFeatureStream` interface ([B, 3, H, W] -> [B, D])
so it can be dropped into `DetectorPipeline` as an extra or replacement
stream via its existing constructor injection, e.g.:

    from src.models.npr_stream import NPRStream, OUTPUT_DIM_RESNET_SHALLOW
    from src.models.fusion import FeatureFusion
    from src.models.detector import DetectorPipeline

    detector = DetectorPipeline(
        forensic_stream=NPRStream(),
        fusion=FeatureFusion(forensic_dim=OUTPUT_DIM_RESNET_SHALLOW),
    )

The backbone is swappable via the `backbone` flag, and the forensic frontend
(the module that turns raw pixels into a residual) is swappable via the
`frontend` constructor arg -- see NPRStream's docstring. That is the M3
go/no-go swap point in npr_stream_guide.md SS7/SS10: if NPR's signal
collapses under a resize/downscale stress test, a Bayar+SRM frontend behind
the same interface plugs in with no other code changes.

Important divergence from the other two streams: NPR's signal is destroyed
by resizing (resizing is itself an interpolation and overwrites the
generator's upsampling signature with the resampler's own), so this stream
expects RAW pixel values in [0, 1] at *native-resolution crops* (e.g. a
fixed 256x256 crop, never a resize), NOT the ImageNet-normalized 512x512
tensor the semantic/frequency streams consume. Wiring a dataloader that
produces that crop is a separate, training-pipeline-owned concern (see
npr_stream_guide.md SS2.1/SS3) and intentionally isn't touched here.
"""

from typing import Final, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from src.models.base_stream import BaseFeatureStream

OUTPUT_DIM_RESNET_SHALLOW: Final[int] = 256
OUTPUT_DIM_CONVNEXT_TINY: Final[int] = 768

BackboneName = Literal["resnet_shallow", "convnext_tiny"]


class NPR(nn.Module):
    """Fixed, parameter-free residual operator: NPR(x) = x - upsample(downsample(x)).

    Both the downsample and upsample are nearest-neighbour at `stride`, so
    the residual reduces to "each pixel minus the top-left pixel of its own
    stride x stride block" -- no learned parameters. Forced to run in fp32
    regardless of the surrounding autocast context: the residual's magnitude
    is small, and fp16 can underflow it into noise.
    """

    def __init__(self, stride: int = 2) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % self.stride != 0 or x.shape[-1] % self.stride != 0:
            raise ValueError(
                f"NPR requires H and W divisible by stride={self.stride}, got {tuple(x.shape[-2:])}"
            )
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            down = F.interpolate(x, scale_factor=1 / self.stride, mode="nearest")
            up = F.interpolate(down, scale_factor=float(self.stride), mode="nearest")
            return x - up


def _truncated_resnet50_stem_layer1() -> nn.Module:
    """ResNet-50 stem + layer1 only (~0.2M params).

    Deliberately shallow: the paper found this beats a full-depth backbone
    for reading this residual, since deeper layers discard the low-level
    statistics NPR is built to expose.
    """
    resnet = tv_models.resnet50(weights=None)
    return nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)


def _convnext_tiny_backbone() -> nn.Module:
    """Full ConvNeXt-Tiny feature extractor (~28M params), ablation only."""
    return tv_models.convnext_tiny(weights=None).features


class NPRStream(BaseFeatureStream):
    """Forensic frontend residual -> BatchNorm(3) rescale -> backbone -> global average pool.

    Input contract: [B, 3, H, W], pixel values in [0, 1] (NOT
    ImageNet-normalized -- the residual is computed on raw pixels and
    rescaled into a healthy range for the backbone by BatchNorm2d, not by
    channel-wise standardization beforehand, which would only rescale the
    residual without adding information). H and W must be divisible by
    `stride` and should be a native-resolution crop, never a resized image.

    Output: [B, D] pooled feature vector. D=256 for the default
    "resnet_shallow" backbone, D=768 for the "convnext_tiny" ablation.

    The frontend (the module that turns raw pixels into a forensic residual)
    is injectable via `frontend`, defaulting to the fixed, parameter-free
    `NPR` operator above. This is the swap point npr_stream_guide.md SS7/SS10
    calls out for the M3 go/no-go: if NPR's signal collapses under the
    resize/downscale robustness stress test, the prescribed fix is a Bayar
    constrained-conv + SRM-filter frontend behind the *same* interface
    ([B, 3, H, W] in [0, 1] -> [B, 3, H, W]), swapped in with e.g.
    `NPRStream(frontend=BayarSRMFrontend())` and no other code changes --
    the guide is explicit that this should be a config change, not a
    rewrite. `stride` is only used to build the default frontend; a custom
    frontend is responsible for its own input-shape validation.
    """

    def __init__(
        self,
        backbone: BackboneName = "resnet_shallow",
        stride: int = 2,
        frontend: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.frontend = frontend if frontend is not None else NPR(stride=stride)
        self.rescale = nn.BatchNorm2d(3)

        if backbone == "resnet_shallow":
            self.backbone = _truncated_resnet50_stem_layer1()
            self.output_dim = OUTPUT_DIM_RESNET_SHALLOW
        elif backbone == "convnext_tiny":
            self.backbone = _convnext_tiny_backbone()
            self.output_dim = OUTPUT_DIM_CONVNEXT_TINY
        else:
            raise ValueError(f"backbone must be 'resnet_shallow' or 'convnext_tiny', got {backbone!r}")
        self.backbone_name = backbone

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {tuple(x.shape)}")
        residual = self.frontend(x)  # [B, 3, H, W]
        residual = self.rescale(residual)
        feature_map = self.backbone(residual)  # [B, C, H', W']
        features = self.flatten(self.pool(feature_map))  # [B, output_dim]
        return features

    @staticmethod
    def aggregate_crop_logits(logits: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Top-k mean aggregation across N crops of one image.

        A plain mean dilutes localized forensic evidence and a plain max is
        too noisy, so inference-time scoring across an image's N crops
        should use the mean of its top-k highest logits instead. `logits` is
        a 1D (or flattenable) tensor of N per-crop logits for a single
        image; returns a scalar aggregated logit.
        """
        flat = logits.reshape(-1)
        k = min(k, flat.numel())
        top_values, _ = torch.topk(flat, k)
        return top_values.mean()

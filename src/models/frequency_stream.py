"""
Module: frequency_stream
Project: AI Image Detector

Stream 2: mid-level frequency-domain feature extractor.
"""

from typing import Final

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from src.models.base_stream import BaseFeatureStream

OUTPUT_DIM: Final[int] = 768


class FrequencyStream(BaseFeatureStream):
    """FFT high-pass representation followed by ConvNeXt-Tiny."""

    def __init__(self, output_dim: int = OUTPUT_DIM, pretrained: bool = False,
                 freeze_backbone: bool = True, highpass_ratio: float = 0.08) -> None:
        super().__init__()
        if not 0.0 <= highpass_ratio < 1.0:
            raise ValueError("highpass_ratio must be in [0, 1)")
        self.output_dim = output_dim
        self.highpass_ratio = highpass_ratio
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        backbone_dim = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()
        self.proj = nn.Identity() if backbone_dim == output_dim else nn.Linear(backbone_dim, output_dim)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _highpass(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        spectrum = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
        magnitude = torch.log1p(torch.abs(spectrum))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=x.device),
                                torch.linspace(-1, 1, width, device=x.device), indexing="ij")
        mask = (torch.sqrt(xx.square() + yy.square()) >= self.highpass_ratio).to(magnitude.dtype)
        filtered = magnitude * mask
        mean = filtered.mean(dim=(-2, -1), keepdim=True)
        std = filtered.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (filtered - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {tuple(x.shape)}")
        return self.proj(self.backbone(self._highpass(x)))

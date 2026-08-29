"""
Module: frequency_stream
Project: AI Image Detector

Stream 2: mid-level frequency-domain feature extractor.
"""

from typing import Final

import torch
import torch.nn as nn

from src.models.base_stream import BaseFeatureStream

OUTPUT_DIM: Final[int] = 768


class FrequencyStream(BaseFeatureStream):
    """Stub mid-level frequency feature stream.

    TODO(integration): drop in a 2D FFT high-pass masking step feeding a
    lightweight ConvNeXt-Tiny backbone here. The stub below takes a real 2D
    FFT magnitude spectrum and pools it, only to satisfy the
    [B, 3, H, W] -> [B, 768] tensor contract so that `FeatureFusion` and
    `DetectorPipeline` can be developed and tested before the real backbone
    lands.
    """

    def __init__(self, output_dim: int = OUTPUT_DIM) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(3, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {tuple(x.shape)}")
        spectrum = torch.fft.fft2(x, norm="ortho")
        magnitude = torch.abs(spectrum)  # [B, 3, H, W] dummy frequency representation
        pooled = self.flatten(self.pool(magnitude))  # [B, 3]
        features = self.proj(pooled)  # [B, output_dim]
        return features

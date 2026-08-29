"""
Module: semantic_stream
Project: AI Image Detector

Stream 1: high-level semantic feature extractor.
"""

from typing import Final

import torch
import torch.nn as nn

from src.models.base_stream import BaseFeatureStream

OUTPUT_DIM: Final[int] = 1024


class SemanticStream(BaseFeatureStream):
    """Stub high-level semantic feature stream.

    TODO(integration): drop in a frozen or fine-tuned DINOv2 / ViT backbone
    here. The stub below (global average pool + linear projection) exists
    only to satisfy the [B, 3, H, W] -> [B, 1024] tensor contract so that
    `FeatureFusion` and `DetectorPipeline` can be developed and tested before
    the real backbone lands.
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
        pooled = self.flatten(self.pool(x))  # [B, 3]
        features = self.proj(pooled)  # [B, output_dim]
        return features

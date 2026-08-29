"""
Module: fusion
Project: AI Image Detector

Cross-stream feature fusion layer.
"""

from typing import Final

import torch
import torch.nn as nn

FUSED_DIM: Final[int] = 512


class FeatureFusion(nn.Module):
    """Fuses semantic and frequency feature vectors into one representation.

    Concatenates the two input feature vectors along the feature dimension
    and projects the result down to a fixed fused dimensionality.

    TODO(integration): swap the concat + linear projection below for
    cross-attention between the two streams once both backbones are final.
    """

    def __init__(self, semantic_dim: int = 1024, freq_dim: int = 768, fused_dim: int = FUSED_DIM) -> None:
        super().__init__()
        self.fused_dim = fused_dim
        self.proj = nn.Linear(semantic_dim + freq_dim, fused_dim)
        self.norm = nn.LayerNorm(fused_dim)
        self.act = nn.GELU()

    def forward(self, feat_semantic: torch.Tensor, feat_freq: torch.Tensor) -> torch.Tensor:
        if feat_semantic.shape[0] != feat_freq.shape[0]:
            raise ValueError("Batch size mismatch between semantic and frequency features")
        combined = torch.cat([feat_semantic, feat_freq], dim=1)  # [B, D1 + D2]
        fused = self.act(self.norm(self.proj(combined)))  # [B, fused_dim]
        return fused

"""
Module: semantic_stream
Project: AI Image Detector

Stream 1: high-level semantic feature extractor.
"""

from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16

from src.models.base_stream import BaseFeatureStream

OUTPUT_DIM: Final[int] = 1024


class SemanticStream(BaseFeatureStream):
    """ViT-based semantic stream preserving the ``[B, 3, H, W] -> [B, 1024]`` contract."""

    def __init__(self, output_dim: int = OUTPUT_DIM, pretrained: bool = False,
                 freeze_backbone: bool = True, unfreeze_last_n_blocks: int = 0) -> None:
        super().__init__()
        self.output_dim = output_dim
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights)
        backbone_dim = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()
        self.proj = nn.Linear(backbone_dim, output_dim)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            if unfreeze_last_n_blocks > 0:
                blocks = self.backbone.encoder.layers
                for block in blocks[-unfreeze_last_n_blocks:]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True
                for parameter in self.backbone.encoder.ln.parameters():
                    parameter.requires_grad = True
        elif unfreeze_last_n_blocks != 0:
            raise ValueError("unfreeze_last_n_blocks requires freeze_backbone=True")

    def parameter_counts(self) -> dict[str, int]:
        """Return total and trainable parameter counts for experiment logging."""
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def train(self, mode: bool = True) -> "SemanticStream":
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {tuple(x.shape)}")
        resized = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        backbone_features = self.backbone(resized)  # [B, 768]
        features = self.proj(backbone_features)  # [B, output_dim]
        return features

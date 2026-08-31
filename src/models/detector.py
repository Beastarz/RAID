"""
Module: detector
Project: AI Image Detector

End-to-end detector pipeline: semantic stream + forensic stream + fusion +
classification head.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.npr_stream import NPRStream
from src.models.fusion import FeatureFusion
from src.models.semantic_stream import SemanticStream


class DetectorPipeline(nn.Module):
    """Wraps the semantic stream, forensic stream, fusion layer, and MLP head.

    `forward` always returns a dict with keys "logit", "prob", and "features",
    matching the data contract fixed in CLAUDE.md:
        {
            "logit": torch.Tensor,     # [B, 1]
            "prob": torch.Tensor,      # [B, 1]
            "features": torch.Tensor,  # [B, D_fused]
        }

    The default forensic stream is `NPRStream`, which expects raw [0, 1]
    native-resolution crops rather than the ImageNet-normalized tensor the
    semantic stream consumes -- `forward` still feeds both streams the same
    `x` today, so this default path is a shape-correct stub, not a jointly
    trained model (see TODO.md SS3). `predict.py --bayar` is the real,
    jointly-trained inference path.
    """

    def __init__(
        self,
        semantic_stream: Optional[nn.Module] = None,
        forensic_stream: Optional[nn.Module] = None,
        fusion: Optional[nn.Module] = None,
        fused_dim: int = 512,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.semantic_stream = semantic_stream if semantic_stream is not None else SemanticStream()
        self.forensic_stream = forensic_stream if forensic_stream is not None else NPRStream()
        self.fusion = fusion if fusion is not None else FeatureFusion(fused_dim=fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_semantic = self.semantic_stream(x)  # [B, D1]
        feat_forensic = self.forensic_stream(x)  # [B, D2]
        features = self.fusion(feat_semantic, feat_forensic)  # [B, D_fused]
        logit = self.classifier(features)  # [B, 1]
        prob = torch.sigmoid(logit)  # [B, 1]
        return {"logit": logit, "prob": prob, "features": features}

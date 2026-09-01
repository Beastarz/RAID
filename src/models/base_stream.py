"""
Module: base_stream
Project: AI Image Detector

Abstract interface shared by every feature-extraction stream (Stream 1:
semantic, Stream 2: forensic). DO NOT modify without team consensus --
`FeatureFusion` and `DetectorPipeline` depend on this exact contract.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseFeatureStream(nn.Module, ABC):
    """Abstract base class for a feature-extraction stream.

    Concrete subclasses accept an image tensor ``[B, 3, H, W]`` and return a
    fixed-size feature vector ``[B, D]``, where ``D`` is defined by the
    subclass (e.g. D=1024 for the semantic stream, D=256 for the default
    NPR forensic stream).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts a feature vector from a batch of images.

        Args:
            x: Image tensor of shape [B, 3, H, W].

        Returns:
            Feature tensor of shape [B, D].
        """
        raise NotImplementedError

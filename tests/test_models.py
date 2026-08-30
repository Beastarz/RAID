"""
Module: test_models
Project: AI Image Detector
"""

import pytest
import torch

from src.models.semantic_stream import SemanticStream


def test_semantic_stream_output_contract():
    model = SemanticStream(pretrained=False)
    output = model(torch.randn(2, 3, 64, 64))
    assert tuple(output.shape) == (2, 1024)


def test_semantic_stream_rejects_non_rgb_input():
    model = SemanticStream(pretrained=False)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 64, 64))


def test_semantic_stream_freezes_backbone_but_trains_projection():
    model = SemanticStream(pretrained=False, freeze_backbone=True)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.proj.parameters())

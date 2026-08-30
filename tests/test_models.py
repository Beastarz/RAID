"""
Module: test_models
Project: AI Image Detector
"""

import pytest
import torch

from src.models.semantic_stream import SemanticStream
from src.models.detector import DetectorPipeline


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


def test_semantic_stream_can_unfreeze_last_blocks():
    model = SemanticStream(pretrained=False, freeze_backbone=True, unfreeze_last_n_blocks=2)
    assert any(parameter.requires_grad for parameter in model.backbone.encoder.layers[-1].parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.encoder.layers[0].parameters())


def test_semantic_stream_parameter_counts_are_consistent():
    model = SemanticStream(pretrained=False)
    counts = model.parameter_counts()
    assert counts["total"] == counts["trainable"] + counts["frozen"]
    assert counts["trainable"] > 0


def test_detector_output_contract():
    model = DetectorPipeline()
    output = model(torch.randn(1, 3, 64, 64))
    assert set(output) == {"logit", "prob", "features"}
    assert tuple(output["logit"].shape) == (1, 1)
    assert tuple(output["prob"].shape) == (1, 1)
    assert tuple(output["features"].shape) == (1, 512)
    assert torch.all((output["prob"] >= 0) & (output["prob"] <= 1))

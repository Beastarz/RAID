"""Tests for the canonical fused detector and shared source preparation."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.models.frontend_bayar import BayarSRMFrontend
from src.models.fused_detector import (
    CanonicalFusedDetector,
    FORENSIC_DIM,
    IMAGE_SIZE,
    SEMANTIC_DIM,
    prepare_fused_inputs,
)
from src.models.npr_stream import NPRStream


class _MeanFeatures(torch.nn.Module):
    """Small deterministic stream used to test the fused model contract."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summary = x.mean(dim=(2, 3))
        repeats = (self.output_dim + summary.shape[1] - 1) // summary.shape[1]
        return summary.repeat(1, repeats)[:, : self.output_dim]


def test_shared_preparation_derives_both_views_from_one_resize():
    source = Image.fromarray(
        np.array(
            [
                [[0, 32, 255], [64, 128, 192], [255, 224, 16]],
                [[16, 80, 144], [96, 160, 224], [240, 48, 112]],
            ],
            dtype=np.uint8,
        ),
        mode="RGB",
    )
    prepared = prepare_fused_inputs(source)
    repeated = prepare_fused_inputs(source)

    assert prepared.original_size == (3, 2)
    assert prepared.resized_size == (IMAGE_SIZE, IMAGE_SIZE)
    assert prepared.interpolation == "bilinear"
    assert tuple(prepared.semantic.shape) == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert tuple(prepared.forensic.shape) == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert prepared.semantic.dtype == torch.float32
    assert prepared.forensic.dtype == torch.float32
    assert torch.equal(prepared.semantic, repeated.semantic)
    assert torch.equal(prepared.forensic, repeated.forensic)
    assert float(prepared.forensic.min()) >= 0.0
    assert float(prepared.forensic.max()) <= 1.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(prepared.semantic * std + mean, prepared.forensic, atol=1e-7, rtol=0.0)


def test_preparation_accepts_hwc_array_and_rejects_invalid_source():
    source = np.zeros((4, 5, 3), dtype=np.uint8)
    prepared = prepare_fused_inputs(source)
    assert prepared.original_size == (5, 4)

    with pytest.raises(ValueError, match=r"shape \[H, W, 3\]"):
        prepare_fused_inputs(np.zeros((4, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"in \[0, 255\]"):
        prepare_fused_inputs(np.full((4, 5, 3), 300.0, dtype=np.float32))


def test_bayar_srm_frontend_preserves_shape_and_bayar_constraint():
    frontend = BayarSRMFrontend()
    source = torch.rand(2, 3, 32, 32, requires_grad=True)
    output = frontend(source)
    assert output.shape == source.shape
    kernel = frontend.bayar._constrained_kernel().detach()
    center = kernel[:, 0, 2, 2]
    off_center = kernel.flatten(1)
    off_center = torch.cat((off_center[:, :12], off_center[:, 13:]), dim=1)
    assert torch.allclose(center, torch.full_like(center, -1.0))
    assert torch.allclose(off_center.sum(dim=1), torch.ones(3), atol=1e-6)
    output.square().mean().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()


def test_final_forensic_stream_uses_bayar_srm_and_returns_features():
    stream = NPRStream(backbone="resnet_shallow", frontend=BayarSRMFrontend()).eval()
    source = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        features = stream(source)
    assert features.shape == (1, FORENSIC_DIM)
    assert torch.isfinite(features).all()


def test_canonical_detector_returns_raw_logit_and_preserves_gradients():
    model = CanonicalFusedDetector(
        semantic_stream=_MeanFeatures(SEMANTIC_DIM),
        forensic_stream=_MeanFeatures(FORENSIC_DIM),
    )
    prepared = prepare_fused_inputs(Image.new("RGB", (8, 6), color=(32, 96, 192)))
    semantic = prepared.semantic.requires_grad_()
    forensic = prepared.forensic.requires_grad_()

    logit = model(semantic, forensic)
    assert tuple(logit.shape) == (1, 1)
    assert not torch.allclose(logit, torch.sigmoid(logit))
    logit.sum().backward()
    assert semantic.grad is not None and torch.any(semantic.grad != 0)
    assert forensic.grad is not None and torch.any(forensic.grad != 0)


def test_canonical_detector_accepts_prepared_inputs_and_rejects_wrong_contract():
    model = CanonicalFusedDetector(
        semantic_stream=_MeanFeatures(SEMANTIC_DIM),
        forensic_stream=_MeanFeatures(FORENSIC_DIM),
    )
    prepared = prepare_fused_inputs(Image.new("RGB", (2, 2), color="white"))
    assert torch.equal(model(prepared), model(prepared.semantic, prepared.forensic))

    with pytest.raises(ValueError, match="exactly 512x512"):
        model(torch.zeros(1, 3, 256, 256), torch.zeros(1, 3, 256, 256))
    with pytest.raises(ValueError, match=r"raw pixels in \[0, 1\]"):
        model(prepared.semantic, torch.full_like(prepared.forensic, 2.0))
    with pytest.raises(TypeError, match="float32"):
        model(prepared.semantic.double(), prepared.forensic)


def test_default_topology_parameter_budget_and_branch_dimensions():
    model = CanonicalFusedDetector()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count == 87_534_262
    assert parameter_count < 2_000_000_000
    assert model.semantic_dim == SEMANTIC_DIM
    assert model.forensic_dim == FORENSIC_DIM
    assert model.fused_dim == 512
    assert model.forensic_stream.backbone_name == "resnet_shallow"
    assert model.forensic_stream.frontend.__class__.__name__ == "BayarSRMFrontend"


def test_model_config_declares_final_shared_preprocessing_contract():
    config_path = Path(__file__).parents[1] / "configs" / "model_config.yaml"
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    assert "frequency" not in config
    assert config["forensic"] == {
        "frontend": "bayar_srm",
        "backbone": "resnet_shallow",
        "output_dim": 256,
        "pixel_range": [0.0, 1.0],
    }
    assert config["preprocessing"]["shared_resize"] is True
    assert config["preprocessing"]["image_size"] == 512
    assert config["preprocessing"]["resize_interpolation"] == "bilinear"
    assert config["decision"] == {"threshold": 0.5, "equality": "greater_or_equal"}

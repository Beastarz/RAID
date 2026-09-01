"""Acceptance tests for the canonical final-model explainability adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.explainability.adapters.detector_adapter import (
    DetectorAttributionTarget,
    DetectorExplainabilityAdapter,
    IntermediateRepresentation,
    vit_token_grid,
)
from src.explainability.contracts import Capability, ExplainabilityAdapter, PreparedModelInputs
from src.explainability.gradcam import grad_cam


ROOT = Path(__file__).parents[1]
BUNDLE = ROOT / "checkpoints" / "detector_bundle.pt"
CHECKPOINTS = ROOT / "checkpoints"


def test_vit_token_grid_removes_cls_and_preserves_patch_order():
    tokens = torch.arange(2 * 197 * 3, dtype=torch.float32).reshape(2, 197, 3)
    grid = vit_token_grid(tokens)
    assert grid.shape == (2, 3, 14, 14)
    assert torch.equal(grid[:, :, 0, 0], tokens[:, 1, :])
    assert torch.equal(grid[:, :, -1, -1], tokens[:, -1, :])
    with pytest.raises(ValueError, match="197"):
        vit_token_grid(torch.zeros(1, 196, 3))


@pytest.mark.skipif(not BUNDLE.is_file(), reason="canonical detector bundle is not installed")
def test_real_adapter_prediction_targets_intermediates_and_unsupported_capabilities():
    adapter = DetectorExplainabilityAdapter(BUNDLE, source_dir=CHECKPOINTS)
    assert isinstance(adapter, ExplainabilityAdapter)
    prepared = adapter.prepare_source_image(ROOT / "test_sample.jpg")
    repeated = adapter.prepare_source_image(ROOT / "test_sample.jpg")
    assert torch.equal(prepared.values.semantic, repeated.values.semantic)
    assert torch.equal(prepared.values.forensic, repeated.values.forensic)
    assert prepared.context["original_size"] == list(prepared.values.original_size)
    assert prepared.context["resized_size"] == [512, 512]
    assert prepared.context["forensic_frontend"] == "bayar_srm"

    prediction = adapter.predict(prepared)
    assert prediction.predicted_logit == pytest.approx(-0.3831035197, abs=1e-7)
    assert prediction.predicted_probability == pytest.approx(0.4053786099, abs=1e-7)
    assert prediction.predicted_label == 0
    assert prediction.decision_threshold == 0.5
    assert adapter.report_metadata["weights_id"].startswith("sha256:")

    targets = adapter.attribution_targets(prepared)
    assert targets.status.available
    assert set(targets.value) == {"semantic", "forensic"}
    semantic = targets.value["semantic"].value
    forensic = targets.value["forensic"].value
    assert isinstance(semantic, DetectorAttributionTarget)
    assert isinstance(forensic, DetectorAttributionTarget)
    assert semantic.activation_transform is vit_token_grid
    assert forensic.activation_transform is None

    hooks_before = sum(len(module._forward_hooks) for module in adapter.model.modules())
    result = grad_cam(
        adapter.model,
        forensic.module,
        forensic.scoring_callable,
        lambda value: value,
        activation_transform=forensic.activation_transform,
    )
    assert result.heatmap.shape == (1, 128, 128)
    assert torch.isfinite(result.heatmap).all() and torch.any(result.heatmap > 0)
    intermediate = adapter.intermediate_representations(prepared)
    hooks_after = sum(len(module._forward_hooks) for module in adapter.model.modules())
    assert hooks_after == hooks_before
    assert set(intermediate.value) == {"semantic", "forensic"}
    assert set(intermediate.value["forensic"]) == {
        "frontend.bayar", "frontend.srm", "frontend.fuse", "backbone.4", "pool"
    }
    assert all(
        isinstance(item, IntermediateRepresentation)
        and item.coordinate_space == "forensic_internal"
        and torch.isfinite(item.value).all()
        for item in intermediate.value["forensic"].values()
    )

    attention = adapter.attention_tensors(prepared)
    coalitions = adapter.branch_subset_logits(prepared)
    assert not attention.status.available and "does not expose" in attention.status.reason
    assert not coalitions.status.available and "baseline" in coalitions.status.reason
    assert not adapter.capabilities[Capability.ATTENTION_TENSORS].available
    assert not adapter.capabilities[Capability.BRANCH_SUBSET_LOGITS].available


@pytest.mark.skipif(not BUNDLE.is_file(), reason="canonical detector bundle is not installed")
def test_adapter_rejects_foreign_prepared_values():
    adapter = DetectorExplainabilityAdapter(BUNDLE)
    with pytest.raises(TypeError, match="not created"):
        adapter.predict(PreparedModelInputs(values=torch.zeros(1), context={}))

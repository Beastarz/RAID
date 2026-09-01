"""Tests for strict final-detector bundles and provenance metadata."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import torch

from src.models.checkpoint_bundle import (
    BUNDLE_SCHEMA_VERSION,
    BUNDLE_TYPE,
    MODEL_ID,
    BundleValidationError,
    build_checkpoint_bundle,
    build_manifest,
    load_checkpoint_bundle,
    load_checkpoint_bundle_payload,
    validate_explainability_contract,
    sha256_file,
    state_dict_digest,
    validate_manifest,
)
from src.models.fused_detector import CanonicalFusedDetector, prepare_fused_inputs


ROOT = Path(__file__).parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
SOURCES = {
    "semantic_stream": CHECKPOINT_DIR / "semantic_stream.pt",
    "forensic_stream": CHECKPOINT_DIR / "bayar_srm_stream.pt",
    "detector_fusion": CHECKPOINT_DIR / "detector_fusion.pt",
}
HAS_PUBLISHED_CHECKPOINTS = all(path.is_file() for path in SOURCES.values())


@pytest.mark.skipif(not HAS_PUBLISHED_CHECKPOINTS, reason="published checkpoints are not installed")
def test_source_hash_and_manifest_record_the_published_provenance():
    assert sha256_file(ROOT / "test_sample.jpg") == "b22a7d4ab1587893a5aad31d66e88c372d6892434ea6f7b75f42cffca706a5c2"
    manifest = build_manifest(SOURCES, ["classifier.0.bias", "classifier.2.bias"])
    validate_manifest(manifest)
    assert manifest["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert manifest["bundle_type"] == BUNDLE_TYPE
    assert manifest["model_id"] == MODEL_ID
    assert manifest["topology"]["branches"] == ["semantic", "forensic"]
    assert manifest["topology"]["forensic"]["frontend"] == "bayar_srm"
    assert manifest["topology"]["forensic"]["backbone"] == "resnet_shallow"
    assert manifest["preprocessing"]["shared_resize"] is True
    assert manifest["preprocessing"]["resize"]["size"] == [512, 512]
    assert manifest["decision"]["threshold"] == 0.5
    assert manifest["source_artifacts"]["semantic_stream"]["sha256"] == (
        "2a8e2710e6e4e69ca04d69dd1bebc2522416e49abcaee5950122d1a56974a9b7"
    )
    assert manifest["source_artifacts"]["forensic_stream"]["sha256"] == (
        "77d3dcf429f798660201dbcd7a56ae9212084b2db28f20f083bbbea61e1721f3"
    )
    assert manifest["source_artifacts"]["detector_fusion"]["sha256"] == (
        "c00bf5546291401d6d5ff439d1857d62e8da1391d06a576281a2263a1242516f"
    )


@pytest.mark.skipif(not HAS_PUBLISHED_CHECKPOINTS, reason="published checkpoints are not installed")
def test_manifest_mutation_is_rejected():
    manifest = build_manifest(SOURCES, ["classifier.0.bias"])
    corrupt = copy.deepcopy(manifest)
    corrupt["decision"]["threshold"] = 0.25
    with pytest.raises(BundleValidationError, match="threshold"):
        validate_manifest(corrupt)


@pytest.mark.skipif(not HAS_PUBLISHED_CHECKPOINTS, reason="published checkpoints are not installed")
def test_manifest_identity_is_bound_to_source_hashes():
    manifest = build_manifest(SOURCES, ["classifier.0.bias"])
    corrupt = copy.deepcopy(manifest)
    corrupt["source_artifacts"]["semantic_stream"]["sha256"] = "0" * 64
    with pytest.raises(BundleValidationError, match="weights_id"):
        validate_manifest(corrupt)


def test_bundle_payload_rejects_embedded_state_value_mutation():
    if not HAS_PUBLISHED_CHECKPOINTS:
        pytest.skip("published checkpoints are not installed")
    state = {"classifier.0.bias": torch.zeros(1)}
    manifest = build_manifest(SOURCES, state)
    assert manifest["state"]["embedded_sha256"] == state_dict_digest(state)
    mutated = {"classifier.0.bias": torch.ones(1)}
    with pytest.raises(BundleValidationError, match="embedded_sha256"):
        load_checkpoint_bundle_payload({"manifest": manifest, "state_dict": mutated})


def test_manifest_identity_is_bound_to_embedded_state_digest():
    if not HAS_PUBLISHED_CHECKPOINTS:
        pytest.skip("published checkpoints are not installed")
    state = {"classifier.0.bias": torch.zeros(1)}
    manifest = build_manifest(SOURCES, state)
    corrupt = copy.deepcopy(manifest)
    corrupt["state"]["embedded_sha256"] = "1" * 64
    with pytest.raises(BundleValidationError, match="weights_id"):
        validate_manifest(corrupt)


def test_bundle_payload_rejects_state_keys_that_do_not_match_manifest():
    # A valid manifest lets this test reach the state-key gate without
    # constructing the 87M-parameter detector.
    if not HAS_PUBLISHED_CHECKPOINTS:
        pytest.skip("published checkpoints are not installed")
    manifest = build_manifest(SOURCES, ["classifier.0.bias"])
    with pytest.raises(BundleValidationError, match="state_dict keys"):
        load_checkpoint_bundle_payload(
            {"manifest": manifest, "state_dict": {"wrong.key": torch.zeros(1)}}
        )


def test_bundle_payload_rejects_unknown_top_level_fields():
    if not HAS_PUBLISHED_CHECKPOINTS:
        pytest.skip("published checkpoints are not installed")
    manifest = build_manifest(SOURCES, ["classifier.0.bias"])
    with pytest.raises(BundleValidationError, match="unexpected"):
        load_checkpoint_bundle_payload(
            {"manifest": manifest, "state_dict": {}, "extra": True}
        )


@pytest.mark.skipif(not HAS_PUBLISHED_CHECKPOINTS, reason="published checkpoints are not installed")
def test_declared_explainability_targets_resolve_on_canonical_topology():
    manifest = build_manifest(SOURCES, ["classifier.0.bias"])
    detector = CanonicalFusedDetector()
    resolved = validate_explainability_contract(detector, manifest)

    assert resolved["attribution_targets"]["semantic"].__class__.__name__ == "LayerNorm"
    assert resolved["attribution_targets"]["forensic"].__class__.__name__ == "Conv2d"
    assert set(resolved["intermediate_representations"]["forensic"]) == {
        "frontend.bayar",
        "frontend.srm",
        "frontend.fuse",
        "backbone.4",
        "pool",
    }
    assert manifest["explainability"]["unsupported_capabilities"]["attention_tensors"]
    assert manifest["explainability"]["unsupported_capabilities"]["branch_subset_logits"]


@pytest.mark.skipif(
    os.environ.get("RAID_RUN_REAL_CHECKPOINT_TESTS") != "1" or not HAS_PUBLISHED_CHECKPOINTS,
    reason="set RAID_RUN_REAL_CHECKPOINT_TESTS=1 with published checkpoints for the real parity gate",
)
def test_bundle_matches_independent_three_file_scorer(tmp_path: Path):
    bundle_path = build_checkpoint_bundle(
        SOURCES["semantic_stream"],
        SOURCES["forensic_stream"],
        SOURCES["detector_fusion"],
        tmp_path / "detector_bundle.pt",
    )
    bundled_model, manifest = load_checkpoint_bundle(bundle_path)
    assert manifest["model_id"] == MODEL_ID

    reference = CanonicalFusedDetector()
    reference.semantic_stream.load_state_dict(
        torch.load(SOURCES["semantic_stream"], map_location="cpu", weights_only=True), strict=True
    )
    reference.forensic_stream.load_state_dict(
        torch.load(SOURCES["forensic_stream"], map_location="cpu", weights_only=True), strict=True
    )
    fusion_state = torch.load(SOURCES["detector_fusion"], map_location="cpu", weights_only=True)
    reference.fusion.load_state_dict(fusion_state["fusion"], strict=True)
    reference.classifier.load_state_dict(fusion_state["classifier"], strict=True)
    reference.eval()

    prepared = prepare_fused_inputs(ROOT / "test_sample.jpg")
    with torch.inference_mode():
        bundled_logit = bundled_model(prepared)
        reference_logit = reference(prepared)
    torch.testing.assert_close(bundled_logit, reference_logit, rtol=1e-6, atol=1e-7)
    assert bundled_logit.item() == pytest.approx(-0.3831035197, abs=1e-7)
    assert torch.sigmoid(bundled_logit).item() == pytest.approx(0.4053786099, abs=1e-7)

    resolved = validate_explainability_contract(bundled_model, manifest)
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for branch, target in resolved["attribution_targets"].items():
        def _capture(_module, _inputs, output, branch_name=branch):
            if isinstance(output, torch.Tensor):
                output.retain_grad()
                captured[branch_name] = output

        handles.append(target.register_forward_hook(_capture))
    try:
        semantic = prepared.semantic.detach().requires_grad_()
        forensic = prepared.forensic.detach().requires_grad_()
        bundled_model(semantic, forensic).sum().backward()
    finally:
        for handle in handles:
            handle.remove()
    assert set(captured) == {"semantic", "forensic"}
    assert all(value.grad is not None and torch.any(value.grad != 0) for value in captured.values())

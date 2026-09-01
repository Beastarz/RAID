"""Tests for model-independent explainability contracts."""

import json
from pathlib import Path

import pytest

from src.explainability.contracts import (
    SCHEMA_VERSION,
    AdapterPrediction,
    ArtifactReference,
    AttributionTarget,
    BranchCoalitionLogit,
    Capability,
    CapabilityResult,
    CapabilityStatus,
    ExplainabilityAdapter,
    ExplanationOutputEnvelope,
    ExplanationResult,
    PredictionOutputEnvelope,
    PredictionRecord,
    PreparedModelInputs,
)


def prediction_record() -> PredictionRecord:
    return PredictionRecord(
        sample_id="sample-001",
        source_reference="images/sample-001.png",
        model_id="model-a",
        predicted_logit=1.25,
        predicted_probability=0.7773,
        predicted_label=1,
        decision_threshold=0.5,
        ground_truth_label=0,
        metadata={"condition": "clean"},
    )


def test_representative_prediction_json_shape_is_exact_and_strict_json():
    envelope = PredictionOutputEnvelope((prediction_record(),))

    assert envelope.to_dict() == {
        "schema_version": "1.0",
        "predictions": [
            {
                "schema_version": "1.0",
                "sample_id": "sample-001",
                "source_reference": "images/sample-001.png",
                "model_id": "model-a",
                "predicted_logit": 1.25,
                "predicted_probability": 0.7773,
                "predicted_label": 1,
                "decision_threshold": 0.5,
                "ground_truth_label": 0,
                "metadata": {"condition": "clean"},
            }
        ],
    }
    json.dumps(envelope.to_dict(), allow_nan=False)


def test_representative_explanation_json_shape_is_exact_and_strict_json():
    artifact = ArtifactReference(
        artifact_id="semantic-map",
        path="explanations/sample-001/semantic.npy",
        artifact_type="attribution_map",
        media_type="application/x-npy",
    )
    explanation = ExplanationResult(
        sample_id="sample-001",
        model_id="model-a",
        method_name="integrated_gradients",
        status=CapabilityStatus.supported(),
        artifacts=(artifact,),
        statistics={"minimum": -0.2, "maximum": 0.8},
    )
    envelope = ExplanationOutputEnvelope((explanation,))

    assert envelope.to_dict() == {
        "schema_version": "1.0",
        "explanations": [
            {
                "schema_version": "1.0",
                "sample_id": "sample-001",
                "model_id": "model-a",
                "method_name": "integrated_gradients",
                "status": {"available": True, "reason": None},
                "artifacts": [
                    {
                        "schema_version": "1.0",
                        "artifact_id": "semantic-map",
                        "path": "explanations/sample-001/semantic.npy",
                        "artifact_type": "attribution_map",
                        "media_type": "application/x-npy",
                        "metadata": {},
                    }
                ],
                "statistics": {"minimum": -0.2, "maximum": 0.8},
                "branch_coalition_logits": [],
                "metadata": {},
            }
        ],
    }
    json.dumps(envelope.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "probability", [-0.01, 1.01, float("nan"), float("inf")]
)
def test_prediction_rejects_invalid_probability(probability):
    with pytest.raises(ValueError):
        PredictionRecord(
            sample_id="sample",
            source_reference="image.png",
            model_id="model",
            predicted_logit=0.0,
            predicted_probability=probability,
            predicted_label=0,
            decision_threshold=0.5,
        )


@pytest.mark.parametrize(
    "threshold", [-0.01, 1.01, float("nan"), float("inf")]
)
def test_prediction_rejects_invalid_decision_threshold(threshold):
    with pytest.raises(ValueError):
        PredictionRecord(
            sample_id="sample",
            model_id="model",
            predicted_logit=0.0,
            predicted_probability=0.5,
            predicted_label=0,
            decision_threshold=threshold,
        )


def test_prediction_accepts_in_memory_source_without_reference():
    record = PredictionRecord(
        sample_id="uploaded-sample",
        model_id="model",
        predicted_logit=0.0,
        predicted_probability=0.2,
        predicted_label=1,
        decision_threshold=0.9,
        source_reference=None,
    )

    assert record.to_dict()["source_reference"] is None
    json.dumps(record.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "field,value",
    [("predicted_label", 2), ("predicted_label", 1.0), ("ground_truth_label", -1)],
)
def test_prediction_rejects_non_binary_labels(field, value):
    values = {
        "sample_id": "sample",
        "source_reference": "image.png",
        "model_id": "model",
        "predicted_logit": 0.0,
        "predicted_probability": 0.5,
        "predicted_label": 0,
        "decision_threshold": 0.5,
        "ground_truth_label": None,
    }
    values[field] = value
    with pytest.raises(ValueError):
        PredictionRecord(**values)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_records_reject_non_finite_logits_and_statistics(bad_value):
    with pytest.raises(ValueError):
        BranchCoalitionLogit(("branch",), bad_value)
    with pytest.raises(ValueError):
        ExplanationResult(
            sample_id="sample",
            model_id="model",
            method_name="method",
            status=CapabilityStatus.supported(),
            statistics={"mean": bad_value},
        )


def test_metadata_rejects_runtime_objects():
    with pytest.raises(TypeError):
        ArtifactReference(
            "id", "map.npy", "map", "application/x-npy", {"tensor": object()}
        )


def test_coalition_names_are_canonical_and_duplicates_are_rejected():
    coalition = BranchCoalitionLogit(("zeta", "alpha"), 0.5)
    assert coalition.branch_names == ("alpha", "zeta")
    assert coalition.to_dict()["branch_names"] == ["alpha", "zeta"]

    with pytest.raises(ValueError, match="unique"):
        BranchCoalitionLogit(("alpha", "alpha"), 0.5)
    with pytest.raises(ValueError, match="whitespace"):
        BranchCoalitionLogit((" alpha",), 0.5)


def test_explanation_rejects_duplicate_coalitions():
    coalition = BranchCoalitionLogit(("alpha",), 0.5)
    with pytest.raises(ValueError, match="coalitions must be unique"):
        ExplanationResult(
            sample_id="sample",
            model_id="model",
            method_name="branch_contributions",
            status=CapabilityStatus.supported(),
            branch_coalition_logits=(coalition, coalition),
        )


def test_unavailable_capabilities_require_an_explicit_reason():
    with pytest.raises(ValueError, match="reason"):
        CapabilityStatus(available=False)

    result = CapabilityResult.unsupported("model does not expose attention")
    assert result.status == CapabilityStatus(
        available=False, reason="model does not expose attention"
    )
    assert result.value is None

    with pytest.raises(ValueError, match="cannot contain"):
        CapabilityResult(CapabilityStatus.unsupported("not exposed"), object())


def test_schema_version_and_required_identifiers_are_validated():
    with pytest.raises(ValueError, match="schema_version"):
        PredictionOutputEnvelope((), schema_version="2.0")
    with pytest.raises(ValueError, match="method_name"):
        ExplanationResult(
            sample_id="sample",
            model_id="model",
            method_name=" ",
            status=CapabilityStatus.supported(),
        )
    with pytest.raises(ValueError, match="model_id"):
        ExplanationResult(
            sample_id="sample",
            model_id=" ",
            method_name="method",
            status=CapabilityStatus.supported(),
        )
    with pytest.raises(ValueError, match="source_reference"):
        PredictionRecord(
            sample_id="sample",
            model_id="model",
            predicted_logit=0.0,
            predicted_probability=0.5,
            predicted_label=0,
            decision_threshold=0.5,
            source_reference="",
        )
    with pytest.raises(ValueError, match="path"):
        ArtifactReference("id", "", "map", "image/png")


class MinimalAdapter:
    @property
    def capabilities(self):
        return {
            Capability.PREDICTION: CapabilityStatus.supported(),
            Capability.ATTRIBUTION_TARGETS: CapabilityStatus.supported(),
            Capability.ATTENTION_TENSORS: CapabilityStatus.unsupported(
                "attention is not exposed"
            ),
            Capability.INTERMEDIATE_REPRESENTATIONS: CapabilityStatus.supported(),
            Capability.BRANCH_SUBSET_LOGITS: CapabilityStatus.unsupported(
                "branch ablation is not implemented"
            ),
        }

    def prepare_source_image(self, raw_source_image):
        return PreparedModelInputs(values={"prepared": raw_source_image}, context={})

    def predict(self, prepared_model_inputs):
        return AdapterPrediction(0.0, 0.5, 1, 0.5)

    def attribution_targets(self, prepared_model_inputs):
        return CapabilityResult.supported(
            {
                "semantic": AttributionTarget("semantic target"),
                "forensic": AttributionTarget("forensic target"),
            }
        )

    def attention_tensors(self, prepared_model_inputs):
        return CapabilityResult.unsupported("attention is not exposed")

    def intermediate_representations(self, prepared_model_inputs):
        return CapabilityResult.supported(
            {
                "semantic": {"tokens": "semantic tokens"},
                "forensic": {"features": "forensic features"},
            }
        )

    def branch_subset_logits(self, prepared_model_inputs):
        return CapabilityResult.unsupported("branch ablation is not implemented")


def test_adapter_protocol_is_runtime_checkable_without_model_knowledge():
    adapter = MinimalAdapter()
    assert isinstance(adapter, ExplainabilityAdapter)
    prepared = adapter.prepare_source_image("raw image")
    assert adapter.predict(prepared).predicted_probability == 0.5
    targets = adapter.attribution_targets(prepared)
    assert targets.status.available
    assert targets.value["semantic"].value == "semantic target"
    assert targets.value["forensic"].value == "forensic target"
    assert not adapter.attention_tensors(prepared).status.available
    representations = adapter.intermediate_representations(prepared)
    assert representations.status.available
    assert representations.value["forensic"]["features"] == "forensic features"


def test_contract_module_has_no_model_or_training_imports():
    source = Path("src/explainability/contracts.py").read_text()
    assert "src.models" not in source
    assert "training" not in source
    assert "DetectorPipeline" not in source
    assert "import torch" not in source
    assert "import numpy" not in source
    assert SCHEMA_VERSION == "1.0"

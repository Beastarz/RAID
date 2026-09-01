"""Versioned, model-independent explainability contracts.

Representative prediction envelope::

    {
      "schema_version": "1.0",
      "predictions": [{
        "schema_version": "1.0", "sample_id": "sample-001",
        "source_reference": "images/sample-001.png", "model_id": "model-a",
        "predicted_logit": 1.25, "predicted_probability": 0.7773,
        "predicted_label": 1, "decision_threshold": 0.5,
        "ground_truth_label": 0,
        "metadata": {"condition": "clean"}
      }]
    }

Representative per-image explanation envelope::

    {
      "schema_version": "1.0",
      "explanations": [{
        "schema_version": "1.0", "sample_id": "sample-001",
        "model_id": "model-a", "method_name": "integrated_gradients",
        "status": {"available": true, "reason": null},
        "artifacts": [{
          "schema_version": "1.0", "artifact_id": "semantic-map",
          "path": "explanations/sample-001/semantic.npy",
          "artifact_type": "attribution_map", "media_type": "application/x-npy",
          "metadata": {}
        }],
        "statistics": {"minimum": -0.2, "maximum": 0.8},
        "branch_coalition_logits": [], "metadata": {}
      }]
    }

Raw source images and prepared model inputs are runtime values and are never
embedded in these JSON records. Logits and probabilities have distinct fields;
ground-truth and predicted labels are likewise kept separate. The decision
threshold records the threshold applied by the adapter to produce its label;
the contract does not prescribe a threshold equality policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Mapping, Protocol, Sequence, TypeVar, runtime_checkable


SCHEMA_VERSION = "1.0"

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _require_non_empty(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_schema_version(value: str) -> None:
    if value != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")


def _finite_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _binary_label(value: int | None, field_name: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
    ):
        raise ValueError(f"{field_name} must be 0, 1, or None")


def _json_value(value: Any, field_name: str = "value") -> JsonValue:
    """Return a detached JSON value, rejecting tensors, callables, and NaN/Inf."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            result[key] = _json_value(item, f"{field_name}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, field_name) for item in value]
    raise TypeError(
        f"{field_name} contains a non-JSON value of type {type(value).__name__}"
    )


class Capability(str, Enum):
    PREDICTION = "prediction"
    ATTRIBUTION_TARGETS = "attribution_targets"
    ATTENTION_TENSORS = "attention_tensors"
    INTERMEDIATE_REPRESENTATIONS = "intermediate_representations"
    BRANCH_SUBSET_LOGITS = "branch_subset_logits"


@dataclass(frozen=True)
class CapabilityStatus:
    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise TypeError("available must be a bool")
        if self.available and self.reason is not None:
            raise ValueError(
                "an available capability must not have an unavailable reason"
            )
        if not self.available:
            _require_non_empty(self.reason, "reason")

    @classmethod
    def supported(cls) -> "CapabilityStatus":
        return cls(available=True)

    @classmethod
    def unsupported(cls, reason: str) -> "CapabilityStatus":
        return cls(available=False, reason=reason)

    def to_dict(self) -> dict[str, JsonValue]:
        return {"available": self.available, "reason": self.reason}


T = TypeVar("T")


@dataclass(frozen=True)
class CapabilityResult(Generic[T]):
    status: CapabilityStatus
    value: T | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, CapabilityStatus):
            raise TypeError("status must be a CapabilityStatus")
        if self.status.available and self.value is None:
            raise ValueError("an available capability result requires a value")
        if not self.status.available and self.value is not None:
            raise ValueError("an unavailable capability result cannot contain a value")

    @classmethod
    def supported(cls, value: T) -> "CapabilityResult[T]":
        return cls(CapabilityStatus.supported(), value)

    @classmethod
    def unsupported(cls, reason: str) -> "CapabilityResult[T]":
        return cls(CapabilityStatus.unsupported(reason))


@dataclass(frozen=True)
class PreparedModelInputs:
    """Opaque adapter-owned model inputs and deterministic preparation context."""

    values: object
    context: object


@dataclass(frozen=True)
class AttributionTarget:
    """Opaque adapter-owned target consumed by a future attribution engine."""

    value: object


@dataclass(frozen=True)
class AdapterPrediction:
    predicted_logit: float
    predicted_probability: float
    predicted_label: int
    decision_threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predicted_logit",
            _finite_float(self.predicted_logit, "predicted_logit"),
        )
        probability = _finite_float(self.predicted_probability, "predicted_probability")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("predicted_probability must be between 0 and 1")
        object.__setattr__(self, "predicted_probability", probability)
        _binary_label(self.predicted_label, "predicted_label")
        threshold = _finite_float(self.decision_threshold, "decision_threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("decision_threshold must be between 0 and 1")
        object.__setattr__(self, "decision_threshold", threshold)


@dataclass(frozen=True)
class BranchCoalitionLogit:
    branch_names: tuple[str, ...]
    logit: float
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        if isinstance(self.branch_names, (str, bytes)):
            raise TypeError("branch_names must be an iterable of strings")
        names = tuple(self.branch_names)
        for name in names:
            _require_non_empty(name, "branch_name")
            if name != name.strip():
                raise ValueError("branch_names must not contain surrounding whitespace")
        if len(set(names)) != len(names):
            raise ValueError("branch_names must be unique")
        object.__setattr__(self, "branch_names", tuple(sorted(names)))
        object.__setattr__(self, "logit", _finite_float(self.logit, "logit"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "branch_names": list(self.branch_names),
            "logit": self.logit,
        }


@dataclass(frozen=True)
class ArtifactReference:
    artifact_id: str
    path: str
    artifact_type: str
    media_type: str
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_non_empty(self.artifact_id, "artifact_id")
        _require_non_empty(self.path, "path")
        _require_non_empty(self.artifact_type, "artifact_type")
        _require_non_empty(self.media_type, "media_type")
        object.__setattr__(self, "metadata", _json_value(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "path": self.path,
            "artifact_type": self.artifact_type,
            "media_type": self.media_type,
            "metadata": _json_value(self.metadata, "metadata"),
        }


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    model_id: str
    predicted_logit: float
    predicted_probability: float
    predicted_label: int
    decision_threshold: float
    source_reference: str | None = None
    ground_truth_label: int | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_non_empty(self.sample_id, "sample_id")
        _require_non_empty(self.model_id, "model_id")
        if self.source_reference is not None:
            _require_non_empty(self.source_reference, "source_reference")
        prediction = AdapterPrediction(
            self.predicted_logit,
            self.predicted_probability,
            self.predicted_label,
            self.decision_threshold,
        )
        object.__setattr__(self, "predicted_logit", prediction.predicted_logit)
        object.__setattr__(
            self, "predicted_probability", prediction.predicted_probability
        )
        object.__setattr__(self, "decision_threshold", prediction.decision_threshold)
        _binary_label(self.ground_truth_label, "ground_truth_label")
        object.__setattr__(self, "metadata", _json_value(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "source_reference": self.source_reference,
            "model_id": self.model_id,
            "predicted_logit": self.predicted_logit,
            "predicted_probability": self.predicted_probability,
            "predicted_label": self.predicted_label,
            "decision_threshold": self.decision_threshold,
            "ground_truth_label": self.ground_truth_label,
            "metadata": _json_value(self.metadata, "metadata"),
        }


@dataclass(frozen=True)
class ExplanationResult:
    sample_id: str
    model_id: str
    method_name: str
    status: CapabilityStatus
    artifacts: tuple[ArtifactReference, ...] = ()
    statistics: Mapping[str, JsonValue] = field(default_factory=dict)
    branch_coalition_logits: tuple[BranchCoalitionLogit, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_non_empty(self.sample_id, "sample_id")
        _require_non_empty(self.model_id, "model_id")
        _require_non_empty(self.method_name, "method_name")
        if not isinstance(self.status, CapabilityStatus):
            raise TypeError("status must be a CapabilityStatus")
        if not self.status.available and (
            self.artifacts or self.statistics or self.branch_coalition_logits
        ):
            raise ValueError("an unavailable explanation cannot contain outputs")
        artifacts = tuple(self.artifacts)
        coalitions = tuple(self.branch_coalition_logits)
        if any(not isinstance(item, ArtifactReference) for item in artifacts):
            raise TypeError("artifacts must contain ArtifactReference records")
        if any(not isinstance(item, BranchCoalitionLogit) for item in coalitions):
            raise TypeError(
                "branch_coalition_logits must contain BranchCoalitionLogit records"
            )
        coalition_keys = [item.branch_names for item in coalitions]
        if len(set(coalition_keys)) != len(coalition_keys):
            raise ValueError("branch coalitions must be unique")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "branch_coalition_logits", coalitions)
        object.__setattr__(
            self, "statistics", _json_value(self.statistics, "statistics")
        )
        object.__setattr__(self, "metadata", _json_value(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "model_id": self.model_id,
            "method_name": self.method_name,
            "status": self.status.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "statistics": _json_value(self.statistics, "statistics"),
            "branch_coalition_logits": [
                item.to_dict() for item in self.branch_coalition_logits
            ],
            "metadata": _json_value(self.metadata, "metadata"),
        }


@dataclass(frozen=True)
class PredictionOutputEnvelope:
    predictions: tuple[PredictionRecord, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        predictions = tuple(self.predictions)
        if any(not isinstance(item, PredictionRecord) for item in predictions):
            raise TypeError("predictions must contain PredictionRecord records")
        object.__setattr__(self, "predictions", predictions)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


@dataclass(frozen=True)
class ExplanationOutputEnvelope:
    explanations: tuple[ExplanationResult, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        explanations = tuple(self.explanations)
        if any(not isinstance(item, ExplanationResult) for item in explanations):
            raise TypeError("explanations must contain ExplanationResult records")
        object.__setattr__(self, "explanations", explanations)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "explanations": [
                explanation.to_dict() for explanation in self.explanations
            ],
        }


@runtime_checkable
class ExplainabilityAdapter(Protocol):
    """Architecture boundary for prediction and optional explanation inputs.

    Capability payloads are keyed by canonical, adapter-defined branch names.
    Generic explainability algorithms must not assume a particular forensic
    frontend or branch topology.
    """

    @property
    def capabilities(self) -> Mapping[Capability, CapabilityStatus]: ...

    def prepare_source_image(self, raw_source_image: object) -> PreparedModelInputs: ...

    def predict(
        self, prepared_model_inputs: PreparedModelInputs
    ) -> AdapterPrediction: ...

    def attribution_targets(
        self, prepared_model_inputs: PreparedModelInputs
    ) -> CapabilityResult[Mapping[str, AttributionTarget]]:
        """Return one opaque attribution target per supported branch."""
        ...

    def attention_tensors(
        self, prepared_model_inputs: PreparedModelInputs
    ) -> CapabilityResult[Sequence[object]]: ...

    def intermediate_representations(
        self, prepared_model_inputs: PreparedModelInputs
    ) -> CapabilityResult[Mapping[str, Mapping[str, object]]]:
        """Return branch-keyed maps of named opaque intermediate values."""
        ...

    def branch_subset_logits(
        self, prepared_model_inputs: PreparedModelInputs
    ) -> CapabilityResult[Sequence[BranchCoalitionLogit]]: ...

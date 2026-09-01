"""Strict deterministic JSON serialization for explainability envelopes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    ArtifactReference,
    BranchCoalitionLogit,
    CapabilityStatus,
    ExplanationOutputEnvelope,
    ExplanationResult,
    PredictionOutputEnvelope,
    PredictionRecord,
    SCHEMA_VERSION,
)


Envelope = PredictionOutputEnvelope | ExplanationOutputEnvelope


def envelope_to_json(envelope: Envelope) -> str:
    """Serialize an existing envelope as deterministic strict JSON."""

    if not isinstance(envelope, (PredictionOutputEnvelope, ExplanationOutputEnvelope)):
        raise TypeError("envelope must be a prediction or explanation output envelope")
    return json.dumps(
        envelope.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def write_envelope_json(envelope: Envelope, path: str | Path) -> Path:
    """Write UTF-8 strict JSON, creating its parent directory when needed."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(envelope_to_json(envelope) + "\n", encoding="utf-8")
    return destination


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _load_json_object(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{field_name}.schema_version must be {SCHEMA_VERSION!r}")
    return value


def _reject_unknown_keys(
    value: Mapping[str, Any], allowed: set[str], *, field_name: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{field_name} contains unknown fields: {sorted(unknown)}")


def prediction_record_from_dict(value: object) -> PredictionRecord:
    """Construct one validated prediction record from a JSON object."""

    payload = _load_json_object(value, field_name="prediction")
    try:
        return PredictionRecord(**payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid prediction record: {exc}") from exc


def read_prediction_jsonl(path: str | Path) -> tuple[PredictionRecord, ...]:
    """Read one strict ``PredictionRecord`` JSON object per nonblank line."""

    source = Path(path)
    records: list[PredictionRecord] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                )
                records.append(prediction_record_from_dict(payload))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid prediction JSONL at line {line_number}: {exc}") from exc
    if not records:
        raise ValueError("prediction JSONL must contain at least one record")
    return tuple(records)


def write_prediction_jsonl(
    records: Sequence[PredictionRecord] | Iterable[PredictionRecord],
    path: str | Path,
) -> Path:
    """Write validated prediction records as deterministic strict JSONL."""

    typed_records = tuple(records)
    if not typed_records:
        raise ValueError("records must contain at least one PredictionRecord")
    if any(not isinstance(record, PredictionRecord) for record in typed_records):
        raise TypeError("records must contain only PredictionRecord values")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        for record in typed_records:
            stream.write(
                json.dumps(
                    record.to_dict(),
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return destination


def explanation_envelope_from_dict(value: object) -> ExplanationOutputEnvelope:
    """Construct an explanation envelope from its strict JSON representation."""

    payload = _load_json_object(value, field_name="explanation envelope")
    _reject_unknown_keys(
        payload,
        {"schema_version", "explanations"},
        field_name="explanation envelope",
    )
    raw_explanations = payload.get("explanations")
    if not isinstance(raw_explanations, list):
        raise ValueError("explanation envelope.explanations must be a list")
    explanations: list[ExplanationResult] = []
    for index, raw in enumerate(raw_explanations):
        item = _load_json_object(raw, field_name=f"explanations[{index}]")
        _reject_unknown_keys(
            item,
            {
                "schema_version",
                "sample_id",
                "model_id",
                "method_name",
                "status",
                "artifacts",
                "statistics",
                "branch_coalition_logits",
                "metadata",
            },
            field_name=f"explanations[{index}]",
        )
        status_value = item.get("status")
        if not isinstance(status_value, dict):
            raise ValueError(f"explanations[{index}].status must be an object")
        _reject_unknown_keys(
            status_value,
            {"available", "reason"},
            field_name=f"explanations[{index}].status",
        )
        try:
            raw_artifacts = item.get("artifacts", [])
            raw_coalitions = item.get("branch_coalition_logits", [])
            if not isinstance(raw_artifacts, list):
                raise ValueError("artifacts must be a list")
            if not isinstance(raw_coalitions, list):
                raise ValueError("branch_coalition_logits must be a list")
            for artifact_index, artifact in enumerate(raw_artifacts):
                artifact_payload = _load_json_object(
                    artifact,
                    field_name=f"explanations[{index}].artifacts[{artifact_index}]",
                )
                _reject_unknown_keys(
                    artifact_payload,
                    {
                        "schema_version",
                        "artifact_id",
                        "path",
                        "artifact_type",
                        "media_type",
                        "metadata",
                    },
                    field_name=f"explanations[{index}].artifacts[{artifact_index}]",
                )
            for coalition_index, coalition in enumerate(raw_coalitions):
                coalition_payload = _load_json_object(
                    coalition,
                    field_name=f"explanations[{index}].branch_coalition_logits[{coalition_index}]",
                )
                _reject_unknown_keys(
                    coalition_payload,
                    {"schema_version", "branch_names", "logit"},
                    field_name=f"explanations[{index}].branch_coalition_logits[{coalition_index}]",
                )
            status = CapabilityStatus(**status_value)
            artifacts = tuple(
                ArtifactReference(**artifact) for artifact in raw_artifacts
            )
            coalitions = tuple(
                BranchCoalitionLogit(**coalition) for coalition in raw_coalitions
            )
            explanations.append(
                ExplanationResult(
                    sample_id=item["sample_id"],
                    model_id=item["model_id"],
                    method_name=item["method_name"],
                    status=status,
                    artifacts=artifacts,
                    statistics=item.get("statistics", {}),
                    branch_coalition_logits=coalitions,
                    metadata=item.get("metadata", {}),
                    schema_version=item["schema_version"],
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid explanation at index {index}: {exc}") from exc
    try:
        return ExplanationOutputEnvelope(
            tuple(explanations), schema_version=payload["schema_version"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid explanation envelope: {exc}") from exc


def read_explanation_json(path: str | Path) -> ExplanationOutputEnvelope:
    """Read and validate an explanation output envelope from strict JSON."""

    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise ValueError(f"could not read explanation JSON {source}: {exc}") from exc
    return explanation_envelope_from_dict(payload)


__all__ = [
    "envelope_to_json",
    "explanation_envelope_from_dict",
    "prediction_record_from_dict",
    "read_explanation_json",
    "read_prediction_jsonl",
    "write_envelope_json",
    "write_prediction_jsonl",
]

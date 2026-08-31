"""Strict deterministic JSON serialization for explainability envelopes."""

from __future__ import annotations

import json
from pathlib import Path

from .contracts import ExplanationOutputEnvelope, PredictionOutputEnvelope


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

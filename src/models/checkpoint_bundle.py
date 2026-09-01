"""Self-describing checkpoint bundle for the canonical fused detector.

The published weights are currently distributed as three independent files.
This module turns those provenance artifacts into one complete, strictly
validated detector state plus a manifest that records the topology and input
contract needed by inference and explainability.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

from src.models.fused_detector import (
    CanonicalFusedDetector,
    FORENSIC_DIM,
    FUSED_DIM,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RawImage,
    SEMANTIC_DIM,
    prepare_fused_inputs,
)


BUNDLE_SCHEMA_VERSION = "1.0"
BUNDLE_TYPE = "raid.canonical_fused_detector"
MODEL_ID = "raid-vit-b16-bayar-srm-resnet-shallow-v1"
DECISION_THRESHOLD = 0.5
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
WEIGHTS_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

_SOURCE_ROLES = ("semantic_stream", "forensic_stream", "detector_fusion")
_TOP_LEVEL_KEYS = frozenset({"manifest", "state_dict"})


class BundleValidationError(ValueError):
    """Raised when a bundle or one of its manifests is not safe to load."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"checkpoint source is not a file: {file_path}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint source is not a file: {source}")
    return {
        "filename": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash state entries deterministically, including values and metadata.

    Hashing only state keys is insufficient: a bundle with replaced tensors would
    otherwise retain the same manifest identity.  Entries are normalized to
    contiguous CPU byte views so the digest is independent of device and tensor
    storage layout.
    """

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise BundleValidationError("state_dict must be a non-empty mapping")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state_dict.items()):
        raise BundleValidationError("state_dict must contain only string keys and tensors")
    digest = hashlib.sha256()
    digest.update(b"raid-state-digest-v1\0")
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        dtype = str(tensor.dtype).encode("utf-8")
        shape = json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        key_bytes = key.encode("utf-8")
        # Flatten first because PyTorch disallows byte-viewing a 0-D tensor
        # (BatchNorm counters are scalar state entries).
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        for value in (key_bytes, dtype, shape, raw):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def _keys_only_digest(state_keys: list[str]) -> str:
    """Provide a deterministic placeholder for legacy ``build_manifest`` callers."""

    digest = hashlib.sha256()
    digest.update(b"raid-state-keys-v1\0")
    for key in sorted(set(state_keys)):
        encoded = key.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _weights_id(
    source_artifacts: Mapping[str, Mapping[str, Any]], state_digest: str
) -> str:
    digest = hashlib.sha256()
    digest.update(b"raid-source-identity-v1\0")
    for role in _SOURCE_ROLES:
        role_bytes = role.encode("utf-8")
        hash_bytes = str(source_artifacts[role]["sha256"]).encode("ascii")
        digest.update(len(role_bytes).to_bytes(8, "big"))
        digest.update(role_bytes)
        digest.update(len(hash_bytes).to_bytes(8, "big"))
        digest.update(hash_bytes)
    state_bytes = state_digest.encode("ascii")
    digest.update(len(state_bytes).to_bytes(8, "big"))
    digest.update(state_bytes)
    return f"sha256:{digest.hexdigest()}"


def _canonical_manifest(
    source_paths: Mapping[str, str | Path], state_keys: list[str], state_digest: str
) -> dict[str, Any]:
    if set(source_paths) != set(_SOURCE_ROLES):
        raise BundleValidationError(
            f"source_paths must contain exactly {list(_SOURCE_ROLES)}, got {sorted(source_paths)}"
        )
    source_artifacts = {role: _source_metadata(source_paths[role]) for role in _SOURCE_ROLES}
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_type": BUNDLE_TYPE,
        "model_id": MODEL_ID,
        "weights_id": _weights_id(source_artifacts, state_digest),
        "topology": {
            "branches": ["semantic", "forensic"],
            "semantic": {
                "stream": "SemanticStream",
                "architecture": "ViT-B/16",
                "feature_dim": SEMANTIC_DIM,
                "internal_input_size": [224, 224],
                "internal_interpolation": "bilinear",
                "internal_align_corners": False,
            },
            "forensic": {
                "stream": "NPRStream",
                "frontend": "bayar_srm",
                "backbone": "resnet_shallow",
                "feature_dim": FORENSIC_DIM,
            },
            "fusion": {
                "architecture": "concat_linear_layernorm_gelu",
                "semantic_dim": SEMANTIC_DIM,
                "forensic_dim": FORENSIC_DIM,
                "fused_dim": FUSED_DIM,
            },
            "classifier": {
                "architecture": "linear_gelu_linear",
                "input_dim": FUSED_DIM,
                "hidden_dim": 128,
                "output_dim": 1,
            },
        },
        "preprocessing": {
            "shared_resize": True,
            "resize": {
                "backend": "Pillow",
                "size": [IMAGE_SIZE, IMAGE_SIZE],
                "interpolation": "bilinear",
                "resample": "BILINEAR",
            },
            "decode_color": "RGB",
            "dtype": "float32",
            "semantic": {
                "normalization": {
                    "mean": list(IMAGENET_MEAN),
                    "std": list(IMAGENET_STD),
                },
                "pixel_source": "shared_resized_pixels",
            },
            "forensic": {
                "pixel_range": [0.0, 1.0],
                "pixel_source": "shared_resized_pixels",
            },
        },
        "decision": {
            "threshold": DECISION_THRESHOLD,
            "equality": "greater_or_equal",
        },
        "source_artifacts": source_artifacts,
        "state": {
            "format": "torch_state_dict",
            "keys": sorted(state_keys),
            "embedded_sha256": state_digest,
        },
        "explainability": {
            "branch_names": ["semantic", "forensic"],
            "attribution_targets": {
                "semantic": "semantic_stream.backbone.encoder.layers.encoder_layer_11.ln_1",
                "forensic": "forensic_stream.backbone.4.2.conv3",
            },
            "intermediate_representations": {
                "semantic": ["backbone.encoder.layers.encoder_layer_11.ln_1"],
                "forensic": [
                    "frontend.bayar",
                    "frontend.srm",
                    "frontend.fuse",
                    "backbone.4",
                    "pool",
                ],
            },
            "unsupported_capabilities": {
                "attention_tensors": "torchvision ViT forward does not expose attention matrices",
                "branch_subset_logits": "bundle does not contain an explicit feature-ablation baseline",
            },
        },
    }


def build_manifest(
    source_paths: Mapping[str, str | Path],
    state_keys: list[str] | tuple[str, ...] | Mapping[str, torch.Tensor],
    state_digest: Optional[str] = None,
) -> dict[str, Any]:
    """Build a deterministic manifest for a complete canonical state dict."""

    if isinstance(state_keys, Mapping):
        keys = list(state_keys)
        digest = state_digest or state_dict_digest(state_keys)
    else:
        keys = list(state_keys)
        digest = state_digest or _keys_only_digest(keys)
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise BundleValidationError("state_digest must be a lowercase SHA-256 digest")
    return _canonical_manifest(source_paths, keys, digest)


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BundleValidationError(f"{path} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise BundleValidationError(f"{path} keys must be strings")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unexpected={extra}")
        raise BundleValidationError(f"{path} has invalid keys ({', '.join(details)})")


def _require_string(value: object, expected: str, path: str) -> None:
    if value != expected:
        raise BundleValidationError(f"{path} must be {expected!r}, got {value!r}")


def _require_int(value: object, expected: int, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise BundleValidationError(f"{path} must be {expected}, got {value!r}")


def _require_bool(value: object, expected: bool, path: str) -> None:
    if value is not expected:
        raise BundleValidationError(f"{path} must be {expected}, got {value!r}")


def _require_float(value: object, expected: float, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != expected:
        raise BundleValidationError(f"{path} must be {expected}, got {value!r}")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Strictly validate the final bundle manifest and its model contract."""

    root = _require_mapping(manifest, "manifest")
    _require_exact_keys(
        root,
        {
            "schema_version",
            "bundle_type",
            "model_id",
            "weights_id",
            "topology",
            "preprocessing",
            "decision",
            "source_artifacts",
            "state",
            "explainability",
        },
        "manifest",
    )
    _require_string(root["schema_version"], BUNDLE_SCHEMA_VERSION, "manifest.schema_version")
    _require_string(root["bundle_type"], BUNDLE_TYPE, "manifest.bundle_type")
    _require_string(root["model_id"], MODEL_ID, "manifest.model_id")
    if not isinstance(root["weights_id"], str) or not WEIGHTS_ID_PATTERN.fullmatch(root["weights_id"]):
        raise BundleValidationError("manifest.weights_id must be a sha256 identity")

    topology = _require_mapping(root["topology"], "manifest.topology")
    _require_exact_keys(
        topology,
        {"branches", "semantic", "forensic", "fusion", "classifier"},
        "manifest.topology",
    )
    if topology["branches"] != ["semantic", "forensic"]:
        raise BundleValidationError("manifest.topology.branches must be ['semantic', 'forensic']")
    semantic = _require_mapping(topology["semantic"], "manifest.topology.semantic")
    _require_exact_keys(
        semantic,
        {
            "stream",
            "architecture",
            "feature_dim",
            "internal_input_size",
            "internal_interpolation",
            "internal_align_corners",
        },
        "manifest.topology.semantic",
    )
    _require_string(semantic["stream"], "SemanticStream", "manifest.topology.semantic.stream")
    _require_string(semantic["architecture"], "ViT-B/16", "manifest.topology.semantic.architecture")
    _require_int(semantic["feature_dim"], SEMANTIC_DIM, "manifest.topology.semantic.feature_dim")
    if semantic["internal_input_size"] != [224, 224]:
        raise BundleValidationError("manifest.topology.semantic.internal_input_size must be [224, 224]")
    _require_string(
        semantic["internal_interpolation"],
        "bilinear",
        "manifest.topology.semantic.internal_interpolation",
    )
    _require_bool(semantic["internal_align_corners"], False, "manifest.topology.semantic.internal_align_corners")

    forensic = _require_mapping(topology["forensic"], "manifest.topology.forensic")
    _require_exact_keys(
        forensic,
        {"stream", "frontend", "backbone", "feature_dim"},
        "manifest.topology.forensic",
    )
    _require_string(forensic["stream"], "NPRStream", "manifest.topology.forensic.stream")
    _require_string(forensic["frontend"], "bayar_srm", "manifest.topology.forensic.frontend")
    _require_string(forensic["backbone"], "resnet_shallow", "manifest.topology.forensic.backbone")
    _require_int(forensic["feature_dim"], FORENSIC_DIM, "manifest.topology.forensic.feature_dim")

    fusion = _require_mapping(topology["fusion"], "manifest.topology.fusion")
    _require_exact_keys(
        fusion,
        {"architecture", "semantic_dim", "forensic_dim", "fused_dim"},
        "manifest.topology.fusion",
    )
    _require_string(
        fusion["architecture"],
        "concat_linear_layernorm_gelu",
        "manifest.topology.fusion.architecture",
    )
    _require_int(fusion["semantic_dim"], SEMANTIC_DIM, "manifest.topology.fusion.semantic_dim")
    _require_int(fusion["forensic_dim"], FORENSIC_DIM, "manifest.topology.fusion.forensic_dim")
    _require_int(fusion["fused_dim"], FUSED_DIM, "manifest.topology.fusion.fused_dim")

    classifier = _require_mapping(topology["classifier"], "manifest.topology.classifier")
    _require_exact_keys(
        classifier,
        {"architecture", "input_dim", "hidden_dim", "output_dim"},
        "manifest.topology.classifier",
    )
    _require_string(
        classifier["architecture"],
        "linear_gelu_linear",
        "manifest.topology.classifier.architecture",
    )
    _require_int(classifier["input_dim"], FUSED_DIM, "manifest.topology.classifier.input_dim")
    _require_int(classifier["hidden_dim"], 128, "manifest.topology.classifier.hidden_dim")
    _require_int(classifier["output_dim"], 1, "manifest.topology.classifier.output_dim")

    preprocessing = _require_mapping(root["preprocessing"], "manifest.preprocessing")
    _require_exact_keys(
        preprocessing,
        {"shared_resize", "resize", "decode_color", "dtype", "semantic", "forensic"},
        "manifest.preprocessing",
    )
    _require_bool(preprocessing["shared_resize"], True, "manifest.preprocessing.shared_resize")
    resize = _require_mapping(preprocessing["resize"], "manifest.preprocessing.resize")
    _require_exact_keys(
        resize,
        {"backend", "size", "interpolation", "resample"},
        "manifest.preprocessing.resize",
    )
    _require_string(resize["backend"], "Pillow", "manifest.preprocessing.resize.backend")
    if resize["size"] != [IMAGE_SIZE, IMAGE_SIZE]:
        raise BundleValidationError("manifest.preprocessing.resize.size must be [512, 512]")
    _require_string(resize["interpolation"], "bilinear", "manifest.preprocessing.resize.interpolation")
    _require_string(resize["resample"], "BILINEAR", "manifest.preprocessing.resize.resample")
    _require_string(preprocessing["decode_color"], "RGB", "manifest.preprocessing.decode_color")
    _require_string(preprocessing["dtype"], "float32", "manifest.preprocessing.dtype")
    semantic_prep = _require_mapping(preprocessing["semantic"], "manifest.preprocessing.semantic")
    _require_exact_keys(
        semantic_prep,
        {"normalization", "pixel_source"},
        "manifest.preprocessing.semantic",
    )
    normalization = _require_mapping(
        semantic_prep["normalization"],
        "manifest.preprocessing.semantic.normalization",
    )
    _require_exact_keys(
        normalization,
        {"mean", "std"},
        "manifest.preprocessing.semantic.normalization",
    )
    if normalization["mean"] != list(IMAGENET_MEAN) or normalization["std"] != list(IMAGENET_STD):
        raise BundleValidationError("manifest.preprocessing.semantic.normalization does not match ImageNet constants")
    _require_string(
        semantic_prep["pixel_source"],
        "shared_resized_pixels",
        "manifest.preprocessing.semantic.pixel_source",
    )
    forensic_prep = _require_mapping(preprocessing["forensic"], "manifest.preprocessing.forensic")
    _require_exact_keys(forensic_prep, {"pixel_range", "pixel_source"}, "manifest.preprocessing.forensic")
    if forensic_prep["pixel_range"] != [0.0, 1.0]:
        raise BundleValidationError("manifest.preprocessing.forensic.pixel_range must be [0.0, 1.0]")
    _require_string(
        forensic_prep["pixel_source"],
        "shared_resized_pixels",
        "manifest.preprocessing.forensic.pixel_source",
    )

    decision = _require_mapping(root["decision"], "manifest.decision")
    _require_exact_keys(decision, {"threshold", "equality"}, "manifest.decision")
    _require_float(decision["threshold"], DECISION_THRESHOLD, "manifest.decision.threshold")
    _require_string(decision["equality"], "greater_or_equal", "manifest.decision.equality")

    source_artifacts = _require_mapping(root["source_artifacts"], "manifest.source_artifacts")
    _require_exact_keys(source_artifacts, set(_SOURCE_ROLES), "manifest.source_artifacts")
    for role in _SOURCE_ROLES:
        artifact = _require_mapping(source_artifacts[role], f"manifest.source_artifacts.{role}")
        _require_exact_keys(
            artifact,
            {"filename", "size_bytes", "sha256"},
            f"manifest.source_artifacts.{role}",
        )
        filename = artifact["filename"]
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise BundleValidationError(f"manifest.source_artifacts.{role}.filename must be a basename")
        size = artifact["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise BundleValidationError(
                f"manifest.source_artifacts.{role}.size_bytes must be a positive integer"
            )
        digest = artifact["sha256"]
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise BundleValidationError(
                f"manifest.source_artifacts.{role}.sha256 must be a lowercase SHA-256 digest"
            )
    state = _require_mapping(root["state"], "manifest.state")
    _require_exact_keys(state, {"format", "keys", "embedded_sha256"}, "manifest.state")
    _require_string(state["format"], "torch_state_dict", "manifest.state.format")
    state_keys = state["keys"]
    if (
        not isinstance(state_keys, list)
        or not state_keys
        or any(not isinstance(key, str) for key in state_keys)
    ):
        raise BundleValidationError("manifest.state.keys must be a non-empty list of strings")
    if state_keys != sorted(set(state_keys)):
        raise BundleValidationError("manifest.state.keys must be sorted and unique")
    if not isinstance(state["embedded_sha256"], str) or not SHA256_PATTERN.fullmatch(state["embedded_sha256"]):
        raise BundleValidationError("manifest.state.embedded_sha256 must be a lowercase SHA-256 digest")
    if root["weights_id"] != _weights_id(source_artifacts, state["embedded_sha256"]):
        raise BundleValidationError(
            "manifest.weights_id does not match source hashes and embedded state digest"
        )

    explainability = _require_mapping(root["explainability"], "manifest.explainability")
    _require_exact_keys(
        explainability,
        {"branch_names", "attribution_targets", "intermediate_representations", "unsupported_capabilities"},
        "manifest.explainability",
    )
    if explainability["branch_names"] != ["semantic", "forensic"]:
        raise BundleValidationError("manifest.explainability.branch_names must be ['semantic', 'forensic']")
    targets = _require_mapping(
        explainability["attribution_targets"],
        "manifest.explainability.attribution_targets",
    )
    _require_exact_keys(
        targets,
        {"semantic", "forensic"},
        "manifest.explainability.attribution_targets",
    )
    _require_string(
        targets["semantic"],
        "semantic_stream.backbone.encoder.layers.encoder_layer_11.ln_1",
        "manifest.explainability.attribution_targets.semantic",
    )
    _require_string(
        targets["forensic"],
        "forensic_stream.backbone.4.2.conv3",
        "manifest.explainability.attribution_targets.forensic",
    )
    representations = _require_mapping(
        explainability["intermediate_representations"],
        "manifest.explainability.intermediate_representations",
    )
    _require_exact_keys(
        representations,
        {"semantic", "forensic"},
        "manifest.explainability.intermediate_representations",
    )
    if representations["semantic"] != ["backbone.encoder.layers.encoder_layer_11.ln_1"]:
        raise BundleValidationError("manifest.explainability.intermediate_representations.semantic is invalid")
    if representations["forensic"] != ["frontend.bayar", "frontend.srm", "frontend.fuse", "backbone.4", "pool"]:
        raise BundleValidationError("manifest.explainability.intermediate_representations.forensic is invalid")
    unsupported = _require_mapping(
        explainability["unsupported_capabilities"],
        "manifest.explainability.unsupported_capabilities",
    )
    _require_exact_keys(
        unsupported,
        {"attention_tensors", "branch_subset_logits"},
        "manifest.explainability.unsupported_capabilities",
    )
    for name, reason in unsupported.items():
        if not isinstance(reason, str) or not reason.strip():
            raise BundleValidationError(
                f"manifest.explainability.unsupported_capabilities.{name} must be a non-empty reason"
            )


def resolve_module_path(module: nn.Module, dotted_path: str) -> nn.Module:
    """Resolve a registered child-module path without allowing arbitrary attrs."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    if not isinstance(dotted_path, str) or not dotted_path.strip():
        raise BundleValidationError("module path must be a non-empty string")
    current = module
    for component in dotted_path.split("."):
        if not component or component in {".", ".."}:
            raise BundleValidationError(f"invalid module path: {dotted_path!r}")
        child = current._modules.get(component)
        if child is None:
            raise BundleValidationError(
                f"module path {dotted_path!r} does not exist at {component!r}"
            )
        current = child
    return current


def validate_explainability_contract(
    detector: nn.Module, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve declared targets and representations against a final detector.

    The returned module references are intended for the Wave 4 adapter.  This
    check deliberately does not run the model or synthesize unsupported
    attention/ablation outputs.
    """

    if not isinstance(detector, nn.Module):
        raise TypeError("detector must be a torch.nn.Module")
    validate_manifest(manifest)
    explainability = _require_mapping(manifest["explainability"], "manifest.explainability")
    targets = _require_mapping(
        explainability["attribution_targets"],
        "manifest.explainability.attribution_targets",
    )
    representations = _require_mapping(
        explainability["intermediate_representations"],
        "manifest.explainability.intermediate_representations",
    )
    resolved_targets: dict[str, nn.Module] = {}
    resolved_representations: dict[str, dict[str, nn.Module]] = {}
    for branch_name in ("semantic", "forensic"):
        branch = detector._modules.get(f"{branch_name}_stream")
        if branch is None:
            raise BundleValidationError(
                f"canonical detector is missing the {branch_name}_stream module"
            )
        target_path = targets[branch_name]
        resolved_targets[branch_name] = resolve_module_path(detector, target_path)
        branch_representations: dict[str, nn.Module] = {}
        for relative_path in representations[branch_name]:
            branch_representations[relative_path] = resolve_module_path(branch, relative_path)
        resolved_representations[branch_name] = branch_representations
    return {
        "attribution_targets": resolved_targets,
        "intermediate_representations": resolved_representations,
    }


def _load_checkpoint_object(path: str | Path, role: str) -> object:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - torch's deserializer varies by version
        raise BundleValidationError(f"could not load {role} checkpoint {path}: {exc}") from exc
    return value


def _state_dict_from_checkpoint(path: str | Path, role: str) -> Mapping[str, torch.Tensor]:
    value = _load_checkpoint_object(path, role)
    if isinstance(value, Mapping) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, Mapping) or not value:
        raise BundleValidationError(f"{role} checkpoint must be a non-empty state-dict mapping")
    if any(not isinstance(key, str) or not isinstance(tensor, torch.Tensor) for key, tensor in value.items()):
        raise BundleValidationError(f"{role} checkpoint contains non-tensor state entries")
    return value


def _load_strict(module: nn.Module, state: Mapping[str, torch.Tensor], role: str) -> None:
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise BundleValidationError(f"strict {role} checkpoint loading failed: {exc}") from exc


def _detector_from_source_states(
    semantic_state: Mapping[str, torch.Tensor],
    forensic_state: Mapping[str, torch.Tensor],
    fusion_state: Mapping[str, torch.Tensor],
    classifier_state: Mapping[str, torch.Tensor],
) -> CanonicalFusedDetector:
    """Construct a detector from the four independently published state maps."""

    detector = CanonicalFusedDetector()
    _load_strict(detector.semantic_stream, semantic_state, "semantic")
    _load_strict(detector.forensic_stream, forensic_state, "forensic")
    _load_strict(detector.fusion, fusion_state, "fusion")
    _load_strict(detector.classifier, classifier_state, "classifier")
    return detector


def verify_three_file_parity(
    bundle_detector: CanonicalFusedDetector,
    semantic_checkpoint: str | Path,
    forensic_checkpoint: str | Path,
    fusion_checkpoint: str | Path,
    parity_image: RawImage,
) -> tuple[float, float]:
    """Compare a bundled detector with an independently loaded three-file scorer.

    This check intentionally constructs a second model from the source files so
    it cannot pass merely because the bundle is compared with itself.  The same
    canonical source-image preparation is used for both models.
    """

    semantic_state = _state_dict_from_checkpoint(semantic_checkpoint, "semantic")
    forensic_state = _state_dict_from_checkpoint(forensic_checkpoint, "forensic")
    fusion_payload = _require_mapping(_load_checkpoint_object(fusion_checkpoint, "fusion"), "fusion checkpoint")
    if set(fusion_payload) != {"fusion", "classifier"}:
        raise BundleValidationError(
            "fusion checkpoint must contain exactly 'fusion' and 'classifier' state dictionaries"
        )
    fusion_state = _require_mapping(fusion_payload["fusion"], "fusion checkpoint.fusion")
    classifier_state = _require_mapping(fusion_payload["classifier"], "fusion checkpoint.classifier")
    reference = _detector_from_source_states(semantic_state, forensic_state, fusion_state, classifier_state)
    prepared = prepare_fused_inputs(parity_image)
    bundle_detector.eval()
    reference.eval()
    with torch.inference_mode():
        bundled_logit = bundle_detector(prepared)
        reference_logit = reference(prepared)
    try:
        torch.testing.assert_close(bundled_logit, reference_logit, rtol=1e-6, atol=1e-7)
    except AssertionError as exc:
        raise BundleValidationError(f"bundle parity check failed: {exc}") from exc
    return float(bundled_logit.item()), float(reference_logit.item())


def build_checkpoint_bundle(
    semantic_checkpoint: str | Path,
    forensic_checkpoint: str | Path,
    fusion_checkpoint: str | Path,
    output_path: str | Path,
    parity_image: Optional[RawImage] = None,
) -> Path:
    """Build and write a complete canonical detector bundle.

    The three input files are read only to construct and validate the complete
    detector state.  They are recorded as provenance in the manifest and are
    never needed by the bundle loader.
    """

    semantic_path = Path(semantic_checkpoint)
    forensic_path = Path(forensic_checkpoint)
    fusion_path = Path(fusion_checkpoint)
    destination = Path(output_path)
    source_resolved = {path.resolve() for path in (semantic_path, forensic_path, fusion_path)}
    if destination.resolve() in source_resolved:
        raise BundleValidationError("output bundle must not overwrite a source checkpoint")

    semantic_state = _state_dict_from_checkpoint(semantic_path, "semantic")
    forensic_state = _state_dict_from_checkpoint(forensic_path, "forensic")
    fusion_payload = _load_checkpoint_object(fusion_path, "fusion")
    fusion_payload = _require_mapping(fusion_payload, "fusion checkpoint")
    if set(fusion_payload) != {"fusion", "classifier"}:
        raise BundleValidationError(
            "fusion checkpoint must contain exactly 'fusion' and 'classifier' state dictionaries"
        )
    fusion_state = _require_mapping(fusion_payload["fusion"], "fusion checkpoint.fusion")
    classifier_state = _require_mapping(fusion_payload["classifier"], "fusion checkpoint.classifier")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in fusion_state.items()):
        raise BundleValidationError("fusion checkpoint.fusion contains non-tensor state entries")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in classifier_state.items()):
        raise BundleValidationError("fusion checkpoint.classifier contains non-tensor state entries")

    detector = _detector_from_source_states(semantic_state, forensic_state, fusion_state, classifier_state)
    state_dict = detector.state_dict()
    source_paths = {
        "semantic_stream": semantic_path,
        "forensic_stream": forensic_path,
        "detector_fusion": fusion_path,
    }
    if parity_image is not None:
        verify_three_file_parity(
            detector,
            semantic_path,
            forensic_path,
            fusion_path,
            parity_image,
        )
    manifest = build_manifest(source_paths, state_dict)
    validate_explainability_contract(detector, manifest)
    payload = {"manifest": manifest, "state_dict": state_dict}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return destination


def _verify_source_files(manifest: Mapping[str, Any], source_dir: str | Path) -> None:
    root = Path(source_dir).resolve()
    source_artifacts = _require_mapping(manifest["source_artifacts"], "manifest.source_artifacts")
    for role in _SOURCE_ROLES:
        artifact = _require_mapping(source_artifacts[role], f"manifest.source_artifacts.{role}")
        filename = artifact["filename"]
        candidate = (root / filename).resolve()
        if candidate.parent != root:
            raise BundleValidationError(f"source artifact {role} escapes source_dir")
        if not candidate.is_file():
            raise FileNotFoundError(f"manifest source artifact is missing: {candidate}")
        expected_size = artifact["size_bytes"]
        expected_hash = artifact["sha256"]
        actual_size = candidate.stat().st_size
        actual_hash = sha256_file(candidate)
        if actual_size != expected_size or actual_hash != expected_hash:
            raise BundleValidationError(
                f"source artifact hash mismatch for {role}: "
                f"expected size/hash {expected_size}/{expected_hash}, "
                f"got {actual_size}/{actual_hash}"
            )


def load_checkpoint_bundle(
    bundle_path: str | Path,
    map_location: str | torch.device = "cpu",
    source_dir: Optional[str | Path] = None,
) -> tuple[CanonicalFusedDetector, dict[str, Any]]:
    """Strictly load a self-describing canonical detector bundle.

    ``source_dir`` is optional and only verifies provenance hashes when
    supplied; standalone source files are not loaded to run inference.
    """

    path = Path(bundle_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint bundle is not a file: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # pragma: no cover - torch's deserializer varies by version
        raise BundleValidationError(f"could not load checkpoint bundle {path}: {exc}") from exc
    return load_checkpoint_bundle_payload(payload, source_dir=source_dir, map_location=map_location)


def load_checkpoint_bundle_payload(
    payload: Mapping[str, Any],
    source_dir: Optional[str | Path] = None,
    map_location: str | torch.device = "cpu",
) -> tuple[CanonicalFusedDetector, dict[str, Any]]:
    """Load a bundle payload, exposed for deterministic in-memory tests."""

    root = _require_mapping(payload, "bundle")
    _require_exact_keys(root, set(_TOP_LEVEL_KEYS), "bundle")
    manifest = _require_mapping(root["manifest"], "bundle.manifest")
    validate_manifest(manifest)
    if source_dir is not None:
        _verify_source_files(manifest, source_dir)

    state = _require_mapping(root["state_dict"], "bundle.state_dict")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise BundleValidationError("bundle.state_dict must contain only string keys and tensors")
    manifest_keys = manifest["state"]["keys"]
    state_keys = sorted(state)
    if state_keys != manifest_keys:
        raise BundleValidationError("bundle.state_dict keys do not match manifest.state.keys")
    expected_digest = manifest["state"]["embedded_sha256"]
    actual_digest = state_dict_digest(state)
    if actual_digest != expected_digest:
        raise BundleValidationError(
            "bundle.state_dict values do not match manifest.state.embedded_sha256: "
            f"expected {expected_digest}, got {actual_digest}"
        )

    try:
        device = torch.device(map_location)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise BundleValidationError(f"invalid map_location {map_location!r}") from exc
    detector = CanonicalFusedDetector().to(device)
    _load_strict(detector, state, "canonical detector")
    validate_explainability_contract(detector, manifest)
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    detector.eval()
    return detector, copy.deepcopy(dict(manifest))


# Concise aliases for callers that describe this as a final-model loader.
load_bundle = load_checkpoint_bundle
build_bundle = build_checkpoint_bundle


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "BUNDLE_TYPE",
    "MODEL_ID",
    "DECISION_THRESHOLD",
    "BundleValidationError",
    "sha256_file",
    "state_dict_digest",
    "build_manifest",
    "validate_manifest",
    "resolve_module_path",
    "validate_explainability_contract",
    "verify_three_file_parity",
    "build_checkpoint_bundle",
    "build_bundle",
    "load_checkpoint_bundle",
    "load_checkpoint_bundle_payload",
    "load_bundle",
]

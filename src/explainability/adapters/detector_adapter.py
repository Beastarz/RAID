"""Explainability boundary for the canonical published fused detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import torch
from torch import nn

from src.explainability.contracts import (
    AdapterPrediction,
    AttributionTarget,
    BranchCoalitionLogit,
    Capability,
    CapabilityResult,
    CapabilityStatus,
    PreparedModelInputs,
)
from src.models.checkpoint_bundle import (
    BUNDLE_SCHEMA_VERSION,
    BUNDLE_TYPE,
    DECISION_THRESHOLD,
    MODEL_ID,
    BundleValidationError,
    load_checkpoint_bundle,
    validate_explainability_contract,
)
from src.models.fused_detector import (
    FORENSIC_DIM,
    FUSED_DIM,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEMANTIC_DIM,
    CanonicalFusedDetector,
    PreparedFusedInputs,
    prepare_fused_inputs,
)


def vit_token_grid(value: torch.Tensor) -> torch.Tensor:
    """Convert the final ViT target's ``[B,197,C]`` tokens to ``[B,C,14,14]``."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("ViT target activation must be a tensor")
    if value.ndim != 3 or value.shape[1] != 197:
        raise ValueError("ViT target activation must have shape [B, 197, C]")
    patches = value[:, 1:, :]
    return patches.transpose(1, 2).reshape(value.shape[0], value.shape[2], 14, 14)


@dataclass(frozen=True)
class DetectorAttributionTarget:
    """Opaque target payload used by generic Grad-CAM and attribution code."""

    branch_name: str
    module: nn.Module
    scoring_callable: Callable[[], torch.Tensor]
    activation_transform: Callable[[torch.Tensor], torch.Tensor] | None
    coordinate_space: str


@dataclass(frozen=True)
class IntermediateRepresentation:
    """A raw internal tensor plus the metadata required to interpret it safely."""

    value: torch.Tensor
    module_path: str
    coordinate_space: str
    raw_scale: str


class DetectorExplainabilityAdapter:
    """Strict adapter for the final semantic + forensic detector bundle."""

    def __init__(
        self,
        bundle_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        source_dir: str | Path | None = None,
    ) -> None:
        try:
            resolved_device = torch.device(device)
            if resolved_device.type == "cuda" and resolved_device.index is None:
                # A bare "cuda" device normalizes to "cuda:0" once a tensor is
                # actually moved onto it via .to(); resolve the index up front
                # so later device equality checks compare like with like.
                resolved_device = torch.device("cuda", torch.cuda.current_device())
            self.device = resolved_device
        except (TypeError, RuntimeError, ValueError) as exc:
            raise BundleValidationError(f"invalid adapter device {device!r}") from exc
        model, manifest = load_checkpoint_bundle(
            bundle_path, map_location=self.device, source_dir=source_dir
        )
        self._validate_final_contract(model, manifest)
        self.model = model
        self.manifest = manifest
        resolved = validate_explainability_contract(model, manifest)
        self._targets: Mapping[str, nn.Module] = MappingProxyType(
            dict(resolved["attribution_targets"])
        )
        self._representations: Mapping[str, Mapping[str, nn.Module]] = MappingProxyType(
            {
                branch: MappingProxyType(dict(values))
                for branch, values in resolved["intermediate_representations"].items()
            }
        )
        unsupported = manifest["explainability"]["unsupported_capabilities"]
        self._capabilities = MappingProxyType(
            {
                Capability.PREDICTION: CapabilityStatus.supported(),
                Capability.ATTRIBUTION_TARGETS: CapabilityStatus.supported(),
                Capability.ATTENTION_TENSORS: CapabilityStatus.unsupported(
                    unsupported["attention_tensors"]
                ),
                Capability.INTERMEDIATE_REPRESENTATIONS: CapabilityStatus.supported(),
                Capability.BRANCH_SUBSET_LOGITS: CapabilityStatus.unsupported(
                    unsupported["branch_subset_logits"]
                ),
            }
        )

    @staticmethod
    def _validate_final_contract(
        model: CanonicalFusedDetector, manifest: Mapping[str, object]
    ) -> None:
        topology = manifest["topology"]
        preprocessing = manifest["preprocessing"]
        decision = manifest["decision"]
        expected = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_type": BUNDLE_TYPE,
            "model_id": MODEL_ID,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise BundleValidationError(f"adapter requires {key}={value!r}")
        if topology["branches"] != ["semantic", "forensic"]:
            raise BundleValidationError("adapter requires semantic + forensic topology")
        semantic = topology["semantic"]
        forensic = topology["forensic"]
        fusion = topology["fusion"]
        if semantic["architecture"] != "ViT-B/16" or semantic["feature_dim"] != SEMANTIC_DIM:
            raise BundleValidationError("adapter requires the published ViT-B/16 semantic branch")
        if (
            forensic["frontend"] != "bayar_srm"
            or forensic["backbone"] != "resnet_shallow"
            or forensic["feature_dim"] != FORENSIC_DIM
        ):
            raise BundleValidationError("adapter requires the Bayar+SRM shallow-ResNet branch")
        if fusion["fused_dim"] != FUSED_DIM:
            raise BundleValidationError("adapter requires the published fusion dimensions")
        if (
            preprocessing["shared_resize"] is not True
            or preprocessing["resize"]["backend"] != "Pillow"
            or preprocessing["resize"]["size"] != [IMAGE_SIZE, IMAGE_SIZE]
            or preprocessing["resize"]["interpolation"] != "bilinear"
            or preprocessing["semantic"]["normalization"]["mean"] != list(IMAGENET_MEAN)
            or preprocessing["semantic"]["normalization"]["std"] != list(IMAGENET_STD)
            or preprocessing["forensic"]["pixel_range"] != [0.0, 1.0]
        ):
            raise BundleValidationError("adapter requires the canonical shared preprocessing")
        if decision != {"threshold": DECISION_THRESHOLD, "equality": "greater_or_equal"}:
            raise BundleValidationError("adapter requires the published decision policy")
        if not isinstance(model, CanonicalFusedDetector):
            raise BundleValidationError("bundle did not load a canonical fused detector")

    @property
    def capabilities(self) -> Mapping[Capability, CapabilityStatus]:
        return self._capabilities

    @property
    def report_metadata(self) -> dict[str, object]:
        """Return JSON-compatible identity and preprocessing facts for reports."""

        return {
            "model_id": self.manifest["model_id"],
            "weights_id": self.manifest["weights_id"],
            "bundle_schema_version": self.manifest["schema_version"],
            "preprocessing": self.manifest["preprocessing"],
        }

    def prepare_source_image(self, raw_source_image: object) -> PreparedModelInputs:
        prepared = prepare_fused_inputs(raw_source_image)  # type: ignore[arg-type]
        values = PreparedFusedInputs(
            semantic=prepared.semantic.to(self.device),
            forensic=prepared.forensic.to(self.device),
            original_size=prepared.original_size,
            resized_size=prepared.resized_size,
            interpolation=prepared.interpolation,
        )
        context = {
            "weights_id": self.manifest["weights_id"],
            "original_size": list(prepared.original_size),
            "resized_size": list(prepared.resized_size),
            "resize_backend": "Pillow",
            "interpolation": prepared.interpolation,
            "semantic_normalization": {
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
            },
            "forensic_pixel_range": [0.0, 1.0],
            "forensic_frontend": "bayar_srm",
        }
        return PreparedModelInputs(values=values, context=context)

    def _prepared_values(self, prepared: PreparedModelInputs) -> PreparedFusedInputs:
        if not isinstance(prepared, PreparedModelInputs):
            raise TypeError("prepared_model_inputs must be PreparedModelInputs")
        if not isinstance(prepared.values, PreparedFusedInputs):
            raise TypeError("prepared inputs were not created by the detector adapter")
        if not isinstance(prepared.context, Mapping) or prepared.context.get("weights_id") != self.manifest["weights_id"]:
            raise ValueError("prepared inputs do not match this detector bundle")
        if prepared.values.semantic.device != self.device or prepared.values.forensic.device != self.device:
            raise ValueError("prepared inputs are not on the adapter device")
        return prepared.values

    def score_logits(
        self, prepared_model_inputs: PreparedModelInputs, *, require_input_grad: bool = False
    ) -> torch.Tensor:
        values = self._prepared_values(prepared_model_inputs)
        if require_input_grad:
            semantic = values.semantic.detach().clone().requires_grad_(True)
            forensic = values.forensic.detach().clone().requires_grad_(True)
            return self.model(semantic, forensic)
        return self.model(values)

    def score_raw_image(self, raw_source_image: object) -> float:
        prepared = self.prepare_source_image(raw_source_image)
        with torch.inference_mode():
            return float(self.score_logits(prepared).reshape(-1)[0].item())

    def predict(self, prepared_model_inputs: PreparedModelInputs) -> AdapterPrediction:
        with torch.inference_mode():
            logits = self.score_logits(prepared_model_inputs)
        if logits.numel() != 1:
            raise ValueError("AdapterPrediction requires exactly one prepared image")
        logit = float(logits.item())
        probability = float(torch.sigmoid(logits).item())
        threshold = float(self.manifest["decision"]["threshold"])
        return AdapterPrediction(logit, probability, int(probability >= threshold), threshold)

    def attribution_targets(
        self, prepared_model_inputs: PreparedModelInputs
    ) -> CapabilityResult[Mapping[str, AttributionTarget]]:
        self._prepared_values(prepared_model_inputs)
        result: dict[str, AttributionTarget] = {}
        for branch, module in self._targets.items():
            payload = DetectorAttributionTarget(
                branch_name=branch,
                module=module,
                scoring_callable=lambda p=prepared_model_inputs: self.score_logits(
                    p, require_input_grad=True
                ),
                activation_transform=vit_token_grid if branch == "semantic" else None,
                coordinate_space="semantic_patch_grid" if branch == "semantic" else "forensic_feature_grid",
            )
            result[branch] = AttributionTarget(payload)
        return CapabilityResult.supported(MappingProxyType(result))

    def attention_tensors(
        self, prepared_model_inputs: PreparedModelInputs
    ) -> CapabilityResult[tuple[object, ...]]:
        self._prepared_values(prepared_model_inputs)
        reason = self._capabilities[Capability.ATTENTION_TENSORS].reason
        assert reason is not None
        return CapabilityResult.unsupported(reason)

    def intermediate_representations(
        self, prepared_model_inputs: PreparedModelInputs
    ) -> CapabilityResult[Mapping[str, Mapping[str, object]]]:
        self._prepared_values(prepared_model_inputs)
        captured: dict[str, dict[str, IntermediateRepresentation]] = {
            "semantic": {},
            "forensic": {},
        }
        handles: list[torch.utils.hooks.RemovableHandle] = []
        training_states = [(module, module.training) for module in self.model.modules()]

        def hook(branch: str, path: str):
            def capture(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"intermediate {branch}.{path} did not return a tensor")
                captured[branch][path] = IntermediateRepresentation(
                    value=output.detach().clone(),
                    module_path=path,
                    coordinate_space=("semantic_token_grid" if branch == "semantic" else "forensic_internal"),
                    raw_scale="model_activation",
                )
            return capture

        try:
            for branch, representations in self._representations.items():
                for path, module in representations.items():
                    handles.append(module.register_forward_hook(hook(branch, path)))
            self.model.eval()
            with torch.inference_mode():
                self.score_logits(prepared_model_inputs)
        finally:
            for handle in handles:
                handle.remove()
            for module, training in training_states:
                module.train(training)
        expected = {branch: set(paths) for branch, paths in self._representations.items()}
        if any(set(captured[branch]) != paths for branch, paths in expected.items()):
            raise RuntimeError("not every declared intermediate representation was captured")
        return CapabilityResult.supported(
            MappingProxyType(
                {branch: MappingProxyType(dict(values)) for branch, values in captured.items()}
            )
        )

    def branch_subset_logits(
        self, prepared_model_inputs: PreparedModelInputs
    ) -> CapabilityResult[tuple[BranchCoalitionLogit, ...]]:
        self._prepared_values(prepared_model_inputs)
        reason = self._capabilities[Capability.BRANCH_SUBSET_LOGITS].reason
        assert reason is not None
        return CapabilityResult.unsupported(reason)


__all__ = [
    "DetectorAttributionTarget",
    "DetectorExplainabilityAdapter",
    "IntermediateRepresentation",
    "vit_token_grid",
]

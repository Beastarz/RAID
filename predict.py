"""Single-image and batch inference for the canonical final detector."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.explainability.adapters.detector_adapter import (
    DetectorAttributionTarget,
    DetectorExplainabilityAdapter,
    IntermediateRepresentation,
    vit_token_grid,
)
from src.explainability.artifacts import ArtifactStore
from src.explainability.contracts import (
    CapabilityStatus,
    ExplanationOutputEnvelope,
    ExplanationResult,
)
from src.explainability.gradcam import grad_cam
from src.explainability.attribution import integrated_gradients
from src.explainability.faithfulness import deletion_insertion
from src.explainability.rendering import colorize_heatmap
from src.explainability.serialization import write_envelope_json
from src.models.checkpoint_bundle import load_checkpoint_bundle
from src.models.fused_detector import (
    CanonicalFusedDetector,
    IMAGE_SIZE,
    PreparedFusedInputs,
    prepare_fused_inputs,
)


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXPLANATION_METHODS = (
    "none", "semantic-gradcam", "forensic-gradcam", "intermediates", "attention",
    "semantic-integrated-gradients", "forensic-gradcam-faithfulness",
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_inputs(image_path: Path) -> PreparedFusedInputs:
    """Load one source image using the canonical shared-resize contract."""

    return prepare_fused_inputs(image_path)


def load_image_tensor(image_path: Path) -> torch.Tensor:
    """Compatibility helper returning the semantic view of shared inputs."""

    return load_image_inputs(image_path).semantic


def build_model(checkpoint: Optional[str], device: torch.device) -> CanonicalFusedDetector:
    """Load the strict final-model bundle onto ``device``.

    Randomly initialized fallback models are deliberately unsupported in normal
    inference: a missing or malformed bundle must fail loudly rather than
    produce an apparently meaningful prediction.
    """

    if not checkpoint:
        raise ValueError("--checkpoint must point to a canonical detector bundle")
    model, _manifest = load_checkpoint_bundle(checkpoint, map_location=device)
    return model


def build_adapter(checkpoint: Optional[str], device: torch.device) -> DetectorExplainabilityAdapter:
    if not checkpoint:
        raise ValueError("--checkpoint must point to a canonical detector bundle")
    return DetectorExplainabilityAdapter(checkpoint, device=device)


def predict_single(
    model: torch.nn.Module | DetectorExplainabilityAdapter,
    image_path: Path,
    device: torch.device,
    bayar_mode: bool = False,
) -> Dict[str, object]:
    """Score one source image using one shared 512x512 preparation.

    ``bayar_mode`` remains accepted for source compatibility but no longer
    selects a separate preprocessing path.
    """

    del bayar_mode
    start = time.perf_counter()
    if isinstance(model, DetectorExplainabilityAdapter):
        prediction = model.predict(model.prepare_source_image(image_path))
        prob = prediction.predicted_probability
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "filename": str(image_path),
            "ai_probability": round(prob, 6),
            "label": "AI-Generated" if prob >= 0.5 else "Authentic",
            "execution_time_ms": round(elapsed_ms, 3),
        }
    prepared = load_image_inputs(image_path)
    prepared = PreparedFusedInputs(
        semantic=prepared.semantic.to(device),
        forensic=prepared.forensic.to(device),
        original_size=prepared.original_size,
        resized_size=prepared.resized_size,
        interpolation=prepared.interpolation,
    )
    with torch.no_grad():
        if isinstance(model, CanonicalFusedDetector):
            prob = float(torch.sigmoid(model(prepared)).item())
        else:
            raise TypeError("predict_single requires a CanonicalFusedDetector loaded from a bundle")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "filename": str(image_path),
        "ai_probability": round(prob, 6),
        "label": "AI-Generated" if prob >= 0.5 else "Authentic",
        "execution_time_ms": round(elapsed_ms, 3),
    }


def _storage_id(image_path: Path) -> str:
    digest = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:12]
    stem = "".join(character if character.isalnum() or character in "._-" else "-" for character in image_path.stem)
    stem = stem.strip(".-") or "image"
    return f"{stem}-{digest}"


def _scalar_representation(value: torch.Tensor) -> torch.Tensor:
    tensor = value.detach().cpu()
    if tensor.ndim == 4:
        return tensor[0].abs().mean(dim=0)
    if tensor.ndim == 3 and tensor.shape[1] == 197:
        return vit_token_grid(tensor)[0].abs().mean(dim=0)
    if tensor.ndim == 3:
        return tensor[0].abs().mean(dim=0, keepdim=True)
    if tensor.ndim == 2:
        return tensor[0].abs().reshape(1, -1)
    raise ValueError(f"cannot render intermediate tensor with shape {tuple(tensor.shape)}")


def explain_single(
    adapter: DetectorExplainabilityAdapter,
    image_path: Path,
    method: str,
    output_directory: Path,
    *,
    ig_steps: int = 32,
    faithfulness_steps: int = 8,
    faithfulness_patch_size: int = 64,
) -> Path:
    """Write one strict explanation envelope and its lossless artifacts."""

    if method not in EXPLANATION_METHODS or method == "none":
        raise ValueError("method must select a supported or explicitly unsupported explanation")
    prepared = adapter.prepare_source_image(image_path)
    sample_id = _storage_id(image_path)
    store = ArtifactStore(output_directory)
    artifacts = []
    status = CapabilityStatus.supported()
    statistics: dict[str, object] = {}
    if method in {"semantic-gradcam", "forensic-gradcam"}:
        branch = method.split("-", 1)[0]
        target_result = adapter.attribution_targets(prepared)
        assert target_result.value is not None
        payload = target_result.value[branch].value
        assert isinstance(payload, DetectorAttributionTarget)
        result = grad_cam(
            adapter.model,
            payload.module,
            payload.scoring_callable,
            lambda value: value,
            activation_transform=payload.activation_transform,
        )
        raw = result.heatmap[0]
        rendered = colorize_heatmap(
            raw,
            coordinate_space=payload.coordinate_space,
            method_name=method,
            colormap="turbo",
            lower_percentile=0.0,
            upper_percentile=100.0,
        )
        artifacts.append(store.save_png(sample_id, method, rendered, artifact_type="gradcam_visualization"))
        artifacts.append(store.save_npy(
            sample_id, f"{method}-raw", raw, artifact_type="gradcam_data",
            coordinate_space=payload.coordinate_space, method_name=method,
            metadata={"raw_scale": "gradcam_relu_weighted_activation"},
        ))
        statistics["selected_logit"] = float(result.selected_logits[0].item())
    elif method == "semantic-integrated-gradients":
        values = prepared.values
        result = integrated_gradients(
            values.semantic,
            lambda semantic: adapter.model(semantic, values.forensic),
            lambda value: value,
            baseline=torch.zeros_like(values.semantic),
            steps=ig_steps,
        )
        raw = result.attribution[0].abs().sum(dim=0)
        rendered = colorize_heatmap(
            raw, coordinate_space="image", method_name=method, colormap="magma",
            lower_percentile=0.0, upper_percentile=100.0,
        )
        artifacts.append(store.save_png(sample_id, method, rendered, artifact_type="attribution_visualization"))
        artifacts.append(store.save_npy(
            sample_id, f"{method}-raw", result.attribution[0],
            artifact_type="attribution_data", coordinate_space="image", method_name=method,
            metadata={"raw_scale": "signed_integrated_gradient", "steps": ig_steps},
        ))
        statistics.update({
            "steps": ig_steps,
            "input_logit": float(result.input_logits[0].item()),
            "baseline_logit": float(result.baseline_logits[0].item()),
            "completeness_delta": float(result.completeness_delta[0].item()),
        })
    elif method == "forensic-gradcam-faithfulness":
        targets = adapter.attribution_targets(prepared)
        assert targets.value is not None
        payload = targets.value["forensic"].value
        assert isinstance(payload, DetectorAttributionTarget)
        cam = grad_cam(
            adapter.model, payload.module, payload.scoring_callable, lambda value: value,
            activation_transform=payload.activation_transform,
        )
        raw = cam.heatmap[0]
        result = deletion_insertion(
            image_path, raw, adapter.score_raw_image,
            patch_size=faithfulness_patch_size,
            perturbation_count=faithfulness_steps,
            baseline="blur",
        )
        rendered = colorize_heatmap(
            raw, coordinate_space=payload.coordinate_space, method_name="forensic-gradcam",
            colormap="turbo", lower_percentile=0.0, upper_percentile=100.0,
        )
        artifacts.append(store.save_png(sample_id, "forensic-gradcam", rendered, artifact_type="gradcam_visualization"))
        artifacts.append(store.save_npy(
            sample_id, "forensic-gradcam-raw", raw, artifact_type="gradcam_data",
            coordinate_space=payload.coordinate_space, method_name="forensic-gradcam",
            metadata={"raw_scale": "gradcam_relu_weighted_activation"},
        ))
        statistics.update({
            "patch_size": result.patch_size,
            "perturbation_count": result.perturbation_count,
            "baseline_policy": result.baseline_policy,
            "deletion": {
                "fractions": list(result.deletion.fractions),
                "raw_scores": list(result.deletion.raw_scores),
                "normalized_scores": list(result.deletion.normalized_scores),
                "normalized_auc": result.deletion.normalized_auc,
            },
            "insertion": {
                "fractions": list(result.insertion.fractions),
                "raw_scores": list(result.insertion.raw_scores),
                "normalized_scores": list(result.insertion.normalized_scores),
                "normalized_auc": result.insertion.normalized_auc,
            },
        })
    elif method == "intermediates":
        result = adapter.intermediate_representations(prepared)
        assert result.value is not None
        for branch, values in result.value.items():
            for name, opaque in values.items():
                assert isinstance(opaque, IntermediateRepresentation)
                artifact_id = f"{branch}-{name.replace('.', '-')}"
                raw = _scalar_representation(opaque.value)
                rendered = colorize_heatmap(
                    raw, coordinate_space=opaque.coordinate_space,
                    method_name=f"intermediate:{branch}:{name}",
                )
                artifacts.append(store.save_png(
                    sample_id, artifact_id, rendered, artifact_type="intermediate_visualization",
                    metadata={"raw_scale": opaque.raw_scale, "module_path": opaque.module_path},
                ))
                artifacts.append(store.save_npy(
                    sample_id, f"{artifact_id}-raw", opaque.value,
                    artifact_type="intermediate_data", coordinate_space=opaque.coordinate_space,
                    method_name=f"intermediate:{branch}:{name}",
                    metadata={"raw_scale": opaque.raw_scale, "module_path": opaque.module_path},
                ))
    else:
        unsupported = adapter.attention_tensors(prepared)
        status = unsupported.status
    explanation = ExplanationResult(
        sample_id=sample_id,
        model_id=str(adapter.manifest["model_id"]),
        method_name=method,
        status=status,
        artifacts=tuple(artifacts),
        statistics=statistics,
        metadata={
            **adapter.report_metadata,
            "source_reference": str(image_path),
            "preparation_context": prepared.context,
        },
    )
    destination = output_directory / sample_id / "explanation.json"
    write_envelope_json(ExplanationOutputEnvelope((explanation,)), destination)
    return destination


def collect_image_paths(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
    return [input_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI image detector inference")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file or a directory of images")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/detector_bundle.pt",
        help="Path to the self-describing canonical detector bundle",
    )
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--faithfulness-steps", type=int, default=8)
    parser.add_argument("--faithfulness-patch-size", type=int, default=64)
    parser.add_argument("--explanation-method", choices=EXPLANATION_METHODS, default="none")
    parser.add_argument("--output-directory", type=Path, default=Path("outputs/explanations"))
    parser.add_argument(
        "--save_heatmap", action="store_true",
        help="Compatibility alias for --explanation-method forensic-gradcam",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    adapter = build_adapter(args.checkpoint, device)
    method = "forensic-gradcam" if args.save_heatmap and args.explanation_method == "none" else args.explanation_method

    image_paths = collect_image_paths(Path(args.image))
    if not image_paths:
        raise FileNotFoundError(f"No valid images found at {args.image}")

    for image_path in image_paths:
        result = predict_single(adapter, image_path, device)
        if method != "none":
            result["explanation_json"] = str(
                explain_single(
                    adapter, image_path, method, args.output_directory,
                    ig_steps=args.ig_steps,
                    faithfulness_steps=args.faithfulness_steps,
                    faithfulness_patch_size=args.faithfulness_patch_size,
                )
            )
        print(json.dumps(result))


if __name__ == "__main__":
    main()

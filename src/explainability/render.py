"""Standalone rendering command for explanation artifact metadata and maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Sequence

import numpy as np
from PIL import Image

from .artifacts import ArtifactStore, validate_artifact_id
from .contracts import ExplanationOutputEnvelope, ExplanationResult
from .rendering import (
    RenderedImage,
    colorize_heatmap,
    render_residual_magnitude,
    render_signed_residual,
    to_numpy,
)
from .serialization import read_explanation_json, write_envelope_json


def _resolve_artifact(root: Path, declared_path: str) -> Path:
    """Resolve a declared POSIX artifact path while preventing traversal."""

    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError("artifact path must be a non-empty string")
    relative = PurePosixPath(declared_path)
    if (
        relative.is_absolute()
        or "\\" in declared_path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"artifact path is not a safe relative path: {declared_path!r}")
    resolved_root = root.resolve()
    resolved = (root / Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("artifact path must remain within artifact_root")
    return resolved


def _storage_id(value: str, *, field_name: str) -> str:
    """Return a stable path-safe component without lossy sanitization."""

    try:
        return validate_artifact_id(value, field_name=field_name)
    except ValueError:
        # Keep the original contract value in the output envelope, while using
        # a reversible UTF-8 encoding for the filesystem component.
        encoded = f"id-{value.encode('utf-8').hex()}"
        return validate_artifact_id(encoded, field_name=field_name)


def _scalar_map(array: np.ndarray, *, metadata: dict, artifact_id: str) -> tuple[np.ndarray, str]:
    """Reduce one stored map to a renderable scalar map and choose its view."""

    mode = str(metadata.get("render_mode", "heatmap"))
    if mode not in {"heatmap", "signed", "magnitude"}:
        raise ValueError("render_mode must be 'heatmap', 'signed', or 'magnitude'")
    if array.ndim == 2:
        scalar = array
    elif array.ndim == 3:
        channel_layout = metadata.get("channel_layout")
        if channel_layout is not None and channel_layout not in {
            "channel_first",
            "channel_last",
        }:
            raise ValueError("channel_layout must be 'channel_first' or 'channel_last'")
        if channel_layout == "channel_last":
            if not 1 <= array.shape[-1] <= 4:
                raise ValueError("the declared channel dimension must contain 1 to 4 channels")
            # Signed and magnitude residual renderers perform their own
            # channel-aware reduction.  Preserve the original tensor so
            # magnitude uses per-pixel RMS rather than mean-then-absolute.
            if mode in {"signed", "magnitude"}:
                return np.asarray(array, dtype=np.float64), mode
            scalar = np.mean(array, axis=-1)
        elif channel_layout == "channel_first":
            if not 1 <= array.shape[0] <= 4:
                raise ValueError("the declared channel dimension must contain 1 to 4 channels")
            if mode in {"signed", "magnitude"}:
                return np.asarray(array, dtype=np.float64), mode
            scalar = np.mean(array, axis=0)
        elif array.shape[0] == 1:
            scalar = array[0]
        elif array.shape[-1] == 1:
            scalar = array[..., 0]
        else:
            raise ValueError(
                f"artifact {artifact_id!r} has a 3D map; declare channel_layout explicitly"
            )
    else:
        raise ValueError(f"artifact {artifact_id!r} must contain a 2D or 3D map")
    return np.asarray(scalar, dtype=np.float64), mode


def _render_map(
    array: np.ndarray,
    *,
    coordinate_space: str,
    method_name: str,
    metadata: dict,
    artifact_id: str,
) -> RenderedImage:
    scalar, mode = _scalar_map(array, metadata=metadata, artifact_id=artifact_id)
    channel_layout = metadata.get("channel_layout") if scalar.ndim == 3 else None
    if mode == "signed":
        return render_signed_residual(
            scalar,
            coordinate_space=coordinate_space,
            method_name=method_name,
            channel_layout=channel_layout,
        )
    if mode == "magnitude":
        return render_residual_magnitude(
            scalar,
            coordinate_space=coordinate_space,
            method_name=method_name,
            channel_layout=channel_layout,
        )
    return colorize_heatmap(
        scalar, coordinate_space=coordinate_space, method_name=method_name
    )


def render_explanation_file(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    """Render all referenced NPY maps and write a new explanation envelope."""

    source = Path(input_path)
    envelope = read_explanation_json(source)
    root = Path(artifact_root) if artifact_root is not None else source.parent
    store = ArtifactStore(Path(output_directory) / "explanations", declared_path_prefix="explanations")
    rendered_results: list[ExplanationResult] = []
    storage_samples: dict[str, str] = {}
    rendered_keys: set[tuple[str, str]] = set()
    for explanation in envelope.explanations:
        storage_sample_id = _storage_id(explanation.sample_id, field_name="sample_id")
        previous_sample_id = storage_samples.setdefault(
            storage_sample_id, explanation.sample_id
        )
        if previous_sample_id != explanation.sample_id:
            raise ValueError(
                "distinct sample IDs map to the same rendered directory: "
                f"{previous_sample_id!r} and {explanation.sample_id!r}"
            )
        references = []
        for artifact in explanation.artifacts:
            storage_artifact_id = _storage_id(artifact.artifact_id, field_name="artifact_id")
            rendered_key = (storage_sample_id, storage_artifact_id)
            if rendered_key in rendered_keys:
                raise ValueError(
                    "duplicate rendered artifact destination for sample "
                    f"{explanation.sample_id!r} and artifact {artifact.artifact_id!r}"
                )
            rendered_keys.add(rendered_key)
            artifact_path = _resolve_artifact(root, artifact.path)
            suffix = artifact_path.suffix.lower()
            if artifact.media_type == "application/x-npy":
                if suffix != ".npy":
                    raise ValueError(
                        f"NPY artifact {artifact.path!r} must use a .npy path"
                    )
                try:
                    value = np.load(artifact_path, allow_pickle=False)
                except (OSError, ValueError) as exc:
                    raise ValueError(f"could not load map artifact {artifact.path!r}: {exc}") from exc
                metadata = dict(artifact.metadata)
                coordinate_space = metadata.get("coordinate_space")
                if not isinstance(coordinate_space, str) or not coordinate_space.strip():
                    raise ValueError(f"map artifact {artifact.path!r} lacks coordinate_space metadata")
                method_name = metadata.get("method_name", explanation.method_name)
                if not isinstance(method_name, str) or not method_name.strip():
                    raise ValueError(f"map artifact {artifact.path!r} has an invalid method_name")
                rendered = _render_map(
                    to_numpy(value),
                    coordinate_space=coordinate_space,
                    method_name=method_name,
                    metadata=metadata,
                    artifact_id=artifact.artifact_id,
                )
                reference = store.save_png(
                    storage_sample_id,
                    storage_artifact_id,
                    rendered,
                    artifact_type="rendered_visualization",
                    metadata={
                        "source_artifact": artifact.path,
                        "source_metadata": dict(artifact.metadata),
                        **(
                            {"source_artifact_id": artifact.artifact_id}
                            if storage_artifact_id != artifact.artifact_id
                            else {}
                        ),
                    },
                )
            elif artifact.media_type == "image/png":
                if suffix != ".png":
                    raise ValueError(
                        f"PNG artifact {artifact.path!r} must use a .png path"
                    )
                try:
                    with Image.open(artifact_path) as image:
                        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
                except (OSError, ValueError) as exc:
                    raise ValueError(f"could not load image artifact {artifact.path!r}: {exc}") from exc
                reference = store.save_png(
                    storage_sample_id,
                    storage_artifact_id,
                    rgb,
                    artifact_type="rendered_visualization",
                    metadata={
                        **dict(artifact.metadata),
                        "source_artifact": artifact.path,
                        **(
                            {"source_artifact_id": artifact.artifact_id}
                            if storage_artifact_id != artifact.artifact_id
                            else {}
                        ),
                    },
                )
            else:
                raise ValueError(
                    f"unsupported artifact media type {artifact.media_type!r} for {artifact.path!r}"
                )
            references.append(reference)
        rendered_results.append(
            ExplanationResult(
                sample_id=explanation.sample_id,
                model_id=explanation.model_id,
                method_name=explanation.method_name,
                status=explanation.status,
                artifacts=tuple(references),
                statistics=explanation.statistics,
                branch_coalition_logits=explanation.branch_coalition_logits,
                # Keep provenance portable and deterministic across invocation
                # directories; artifact paths themselves are declared relative
                # references in the envelope.
                metadata={**dict(explanation.metadata), "rendered_from": source.name},
            )
        )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    return write_envelope_json(
        ExplanationOutputEnvelope(tuple(rendered_results)),
        output / "rendered_explanations.json",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render explanation metadata/maps into per-sample PNG artifacts"
    )
    parser.add_argument("--input", required=True, help="Explanation JSON envelope")
    parser.add_argument("--output", required=True, help="Output artifact directory")
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Root containing paths declared by the input envelope (defaults to input's directory)",
    )
    args = parser.parse_args(argv)
    output = render_explanation_file(
        args.input,
        args.output,
        artifact_root=args.artifact_root,
    )
    print(json.dumps({"rendered_explanations": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render_explanation_file"]

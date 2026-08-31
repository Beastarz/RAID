"""Persistence of rendered explainability artifacts and lossless arrays."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .contracts import ArtifactReference
from .rendering import RenderedImage, raw_statistics, to_numpy, validate_rgb_image


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_artifact_id(value: object, *, field_name: str) -> str:
    """Return a path-safe stable ID, rejecting traversal and lossy sanitizing."""

    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            f"{field_name} must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_', or '-'"
        )
    return value


def _validate_declared_prefix(value: str | None) -> PurePosixPath | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("declared_path_prefix must be a string or None")
    path = PurePosixPath(value)
    windows_drive = len(value) >= 2 and value[0].isalpha() and value[1] == ":"
    if (
        not value
        or "\\" in value
        or windows_drive
        or path.is_absolute()
        or any(part in {".", ".."} for part in value.split("/"))
        or path.as_posix() != value
    ):
        raise ValueError("declared_path_prefix must be a safe relative path")
    return path


class ArtifactStore:
    """Write per-sample artifacts below one output root."""

    def __init__(self, output_directory: str | Path, *, declared_path_prefix: str | None = None):
        self.output_directory = Path(output_directory)
        self.declared_path_prefix = _validate_declared_prefix(declared_path_prefix)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if not self.output_directory.is_dir():
            raise ValueError("output_directory must be a directory")

    def _paths(self, sample_id: str, filename: str) -> tuple[Path, str]:
        relative = Path(sample_id) / filename
        declared = PurePosixPath(sample_id, filename)
        if self.declared_path_prefix is not None:
            declared = self.declared_path_prefix / declared
        destination = self.output_directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        root = self.output_directory.resolve()
        if not destination.resolve().is_relative_to(root):
            raise ValueError("artifact destination must remain within output_directory")
        return destination, declared.as_posix()

    def save_png(
        self,
        sample_id: str,
        artifact_id: str,
        rendered: RenderedImage | object,
        *,
        artifact_type: str = "visualization",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactReference:
        sample = validate_artifact_id(sample_id, field_name="sample_id")
        artifact = validate_artifact_id(artifact_id, field_name="artifact_id")
        if isinstance(rendered, RenderedImage):
            image = validate_rgb_image(rendered.image)
            generated_metadata = dict(rendered.metadata)
        else:
            image = validate_rgb_image(rendered)
            generated_metadata = {}
        combined_metadata = dict(metadata or {})
        combined_metadata.update(generated_metadata)
        destination, declared_path = self._paths(sample, f"{artifact}.png")
        Image.fromarray(image, mode="RGB").save(
            destination, format="PNG", optimize=False, compress_level=9
        )
        return ArtifactReference(
            artifact_id=artifact,
            path=declared_path,
            artifact_type=artifact_type,
            media_type="image/png",
            metadata=combined_metadata,
        )

    def save_npy(
        self,
        sample_id: str,
        artifact_id: str,
        value: object,
        *,
        artifact_type: str = "attribution_map",
        coordinate_space: str,
        method_name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactReference:
        sample = validate_artifact_id(sample_id, field_name="sample_id")
        artifact = validate_artifact_id(artifact_id, field_name="artifact_id")
        if not isinstance(coordinate_space, str) or not coordinate_space.strip():
            raise ValueError("coordinate_space must be a non-empty string")
        if not isinstance(method_name, str) or not method_name.strip():
            raise ValueError("method_name must be a non-empty string")
        array = to_numpy(value)
        generated_metadata: dict[str, Any] = {
            "coordinate_space": coordinate_space,
            "method_name": method_name,
            "raw_statistics": raw_statistics(array),
            "display_normalization": {"type": "none"},
        }
        combined_metadata = dict(metadata or {})
        combined_metadata.update(generated_metadata)
        destination, declared_path = self._paths(sample, f"{artifact}.npy")
        with destination.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        return ArtifactReference(
            artifact_id=artifact,
            path=declared_path,
            artifact_type=artifact_type,
            media_type="application/x-npy",
            metadata=combined_metadata,
        )

    def save_rendered(
        self,
        sample_id: str,
        artifact_id: str,
        rendered: RenderedImage,
        *,
        raw_value: object | None = None,
        artifact_type: str = "visualization",
    ) -> tuple[ArtifactReference, ...]:
        """Save PNG and, when supplied, the original array as a sibling NPY."""

        png = self.save_png(sample_id, artifact_id, rendered, artifact_type=artifact_type)
        if raw_value is None:
            return (png,)
        npy = self.save_npy(
            sample_id,
            f"{artifact_id}-raw",
            raw_value,
            artifact_type=f"{artifact_type}_data",
            coordinate_space=str(rendered.metadata.get("coordinate_space", "unspecified")),
            method_name=str(rendered.metadata.get("method_name", "unspecified")),
        )
        return png, npy

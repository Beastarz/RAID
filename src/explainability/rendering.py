"""Deterministic, model-independent rendering from array-like values."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from matplotlib import colormaps
from PIL import Image


DEFAULT_PERCENTILES = (1.0, 5.0, 50.0, 95.0, 99.0)
ChannelLayout = Literal["channel_first", "channel_last"]


@dataclass(frozen=True)
class RenderedImage:
    """An RGB image and JSON-safe metadata describing its display transform."""

    image: np.ndarray
    metadata: Mapping[str, Any]


def to_numpy(value: object, *, name: str = "value") -> np.ndarray:
    """Detach a finite numeric array from NumPy or a tensor-like object.

    Tensor support is deliberately duck typed and does not require importing a
    tensor framework. The returned array is always a caller-independent copy.
    """

    converted = value
    for method_name in ("detach", "cpu", "numpy"):
        method = getattr(converted, method_name, None)
        if method is not None:
            if not callable(method):
                raise TypeError(f"{name}.{method_name} must be callable")
            converted = method()
    try:
        array = np.array(converted, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array or tensor-like object") from exc
    if array.dtype.kind == "b":
        raise TypeError(f"{name} must not contain boolean values")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_scalar_heatmap(value: object, *, name: str = "heatmap") -> np.ndarray:
    """Return a detached float64, non-empty 2D scalar map."""

    array = to_numpy(value, name=name)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty 2D scalar map")
    return array.astype(np.float64, copy=True)


def _validate_percentiles(percentiles: Sequence[float]) -> tuple[float, ...]:
    values: list[float] = []
    for value in percentiles:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("percentiles must contain numbers")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 100.0:
            raise ValueError("percentiles must be finite and between 0 and 100")
        values.append(number)
    if len(set(values)) != len(values):
        raise ValueError("percentiles must be unique")
    return tuple(values)


def raw_statistics(
    value: object, *, percentiles: Sequence[float] = DEFAULT_PERCENTILES
) -> dict[str, Any]:
    """Compute finite, JSON-native statistics before display normalization."""

    array = to_numpy(value).astype(np.float64, copy=False)
    if array.size == 0:
        raise ValueError("value must not be empty")
    requested = _validate_percentiles(percentiles)
    percentile_values = np.percentile(array, requested) if requested else []
    return {
        "shape": [int(dimension) for dimension in array.shape],
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "percentiles": {
            _percentile_key(percentile): float(result)
            for percentile, result in zip(requested, percentile_values)
        },
    }


def _percentile_key(value: float) -> str:
    return f"p{value:g}"


def resize_scalar_heatmap(
    heatmap: object, size: tuple[int, int], *, resampling: Literal["bilinear", "nearest"] = "bilinear"
) -> np.ndarray:
    """Resize a scalar map to ``(height, width)`` without normalizing values."""

    array = validate_scalar_heatmap(heatmap)
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in size)
    ):
        raise ValueError("size must be a (height, width) tuple of positive integers")
    filters = {"bilinear": Image.Resampling.BILINEAR, "nearest": Image.Resampling.NEAREST}
    if resampling not in filters:
        raise ValueError("resampling must be 'bilinear' or 'nearest'")
    height, width = size
    image = Image.fromarray(array.astype(np.float32), mode="F")
    return np.asarray(image.resize((width, height), filters[resampling]), dtype=np.float64).copy()


def percentile_normalize(
    heatmap: object, *, lower_percentile: float = 1.0, upper_percentile: float = 99.0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize a scalar map to [0, 1], mapping constant ranges to all zeros."""

    lower, upper = _validate_percentiles((lower_percentile, upper_percentile))
    if lower >= upper:
        raise ValueError("lower_percentile must be less than upper_percentile")
    array = validate_scalar_heatmap(heatmap)
    lower_value, upper_value = np.percentile(array, (lower, upper))
    constant = bool(upper_value <= lower_value)
    if constant:
        normalized = np.zeros(array.shape, dtype=np.float64)
    else:
        normalized = np.clip((array - lower_value) / (upper_value - lower_value), 0.0, 1.0)
    metadata = {
        "type": "percentile",
        "lower_percentile": lower,
        "upper_percentile": upper,
        "lower_value": float(lower_value),
        "upper_value": float(upper_value),
        "constant_map": constant,
    }
    return normalized, metadata


def _apply_colormap(values: np.ndarray, colormap: str) -> np.ndarray:
    try:
        rgba = colormaps[colormap](values)
    except KeyError as exc:
        raise ValueError(f"unknown colormap: {colormap!r}") from exc
    return np.rint(np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)


def _validate_render_labels(coordinate_space: object, method_name: object) -> None:
    if not isinstance(coordinate_space, str) or not coordinate_space.strip():
        raise ValueError("coordinate_space must be a non-empty string")
    if not isinstance(method_name, str) or not method_name.strip():
        raise ValueError("method_name must be a non-empty string")


def colorize_heatmap(
    heatmap: object,
    *,
    coordinate_space: str,
    method_name: str,
    colormap: str = "viridis",
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> RenderedImage:
    """Render a standalone scalar map without interpreting its coordinates."""

    _validate_render_labels(coordinate_space, method_name)
    array = validate_scalar_heatmap(heatmap)
    normalized, display = percentile_normalize(
        array, lower_percentile=lower_percentile, upper_percentile=upper_percentile
    )
    return RenderedImage(
        image=_apply_colormap(normalized, colormap),
        metadata={
            "coordinate_space": coordinate_space,
            "method_name": method_name,
            "raw_statistics": raw_statistics(array),
            "display_normalization": display,
            "colormap": colormap,
        },
    )


def validate_rgb_image(value: object, *, name: str = "image") -> np.ndarray:
    """Validate HWC RGB input and convert it deterministically to uint8."""

    array = to_numpy(value, name=name)
    if array.ndim != 3 or array.shape[2] != 3 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty HWC RGB image")
    if array.dtype.kind == "f":
        if np.min(array) < 0.0 or np.max(array) > 1.0:
            raise ValueError(f"floating-point {name} values must be in [0, 1]")
        return np.rint(array * 255.0).astype(np.uint8)
    if np.min(array) < 0 or np.max(array) > 255:
        raise ValueError(f"integer {name} values must be in [0, 255]")
    return array.astype(np.uint8, copy=True)


def overlay_heatmap(
    source_image: object,
    heatmap: object,
    *,
    coordinate_space: str,
    method_name: str,
    alpha: float = 0.5,
    colormap: str = "turbo",
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> RenderedImage:
    """Overlay an explicitly image-coordinate scalar map on an RGB image."""

    if coordinate_space != "image":
        raise ValueError("overlays require coordinate_space='image'")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)):
        raise TypeError("alpha must be a finite number")
    alpha_value = float(alpha)
    if not 0.0 <= alpha_value <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    source = validate_rgb_image(source_image, name="source_image")
    raw_map = validate_scalar_heatmap(heatmap)
    resized = resize_scalar_heatmap(raw_map, source.shape[:2])
    colored = colorize_heatmap(
        resized,
        coordinate_space=coordinate_space,
        method_name=method_name,
        colormap=colormap,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    blended = np.rint(
        (1.0 - alpha_value) * source.astype(np.float64)
        + alpha_value * colored.image.astype(np.float64)
    ).astype(np.uint8)
    metadata = dict(colored.metadata)
    metadata["raw_statistics"] = raw_statistics(raw_map)
    metadata["display_normalization"] = {
        **colored.metadata["display_normalization"],
        "resized_shape": [int(source.shape[0]), int(source.shape[1])],
    }
    metadata["alpha"] = alpha_value
    return RenderedImage(blended, metadata)


def _residual_scalar_maps(
    residual: object, *, channel_layout: ChannelLayout | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = to_numpy(residual, name="residual").astype(np.float64, copy=False)
    if array.ndim == 2:
        if channel_layout is not None:
            raise ValueError("channel_layout must be omitted for a 2D residual")
        signed = array.copy()
        magnitude = np.abs(array)
    elif array.ndim == 3:
        if channel_layout not in ("channel_first", "channel_last"):
            raise ValueError("3D residuals require an explicit channel_layout")
        axis = 0 if channel_layout == "channel_first" else 2
        if not 1 <= array.shape[axis] <= 4:
            raise ValueError("the declared residual channel dimension must contain 1 to 4 channels")
        signed = np.mean(array, axis=axis)
        magnitude = np.sqrt(np.mean(np.square(array), axis=axis))
    else:
        raise ValueError("residual must be 2D or 3D")
    if 0 in array.shape:
        raise ValueError("residual must not be empty")
    return array.copy(), signed, magnitude


def render_signed_residual(
    residual: object,
    *,
    coordinate_space: str,
    method_name: str,
    channel_layout: ChannelLayout | None = None,
    colormap: str = "coolwarm",
) -> RenderedImage:
    """Render signed residuals using channel mean and a symmetric zero scale."""

    _validate_render_labels(coordinate_space, method_name)
    raw, signed, _ = _residual_scalar_maps(residual, channel_layout=channel_layout)
    absolute_max = float(np.max(np.abs(signed)))
    normalized = np.full(signed.shape, 0.5, dtype=np.float64)
    if absolute_max > 0.0:
        normalized = np.clip((signed / absolute_max + 1.0) / 2.0, 0.0, 1.0)
    return RenderedImage(
        _apply_colormap(normalized, colormap),
        {
            "coordinate_space": coordinate_space,
            "method_name": method_name,
            "raw_statistics": raw_statistics(raw),
            "signed_map_statistics": raw_statistics(signed),
            "display_normalization": {
                "type": "symmetric_zero_centered",
                "center": 0.0,
                "absolute_scale_maximum": absolute_max,
                "constant_map": absolute_max == 0.0,
            },
            "channel_layout": channel_layout,
            "channel_reduction": "mean" if raw.ndim == 3 else "none",
            "colormap": colormap,
        },
    )


def render_residual_magnitude(
    residual: object,
    *,
    coordinate_space: str,
    method_name: str,
    channel_layout: ChannelLayout | None = None,
    colormap: str = "magma",
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> RenderedImage:
    """Render absolute 2D residuals or per-pixel RMS 3D residual magnitude."""

    _validate_render_labels(coordinate_space, method_name)
    raw, _, magnitude = _residual_scalar_maps(residual, channel_layout=channel_layout)
    normalized, display = percentile_normalize(
        magnitude,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    return RenderedImage(
        _apply_colormap(normalized, colormap),
        {
            "coordinate_space": coordinate_space,
            "method_name": method_name,
            "raw_statistics": raw_statistics(raw),
            "magnitude_statistics": raw_statistics(magnitude),
            "display_normalization": display,
            "channel_layout": channel_layout,
            "channel_reduction": "root_mean_square" if raw.ndim == 3 else "absolute_value",
            "colormap": colormap,
        },
    )


def side_by_side_panel(
    left: object,
    right: object,
    *,
    separator_width: int = 1,
    separator_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Join equal-height RGB images with an explicit solid RGB separator."""

    left_image = validate_rgb_image(left, name="left")
    right_image = validate_rgb_image(right, name="right")
    if left_image.shape[0] != right_image.shape[0]:
        raise ValueError("panel images must have equal heights")
    if isinstance(separator_width, bool) or not isinstance(separator_width, int) or separator_width < 0:
        raise ValueError("separator_width must be a non-negative integer")
    if (
        not isinstance(separator_color, tuple)
        or len(separator_color) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255 for item in separator_color)
    ):
        raise ValueError("separator_color must be an RGB tuple with values in [0, 255]")
    separator = np.empty((left_image.shape[0], separator_width, 3), dtype=np.uint8)
    separator[:] = separator_color
    return np.concatenate((left_image, separator, right_image), axis=1)

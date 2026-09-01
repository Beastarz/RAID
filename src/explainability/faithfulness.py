"""Raw-image deletion/insertion faithfulness for model-independent heatmaps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from PIL import Image, ImageFilter

from src.explainability.rendering import resize_scalar_heatmap, validate_rgb_image, validate_scalar_heatmap


@dataclass(frozen=True)
class FaithfulnessCurve:
    direction: str
    fractions: tuple[float, ...]
    raw_scores: tuple[float, ...]
    normalized_scores: tuple[float, ...]
    normalized_auc: float

    def __post_init__(self) -> None:
        if self.direction not in {"deletion", "insertion"}:
            raise ValueError("direction must be deletion or insertion")
        if not (len(self.fractions) == len(self.raw_scores) == len(self.normalized_scores)):
            raise ValueError("curve arrays must have equal lengths")
        if len(self.fractions) < 2 or self.fractions[0] != 0.0 or self.fractions[-1] != 1.0:
            raise ValueError("curve fractions must include 0 and 1 endpoints")
        if any(not math.isfinite(value) for values in (self.fractions, self.raw_scores, self.normalized_scores) for value in values):
            raise ValueError("curve values must be finite")
        if any(a >= b for a, b in zip(self.fractions, self.fractions[1:])):
            raise ValueError("curve fractions must increase strictly")
        if not math.isfinite(self.normalized_auc):
            raise ValueError("normalized_auc must be finite")


@dataclass(frozen=True)
class FaithfulnessResult:
    deletion: FaithfulnessCurve
    insertion: FaithfulnessCurve
    patch_size: int
    perturbation_count: int
    baseline_policy: str
    image_size: tuple[int, int]


def _rgb_array(value: object, name: str) -> np.ndarray:
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"), dtype=np.uint8).copy()
    try:
        return validate_rgb_image(value, name=name)
    except (TypeError, ValueError) as exc:
        raise type(exc)(str(exc)) from exc


def _baseline(
    image: np.ndarray,
    baseline: str | object,
    *,
    dataset_mean: object | None,
    patch_size: int,
) -> tuple[np.ndarray, str]:
    if isinstance(baseline, str):
        if baseline == "blur":
            blurred = Image.fromarray(image, mode="RGB").filter(
                ImageFilter.GaussianBlur(radius=max(1.0, patch_size / 2.0))
            )
            return np.asarray(blurred, dtype=np.uint8).copy(), "blur"
        if baseline == "dataset_mean":
            if dataset_mean is None:
                raise ValueError("dataset_mean baseline requires dataset_mean")
            mean = np.asarray(dataset_mean, dtype=np.float64)
            if mean.shape not in {(3,), image.shape}:
                raise ValueError("dataset_mean must have shape [3] or match the RGB image")
            if not np.isfinite(mean).all() or np.min(mean) < 0 or np.max(mean) > 255:
                raise ValueError("dataset_mean values must be finite and in [0, 255]")
            return np.broadcast_to(mean, image.shape).round().astype(np.uint8).copy(), "dataset_mean"
        raise ValueError("baseline must be 'blur', 'dataset_mean', or an RGB image")
    value = _rgb_array(baseline, "baseline")
    if value.shape != image.shape:
        raise ValueError("explicit baseline must match the source image shape")
    return value, "explicit"


def _score(value: np.ndarray, scorer: Callable[[object], object], selector: Callable[[object], object]) -> float:
    selected = selector(scorer(value.copy()))
    if isinstance(selected, bool):
        raise TypeError("selected score must be numeric")
    try:
        result = float(selected)
    except (TypeError, ValueError) as exc:
        raise TypeError("selected score must be scalar numeric") from exc
    if not math.isfinite(result):
        raise ValueError("selected score must be finite")
    return result


def _normalize(scores: list[float], baseline_score: float, original_score: float) -> list[float]:
    scale = original_score - baseline_score
    if abs(scale) <= 1e-12:
        return [0.0 for _ in scores]
    return [float(np.clip((score - baseline_score) / scale, 0.0, 1.0)) for score in scores]


def _auc(fractions: list[float], scores: list[float]) -> float:
    return float(sum((x1 - x0) * (y0 + y1) * 0.5 for x0, x1, y0, y1 in zip(fractions, fractions[1:], scores, scores[1:])))


def deletion_insertion(
    raw_source_image: object,
    heatmap: object,
    scoring_callback: Callable[[object], object],
    *,
    patch_size: int = 32,
    perturbation_count: int | None = None,
    baseline: str | object = "blur",
    dataset_mean: object | None = None,
    logit_selector: Callable[[object], object] | None = None,
) -> FaithfulnessResult:
    """Perturb source pixels and rescore every step through the full raw-image path."""

    if not callable(scoring_callback):
        raise TypeError("scoring_callback must be callable")
    selector = (lambda value: value) if logit_selector is None else logit_selector
    if not callable(selector):
        raise TypeError("logit_selector must be callable")
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    image = _rgb_array(raw_source_image, "raw_source_image")
    height, width = image.shape[:2]
    base, baseline_policy = _baseline(
        image, baseline, dataset_mean=dataset_mean, patch_size=patch_size
    )
    saliency = validate_scalar_heatmap(heatmap)
    if saliency.shape != (height, width):
        saliency = resize_scalar_heatmap(saliency, (height, width), resampling="bilinear")
    patches: list[tuple[float, int, int, int, int]] = []
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            bottom, right = min(top + patch_size, height), min(left + patch_size, width)
            patches.append((float(saliency[top:bottom, left:right].mean()), top, left, bottom, right))
    patches.sort(key=lambda item: (-item[0], item[1], item[2]))
    count = len(patches) if perturbation_count is None else perturbation_count
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("perturbation_count must be a positive integer")
    count = min(count, len(patches))
    groups = [list(group) for group in np.array_split(np.arange(len(patches)), count)]
    original_score = _score(image, scoring_callback, selector)
    baseline_score = _score(base, scoring_callback, selector)
    deletion_image, insertion_image = image.copy(), base.copy()
    deletion_scores, insertion_scores = [original_score], [baseline_score]
    changed = 0
    fractions = [0.0]
    for group in groups:
        for index in group:
            _, top, left, bottom, right = patches[int(index)]
            deletion_image[top:bottom, left:right] = base[top:bottom, left:right]
            insertion_image[top:bottom, left:right] = image[top:bottom, left:right]
        changed += len(group)
        fractions.append(changed / len(patches))
        deletion_scores.append(_score(deletion_image, scoring_callback, selector))
        insertion_scores.append(_score(insertion_image, scoring_callback, selector))
    fractions[-1] = 1.0
    deletion_normalized = _normalize(deletion_scores, baseline_score, original_score)
    insertion_normalized = _normalize(insertion_scores, baseline_score, original_score)
    deletion_curve = FaithfulnessCurve(
        "deletion", tuple(fractions), tuple(deletion_scores), tuple(deletion_normalized),
        _auc(fractions, deletion_normalized),
    )
    insertion_curve = FaithfulnessCurve(
        "insertion", tuple(fractions), tuple(insertion_scores), tuple(insertion_normalized),
        _auc(fractions, insertion_normalized),
    )
    return FaithfulnessResult(
        deletion_curve, insertion_curve, patch_size, count, baseline_policy, (width, height)
    )


def evaluate_named_heatmaps(
    raw_source_image: object,
    heatmaps: Mapping[str, object],
    scoring_callback: Callable[[object], object],
    **kwargs: object,
) -> dict[str, FaithfulnessResult]:
    """Evaluate semantic, forensic, combined, or other named maps independently."""

    if not isinstance(heatmaps, Mapping) or not heatmaps:
        raise ValueError("heatmaps must be a non-empty mapping")
    result: dict[str, FaithfulnessResult] = {}
    for name, heatmap in heatmaps.items():
        if not isinstance(name, str) or not name.strip() or name in result:
            raise ValueError("heatmap names must be unique non-empty strings")
        result[name] = deletion_insertion(
            raw_source_image, heatmap, scoring_callback, **kwargs
        )
    return result


__all__ = ["FaithfulnessCurve", "FaithfulnessResult", "deletion_insertion", "evaluate_named_heatmaps"]

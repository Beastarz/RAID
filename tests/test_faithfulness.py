"""Tests for raw-image deletion/insertion faithfulness."""

import numpy as np
import pytest

from src.explainability.faithfulness import deletion_insertion, evaluate_named_heatmaps


def _patch_classifier(image: object) -> float:
    value = np.asarray(image, dtype=np.float64)
    return float(value[:4, :4, 0].mean())


def test_correct_map_ranks_above_wrong_map_without_mutating_inputs():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:4, :4] = 255
    original = image.copy()
    correct = np.zeros((8, 8), dtype=np.float32)
    correct[:4, :4] = 1
    wrong = 1 - correct
    baseline = np.zeros_like(image)
    good = deletion_insertion(image, correct, _patch_classifier, patch_size=4, baseline=baseline)
    bad = deletion_insertion(image, wrong, _patch_classifier, patch_size=4, baseline=baseline)
    assert good.insertion.normalized_auc > bad.insertion.normalized_auc
    assert good.deletion.normalized_auc < bad.deletion.normalized_auc
    assert good.deletion.raw_scores[0] == good.insertion.raw_scores[-1] == 255
    assert good.deletion.raw_scores[-1] == good.insertion.raw_scores[0] == 0
    assert np.array_equal(image, original)


def test_curves_are_deterministic_include_endpoints_and_rescore_raw_images():
    image = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
    heatmap = np.ones((2, 3), dtype=np.float32)
    calls = []

    def scorer(value: object) -> dict[str, float]:
        array = np.asarray(value)
        assert array.shape == image.shape and array.dtype == np.uint8
        calls.append(array.copy())
        return {"logit": float(array.mean())}

    first = deletion_insertion(
        image, heatmap, scorer, patch_size=4, perturbation_count=3,
        baseline="dataset_mean", dataset_mean=[10, 20, 30],
        logit_selector=lambda value: value["logit"],
    )
    second = deletion_insertion(
        image, heatmap, lambda value: float(np.asarray(value).mean()),
        patch_size=4, perturbation_count=3, baseline="dataset_mean",
        dataset_mean=[10, 20, 30],
    )
    assert first == second
    assert first.deletion.fractions == (0.0, 1 / 3, 2 / 3, 1.0)
    assert first.insertion.fractions == first.deletion.fractions
    assert len(calls) == 2 + 2 * first.perturbation_count
    assert 0 <= first.deletion.normalized_auc <= 1
    assert 0 <= first.insertion.normalized_auc <= 1


def test_named_maps_are_kept_separate_and_validation_is_strict():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    maps = {"semantic": np.zeros((4, 4)), "forensic": np.ones((4, 4))}
    results = evaluate_named_heatmaps(
        image, maps, lambda value: float(np.asarray(value).mean()),
        patch_size=2, baseline=np.full_like(image, 20),
    )
    assert set(results) == set(maps)
    with pytest.raises(ValueError, match="patch_size"):
        deletion_insertion(image, maps["semantic"], lambda value: 0.0, patch_size=0)
    with pytest.raises(ValueError, match="dataset_mean"):
        deletion_insertion(image, maps["semantic"], lambda value: 0.0, baseline="dataset_mean")
    with pytest.raises(ValueError, match="finite"):
        deletion_insertion(image, maps["semantic"], lambda value: float("nan"), baseline=image)
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_named_heatmaps(image, {}, lambda value: 0.0)

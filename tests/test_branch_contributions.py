"""Focused tests for exact model-independent branch contributions."""

import json

import pytest

from src.explainability.branch_contributions import (
    BranchContributionResult,
    compute_branch_contributions,
)
from src.explainability.contracts import BranchCoalitionLogit


def _records(values):
    return tuple(
        BranchCoalitionLogit(coalition, logit)
        for coalition, logit in values.items()
    )


def test_two_branch_shapley_values_reconstruct_nonlinear_fusion():
    # v(S) = 1 + 2*a + 3*b + 4*a*b.  The interaction is split equally.
    result = compute_branch_contributions(
        _records(
            {
                (): 1.0,
                ("semantic",): 3.0,
                ("forensic",): 4.0,
                ("semantic", "forensic"): 10.0,
            }
        )
    )

    assert result.branch_names == ("forensic", "semantic")
    assert result.baseline_logit == 1.0
    assert result.combined_logit == 10.0
    assert result.contributions == {"forensic": 5.0, "semantic": 4.0}
    assert result.reconstruction_error == pytest.approx(0.0)
    assert result.absolute_shares == {
        "forensic": pytest.approx(5.0 / 9.0),
        "semantic": pytest.approx(4.0 / 9.0),
    }


def test_three_branch_shapley_values_support_arbitrary_adapter_names():
    names = ("branch-z", "branch-a", "branch-m")

    def value(coalition):
        selected = set(coalition)
        return (
            2.0
            + sum({"branch-z": 1.0, "branch-a": -2.0, "branch-m": 4.0}[name] for name in selected)
            + (3.0 if {"branch-z", "branch-a"}.issubset(selected) else 0.0)
            + (-1.5 if {"branch-a", "branch-m"}.issubset(selected) else 0.0)
        )

    from itertools import combinations

    coalitions = {
        subset: value(subset)
        for size in range(4)
        for subset in combinations(names, size)
    }
    result = compute_branch_contributions(coalitions)

    assert result.branch_names == tuple(sorted(names))
    assert result.contributions == {
        "branch-a": pytest.approx(-1.25),
        "branch-m": pytest.approx(3.25),
        "branch-z": pytest.approx(2.5),
    }
    assert result.reconstruction_error == pytest.approx(0.0)
    assert sum(result.contributions.values()) + result.baseline_logit == pytest.approx(
        result.combined_logit
    )
    assert result.coalition_metadata["branch_count"] == 3
    assert result.coalition_metadata["coalition_count"] == 8


def test_mapping_input_and_optional_display_shares_are_supported():
    result = compute_branch_contributions(
        {(): 2.0, ("x",): 2.0, ("y",): 2.0, ("x", "y"): 2.0},
        include_absolute_shares=False,
    )
    assert result.contributions == {"x": 0.0, "y": 0.0}
    assert result.absolute_shares is None
    json.dumps(result.to_dict(), allow_nan=False)
    json.loads(result.to_json())


@pytest.mark.parametrize("branch_names", ["ab", b"ab"])
def test_branch_names_reject_string_collections(branch_names):
    with pytest.raises(TypeError, match="branch_names must be an iterable"):
        compute_branch_contributions(
            {(): 0.0, ("a",): 1.0},
            branch_names=branch_names,
        )


@pytest.mark.parametrize("branch_names", ["ab", b"ab"])
def test_coalition_record_rejects_string_branch_names(branch_names):
    with pytest.raises(TypeError, match="branch_names must be an iterable"):
        BranchCoalitionLogit(branch_names, 0.0)


def test_validated_result_values_and_metadata_are_immutable_and_detached():
    contributions = {"a": 1.0}
    shares = {"a": 1.0}
    metadata = {"nested": {"items": ["original"]}}
    coalition_metadata = {"nested": {"value": 1}}
    result = BranchContributionResult(
        branch_names=("a",),
        baseline_logit=0.0,
        combined_logit=1.0,
        contributions=contributions,
        reconstruction_error=0.0,
        coalition_logits=_records({(): 0.0, ("a",): 1.0}),
        absolute_shares=shares,
        metadata=metadata,
        coalition_metadata=coalition_metadata,
    )

    contributions["a"] = 99.0
    shares["a"] = 0.0
    metadata["nested"]["items"].append("caller mutation")
    coalition_metadata["nested"]["value"] = 99
    assert result.contributions["a"] == 1.0
    assert result.absolute_shares["a"] == 1.0
    assert result.metadata == {"nested": {"items": ("original",)}}
    assert result.coalition_metadata == {"nested": {"value": 1}}

    with pytest.raises(TypeError):
        result.contributions["a"] = 2.0
    with pytest.raises(TypeError):
        result.absolute_shares["a"] = 0.0
    with pytest.raises(TypeError):
        result.metadata["nested"]["items"] = ("changed",)
    with pytest.raises(TypeError):
        result.coalition_metadata["nested"]["value"] = 2

    serialized = result.to_dict()
    serialized["contributions"]["a"] = 42.0
    serialized["metadata"]["nested"]["items"].append("serialized mutation")
    serialized["coalition_metadata"]["nested"]["value"] = 42
    assert result.contributions["a"] == 1.0
    assert result.metadata == {"nested": {"items": ("original",)}}
    assert result.coalition_metadata == {"nested": {"value": 1}}


@pytest.mark.parametrize(
    "coalitions,match",
    [
        ({(): 0.0, ("a",): 1.0, ("b",): 1.0}, "incomplete"),
        ({(): 0.0, ("a",): 1.0, ("b",): 1.0, ("a", "a"): 2.0}, "unique"),
        ({(): float("nan")}, "finite"),
    ],
)
def test_invalid_power_sets_and_logits_are_rejected(coalitions, match):
    with pytest.raises((TypeError, ValueError), match=match):
        compute_branch_contributions(coalitions)


def test_branch_limit_and_explicit_names_are_validated():
    with pytest.raises(ValueError, match="cannot exceed 10"):
        compute_branch_contributions(
            {(): 0.0, ("a",): 1.0},
            max_branches=11,
        )

    with pytest.raises(ValueError, match="at most 1"):
        compute_branch_contributions(
            {(): 0.0, ("a",): 1.0, ("b",): 1.0, ("a", "b"): 2.0},
            max_branches=1,
        )

    with pytest.raises(ValueError, match="not listed"):
        compute_branch_contributions(
            {(): 0.0, ("a",): 1.0, ("b",): 1.0, ("a", "b"): 2.0},
            branch_names=("a",),
        )


def test_result_schema_rejects_inconsistent_reconstruction_and_power_set():
    records = _records({(): 0.0, ("a",): 1.0})
    with pytest.raises(ValueError, match="complete"):
        BranchContributionResult(
            branch_names=("a", "b"),
            baseline_logit=0.0,
            combined_logit=1.0,
            contributions={"a": 1.0, "b": 0.0},
            reconstruction_error=0.0,
            coalition_logits=records,
        )


def test_result_schema_rejects_non_shapley_contributions_and_bad_metadata():
    records = _records(
        {(): 0.0, ("a",): 0.0, ("b",): 0.0, ("a", "b"): 10.0}
    )
    with pytest.raises(ValueError, match="exact Shapley"):
        BranchContributionResult(
            branch_names=("a", "b"),
            baseline_logit=0.0,
            combined_logit=10.0,
            contributions={"a": 10.0, "b": 0.0},
            reconstruction_error=0.0,
            coalition_logits=records,
        )

    with pytest.raises(TypeError, match="metadata"):
        BranchContributionResult(
            branch_names=("a",),
            baseline_logit=0.0,
            combined_logit=1.0,
            contributions={"a": 1.0},
            reconstruction_error=0.0,
            coalition_logits=_records({(): 0.0, ("a",): 1.0}),
            metadata=[],
        )


def test_result_serialization_detaches_nested_metadata_and_orders_coalitions():
    records = tuple(
        reversed(
            _records({(): 0.0, ("a",): 1.0})
        )
    )
    result = BranchContributionResult(
        branch_names=("a",),
        baseline_logit=0.0,
        combined_logit=1.0,
        contributions={"a": 1.0},
        reconstruction_error=0.0,
        coalition_logits=records,
        metadata={"nested": {"value": 1}},
    )

    assert [item.branch_names for item in result.coalition_logits] == [(), ("a",)]
    serialized = result.to_dict()
    serialized["metadata"]["nested"]["value"] = 99
    assert result.metadata == {"nested": {"value": 1}}


def test_iterable_input_is_bounded_before_unbounded_consumption():
    def records():
        yield BranchCoalitionLogit((), 0.0)
        yield BranchCoalitionLogit(("a",), 1.0)
        yield BranchCoalitionLogit(("b",), 2.0)
        raise AssertionError("the implementation consumed beyond the practical limit")

    with pytest.raises(ValueError, match="practical limit"):
        compute_branch_contributions(records(), max_branches=1)

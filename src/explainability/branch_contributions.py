"""Exact, model-independent Shapley branch contributions.

The adapter owns branch ablation and supplies the logit for every coalition.
This module only validates that those logits form a complete power set and
computes the exact Shapley value of each named branch.  In particular, it
does not know about semantic, NPR, Bayar, SRM, or any other model topology.

The primary entry point is :func:`compute_branch_contributions`.
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass, field
from itertools import combinations
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from src.explainability.contracts import (
    SCHEMA_VERSION,
    BranchCoalitionLogit,
    JsonValue,
    _json_value,
)


# Exact enumeration becomes exponential.  Ten branches require 1,024 supplied
# logits and 5,120 marginal terms, which is a useful upper bound for an
# explanation request while preventing accidental very large jobs.
DEFAULT_MAX_BRANCHES = 10


def _branch_name(value: object, field_name: str = "branch_name") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return value


def _branch_names(
    values: Iterable[object], field_name: str = "branch_names"
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of branch names")
    try:
        names = tuple(_branch_name(value, "branch_name") for value in values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of branch names") from exc
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} must contain unique branch names")
    return tuple(sorted(names))


def _finite_logit(value: object, field_name: str = "logit") -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{field_name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _normalise_coalition_names(value: object) -> frozenset[str]:
    if isinstance(value, (str, bytes)):
        raise TypeError("coalition branch names must be an iterable of strings")
    try:
        names = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("coalition branch names must be an iterable of strings") from exc
    normalised = tuple(_branch_name(name, "branch_name") for name in names)
    if len(set(normalised)) != len(normalised):
        raise ValueError("coalition branch names must be unique")
    return frozenset(normalised)


def _coercions(
    coalition_logits: (
        Sequence[BranchCoalitionLogit]
        | Mapping[Iterable[str], float]
        | Iterable[BranchCoalitionLogit]
    ),
    *,
    max_coalitions: int | None = None,
    max_branches: int | None = None,
) -> dict[frozenset[str], float]:
    """Convert supported input forms to one deterministic coalition mapping."""
    if max_coalitions is not None and (
        isinstance(max_coalitions, bool)
        or not isinstance(max_coalitions, int)
        or max_coalitions < 1
    ):
        raise ValueError("max_coalitions must be a positive integer")
    if isinstance(coalition_logits, Mapping):
        if max_coalitions is not None and len(coalition_logits) > max_coalitions:
            raise ValueError(
                "coalition input exceeds the practical limit: "
                f"at most {max_coalitions} coalition values are allowed"
                + (
                    f" for at most {max_branches} branches"
                    if max_branches is not None
                    else ""
                )
            )
        entries = coalition_logits.items()
    else:
        if isinstance(coalition_logits, (str, bytes)):
            raise TypeError("coalition_logits must contain coalition records")
        entries = []
        try:
            iterator = iter(coalition_logits)
        except TypeError as exc:
            raise TypeError("coalition_logits must be a mapping or iterable") from exc
        for index, record in enumerate(iterator):
            if max_coalitions is not None and index >= max_coalitions:
                raise ValueError(
                    "coalition input exceeds the practical limit: "
                    f"at most {max_coalitions} coalition values are allowed"
                    + (
                        f" for at most {max_branches} branches"
                        if max_branches is not None
                        else ""
                    )
                )
            if not isinstance(record, BranchCoalitionLogit):
                raise TypeError(
                    "coalition_logits iterables must contain BranchCoalitionLogit records"
                )
            entries.append((record.branch_names, record.logit))

    result: dict[frozenset[str], float] = {}
    for raw_names, raw_logit in entries:
        if isinstance(coalition_logits, Mapping):
            key = _normalise_coalition_names(raw_names)
            logit = _finite_logit(raw_logit)
        else:
            if not isinstance(raw_logit, numbers.Real):
                # This branch is mainly defensive; BranchCoalitionLogit itself
                # validates its logit, while it also gives a useful error for a
                # malformed duck-typed record.
                raise TypeError("coalition records must contain finite logits")
            key = _normalise_coalition_names(raw_names)
            logit = _finite_logit(raw_logit)
        if key in result:
            raise ValueError("coalition branch sets must be unique")
        result[key] = logit
    if not result:
        raise ValueError("coalition_logits must be nonempty")
    return result


def _expected_coalitions(names: tuple[str, ...]) -> set[frozenset[str]]:
    expected: set[frozenset[str]] = set()
    for size in range(len(names) + 1):
        expected.update(frozenset(item) for item in combinations(names, size))
    return expected


def _shapley_values(
    values: Mapping[frozenset[str], float], names: tuple[str, ...]
) -> dict[str, float]:
    """Compute exact Shapley values for an already validated power set."""
    factorial_n = math.factorial(len(names))
    result: dict[str, float] = {}
    for branch in names:
        others = tuple(name for name in names if name != branch)
        contribution = 0.0
        for size in range(len(others) + 1):
            weight = (
                math.factorial(size)
                * math.factorial(len(names) - size - 1)
                / factorial_n
            )
            for subset in combinations(others, size):
                coalition = frozenset(subset)
                contribution += weight * (
                    values[coalition | {branch}] - values[coalition]
                )
        result[branch] = float(contribution)
    return result


def _ordered_records(
    values: Mapping[frozenset[str], float]
) -> tuple[BranchCoalitionLogit, ...]:
    return tuple(
        BranchCoalitionLogit(tuple(sorted(names)), logit)
        for names, logit in sorted(
            values.items(), key=lambda item: (len(item[0]), tuple(sorted(item[0])))
        )
    )


@dataclass(frozen=True)
class BranchContributionResult:
    """Validated Shapley decomposition of a fused logit.

    ``baseline_logit`` is the empty-coalition value.  ``combined_logit`` is
    the full-coalition value.  The signed contributions satisfy
    ``baseline_logit + sum(contributions.values()) == combined_logit`` up to
    floating-point error.  ``absolute_shares`` is optional and is a display
    aid, not another decomposition; when all signed values are zero it maps
    every branch to ``0.0``.
    """

    branch_names: tuple[str, ...]
    baseline_logit: float
    combined_logit: float
    contributions: Mapping[str, float]
    reconstruction_error: float
    coalition_logits: tuple[BranchCoalitionLogit, ...]
    absolute_shares: Mapping[str, float] | None = None
    coalition_metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        names = _branch_names(self.branch_names)
        if not names:
            raise ValueError("branch_names must contain at least one branch")
        if len(names) > DEFAULT_MAX_BRANCHES:
            raise ValueError(
                f"exact Shapley attribution supports at most {DEFAULT_MAX_BRANCHES} branches"
            )
        object.__setattr__(self, "branch_names", names)
        object.__setattr__(
            self, "baseline_logit", _finite_logit(self.baseline_logit, "baseline_logit")
        )
        object.__setattr__(
            self, "combined_logit", _finite_logit(self.combined_logit, "combined_logit")
        )
        error = _finite_logit(self.reconstruction_error, "reconstruction_error")
        if error < 0.0:
            raise ValueError("reconstruction_error must be non-negative")
        object.__setattr__(self, "reconstruction_error", error)

        contribution_values = _validate_named_values(
            self.contributions, names, "contributions"
        )
        object.__setattr__(
            self, "contributions", MappingProxyType(dict(contribution_values))
        )
        if self.absolute_shares is not None:
            shares = _validate_named_values(
                self.absolute_shares, names, "absolute_shares"
            )
            if any(value < 0.0 for value in shares.values()):
                raise ValueError("absolute_shares must be non-negative")
            if any(value > 1.0 for value in shares.values()):
                raise ValueError("absolute_shares must be within [0, 1]")
            share_total = sum(shares.values())
            contribution_total = sum(abs(value) for value in contribution_values.values())
            expected_share_total = 0.0 if contribution_total == 0.0 else 1.0
            if not math.isclose(
                share_total, expected_share_total, rel_tol=0.0, abs_tol=1e-10
            ):
                raise ValueError(
                    "absolute_shares must sum to 1 when contributions are nonzero"
                )
            object.__setattr__(
                self, "absolute_shares", MappingProxyType(dict(shares))
            )

        records = tuple(self.coalition_logits)
        if any(not isinstance(item, BranchCoalitionLogit) for item in records):
            raise TypeError("coalition_logits must contain BranchCoalitionLogit records")
        coalition_map = {frozenset(item.branch_names): item.logit for item in records}
        if len(coalition_map) != len(records):
            raise ValueError("coalition_logits must not contain duplicate coalitions")
        expected = _expected_coalitions(names)
        if set(coalition_map) != expected:
            raise ValueError("coalition_logits must contain the complete branch power set")
        if not math.isclose(
            coalition_map[frozenset()], self.baseline_logit, rel_tol=0.0, abs_tol=1e-10
        ):
            raise ValueError("baseline_logit must match the empty coalition logit")
        if not math.isclose(
            coalition_map[frozenset(names)], self.combined_logit, rel_tol=0.0, abs_tol=1e-10
        ):
            raise ValueError("combined_logit must match the full coalition logit")
        expected_contributions = _shapley_values(coalition_map, names)
        scale = max(
            1.0,
            abs(self.baseline_logit),
            abs(self.combined_logit),
            sum(abs(value) for value in expected_contributions.values()),
        )
        if any(
            not math.isclose(
                contribution_values[name], expected_contributions[name],
                rel_tol=0.0, abs_tol=1e-10 * scale,
            )
            for name in names
        ):
            raise ValueError("contributions must contain the exact Shapley values")
        if self.absolute_shares is not None:
            expected_total = sum(abs(value) for value in expected_contributions.values())
            expected_shares = {
                name: (
                    abs(value) / expected_total if expected_total else 0.0
                )
                for name, value in expected_contributions.items()
            }
            if any(
                not math.isclose(
                    self.absolute_shares[name], expected_shares[name],
                    rel_tol=0.0, abs_tol=1e-10,
                )
                for name in names
            ):
                raise ValueError("absolute_shares must match the contributions")
        if not math.isclose(
            abs(
                self.baseline_logit
                + sum(contribution_values.values())
                - self.combined_logit
            ),
            error,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError("reconstruction_error does not match the supplied values")
        object.__setattr__(self, "coalition_logits", _ordered_records(coalition_map))
        if not isinstance(self.coalition_metadata, Mapping):
            raise TypeError("coalition_metadata must be a mapping")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(
            self,
            "coalition_metadata",
            _freeze_json(_json_value(self.coalition_metadata, "coalition_metadata")),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_json(_json_value(self.metadata, "metadata")),
        )

    @property
    def signed_reconstruction_error(self) -> float:
        """Signed form of the reconstruction residual."""
        return (
            self.baseline_logit
            + sum(self.contributions.values())
            - self.combined_logit
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "branch_names": list(self.branch_names),
            "baseline_logit": self.baseline_logit,
            "combined_logit": self.combined_logit,
            "contributions": dict(self.contributions),
            "absolute_shares": (
                None if self.absolute_shares is None else dict(self.absolute_shares)
            ),
            "reconstruction_error": self.reconstruction_error,
            "coalition_logits": [item.to_dict() for item in self.coalition_logits],
            "coalition_metadata": _json_value(
                self.coalition_metadata, "coalition_metadata"
            ),
            "metadata": _json_value(self.metadata, "metadata"),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, indent=indent, sort_keys=True)


def _validate_named_values(
    values: Mapping[str, float], names: tuple[str, ...], field_name: str
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    if set(values) != set(names):
        raise ValueError(f"{field_name} must contain exactly the declared branch names")
    result: dict[str, float] = {}
    for name in names:
        _branch_name(name, "branch_name")
        result[name] = _finite_logit(values[name], f"{field_name}.{name}")
    return result


def _freeze_json(value: JsonValue) -> JsonValue:
    """Recursively freeze an already validated JSON-compatible value."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def compute_branch_contributions(
    coalition_logits: (
        Sequence[BranchCoalitionLogit]
        | Mapping[Iterable[str], float]
        | Iterable[BranchCoalitionLogit]
    ),
    *,
    branch_names: Iterable[str] | None = None,
    max_branches: int = DEFAULT_MAX_BRANCHES,
    include_absolute_shares: bool = True,
    tolerance: float = 1e-10,
    coalition_metadata: Mapping[str, JsonValue] | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> BranchContributionResult:
    """Compute exact Shapley values from a complete coalition power set.

    ``coalition_logits`` may be a sequence of :class:`BranchCoalitionLogit`
    records or a mapping whose keys are iterables of branch names.  The empty
    coalition is required and represents the baseline.  Branch names are
    inferred from the supplied coalitions unless ``branch_names`` is given.
    ``max_branches`` bounds exact enumeration; the default supports the
    intended two-branch detector and modest extensions.
    """
    if (
        isinstance(max_branches, bool)
        or not isinstance(max_branches, int)
        or max_branches < 1
    ):
        raise ValueError("max_branches must be a positive integer")
    if max_branches > DEFAULT_MAX_BRANCHES:
        raise ValueError(
            f"max_branches cannot exceed {DEFAULT_MAX_BRANCHES} for exact Shapley attribution"
        )
    if isinstance(tolerance, bool) or not isinstance(tolerance, numbers.Real):
        raise TypeError("tolerance must be a finite non-negative number")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be a finite non-negative number")
    if not isinstance(include_absolute_shares, bool):
        raise TypeError("include_absolute_shares must be a bool")

    values = _coercions(
        coalition_logits,
        max_coalitions=1 << max_branches,
        max_branches=max_branches,
    )
    inferred = _branch_names(sorted({name for coalition in values for name in coalition}))
    names = inferred if branch_names is None else _branch_names(branch_names)
    if not names:
        raise ValueError("branch_names must contain at least one branch")
    if len(names) > max_branches:
        raise ValueError(
            f"exact Shapley attribution supports at most {max_branches} branches"
        )
    if branch_names is not None and not set(inferred).issubset(names):
        raise ValueError("coalition contains a branch not listed in branch_names")

    expected = _expected_coalitions(names)
    actual = set(values)
    missing = expected - actual
    extra = actual - expected
    if missing:
        missing_display = sorted(
            (tuple(sorted(item)) for item in missing),
            key=lambda item: (len(item), item),
        )
        raise ValueError(f"coalition power set is incomplete; missing {missing_display}")
    if extra:
        extra_display = sorted(
            (tuple(sorted(item)) for item in extra),
            key=lambda item: (len(item), item),
        )
        raise ValueError(f"coalition contains undeclared branch names: {extra_display}")

    full_set = frozenset(names)
    empty_set = frozenset()
    baseline = values[empty_set]
    combined = values[full_set]
    branch_values = _shapley_values(values, names)

    signed_residual = baseline + sum(branch_values.values()) - combined
    reconstruction_error = abs(signed_residual)
    scale = max(
        1.0,
        abs(baseline),
        abs(combined),
        sum(abs(value) for value in branch_values.values()),
    )
    if reconstruction_error > tolerance * scale:
        raise ValueError(
            "Shapley contributions do not reconstruct the combined logit "
            f"within tolerance (error={reconstruction_error:g})"
        )

    shares: dict[str, float] | None = None
    if include_absolute_shares:
        total_absolute = sum(abs(value) for value in branch_values.values())
        shares = {
            name: (abs(value) / total_absolute if total_absolute else 0.0)
            for name, value in branch_values.items()
        }

    supplied_coalition_metadata = {} if coalition_metadata is None else coalition_metadata
    supplied_metadata = {} if metadata is None else metadata
    if not isinstance(supplied_coalition_metadata, Mapping):
        raise TypeError("coalition_metadata must be a mapping")
    if not isinstance(supplied_metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    # Validate caller metadata now, and add deterministic facts in a separate
    # namespace so caller-provided keys cannot change the interpretation.
    _json_value(supplied_coalition_metadata, "coalition_metadata")
    _json_value(supplied_metadata, "metadata")
    generated_coalition_metadata: dict[str, JsonValue] = {}
    generated_coalition_metadata.update(dict(supplied_coalition_metadata))
    # Generated facts are authoritative if a caller uses one of their keys.
    generated_coalition_metadata.update(
        {
            "branch_count": len(names),
            "coalition_count": len(values),
            "complete_power_set": True,
        }
    )

    return BranchContributionResult(
        branch_names=names,
        baseline_logit=baseline,
        combined_logit=combined,
        contributions=branch_values,
        reconstruction_error=reconstruction_error,
        coalition_logits=_ordered_records(values),
        absolute_shares=shares,
        coalition_metadata=generated_coalition_metadata,
        metadata=supplied_metadata,
    )


__all__ = [
    "DEFAULT_MAX_BRANCHES",
    "BranchContributionResult",
    "compute_branch_contributions",
]

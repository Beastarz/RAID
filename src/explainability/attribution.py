"""Model-independent vanilla-gradient and Integrated Gradients attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


_VANILLA_METHOD = "vanilla_gradients"
_INTEGRATED_METHOD = "integrated_gradients"


def _validate_input(input_tensor: object) -> torch.Tensor:
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor must be a torch.Tensor")
    if input_tensor.layout != torch.strided:
        raise TypeError("input_tensor must be a dense torch.Tensor")
    if not input_tensor.is_floating_point():
        raise TypeError("input_tensor must use a floating-point dtype")
    if input_tensor.ndim < 1 or input_tensor.shape[0] < 1:
        raise ValueError("input_tensor must have a nonempty batch dimension")
    if not torch.isfinite(input_tensor).all().item():
        raise ValueError("input_tensor must contain only finite values")
    return input_tensor


def _validate_tensor(
    value: object,
    field_name: str,
    *,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise TypeError(f"{field_name} must be a dense torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{field_name} must use a floating-point dtype")
    if value.ndim < 1:
        raise ValueError(f"{field_name} must have a batch dimension")
    if batch_size is not None and value.shape[0] != batch_size:
        raise ValueError(f"{field_name} must match the input batch size")
    if device is not None and value.device != device:
        raise ValueError(f"{field_name} must be on the input device")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{field_name} must contain only finite values")
    return value


def _validate_steps(steps: object) -> int:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    return steps


def _normalise_logits(
    value: object,
    *,
    batch_size: int,
    device: torch.device,
    field_name: str = "logit_selector output",
) -> torch.Tensor:
    logits = _validate_tensor(
        value,
        field_name,
        batch_size=batch_size,
        device=device,
    )
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    raise ValueError(f"{field_name} must have shape [B] or [B, 1]")


def _score_logits(
    tensor: torch.Tensor,
    scoring_callable: Callable[[torch.Tensor], object],
    logit_selector: Callable[[object], object],
) -> torch.Tensor:
    scored = scoring_callable(tensor)
    selected = logit_selector(scored)
    logits = _normalise_logits(
        selected,
        batch_size=tensor.shape[0],
        device=tensor.device,
    )
    if not logits.requires_grad:
        raise ValueError("selected logits must remain connected to input_tensor")
    return logits


def _gradient_for_logits(
    tensor: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    try:
        gradient = torch.autograd.grad(
            logits.sum(),
            tensor,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]
    except (RuntimeError, ValueError) as exc:
        raise ValueError("could not compute gradients for selected logits") from exc
    if not torch.isfinite(gradient).all().item():
        raise ValueError("gradients must be finite")
    return gradient.detach().clone()


def _validate_callable(value: object, field_name: str) -> None:
    if not callable(value):
        raise TypeError(f"{field_name} must be callable")


def _prepare_baseline(
    baseline: object,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    if baseline is None:
        return torch.zeros_like(input_tensor)
    if isinstance(baseline, bool):
        raise TypeError("baseline must be numeric or a tensor")
    if isinstance(baseline, torch.Tensor):
        if baseline.layout != torch.strided:
            raise TypeError("baseline must be a dense tensor")
        source = baseline
    else:
        try:
            source = torch.as_tensor(
                baseline,
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError("baseline must be numeric or a tensor") from exc
    if source.layout != torch.strided:
        raise TypeError("baseline must be a dense tensor")
    if source.is_complex():
        raise TypeError("baseline must use a real numeric dtype")
    if not source.is_floating_point():
        if source.dtype == torch.bool:
            raise TypeError("baseline must use a numeric dtype")
        source = source.to(dtype=input_tensor.dtype, device=input_tensor.device)
    else:
        source = source.to(dtype=input_tensor.dtype, device=input_tensor.device)
    if not torch.isfinite(source).all().item():
        raise ValueError("baseline must contain only finite values")
    try:
        return torch.broadcast_to(source, input_tensor.shape).clone()
    except (RuntimeError, ValueError) as exc:
        raise ValueError("baseline must be safely broadcastable to input_tensor") from exc


@dataclass(frozen=True)
class GradientAttributionResult:
    """Detached gradient attribution output with method-specific fields."""

    method: str
    attribution: torch.Tensor
    input_logits: torch.Tensor
    baseline_logits: torch.Tensor | None = None
    completeness_delta: torch.Tensor | None = None
    steps: int = 1

    def __post_init__(self) -> None:
        if self.method not in {_VANILLA_METHOD, _INTEGRATED_METHOD}:
            raise ValueError(
                "method must be 'vanilla_gradients' or 'integrated_gradients'"
            )
        steps = _validate_steps(self.steps)
        attribution = _validate_tensor(self.attribution, "attribution")
        batch_size = attribution.shape[0]
        input_logits = _normalise_logits(
            self.input_logits,
            batch_size=batch_size,
            device=attribution.device,
            field_name="input_logits",
        )
        if self.method == _VANILLA_METHOD:
            if steps != 1:
                raise ValueError("vanilla_gradients results must use steps=1")
            if self.baseline_logits is not None:
                raise ValueError("vanilla_gradients must not contain baseline_logits")
            if self.completeness_delta is not None:
                raise ValueError(
                    "vanilla_gradients must not contain completeness_delta"
                )
        else:
            if self.baseline_logits is None:
                raise ValueError("integrated_gradients requires baseline_logits")
            if self.completeness_delta is None:
                raise ValueError("integrated_gradients requires completeness_delta")
        baseline_logits = None
        completeness_delta = None
        if self.baseline_logits is not None:
            baseline_logits = _normalise_logits(
                self.baseline_logits,
                batch_size=batch_size,
                device=attribution.device,
                field_name="baseline_logits",
            )
        if self.completeness_delta is not None:
            completeness_delta = _normalise_logits(
                self.completeness_delta,
                batch_size=batch_size,
                device=attribution.device,
                field_name="completeness_delta",
            )
        if self.method == _INTEGRATED_METHOD:
            assert baseline_logits is not None and completeness_delta is not None
            expected_delta = (
                input_logits
                - baseline_logits
                - attribution.reshape(batch_size, -1).sum(dim=1)
            )
            try:
                torch.testing.assert_close(
                    completeness_delta,
                    expected_delta,
                    check_dtype=False,
                )
            except AssertionError as exc:
                raise ValueError(
                    "completeness_delta does not match the supplied attribution"
                ) from exc
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "attribution", attribution.detach().clone())
        object.__setattr__(self, "input_logits", input_logits.detach().clone())
        object.__setattr__(
            self,
            "baseline_logits",
            None if baseline_logits is None else baseline_logits.detach().clone(),
        )
        object.__setattr__(
            self,
            "completeness_delta",
            None
            if completeness_delta is None
            else completeness_delta.detach().clone(),
        )


def vanilla_gradients(
    input_tensor: torch.Tensor,
    scoring_callable: Callable[[torch.Tensor], object],
    logit_selector: Callable[[object], object],
) -> GradientAttributionResult:
    """Return signed input gradients for one selected raw logit per sample."""
    input_value = _validate_input(input_tensor)
    _validate_callable(scoring_callable, "scoring_callable")
    _validate_callable(logit_selector, "logit_selector")
    with torch.enable_grad():
        working = input_value.detach().clone().requires_grad_(True)
        logits = _score_logits(working, scoring_callable, logit_selector)
        gradient = _gradient_for_logits(working, logits)
    return GradientAttributionResult(
        method=_VANILLA_METHOD,
        attribution=gradient,
        input_logits=logits.detach().clone(),
        steps=1,
    )


def integrated_gradients(
    input_tensor: torch.Tensor,
    scoring_callable: Callable[[torch.Tensor], object],
    logit_selector: Callable[[object], object],
    *,
    baseline: object = None,
    steps: int = 64,
) -> GradientAttributionResult:
    """Return trapezoidal Integrated Gradients along a straight-line path."""
    input_value = _validate_input(input_tensor)
    _validate_callable(scoring_callable, "scoring_callable")
    _validate_callable(logit_selector, "logit_selector")
    step_count = _validate_steps(steps)
    baseline_value = _prepare_baseline(baseline, input_value)
    difference = input_value.detach() - baseline_value.detach()
    gradient_total = torch.zeros_like(input_value)
    baseline_logits = None
    input_logits = None
    previous_gradient = None
    with torch.enable_grad():
        for index in range(step_count + 1):
            alpha = index / step_count
            point = (baseline_value + alpha * difference).detach().clone()
            point.requires_grad_(True)
            logits = _score_logits(point, scoring_callable, logit_selector)
            gradient = _gradient_for_logits(point, logits)
            if index == 0:
                baseline_logits = logits.detach().clone()
            if index == step_count:
                input_logits = logits.detach().clone()
            if previous_gradient is not None:
                gradient_total = gradient_total + (previous_gradient + gradient) * 0.5
            previous_gradient = gradient
    assert baseline_logits is not None and input_logits is not None
    attribution = difference * (gradient_total / step_count)
    completeness_delta = (
        input_logits
        - baseline_logits
        - attribution.reshape(attribution.shape[0], -1).sum(dim=1)
    )
    return GradientAttributionResult(
        method=_INTEGRATED_METHOD,
        attribution=attribution.detach().clone(),
        input_logits=input_logits,
        baseline_logits=baseline_logits,
        completeness_delta=completeness_delta.detach().clone(),
        steps=step_count,
    )


__all__ = [
    "GradientAttributionResult",
    "integrated_gradients",
    "vanilla_gradients",
]

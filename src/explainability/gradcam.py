"""Generic Grad-CAM over an adapter-owned zero-argument scoring callback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


def _validate_tensor(
    value: object,
    field_name: str,
    *,
    ndim: int | None = None,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise TypeError(f"{field_name} must be a dense torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{field_name} must use a floating-point dtype")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{field_name} must have shape with rank {ndim}")
    if value.ndim < 1:
        raise ValueError(f"{field_name} must have a batch dimension")
    if batch_size is not None and value.shape[0] != batch_size:
        raise ValueError(f"{field_name} must match the activation batch size")
    if device is not None and value.device != device:
        raise ValueError(f"{field_name} must be on the activation device")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{field_name} must contain only finite values")
    return value


def _normalise_logits(value: object, *, device: torch.device) -> torch.Tensor:
    logits = _validate_tensor(value, "logit_selector output", device=device)
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    raise ValueError("logit_selector output must have shape [B] or [B, 1]")


def _validate_callable(value: object, field_name: str) -> None:
    if not callable(value):
        raise TypeError(f"{field_name} must be callable")


@dataclass(frozen=True)
class GradCAMResult:
    """Detached, unnormalized Grad-CAM heatmap and selected logits."""

    heatmap: torch.Tensor
    selected_logits: torch.Tensor

    def __post_init__(self) -> None:
        heatmap = _validate_tensor(self.heatmap, "heatmap", ndim=3)
        batch_size, height, width = heatmap.shape
        if batch_size < 1 or height < 1 or width < 1:
            raise ValueError("heatmap must have positive batch and spatial dimensions")
        if (heatmap < 0).any().item():
            raise ValueError("heatmap must be non-negative")
        logits = _normalise_logits(
            self.selected_logits,
            device=heatmap.device,
        )
        if logits.shape[0] != batch_size:
            raise ValueError("selected_logits must match the heatmap batch size")
        object.__setattr__(self, "heatmap", heatmap.detach().clone())
        object.__setattr__(self, "selected_logits", logits.detach().clone())


def grad_cam(
    model: nn.Module,
    target_module: nn.Module,
    scoring_callable: Callable[[], object],
    logit_selector: Callable[[object], object],
    *,
    activation_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> GradCAMResult:
    """Compute Grad-CAM for a selected raw logit from an adapter callback."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(target_module, nn.Module):
        raise TypeError("target_module must be a torch.nn.Module")
    if not any(module is target_module for module in model.modules()):
        raise ValueError("target_module must belong to model")
    _validate_callable(scoring_callable, "scoring_callable")
    _validate_callable(logit_selector, "logit_selector")
    if activation_transform is not None:
        _validate_callable(activation_transform, "activation_transform")

    training_states = [(module, module.training) for module in model.modules()]
    hook_handle: torch.utils.hooks.RemovableHandle | None = None
    activations: list[torch.Tensor] = []

    def capture_activation(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("target_module must return a tensor activation")
        activations.append(output)

    try:
        hook_handle = target_module.register_forward_hook(capture_activation)
        for module, _training in training_states:
            module.eval()
        with torch.enable_grad():
            scored = scoring_callable()
            if len(activations) != 1:
                raise RuntimeError(
                    "target_module must execute exactly once during scoring"
                )
            logits = _normalise_logits(
                logit_selector(scored),
                device=activations[0].device,
            )
            activation = _validate_tensor(
                activations[0],
                "target activation",
                device=logits.device,
            )
            if activation.shape[0] != logits.shape[0]:
                raise ValueError("target activation and selected logits must share a batch")
            if not logits.requires_grad:
                raise ValueError("selected logits must remain connected to target activation")
            try:
                gradient = torch.autograd.grad(
                    logits.sum(),
                    activation,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )[0]
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    "could not compute target activation gradients; selected logits "
                    "must be connected to the target activation"
                ) from exc
            gradient = _validate_tensor(
                gradient,
                "target activation gradient",
                device=activation.device,
            )
            if activation_transform is not None:
                transformed_activation = activation_transform(activation)
                transformed_gradient = activation_transform(gradient)
            else:
                transformed_activation = activation
                transformed_gradient = gradient
            transformed_activation = _validate_tensor(
                transformed_activation,
                "transformed activation",
                ndim=4,
                batch_size=activation.shape[0],
                device=activation.device,
            )
            transformed_gradient = _validate_tensor(
                transformed_gradient,
                "transformed gradient",
                ndim=4,
                batch_size=activation.shape[0],
                device=activation.device,
            )
            if transformed_gradient.shape != transformed_activation.shape:
                raise ValueError(
                    "transformed activation and gradient must have identical shapes"
                )
            if transformed_gradient.dtype != transformed_activation.dtype:
                raise ValueError(
                    "transformed activation and gradient must have identical dtypes"
                )
            weights = transformed_gradient.mean(dim=(-2, -1), keepdim=True)
            heatmap = torch.relu(
                (weights * transformed_activation).sum(dim=1)
            )
            heatmap = _validate_tensor(
                heatmap,
                "heatmap",
                ndim=3,
                batch_size=activation.shape[0],
                device=activation.device,
            )
            return GradCAMResult(
                heatmap=heatmap.detach().clone(),
                selected_logits=logits.detach().clone(),
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        for module, training in training_states:
            module.train(training)


__all__ = ["GradCAMResult", "grad_cam"]

"""Class-agnostic attention rollout for adapter-supplied attention tensors.

The adapter is responsible for removing special tokens and selecting the
patch-token order.  This module only performs the model-independent rollout:
head averaging, residual identity, row normalization, and layer composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def _positive_pair(value: object, field_name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must contain two positive integers")
    try:
        pair = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must contain two positive integers") from exc
    if len(pair) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in pair
    ):
        raise ValueError(f"{field_name} must contain two positive integers")
    return pair[0], pair[1]


def _token_index(value: object, field_name: str, token_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if not 0 <= value < token_count:
        raise ValueError(f"{field_name} must be in [0, {token_count})")
    return value


def _patch_indices(
    value: object,
    *,
    token_count: int,
    cls_token_index: int,
    expected_count: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("patch_token_indices must be a sequence of integers")
    try:
        indices = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("patch_token_indices must be a sequence of integers") from exc
    if len(indices) != expected_count:
        raise ValueError(
            "patch_token_indices must contain exactly "
            f"{expected_count} entries for the requested patch grid"
        )
    if any(isinstance(item, bool) or not isinstance(item, int) for item in indices):
        raise TypeError("patch_token_indices must contain only integers")
    if any(not 0 <= item < token_count for item in indices):
        raise ValueError("patch_token_indices must be in the token range")
    if cls_token_index in indices:
        raise ValueError("patch_token_indices must not include cls_token_index")
    if len(set(indices)) != len(indices):
        raise ValueError("patch_token_indices must contain unique token indices")
    return indices


def _validate_attention_tensor(
    value: object,
    *,
    field_name: str,
    expected_shape: torch.Size | None = None,
    expected_dtype: torch.dtype | None = None,
    expected_device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise TypeError(f"{field_name} must be a dense torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{field_name} must use a floating-point dtype")
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError(f"{field_name} tensors must have identical shapes")
    if expected_dtype is not None and value.dtype != expected_dtype:
        raise ValueError(f"{field_name} tensors must have identical dtypes")
    if expected_device is not None and value.device != expected_device:
        raise ValueError(f"{field_name} tensors must be on the same device")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{field_name} must contain only finite values")
    if (value < 0).any().item():
        raise ValueError(f"{field_name} must contain only non-negative values")
    return value


@dataclass(frozen=True)
class AttentionRolloutResult:
    """Detached, unnormalized class-agnostic attention rollout outputs."""

    heatmap: torch.Tensor
    joint_attention: torch.Tensor
    patch_grid: tuple[int, int]
    cls_token_index: int
    patch_token_indices: tuple[int, ...]
    class_agnostic: bool = True

    def __post_init__(self) -> None:
        grid = _positive_pair(self.patch_grid, "patch_grid")
        if not isinstance(self.joint_attention, torch.Tensor):
            raise TypeError("joint_attention must be a torch.Tensor")
        if self.joint_attention.ndim != 3:
            raise ValueError("joint_attention must have shape [B, T, T]")
        batch_size, token_count, token_count_b = self.joint_attention.shape
        if batch_size < 1 or token_count < 1 or token_count_b != token_count:
            raise ValueError("joint_attention must have shape [B, T, T] with positive sizes")
        joint = _validate_attention_tensor(
            self.joint_attention,
            field_name="joint_attention",
        )
        cls_index = _token_index(
            self.cls_token_index, "cls_token_index", token_count
        )
        expected_count = grid[0] * grid[1]
        indices = _patch_indices(
            self.patch_token_indices,
            token_count=token_count,
            cls_token_index=cls_index,
            expected_count=expected_count,
        )
        if not isinstance(self.heatmap, torch.Tensor):
            raise TypeError("heatmap must be a torch.Tensor")
        if self.heatmap.shape != (batch_size, grid[0], grid[1]):
            raise ValueError(
                "heatmap must have shape "
                f"[B, rows, cols] = {(batch_size, grid[0], grid[1])}"
            )
        heatmap = _validate_attention_tensor(self.heatmap, field_name="heatmap")
        if heatmap.dtype != joint.dtype:
            raise ValueError("heatmap and joint_attention must have the same dtype")
        if heatmap.device != joint.device:
            raise ValueError("heatmap and joint_attention must be on the same device")
        if not isinstance(self.class_agnostic, bool):
            raise TypeError("class_agnostic must be a bool")
        if not self.class_agnostic:
            raise ValueError("attention rollout is class-agnostic")
        try:
            torch.testing.assert_close(
                joint.sum(dim=-1),
                torch.ones_like(joint[..., 0]),
            )
        except AssertionError as exc:
            raise ValueError("joint_attention rows must sum to 1") from exc
        expected_heatmap = joint[:, cls_index, list(indices)].reshape(
            batch_size, grid[0], grid[1]
        )
        try:
            torch.testing.assert_close(heatmap, expected_heatmap)
        except AssertionError as exc:
            raise ValueError(
                "heatmap must match CLS-to-patch joint attention values"
            ) from exc

        # Clone at the contract boundary so neither caller-owned tensors nor
        # tensors retained by a preceding computation can mutate the result.
        object.__setattr__(self, "heatmap", heatmap.detach().clone())
        object.__setattr__(self, "joint_attention", joint.detach().clone())
        object.__setattr__(self, "patch_grid", grid)
        object.__setattr__(self, "cls_token_index", cls_index)
        object.__setattr__(self, "patch_token_indices", indices)
        object.__setattr__(self, "class_agnostic", True)


def attention_rollout(
    attention_tensors: Sequence[torch.Tensor],
    *,
    patch_grid: tuple[int, int],
    cls_token_index: int = 0,
    patch_token_indices: Sequence[int] | None = None,
) -> AttentionRolloutResult:
    """Compose adapter-supplied attention matrices into a patch heatmap.

    Each input tensor has shape ``[batch, heads, tokens, tokens]``.  Layers
    are consumed in forward order, yielding ``A_last @ ... @ A_first`` after
    head averaging, residual identity, and row normalization.  The final CLS
    row is reshaped according to the adapter-supplied patch-token order.
    """
    if isinstance(attention_tensors, (str, bytes, torch.Tensor)):
        raise TypeError("attention_tensors must be a nonempty sequence of tensors")
    try:
        layers = tuple(attention_tensors)
    except TypeError as exc:
        raise TypeError("attention_tensors must be a nonempty sequence of tensors") from exc
    if not layers:
        raise ValueError("attention_tensors must be nonempty")

    first = _validate_attention_tensor(
        layers[0], field_name="attention_tensors[0]"
    )
    if first.ndim != 4:
        raise ValueError("attention tensors must have shape [B, H, T, T]")
    batch_size, head_count, token_count, token_count_b = first.shape
    if batch_size < 1 or head_count < 1 or token_count < 1 or token_count_b != token_count:
        raise ValueError(
            "attention tensors must have shape [B, H, T, T] with positive sizes"
        )
    for index, layer in enumerate(layers[1:], start=1):
        _validate_attention_tensor(
            layer,
            field_name=f"attention_tensors[{index}]",
            expected_shape=first.shape,
            expected_dtype=first.dtype,
            expected_device=first.device,
        )
        if layer.ndim != 4 or layer.shape[-1] != layer.shape[-2]:
            raise ValueError("attention tensors must have shape [B, H, T, T]")

    grid = _positive_pair(patch_grid, "patch_grid")
    cls_index = _token_index(cls_token_index, "cls_token_index", token_count)
    expected_count = grid[0] * grid[1]
    if patch_token_indices is None:
        indices = tuple(index for index in range(token_count) if index != cls_index)
        if len(indices) != expected_count:
            raise ValueError(
                "patch_grid must contain exactly one cell per non-CLS token "
                "when patch_token_indices is omitted"
            )
    else:
        indices = _patch_indices(
            patch_token_indices,
            token_count=token_count,
            cls_token_index=cls_index,
            expected_count=expected_count,
        )

    identity = torch.eye(token_count, dtype=first.dtype, device=first.device)
    identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
    joint = identity
    for layer in layers:
        averaged = layer.detach().mean(dim=1)
        augmented = averaged + identity
        normalized = augmented / augmented.sum(dim=-1, keepdim=True)
        joint = normalized @ joint
    joint = joint.detach().clone()
    heatmap = joint[:, cls_index, list(indices)].reshape(
        batch_size, grid[0], grid[1]
    )
    return AttentionRolloutResult(
        heatmap=heatmap.detach().clone(),
        joint_attention=joint,
        patch_grid=grid,
        cls_token_index=cls_index,
        patch_token_indices=indices,
        class_agnostic=True,
    )


__all__ = ["AttentionRolloutResult", "attention_rollout"]

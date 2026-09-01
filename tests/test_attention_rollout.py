"""Focused tests for model-independent class-agnostic attention rollout."""

import pytest
import torch
from torch import nn

from src.explainability.attention_rollout import (
    AttentionRolloutResult,
    attention_rollout,
)


def test_attention_rollout_is_exported_by_explainability_facade():
    from src.explainability import (
        AttentionRolloutResult as ExportedResult,
        attention_rollout as exported_rollout,
    )

    assert ExportedResult is AttentionRolloutResult
    assert exported_rollout is attention_rollout


def _layer(rows, *, dtype=torch.float32, requires_grad=False):
    return torch.tensor(rows, dtype=dtype, requires_grad=requires_grad).unsqueeze(0).unsqueeze(0)


class MiniatureTransformerFixture(nn.Module):
    """Tiny deterministic transformer-style source of adapter attention matrices."""

    def __init__(self):
        super().__init__()
        layer = torch.zeros((1, 2, 5, 5), dtype=torch.float32)
        layer[:, :, 0, 3] = 3.0
        self.register_buffer("attention_layer", layer)

    def attention_tensors(self):
        return (self.attention_layer.clone(),)

    def forward(self):
        return self.attention_tensors()


def test_miniature_transformer_attention_localizes_known_salient_patch():
    transformer = MiniatureTransformerFixture()
    result = attention_rollout(
        transformer(),
        patch_grid=(2, 2),
    )

    # The supplied matrices route the CLS token most strongly to patch 3,
    # which is the first cell of the second heatmap row.
    assert torch.equal(result.heatmap.argmax(), torch.tensor(2))
    assert torch.allclose(result.heatmap, torch.tensor([[[0.0, 0.0], [0.75, 0.0]]]))
    assert torch.isfinite(result.heatmap).all()


def test_known_matrix_extracts_cls_to_patch_influence():
    # Residual + row normalization gives the CLS row [1/3, 0, 2/3].
    result = attention_rollout(
        [_layer([[0.0, 0.0, 2.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])],
        patch_grid=(1, 2),
    )

    assert result.class_agnostic is True
    assert result.patch_grid == (1, 2)
    assert result.cls_token_index == 0
    assert result.patch_token_indices == (1, 2)
    assert result.heatmap.shape == (1, 1, 2)
    assert torch.allclose(result.heatmap, torch.tensor([[[0.0, 2.0 / 3.0]]]))
    assert torch.allclose(
        result.joint_attention[0, 0], torch.tensor([1.0 / 3.0, 0.0, 2.0 / 3.0])
    )


def test_layers_compose_in_forward_order():
    first = _layer([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    second = _layer([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    result = attention_rollout([first, second], patch_grid=(1, 2))

    # A2 @ A1: [1/2, 0, 1/2] @ A1 = [1/4, 1/4, 1/2].
    assert torch.allclose(
        result.joint_attention[0, 0],
        torch.tensor([1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0]),
    )


def test_batch_and_custom_cls_and_patch_order_are_preserved():
    tensor = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    tensor[0, 0, 2, 0] = 3.0
    tensor[1, 0, 2, 3] = 3.0
    result = attention_rollout(
        [tensor],
        patch_grid=(1, 2),
        cls_token_index=2,
        patch_token_indices=(3, 0),
    )

    assert result.heatmap.shape == (2, 1, 2)
    assert torch.allclose(result.heatmap[0], torch.tensor([[0.0, 3.0 / 4.0]]))
    assert torch.allclose(result.heatmap[1], torch.tensor([[3.0 / 4.0, 0.0]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved_and_repeated_calls_are_deterministic(dtype):
    tensor = _layer([[0.0, 2.0], [1.0, 0.0]], dtype=dtype)
    first = attention_rollout([tensor], patch_grid=(1, 1))
    second = attention_rollout([tensor], patch_grid=(1, 1))

    assert first.heatmap.dtype is dtype
    assert first.joint_attention.dtype is dtype
    assert first.heatmap.device == tensor.device
    assert torch.equal(first.heatmap, second.heatmap)
    assert torch.equal(first.joint_attention, second.joint_attention)


def test_result_is_detached_cloned_and_does_not_mutate_inputs():
    tensor = _layer([[0.0, 1.0], [0.0, 0.0]], requires_grad=True)
    original = tensor.detach().clone()
    result = attention_rollout([tensor], patch_grid=(1, 1))

    with torch.no_grad():
        tensor.add_(10.0)
    assert torch.equal(tensor.detach() - 10.0, original)
    assert not result.heatmap.requires_grad
    assert not result.joint_attention.requires_grad
    assert result.heatmap.data_ptr() != tensor.data_ptr()
    assert result.joint_attention.data_ptr() != tensor.data_ptr()


@pytest.mark.parametrize(
    "bad", [[], "layers", b"layers", None, [torch.ones(2, 2)]]
)
def test_malformed_attention_inputs_are_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        attention_rollout(bad, patch_grid=(1, 1))


def test_attention_inputs_require_matching_shape_dtype_device_and_values():
    valid = _layer([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="identical shapes"):
        attention_rollout([valid, torch.zeros((1, 1, 3, 3))], patch_grid=(1, 1))
    with pytest.raises(ValueError, match="identical dtypes"):
        attention_rollout([valid, valid.to(torch.float64)], patch_grid=(1, 1))
    with pytest.raises(ValueError, match="non-negative"):
        attention_rollout([_layer([[0.0, -1.0], [0.0, 0.0]])], patch_grid=(1, 1))
    with pytest.raises(ValueError, match="finite"):
        attention_rollout([_layer([[0.0, float("nan")], [0.0, 0.0]])], patch_grid=(1, 1))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"patch_grid": (0, 1)}, "patch_grid"),
        ({"patch_grid": (1, 3)}, "patch_grid must contain exactly"),
        ({"patch_grid": "12"}, "patch_grid"),
        ({"patch_grid": (1, 1), "cls_token_index": 2}, "cls_token_index"),
        ({"patch_grid": (1, 1), "patch_token_indices": (0,)}, "must not include"),
        ({"patch_grid": (1, 2), "patch_token_indices": (1, 1)}, "unique"),
        ({"patch_grid": (1, 1), "patch_token_indices": "1"}, "patch_token_indices"),
    ],
)
def test_grid_and_token_selection_are_validated(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        attention_rollout([_layer([[0.0, 1.0], [0.0, 0.0]])], **kwargs)


def test_direct_result_construction_validates_and_clones_outputs():
    heatmap = torch.zeros((1, 1, 1), requires_grad=True)
    joint = torch.eye(2).unsqueeze(0)
    result = AttentionRolloutResult(
        heatmap=heatmap,
        joint_attention=joint,
        patch_grid=(1, 1),
        cls_token_index=0,
        patch_token_indices=(1,),
    )
    assert not result.heatmap.requires_grad
    assert result.heatmap.data_ptr() != heatmap.data_ptr()
    with pytest.raises(ValueError, match="class-agnostic"):
        AttentionRolloutResult(
            heatmap=torch.ones((1, 1, 1)),
            joint_attention=joint,
            patch_grid=(1, 1),
            cls_token_index=0,
            patch_token_indices=(1,),
            class_agnostic=False,
        )


def test_direct_result_rejects_inconsistent_heatmap():
    with pytest.raises(ValueError, match="heatmap must match"):
        AttentionRolloutResult(
            heatmap=torch.ones((1, 1, 1)),
            joint_attention=torch.eye(2).unsqueeze(0),
            patch_grid=(1, 1),
            cls_token_index=0,
            patch_token_indices=(1,),
        )


def test_direct_result_rejects_non_normalized_joint_attention():
    with pytest.raises(ValueError, match="rows must sum to 1"):
        AttentionRolloutResult(
            heatmap=torch.ones((1, 1, 1)),
            joint_attention=torch.tensor([[[1.0, 1.0], [0.0, 1.0]]]),
            patch_grid=(1, 1),
            cls_token_index=0,
            patch_token_indices=(1,),
        )


def test_nonfinite_or_negative_direct_result_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        AttentionRolloutResult(
            heatmap=torch.tensor([[[-1.0]]]),
            joint_attention=torch.eye(2).unsqueeze(0),
            patch_grid=(1, 1),
            cls_token_index=0,
            patch_token_indices=(1,),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_dtype_and_device_are_preserved():
    tensor = _layer([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64).cuda()
    result = attention_rollout([tensor], patch_grid=(1, 1))
    assert result.heatmap.device == tensor.device
    assert result.joint_attention.device == tensor.device
    assert result.heatmap.dtype is torch.float64

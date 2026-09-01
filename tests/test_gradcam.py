"""Focused tests for generic Grad-CAM."""

import pytest
import torch
from torch import nn

from src.explainability.gradcam import GradCAMResult, grad_cam


def test_gradcam_is_exported_by_explainability_facade():
    from src.explainability import GradCAMResult as ExportedResult, grad_cam as exported_grad_cam

    assert ExportedResult is GradCAMResult
    assert exported_grad_cam is grad_cam


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=2, bias=False)

    def forward(self, value):
        return self.conv(value)


def _tiny_model():
    model = TinyCNN()
    with torch.no_grad():
        model.conv.weight.fill_(1.0)
    return model


def test_known_cnn_localization_and_raw_logit_selection():
    model = _tiny_model()
    value = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)

    def score():
        raw_logit = model(value).flatten(1).sum(dim=1) / 10.0
        return {
            "raw_logit": raw_logit,
            "sigmoid_probability": torch.sigmoid(raw_logit),
        }

    result = grad_cam(model, model.conv, score, lambda output: output["raw_logit"])

    assert torch.equal(result.heatmap, torch.tensor([[[1.2, 1.6], [2.4, 2.8]]]))
    assert torch.equal(result.selected_logits, torch.tensor([8.0]))
    assert result.heatmap.max() > 1.0
    assert not model.conv._forward_hooks


def test_batch_transform_and_dtype_are_supported_without_normalization():
    model = _tiny_model().double()
    value = torch.ones((2, 1, 3, 3), dtype=torch.float64)

    def score():
        return model(value).mean(dim=(1, 2, 3))[:, None]

    result = grad_cam(
        model,
        model.conv,
        score,
        lambda output: output,
        activation_transform=lambda tensor: tensor * 2.0,
    )

    assert result.heatmap.shape == (2, 2, 2)
    assert result.heatmap.dtype is torch.float64
    assert result.selected_logits.dtype is torch.float64
    assert torch.equal(result.heatmap, torch.full((2, 2, 2), 4.0, dtype=torch.float64))
    assert torch.equal(result.heatmap[0], result.heatmap[1])


def test_outer_no_grad_and_repeated_calls_are_deterministic():
    model = _tiny_model()
    value = torch.ones((1, 1, 3, 3))
    score = lambda: model(value).sum(dim=(1, 2, 3))

    with torch.no_grad():
        first = grad_cam(model, model.conv, score, lambda output: output)
        second = grad_cam(model, model.conv, score, lambda output: output)
    assert torch.equal(first.heatmap, second.heatmap)
    assert torch.equal(first.selected_logits, second.selected_logits)


def test_mixed_training_flags_and_parameter_grads_are_restored():
    model = nn.Sequential(nn.BatchNorm2d(1), TinyCNN())
    target = model[1].conv
    model.train(True)
    model[0].train(False)
    model[1].train(True)
    model[1].conv.train(False)
    model[0].weight.grad = torch.full_like(model[0].weight, 3.0)
    model[0].bias.grad = torch.full_like(model[0].bias, -2.0)
    states = {module: module.training for module in model.modules()}
    grads = {
        parameter: None if parameter.grad is None else parameter.grad.clone()
        for parameter in model.parameters()
    }
    value = torch.ones((1, 1, 3, 3))

    result = grad_cam(
        model,
        target,
        lambda: model(value).sum(dim=(1, 2, 3)),
        lambda output: output,
    )
    assert result.heatmap.isfinite().all()
    assert all(module.training == state for module, state in states.items())
    assert all(
        (parameter.grad is None if grads[parameter] is None else torch.equal(parameter.grad, grads[parameter]))
        for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    "scoring,match",
    [
        (lambda model, value: lambda: value.sum(), "exactly once"),
        (lambda model, value: lambda: (model(value), model(value))[1], "exactly once"),
    ],
)
def test_target_must_execute_exactly_once(scoring, match):
    model = _tiny_model()
    value = torch.ones((1, 1, 3, 3))
    with pytest.raises(RuntimeError, match=match):
        grad_cam(model, model.conv, scoring(model, value), lambda output: output)
    assert not model.conv._forward_hooks


def test_target_membership_and_callable_inputs_are_validated():
    model = _tiny_model()
    other = nn.Conv2d(1, 1, 1)
    with pytest.raises(ValueError, match="belong"):
        grad_cam(model, other, lambda: None, lambda output: output)
    with pytest.raises(TypeError, match="scoring_callable"):
        grad_cam(model, model.conv, None, lambda output: output)
    with pytest.raises(TypeError, match="logit_selector"):
        grad_cam(model, model.conv, lambda: None, None)


def test_malformed_activations_transforms_selectors_and_gradients_are_rejected():
    class IntegerTarget(nn.Module):
        def forward(self, value):
            return value.to(torch.int64)

    model = nn.Sequential(IntegerTarget())
    value = torch.ones((1, 1, 2, 2))
    with pytest.raises(TypeError, match="floating"):
        grad_cam(model, model[0], lambda: model(value), lambda output: output)
    assert not model[0]._forward_hooks

    model = _tiny_model()
    with pytest.raises(ValueError, match="shape"):
        grad_cam(
            model,
            model.conv,
            lambda: model(value).sum(dim=(1, 2, 3)),
            lambda output: output,
            activation_transform=lambda tensor: tensor.flatten(1),
        )

    with pytest.raises(ValueError, match="shape"):
        grad_cam(
            model,
            model.conv,
            lambda: model(value).sum(dim=(1, 2, 3)),
            lambda output: torch.ones((1, 2)),
        )

    def disconnected_score():
        model(value)
        return torch.ones((1,), requires_grad=True)

    with pytest.raises(ValueError, match="connected"):
        grad_cam(model, model.conv, disconnected_score, lambda output: output)


def test_hook_cleanup_and_state_restore_on_scoring_exception():
    model = _tiny_model()
    model.train(True)
    value = torch.ones((1, 1, 3, 3))

    def fail():
        model(value)
        raise RuntimeError("scoring failed")

    with pytest.raises(RuntimeError, match="scoring failed"):
        grad_cam(model, model.conv, fail, lambda output: output)
    assert model.training
    assert not model.conv._forward_hooks


def test_direct_result_is_detached_cloned_and_validated():
    heatmap = torch.ones((1, 2, 2), requires_grad=True)
    logits = torch.tensor([[3.0]])
    result = GradCAMResult(heatmap, logits)
    assert result.selected_logits.shape == (1,)
    assert not result.heatmap.requires_grad
    assert result.heatmap.data_ptr() != heatmap.data_ptr()

    with pytest.raises(ValueError, match="non-negative"):
        GradCAMResult(torch.tensor([[[-1.0]]]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="shape"):
        GradCAMResult(torch.ones((1, 1, 1)), torch.ones((1, 2)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_device_is_preserved():
    model = _tiny_model().cuda()
    value = torch.ones((1, 1, 3, 3), device="cuda")
    result = grad_cam(
        model,
        model.conv,
        lambda: model(value).sum(dim=(1, 2, 3)),
        lambda output: output,
    )
    assert result.heatmap.device.type == "cuda"
    assert result.selected_logits.device.type == "cuda"

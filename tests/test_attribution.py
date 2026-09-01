"""Focused tests for vanilla gradients and Integrated Gradients."""

import pytest
import torch

from src.explainability.attribution import (
    GradientAttributionResult,
    integrated_gradients,
    vanilla_gradients,
)


def test_attribution_is_exported_by_explainability_facade():
    from src.explainability import (
        GradientAttributionResult as ExportedResult,
        integrated_gradients as exported_integrated_gradients,
        vanilla_gradients as exported_vanilla_gradients,
    )

    assert ExportedResult is GradientAttributionResult
    assert exported_vanilla_gradients is vanilla_gradients
    assert exported_integrated_gradients is integrated_gradients


def _selector(output):
    return output["raw_logit"]


def _linear_score(weights):
    def score(tensor):
        return {"raw_logit": (tensor * weights).flatten(1).sum(dim=1)}

    return score


def test_vanilla_gradients_uses_selector_raw_logit_and_preserves_input():
    input_tensor = torch.tensor([[2.0, -3.0], [1.0, 4.0]], requires_grad=True)
    original = input_tensor.detach().clone()
    weights = torch.tensor([[3.0, -2.0]])

    def score(tensor):
        raw_logit = (tensor * weights).flatten(1).sum(dim=1)
        return {
            "raw_logit": raw_logit,
            "sigmoid_probability": torch.sigmoid(raw_logit),
        }

    result = vanilla_gradients(input_tensor, score, _selector)

    assert result.method == "vanilla_gradients"
    assert result.steps == 1
    assert result.baseline_logits is None
    assert result.completeness_delta is None
    assert torch.equal(result.attribution, weights.expand_as(input_tensor))
    probability_gradient = (
        torch.sigmoid(result.input_logits)
        * (1.0 - torch.sigmoid(result.input_logits))
        * weights.flatten()
    )
    assert not torch.allclose(result.attribution, probability_gradient[:, None])
    assert torch.equal(input_tensor.detach(), original)
    assert input_tensor.requires_grad


def test_integrated_gradients_is_exact_for_linear_model_and_supports_broadcast_baseline():
    input_tensor = torch.tensor([[2.0, 4.0], [3.0, -1.0]], dtype=torch.float64)
    baseline = torch.tensor([1.0, 2.0], dtype=torch.float32)
    weights = torch.tensor([[3.0, -2.0]], dtype=torch.float64)
    result = integrated_gradients(
        input_tensor,
        _linear_score(weights),
        _selector,
        baseline=baseline,
        steps=8,
    )

    expected = (input_tensor - baseline.to(torch.float64)) * weights
    assert result.method == "integrated_gradients"
    assert result.steps == 8
    assert torch.allclose(result.attribution, expected)
    assert torch.allclose(result.baseline_logits, _linear_score(weights)(baseline.to(torch.float64))["raw_logit"])
    assert torch.allclose(result.input_logits, _linear_score(weights)(input_tensor)["raw_logit"])
    assert torch.allclose(result.completeness_delta, torch.zeros(2, dtype=torch.float64))
    assert result.attribution.dtype is input_tensor.dtype


def test_integrated_gradients_nonlinear_completeness_and_batch_shape():
    input_tensor = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float32)

    def score(tensor):
        return {"raw_logit": (tensor.square()[:, 0] + 2.0 * tensor[:, 1])}

    result = integrated_gradients(input_tensor, score, _selector, steps=128)
    assert result.attribution.shape == input_tensor.shape
    assert result.input_logits.shape == (2,)
    assert result.baseline_logits.shape == (2,)
    assert torch.allclose(result.completeness_delta, torch.zeros(2), atol=2e-5)


@pytest.mark.parametrize("selector_shape", ["vector", "column"])
def test_selectors_accept_b_or_b_one(selector_shape):
    tensor = torch.tensor([[1.0], [2.0]])

    def score(value):
        logits = value[:, 0] * 4.0
        return {"raw_logit": logits if selector_shape == "vector" else logits[:, None]}

    result = vanilla_gradients(tensor, score, _selector)
    assert result.input_logits.shape == (2,)
    assert torch.equal(result.attribution, torch.full_like(tensor, 4.0))


def test_gradient_calls_do_not_modify_parameter_grads_or_model_mode():
    model = torch.nn.Linear(2, 1)
    model.train(False)
    model.weight.grad = torch.full_like(model.weight, 7.0)
    model.bias.grad = torch.full_like(model.bias, -3.0)
    input_tensor = torch.tensor([[1.0, 2.0]])
    old_weight_grad = model.weight.grad.clone()
    old_bias_grad = model.bias.grad.clone()

    def score(value):
        return {"raw_logit": model(value)}

    result = integrated_gradients(input_tensor, score, _selector, steps=4)
    assert torch.equal(model.weight.grad, old_weight_grad)
    assert torch.equal(model.bias.grad, old_bias_grad)
    assert not model.training
    assert result.attribution.isfinite().all()


def test_both_apis_enable_grad_inside_outer_no_grad():
    value = torch.tensor([[2.0, -1.0]])
    score = _linear_score(torch.tensor([[3.0, 2.0]]))
    with torch.no_grad():
        vanilla = vanilla_gradients(value, score, _selector)
        integrated = integrated_gradients(value, score, _selector, steps=4)

    assert torch.equal(vanilla.attribution, torch.tensor([[3.0, 2.0]]))
    assert torch.allclose(integrated.attribution, value * torch.tensor([[3.0, 2.0]]))
    assert torch.allclose(integrated.completeness_delta, torch.zeros(1))


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda x: vanilla_gradients("not a tensor", _linear_score(torch.ones(1, 1)), _selector), "input_tensor"),
        (lambda x: vanilla_gradients(x, _linear_score(torch.ones(1, 1)), None), "callable"),
    ],
)
def test_invalid_callable_or_input_is_rejected(call, match):
    value = torch.ones((1, 1))
    with pytest.raises((TypeError, ValueError), match=match):
        call(value)


def test_invalid_logits_gradients_baselines_and_steps_are_rejected():
    input_tensor = torch.ones((2, 1))

    def bad_shape(value):
        return {"raw_logit": torch.ones((2, 2), device=value.device)}

    with pytest.raises(ValueError, match="shape"):
        vanilla_gradients(input_tensor, bad_shape, _selector)

    def bad_nonfinite(value):
        return {"raw_logit": torch.full((2,), float("nan"), device=value.device)}

    with pytest.raises(ValueError, match="finite"):
        vanilla_gradients(input_tensor, bad_nonfinite, _selector)

    def bad_gradient(value):
        return {"raw_logit": torch.sqrt(value[:, 0])}

    with pytest.raises(ValueError, match="finite"):
        vanilla_gradients(torch.zeros((2, 1)), bad_gradient, _selector)

    with pytest.raises(ValueError, match="broadcastable"):
        integrated_gradients(input_tensor, _linear_score(torch.ones(1, 1)), _selector, baseline=torch.ones(3), steps=2)
    with pytest.raises(ValueError, match="steps"):
        integrated_gradients(input_tensor, _linear_score(torch.ones(1, 1)), _selector, steps=0)
    with pytest.raises(ValueError, match="steps"):
        integrated_gradients(input_tensor, _linear_score(torch.ones(1, 1)), _selector, steps=True)


def test_direct_result_contract_validates_method_fields_shapes_and_clones():
    attribution = torch.ones((2, 1), requires_grad=True)
    input_logits = torch.tensor([1.0, 2.0])
    result = GradientAttributionResult(
        method="vanilla_gradients",
        attribution=attribution,
        input_logits=input_logits[:, None],
    )
    assert result.input_logits.shape == (2,)
    assert not result.attribution.requires_grad
    assert result.attribution.data_ptr() != attribution.data_ptr()

    with pytest.raises(ValueError, match="requires baseline"):
        GradientAttributionResult(
            method="integrated_gradients",
            attribution=torch.ones((2, 1)),
            input_logits=input_logits,
        )

    with pytest.raises(ValueError, match="completeness_delta"):
        GradientAttributionResult(
            method="integrated_gradients",
            attribution=torch.ones((2, 1)),
            input_logits=input_logits,
            baseline_logits=torch.zeros(2),
            completeness_delta=torch.zeros(2),
            steps=4,
        )
    with pytest.raises(ValueError, match="must not contain baseline"):
        GradientAttributionResult(
            method="vanilla_gradients",
            attribution=torch.ones((2, 1)),
            input_logits=input_logits,
            baseline_logits=torch.zeros(2),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_attribution_dtype_and_determinism(dtype):
    value = torch.tensor([[2.0], [3.0]], dtype=dtype)

    def score(tensor):
        return {"raw_logit": tensor[:, 0].pow(3)}

    first = integrated_gradients(value, score, _selector, steps=16)
    second = integrated_gradients(value, score, _selector, steps=16)
    assert first.attribution.dtype is dtype
    assert torch.equal(first.attribution, second.attribution)
    assert torch.equal(first.completeness_delta, second.completeness_delta)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_device_is_preserved():
    value = torch.tensor([[2.0]], device="cuda")
    result = vanilla_gradients(value, _linear_score(torch.ones(1, 1, device="cuda")), _selector)
    assert result.attribution.device.type == "cuda"

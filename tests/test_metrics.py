"""Metrics, the Metric protocol, and the model wrappers they are handed."""

import copy

import numpy as np
import pytest
import torch

from loss_landscapes.metrics import Loss, LossGradient, LossPerturbations, Metric, MetricPipeline
from loss_landscapes.model_interface.model_wrapper import (
    GeneralModelWrapper,
    SimpleModelWrapper,
    wrap_model,
)


@pytest.fixture
def wrapper(model):
    return wrap_model(model)


# --- Loss ------------------------------------------------------------------


def test_loss_matches_a_direct_evaluation(model, data, wrapper):
    inputs, target = data
    loss_fn = torch.nn.MSELoss()

    expected = loss_fn(model(inputs), target).item()

    assert Loss(loss_fn, inputs, target)(wrapper) == pytest.approx(expected)


def test_loss_returns_a_python_float(data, wrapper):
    inputs, target = data
    assert isinstance(Loss(torch.nn.MSELoss(), inputs, target)(wrapper), float)


# --- LossPerturbations -----------------------------------------------------


def test_loss_perturbations_returns_one_delta_per_direction(data, wrapper):
    inputs, target = data
    metric = LossPerturbations(torch.nn.MSELoss(), inputs, target, n_directions=5, alpha=0.1)

    assert metric(wrapper).shape == (5,)


def test_loss_perturbations_restores_the_starting_parameters(model, data, wrapper):
    inputs, target = data
    before = [p.detach().clone() for p in model.parameters()]

    LossPerturbations(torch.nn.MSELoss(), inputs, target, n_directions=3, alpha=0.1)(wrapper)

    for old, new in zip(before, model.parameters()):
        assert torch.allclose(old, new, atol=1e-6)


# --- MetricPipeline --------------------------------------------------------


def test_metric_pipeline_returns_each_metric_in_order(data, wrapper):
    inputs, target = data

    class Constant(Metric):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def __call__(self, model_wrapper):
            return self.value

    assert MetricPipeline([Constant(1), Constant(2)])(wrapper) == (1, 2)


def test_metric_is_abstract():
    with pytest.raises(TypeError):
        Metric()


# --- wrappers --------------------------------------------------------------


def test_wrap_model_accepts_a_module(model):
    assert isinstance(wrap_model(model), SimpleModelWrapper)


def test_wrap_model_passes_wrappers_through(model):
    wrapped = SimpleModelWrapper(model)
    assert wrap_model(wrapped) is wrapped


def test_wrap_model_rejects_anything_else():
    with pytest.raises(ValueError):
        wrap_model(object())


def test_wrapping_disables_gradient_tracking(model):
    wrap_model(model)
    assert all(not p.requires_grad for p in model.parameters())


def test_simple_wrapper_forward_matches_the_model(model, data):
    inputs, _ = data
    assert torch.allclose(SimpleModelWrapper(model).forward(inputs), model(inputs))


def test_general_wrapper_uses_the_supplied_forward_function(model, data):
    inputs, _ = data
    wrapper = GeneralModelWrapper(model, [model], lambda m, x: m(x) * 0)

    assert torch.allclose(wrapper.forward(inputs), torch.zeros_like(model(inputs)))


def test_general_wrapper_collects_parameters_from_every_module(model, other_model):
    wrapper = GeneralModelWrapper(None, [model, other_model], lambda m, x: x)
    expected = len(list(model.parameters())) + len(list(other_model.parameters()))

    assert len(wrapper.get_module_parameters()) == expected


def test_train_and_eval_propagate_to_every_module(model, other_model):
    wrapper = GeneralModelWrapper(None, [model, other_model], lambda m, x: x)

    wrapper.eval()
    assert not model.training and not other_model.training

    wrapper.train()
    assert model.training and other_model.training


def test_wrapper_parameters_yields_every_module_parameter_in_order(model, other_model):
    wrapper = GeneralModelWrapper(None, [model, other_model], lambda m, x: x)
    expected = [*model.parameters(), *other_model.parameters()]

    actual = list(wrapper.parameters())

    assert len(actual) == len(expected)
    assert all(a is e for a, e in zip(actual, expected, strict=True))


def test_wrapper_named_parameters_yields_every_module_name_and_parameter(model, other_model):
    wrapper = GeneralModelWrapper(None, [model, other_model], lambda m, x: x)
    expected = [*model.named_parameters(), *other_model.named_parameters()]

    actual = list(wrapper.named_parameters())

    assert [name for name, _ in actual] == [name for name, _ in expected]
    assert all(a is e for (_, a), (_, e) in zip(actual, expected, strict=True))


# --- LossGradient ----------------------------------------------------------


def test_loss_gradient_matches_a_direct_evaluation(model, data, wrapper):
    inputs, target = data
    reference = copy.deepcopy(model).requires_grad_(True)
    torch.nn.MSELoss()(reference(inputs), target).backward()
    expected = np.concatenate([p.grad.numpy().flatten() for p in reference.parameters()])

    gradient = LossGradient(torch.nn.MSELoss(), inputs, target)(wrapper)

    assert isinstance(gradient, np.ndarray)
    np.testing.assert_allclose(gradient, expected, rtol=1e-6)


def test_loss_gradient_restores_gradient_tracking(model, data, wrapper):
    inputs, target = data
    LossGradient(torch.nn.MSELoss(), inputs, target)(wrapper)

    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_loss_gradient_works_under_no_grad(data, wrapper):
    inputs, target = data
    with torch.no_grad():
        gradient = LossGradient(torch.nn.MSELoss(), inputs, target)(wrapper)

    assert np.any(gradient != 0)


def test_loss_gradient_is_zero_for_modules_the_loss_does_not_use(model, other_model, data):
    inputs, target = data
    wrapper = wrap_model(GeneralModelWrapper(model, [model, other_model], lambda m, x: m(x)))
    n_used = sum(p.numel() for p in model.parameters())

    gradient = LossGradient(torch.nn.MSELoss(), inputs, target)(wrapper)

    assert gradient.shape == (n_used + sum(p.numel() for p in other_model.parameters()),)
    assert np.any(gradient[:n_used] != 0)
    assert np.all(gradient[n_used:] == 0)


def test_loss_gradient_rejects_a_loss_that_uses_no_wrapped_parameter(model, data):
    inputs, target = data
    wrapper = wrap_model(GeneralModelWrapper(model, [model], lambda m, x: x[:, :2]))

    with pytest.raises(ValueError, match="does not depend on any of the wrapped parameters"):
        LossGradient(torch.nn.MSELoss(), inputs, target)(wrapper)

    assert all(not p.requires_grad for p in model.parameters())


def test_loss_gradient_densifies_sparse_gradients():
    sparse = torch.nn.Embedding(10, 3, sparse=True)
    dense = torch.nn.Embedding(10, 3)
    dense.load_state_dict(sparse.state_dict())
    indices, target = torch.tensor([1, 4, 1, 7]), torch.rand(4, 3)

    dense.requires_grad_(True)
    torch.nn.MSELoss()(dense(indices), target).backward()

    gradient = LossGradient(torch.nn.MSELoss(), indices, target)(wrap_model(sparse))

    np.testing.assert_allclose(gradient, dense.weight.grad.numpy().flatten(), rtol=1e-6)

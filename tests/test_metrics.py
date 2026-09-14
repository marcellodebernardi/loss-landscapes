"""Metrics, the Metric protocol, and the model wrappers they are handed."""

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


# TODO: fix ModelWrapper.parameters()/named_parameters(), then drop this marker.
@pytest.mark.xfail(
    reason="ModelWrapper.parameters()/named_parameters() call itertools.chain on a list of "
    "generators instead of chaining the generators themselves, so they yield generator "
    "objects rather than parameters.",
    strict=True,
)
def test_wrapper_parameters_yields_tensors(model):
    assert all(isinstance(p, torch.Tensor) for p in wrap_model(model).parameters())


# TODO: let LossGradient keep gradients enabled, then drop this marker.
@pytest.mark.xfail(
    reason="wrap_model() calls requires_grad_(False) on every parameter, so nothing reached "
    "through a wrapper is part of an autograd graph and LossGradient cannot differentiate "
    "at all. This fires before the named_parameters() defect above is even reached, so both "
    "have to be fixed before LossGradient works.",
    strict=True,
    raises=RuntimeError,
)
def test_loss_gradient_returns_a_gradient(model, data, wrapper):
    inputs, target = data
    gradient = LossGradient(torch.nn.MSELoss(), inputs, target)(wrapper)

    assert isinstance(gradient, np.ndarray)


def test_wrapping_makes_gradients_uncomputable(model, data, wrapper):
    """Pin the cause of the xfail above, so fixing one defect does not mask the other.

    Once requires_grad is restored this test fails, which is the signal that the
    LossGradient xfail can be revisited.
    """
    inputs, target = data
    loss = torch.nn.MSELoss()(wrapper.forward(inputs), target)

    assert not loss.requires_grad

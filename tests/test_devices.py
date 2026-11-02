"""Everything runs on the device the model's parameters are on.

Each test runs once per device in `available_devices()`: CPU everywhere, plus CUDA and MPS
where present. Results computed on an accelerator are compared with the same computation on
CPU wherever the library is deterministic.
"""

import copy

import numpy as np
import pytest
import torch

import loss_landscapes
from loss_landscapes.metrics import Loss, LossGradient, LossPerturbations
from loss_landscapes.model_interface.model_parameters import ModelParameters, rand_n_like, rand_u_like

NORMALIZATIONS = ["filter", "layer", "model", None]


def on(device, module):
    return copy.deepcopy(module).to(device)


def assert_on(device, module):
    assert all(p.device.type == device.type for p in module.parameters())


@pytest.fixture
def loss_fn():
    return torch.nn.MSELoss()


# --- ModelParameters -------------------------------------------------------


@pytest.mark.parametrize("sampler", [rand_u_like, rand_n_like])
def test_random_vectors_match_each_tensor_device_and_dtype(device, model, sampler):
    parameters = ModelParameters([p.detach() for p in on(device, model).parameters()])

    for new, old in zip(sampler(parameters), parameters, strict=True):
        assert new.device == old.device
        assert new.dtype == old.dtype
        assert new.shape == old.shape


def test_norms_and_dot_match_cpu(device, parameters, other_parameters):
    def to(params):
        return ModelParameters([p.to(device) for p in params])

    moved, other = to(parameters), to(other_parameters)

    assert moved.model_norm() == pytest.approx(parameters.model_norm(), rel=1e-6)
    assert moved.layer_norm(0) == pytest.approx(parameters.layer_norm(0), rel=1e-6)
    assert moved.filter_norm((0, 1)) == pytest.approx(parameters.filter_norm((0, 1)), rel=1e-6)
    assert moved.dot(other) == pytest.approx(parameters.dot(other_parameters), rel=1e-6)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_float16_norms_and_dot_do_not_overflow(device, order):
    """float16 tops out at 65,504, so summing 100,000 ones in float16 would overflow.

    MPS has no float64 and accumulates in float32, so this covers that path separately.
    """
    ones = ModelParameters([torch.ones(2, 100_000, dtype=torch.float16, device=device)])
    row = 100_000 ** (1 / order)

    assert ones.model_norm(order) == pytest.approx(2 ** (1 / order) * row)
    assert ones.layer_norm(0, order) == pytest.approx(2 ** (1 / order) * row)
    assert ones.filter_norm((0, 1), order) == pytest.approx(row)
    assert ones.dot(ones) == pytest.approx(200_000)


def test_as_numpy_copies_to_host_memory(device, parameters):
    moved = ModelParameters([p.to(device) for p in parameters])
    np.testing.assert_array_equal(moved.as_numpy(), parameters.as_numpy())


# --- landscapes ------------------------------------------------------------


def test_point_matches_cpu(device, model, data, loss_fn):
    inputs, target = data
    expected = loss_landscapes.point(model, Loss(loss_fn, inputs, target))

    moved = on(device, model)
    actual = loss_landscapes.point(moved, Loss(loss_fn, inputs.to(device), target.to(device)))

    assert actual == pytest.approx(expected, rel=1e-5)
    assert_on(device, moved)


def test_linear_interpolation_matches_cpu(device, model, other_model, data, loss_fn):
    inputs, target = data
    expected = loss_landscapes.linear_interpolation(
        model, other_model, Loss(loss_fn, inputs, target), steps=5, deepcopy_model=True
    )

    moved = on(device, model)
    metric = Loss(loss_fn, inputs.to(device), target.to(device))
    actual = loss_landscapes.linear_interpolation(moved, on(device, other_model), metric, steps=5)

    np.testing.assert_allclose(actual, expected, rtol=1e-5)
    assert_on(device, moved)


def test_planar_interpolation_matches_cpu(device, model, other_model, data, loss_fn):
    inputs, target = data
    torch.manual_seed(0)
    third = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    expected = loss_landscapes.planar_interpolation(
        model, other_model, third, Loss(loss_fn, inputs, target), steps=4, deepcopy_model=True
    )

    moved = on(device, model)
    metric = Loss(loss_fn, inputs.to(device), target.to(device))
    actual = loss_landscapes.planar_interpolation(moved, on(device, other_model), on(device, third), metric, steps=4)

    np.testing.assert_allclose(actual, expected, rtol=1e-5)
    assert_on(device, moved)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_random_line_travels_the_requested_distance(device, model, data, loss_fn, normalization):
    inputs, target = data
    moved = on(device, model)
    before = ModelParameters([p.detach().clone() for p in moved.parameters()])
    metric = Loss(loss_fn, inputs.to(device), target.to(device))

    result = loss_landscapes.random_line(moved, metric, distance=0.25, steps=4, normalization=normalization)

    assert result.shape == (4,) and np.isfinite(result).all()
    displacement = np.linalg.norm(
        ModelParameters([p.detach() for p in moved.parameters()]).as_numpy() - before.as_numpy()
    )
    assert displacement == pytest.approx(before.model_norm() * 0.25, rel=1e-4)
    assert_on(device, moved)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_random_plane_returns_a_finite_grid(device, model, data, loss_fn, normalization):
    inputs, target = data
    moved = on(device, model)
    metric = Loss(loss_fn, inputs.to(device), target.to(device))

    result = loss_landscapes.random_plane(moved, metric, distance=1, steps=4, normalization=normalization)

    assert result.shape == (4, 4) and np.isfinite(result).all()
    assert_on(device, moved)


# --- metrics ---------------------------------------------------------------


def test_loss_gradient_matches_cpu(device, model, data, loss_fn):
    inputs, target = data
    expected = loss_landscapes.point(copy.deepcopy(model), LossGradient(loss_fn, inputs, target))

    moved = on(device, model)
    actual = loss_landscapes.point(moved, LossGradient(loss_fn, inputs.to(device), target.to(device)))

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
    assert_on(device, moved)


def test_loss_perturbations_restores_the_parameters(device, model, data, loss_fn):
    inputs, target = data
    moved = on(device, model)
    before = [p.detach().clone() for p in moved.parameters()]
    metric = LossPerturbations(loss_fn, inputs.to(device), target.to(device), n_directions=3, alpha=0.1)

    deltas = loss_landscapes.point(moved, metric)

    assert deltas.shape == (3, 2) and np.isfinite(deltas).all()
    for old, new in zip(before, moved.parameters(), strict=True):
        assert torch.allclose(old, new, atol=1e-6)
    assert_on(device, moved)

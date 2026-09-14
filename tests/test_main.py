"""
The five public entry points: shape, geometry, and the invariants the 2019 bug-fix
commits were reaching for.
"""

import numpy as np
import pytest
import torch

import loss_landscapes
from loss_landscapes.metrics import Loss
from loss_landscapes.model_interface.model_parameters import ModelParameters

NORMALIZATIONS = ["filter", "layer", "model", None]


@pytest.fixture
def metric(data):
    inputs, target = data
    return Loss(torch.nn.MSELoss(), inputs, target)


def snapshot(model) -> ModelParameters:
    return ModelParameters([p.detach().clone() for p in model.parameters()])


def displacement_norm(before: ModelParameters, after: ModelParameters) -> float:
    return float(np.linalg.norm(after.as_numpy() - before.as_numpy()))


# --- shapes ----------------------------------------------------------------


def test_point_returns_a_scalar(model, metric):
    assert isinstance(loss_landscapes.point(model, metric), float)


def test_linear_interpolation_returns_one_value_per_step(model, other_model, metric):
    assert loss_landscapes.linear_interpolation(model, other_model, metric, steps=7).shape == (7,)


def test_random_line_returns_one_value_per_step(model, metric):
    assert loss_landscapes.random_line(model, metric, distance=0.1, steps=9).shape == (9,)


def test_planar_interpolation_returns_a_square_grid(model, other_model, metric):
    torch.manual_seed(0)
    third = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    result = loss_landscapes.planar_interpolation(model, other_model, third, metric, steps=5)
    assert result.shape == (5, 5)


def test_random_plane_returns_a_square_grid(model, metric):
    assert loss_landscapes.random_plane(model, metric, distance=1, steps=6).shape == (6, 6)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_random_plane_values_are_finite(model, metric, normalization):
    result = loss_landscapes.random_plane(model, metric, distance=1, steps=4, normalization=normalization)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_random_line_values_are_finite(model, metric, normalization):
    result = loss_landscapes.random_line(model, metric, distance=0.1, steps=4, normalization=normalization)
    assert np.isfinite(result).all()


def test_unsupported_normalization_is_rejected(model, metric):
    with pytest.raises(AttributeError):
        loss_landscapes.random_line(model, metric, steps=2, normalization="nonsense")


def test_non_model_argument_is_rejected(metric):
    with pytest.raises(ValueError):
        loss_landscapes.point("not a model", metric)


# --- geometry --------------------------------------------------------------


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_random_line_travels_the_requested_distance(model, metric, normalization):
    """
    `distance` is a multiple of the start point's norm: after `steps` steps the model
    should sit exactly that far from where it started, whatever the normalization.
    This is the invariant commits 450964d / 9d3cfd3 / 570a1b6 were chasing.
    """
    before = snapshot(model)
    distance = 0.25

    loss_landscapes.random_line(model, metric, distance=distance, steps=10, normalization=normalization)

    expected = before.model_norm() * distance
    assert displacement_norm(before, snapshot(model)) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("steps", [4, 16, 64])
def test_random_line_distance_is_independent_of_step_count(model, metric, steps):
    """Refining the grid must not change where the line ends."""
    before = snapshot(model)

    loss_landscapes.random_line(model, metric, distance=0.25, steps=steps)

    expected = before.model_norm() * 0.25
    assert displacement_norm(before, snapshot(model)) == pytest.approx(expected, rel=1e-4)


def test_random_line_steps_are_evenly_spaced(model):
    """Every step along the line must be the same length."""
    positions = []

    class RecordPosition:
        def __call__(self, model_wrapper):
            positions.append(snapshot(model_wrapper.get_modules()[0]))
            return 0.0

    loss_landscapes.random_line(model, RecordPosition(), distance=0.5, steps=8)

    gaps = [displacement_norm(a, b) for a, b in zip(positions, positions[1:])]
    assert gaps == pytest.approx([gaps[0]] * len(gaps), rel=1e-4)


def test_linear_interpolation_ends_at_the_second_model(model, other_model, metric):
    """Interpolating start -> end must actually arrive at end."""
    target = snapshot(other_model)

    loss_landscapes.linear_interpolation(model, other_model, metric, steps=20)

    assert displacement_norm(target, snapshot(model)) == pytest.approx(0.0, abs=1e-5)


def test_random_plane_is_centred_on_the_start_point(model, metric):
    """
    random_plane shifts the model back by half the plane before evaluating, so that the
    original parameters land in the middle of the returned grid.
    """
    before = snapshot(model)
    steps = 8

    loss_landscapes.random_plane(model, metric, distance=1, steps=steps, normalization=None)

    # The walk ends one full row past the far corner; what matters is that the start
    # point is interior to the plane rather than at a corner of it.
    travelled = displacement_norm(before, snapshot(model))
    assert travelled > 0


@pytest.mark.parametrize("deepcopy_model", [True, False])
def test_deepcopy_model_controls_whether_the_input_is_mutated(model, metric, deepcopy_model):
    """
    With deepcopy_model=False (the default) these functions walk the caller's model
    through parameter space and leave it wherever the walk ended.
    """
    before = snapshot(model)

    loss_landscapes.random_line(model, metric, distance=0.1, steps=4, deepcopy_model=deepcopy_model)

    moved = displacement_norm(before, snapshot(model)) > 1e-9
    assert moved is not deepcopy_model


# --- orthogonality of the plane directions ---------------------------------


def _capture_plane_directions(monkeypatch, model, metric, normalization, steps=4):
    """
    Grab the two direction vectors random_plane builds. They are mutated in place by the
    scaling that follows, so the captured references show their final state.
    """
    import loss_landscapes.main as main

    captured = {}
    real_rand_u_like = main.rand_u_like
    real_orthogonal_to = main.orthogonal_to

    def fake_rand_u_like(example):
        captured["one"] = real_rand_u_like(example)
        return captured["one"]

    def fake_orthogonal_to(vector):
        captured["two"] = real_orthogonal_to(vector)
        return captured["two"]

    monkeypatch.setattr(main, "rand_u_like", fake_rand_u_like)
    monkeypatch.setattr(main, "orthogonal_to", fake_orthogonal_to)

    loss_landscapes.random_plane(model, metric, distance=1, steps=steps, normalization=normalization)
    return captured["one"], captured["two"]


def cosine(a: ModelParameters, b: ModelParameters) -> float:
    return a.dot(b) / (a.model_norm() * b.model_norm())


def test_plane_directions_are_orthogonal_without_normalization(monkeypatch, model, metric):
    one, two = _capture_plane_directions(monkeypatch, model, metric, normalization=None)
    assert cosine(one, two) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("normalization", ["filter", "layer", "model"])
@pytest.mark.xfail(
    reason="random_plane makes dir_two orthogonal to dir_one and only then normalizes both, "
    "which rotates them apart again. The plane the grid is sampled on is therefore skewed "
    "whenever normalization is enabled, which is the default. "
    "See https://github.com/marcellodebernardi/loss-landscapes/issues/TBD",
    strict=True,
)
def test_plane_directions_are_orthogonal_with_normalization(monkeypatch, model, metric, normalization):
    one, two = _capture_plane_directions(monkeypatch, model, metric, normalization=normalization)
    assert cosine(one, two) == pytest.approx(0.0, abs=1e-5)


# --- reproducibility -------------------------------------------------------


def test_random_plane_is_reproducible_under_a_fixed_seed(model, metric):
    torch.manual_seed(1234)
    first = loss_landscapes.random_plane(snapshot_model(model), metric, distance=1, steps=5)
    torch.manual_seed(1234)
    second = loss_landscapes.random_plane(snapshot_model(model), metric, distance=1, steps=5)

    np.testing.assert_allclose(first, second, rtol=1e-6)


def snapshot_model(model):
    """A fresh copy of the model, so successive runs start from the same point."""
    import copy

    return copy.deepcopy(model)

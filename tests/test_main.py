"""Tests for the five public entry points.

Covers shape, geometry, and the invariants the 2019 bug-fix commits were reaching for.
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
def test_random_line_direction_has_entries_of_both_signs(model, metric, normalization):
    """Directions are Gaussian, so the line must not move every parameter the same way.

    Normalisation only rescales blocks by positive factors, so it preserves signs, and a
    uniform [0, 1) direction would move every parameter upwards.
    """
    before = snapshot(model)

    loss_landscapes.random_line(model, metric, distance=0.25, steps=4, normalization=normalization)

    displacement = snapshot(model).as_numpy() - before.as_numpy()
    assert (displacement > 0).any() and (displacement < 0).any()


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_random_line_travels_the_requested_distance(model, metric, normalization):
    """The line must end exactly `distance` from where it started.

    `distance` is a multiple of the start point's norm, and the endpoint should sit that
    far away whatever the normalization. This is the invariant that commits 450964d,
    9d3cfd3 and 570a1b6 were chasing.
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
    """The original parameters must land in the middle of the sampled grid.

    random_plane shifts the model back by half the plane before evaluating. The centroid
    of the points it evaluates should therefore sit on the original start point, give or
    take the half-step the grid is offset by.
    """
    before = snapshot(model)
    steps = 8
    positions = []

    class RecordPosition:
        def __call__(self, model_wrapper):
            positions.append(snapshot(model_wrapper.get_modules()[0]))
            return 0.0

    loss_landscapes.random_plane(model, RecordPosition(), distance=1, steps=steps, normalization=None)

    assert len(positions) == steps * steps

    centroid = np.mean([p.as_numpy() for p in positions], axis=0)
    offset = float(np.linalg.norm(centroid - before.as_numpy()))

    # One grid step, as a generous tolerance on "the middle": the walk starts a half-step
    # into the plane, so the centroid is displaced by that much and no more.
    grid_step = before.model_norm() * 1.0 / steps
    assert offset < grid_step


@pytest.mark.parametrize("deepcopy_model", [True, False])
def test_deepcopy_model_controls_whether_the_input_is_mutated(model, metric, deepcopy_model):
    """deepcopy_model decides whether the caller's model is left where the walk ended.

    With deepcopy_model=False, the default, these functions walk the caller's own model
    through parameter space rather than a copy of it.
    """
    before = snapshot(model)

    loss_landscapes.random_line(model, metric, distance=0.1, steps=4, deepcopy_model=deepcopy_model)

    moved = displacement_norm(before, snapshot(model)) > 1e-9
    assert moved is not deepcopy_model


# --- orthogonality of the plane directions ---------------------------------


def _capture_plane_directions(monkeypatch, model, metric, normalization, steps=4, ignore_1d=None):
    """Grab the two direction vectors random_plane builds.

    They are mutated in place by the scaling that follows, so the captured references
    show their final, scaled state.

    Args:
        monkeypatch: pytest fixture used to intercept the direction constructors.
        model: model defining the origin of the plane.
        metric: metric to evaluate at each grid point.
        normalization: normalization passed through to random_plane.
        steps: grid resolution.
        ignore_1d: ignore_1d passed through to random_plane.

    Returns:
        The two direction vectors, as they stood after random_plane finished.
    """
    import loss_landscapes.main as main

    captured = {}
    real_rand_n_like = main.rand_n_like
    real_orthogonal_to = main.orthogonal_to

    def fake_rand_n_like(example):
        captured["one"] = real_rand_n_like(example)
        return captured["one"]

    def fake_orthogonal_to(vector):
        captured["two"] = real_orthogonal_to(vector)
        return captured["two"]

    monkeypatch.setattr(main, "rand_n_like", fake_rand_n_like)
    monkeypatch.setattr(main, "orthogonal_to", fake_orthogonal_to)

    loss_landscapes.random_plane(
        model, metric, distance=1, steps=steps, normalization=normalization, ignore_1d=ignore_1d
    )
    return captured["one"], captured["two"]


def cosine(a: ModelParameters, b: ModelParameters) -> float:
    return a.dot(b) / (a.model_norm() * b.model_norm())


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_plane_directions_are_orthogonal(monkeypatch, model, metric, normalization):
    """The grid is plotted on perpendicular axes, so the directions must be orthogonal."""
    one, two = _capture_plane_directions(monkeypatch, model, metric, normalization=normalization)
    assert cosine(one, two) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_plane_directions_have_entries_of_both_signs(monkeypatch, model, metric, normalization):
    """Both directions are Gaussian, not uniform on [0, 1), so neither is all-positive."""
    one, two = _capture_plane_directions(monkeypatch, model, metric, normalization=normalization)
    for direction in (one.as_numpy(), two.as_numpy()):
        assert (direction > 0).any() and (direction < 0).any()


def _block_norm_ratios(direction: ModelParameters, reference: ModelParameters, normalization: str) -> list:
    """Ratio of each layer or filter norm of `direction` to the matching norm of `reference`."""
    if normalization == "layer":
        return [direction.layer_norm(i) / reference.layer_norm(i) for i in range(len(reference))]
    ratios = []
    for i, parameter in enumerate(reference):
        if parameter.dim() > 1:
            ratios.extend(direction.filter_norm((i, f)) / reference.filter_norm((i, f)) for f in range(len(parameter)))
    return ratios


@pytest.mark.parametrize("normalization", ["filter", "layer"])
def test_plane_directions_keep_their_normalization(monkeypatch, model, metric, normalization):
    """Orthogonalising must not undo layer/filter normalisation of either direction.

    Both directions are scaled by one factor each after normalisation, so a normalised
    direction has the same ratio to the start point in every layer or filter.
    """
    start = snapshot(model)
    one, two = _capture_plane_directions(monkeypatch, model, metric, normalization=normalization)
    for direction in (one, two):
        ratios = _block_norm_ratios(direction, start, normalization)
        assert ratios == pytest.approx([ratios[0]] * len(ratios), rel=1e-4)
        if normalization == "filter":  # 1-D tensors are left unperturbed by default
            assert all(not direction[i].any() for i, p in enumerate(start) if p.dim() <= 1)


def test_plane_directions_are_orthogonal_when_normalization_cannot_converge(monkeypatch):
    """Single-element filters can stall the iteration; orthogonality must still hold."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    inputs, targets = torch.rand(4, 1), torch.rand(4, 1)
    metric = Loss(torch.nn.MSELoss(), inputs, targets)

    one, two = _capture_plane_directions(monkeypatch, model, metric, normalization="filter")
    assert cosine(one, two) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("normalization", ["layer", "filter"])
def test_random_plane_handles_zero_biases(monkeypatch, model, metric, normalization):
    """Zero-initialised biases have zero reference norms, which repeated normalisation must tolerate."""
    with torch.no_grad():
        for layer in (model[0], model[2]):
            layer.bias.zero_()

    one, two = _capture_plane_directions(monkeypatch, model, metric, normalization=normalization)
    assert cosine(one, two) == pytest.approx(0.0, abs=1e-5)
    assert np.isfinite(loss_landscapes.random_plane(model, metric, steps=3, normalization=normalization)).all()


def test_random_plane_rejects_a_model_with_one_parameter():
    model = torch.nn.Linear(1, 1, bias=False)
    metric = Loss(torch.nn.MSELoss(), torch.rand(4, 1), torch.rand(4, 1))
    with pytest.raises(ValueError):
        loss_landscapes.random_plane(model, metric, steps=3)


@pytest.mark.parametrize("sampler", [loss_landscapes.random_line, loss_landscapes.random_plane])
def test_model_normalization_preserves_the_sampled_subspace(sampler, model, metric):
    """Global normalisation must preserve samples after the common distance scaling."""
    torch.manual_seed(123)
    unnormalised = sampler(model, metric, distance=0.5, steps=4, normalization=None, deepcopy_model=True)
    torch.manual_seed(123)
    normalised = sampler(model, metric, distance=0.5, steps=4, normalization="model", deepcopy_model=True)

    np.testing.assert_allclose(normalised, unnormalised, rtol=1e-5, atol=1e-6)


# --- low precision ---------------------------------------------------------


@pytest.fixture
def half_model():
    """A float16 model with a 100,000-element weight, whose squared norm overflows float16."""
    torch.manual_seed(0)
    return torch.nn.Linear(1000, 100).half()


@pytest.fixture
def half_metric():
    torch.manual_seed(0)
    return Loss(torch.nn.MSELoss(), torch.rand(4, 1000).half(), torch.rand(4, 100).half())


def _as_float64(parameters: ModelParameters) -> np.ndarray:
    return parameters.as_numpy().astype(np.float64)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_random_line_travels_the_requested_distance_for_float16_models(half_model, half_metric, normalization):
    """The line must move every tensor it perturbs, not just the ones whose norm does not overflow.

    Filter normalization leaves 1-D tensors unperturbed by default, as in Li et al. (2018).
    """
    before_tensors = [p.detach().clone() for p in half_model.parameters()]
    before = _as_float64(snapshot(half_model))

    result = loss_landscapes.random_line(half_model, half_metric, distance=0.25, steps=4, normalization=normalization)

    assert np.isfinite(result).all()
    displacement = np.linalg.norm(_as_float64(snapshot(half_model)) - before)
    assert displacement == pytest.approx(np.linalg.norm(before) * 0.25, rel=1e-2)
    for old, new in zip(before_tensors, half_model.parameters(), strict=True):
        assert torch.equal(old, new.detach()) == (normalization == "filter" and old.dim() <= 1)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_random_plane_is_finite_and_not_flat_for_float16_models(half_model, half_metric, normalization):
    result = loss_landscapes.random_plane(half_model, half_metric, distance=1, steps=3, normalization=normalization)

    assert np.isfinite(result).all()
    assert np.ptp(result) > 0


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


# --- ignore_1d -------------------------------------------------------------


def _moved_tensors(model, sampler, **kwargs) -> list:
    """Run a sampler on the model and report, per parameter tensor, whether it moved."""
    before = [p.detach().clone() for p in model.parameters()]
    inputs, targets = torch.rand(8, 4), torch.rand(8, 2)
    sampler(model, Loss(torch.nn.MSELoss(), inputs, targets), steps=3, **kwargs)
    return [not torch.equal(old, new.detach()) for old, new in zip(before, model.parameters(), strict=True)]


@pytest.mark.parametrize("sampler", [loss_landscapes.random_line, loss_landscapes.random_plane])
@pytest.mark.parametrize(
    ("normalization", "ignore_1d", "biases_move"),
    [
        ("filter", None, False),
        ("filter", False, True),
        ("layer", None, True),
        ("layer", True, False),
        ("model", None, True),
        ("model", True, False),
        (None, None, True),
        (None, True, False),
    ],
)
def test_ignore_1d_decides_whether_1d_tensors_move(model, sampler, normalization, ignore_1d, biases_move):
    """By default only filter normalization leaves 1-D tensors in place, as in Li et al. (2018)."""
    moved = _moved_tensors(model, sampler, normalization=normalization, ignore_1d=ignore_1d)

    for p, did_move in zip(model.parameters(), moved, strict=True):
        assert did_move == (biases_move or p.dim() > 1)


@pytest.mark.parametrize("sampler", [loss_landscapes.random_line, loss_landscapes.random_plane])
def test_ignore_1d_rejects_models_with_nothing_left_to_perturb(sampler):
    class BiasOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([0.5, -1.0]))

        def forward(self, x):
            return x + self.bias

    metric = Loss(torch.nn.MSELoss(), torch.rand(4, 2), torch.rand(4, 2))

    with pytest.raises(ValueError, match="ignore_1d"):
        sampler(BiasOnly(), metric, steps=2)
    assert np.isfinite(sampler(BiasOnly(), metric, steps=2, ignore_1d=False)).all()


@pytest.mark.parametrize("normalization", ["filter", "layer", "model", None])
def test_plane_directions_are_orthogonal_when_ignoring_1d_tensors(monkeypatch, model, metric, normalization):
    """Zeroing 1-D tensors changes the angle between the directions, which must be corrected."""
    one, two = _capture_plane_directions(monkeypatch, model, metric, normalization=normalization, ignore_1d=True)
    assert cosine(one, two) == pytest.approx(0.0, abs=1e-5)


def _linear(n_in, n_out, weight, bias=None):
    model = torch.nn.Linear(n_in, n_out)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(weight))
        if bias is not None:
            model.bias.copy_(torch.tensor(bias))
    return model


@pytest.mark.parametrize("sampler", [loss_landscapes.random_line, loss_landscapes.random_plane])
def test_zero_weights_with_ignored_biases_leave_nothing_to_perturb(sampler):
    """Filter normalization zeroes zero filters and, by default, biases, leaving no direction."""
    metric = Loss(torch.nn.MSELoss(), torch.rand(4, 2), torch.rand(4, 2))

    def model():
        return _linear(2, 2, [[0.0, 0.0], [0.0, 0.0]], [0.5, -1.0])

    with pytest.raises(ValueError, match="no parameters to perturb after normalization"):
        sampler(model(), metric, steps=2)
    assert np.isfinite(sampler(model(), metric, steps=2, ignore_1d=False)).all()


def test_plane_directions_stay_orthogonal_when_normalization_leaves_only_signs(monkeypatch):
    """Normalised directions that coincide are resampled, not collapsed to zero.

    Single-element filters keep only the signs of a sample, so normalised directions often coincide.
    """
    metric = Loss(torch.nn.MSELoss(), torch.rand(4, 1), torch.rand(4, 2))

    for seed in range(50):
        torch.manual_seed(seed)
        one, two = _capture_plane_directions(monkeypatch, _linear(1, 2, [[1.0], [1.0]]), metric, "filter")
        assert two.model_norm() > 0
        assert cosine(one, two) == pytest.approx(0.0, abs=1e-5)


def test_random_plane_keeps_float16_directions_that_float16_resolves(monkeypatch):
    """A second direction that float16 can resolve must be kept, not rejected as collapsed.

    Weights 1 and 1/32 are exact in float16 and leave ~6% of the second direction after
    projection, about 60 machine epsilons.
    """
    metric = Loss(torch.nn.MSELoss(), torch.rand(4, 1).half(), torch.rand(4, 2).half())

    for seed in range(20):
        torch.manual_seed(seed)
        model = _linear(1, 2, [[1.0], [1 / 32]]).half()
        one, two = _capture_plane_directions(monkeypatch, model, metric, "filter")
        assert cosine(one, two) == pytest.approx(0.0, abs=5 * torch.finfo(torch.float16).eps)


def test_random_plane_rejects_directions_that_span_one_dimension():
    """One nonzero single-element filter normalizes to plus or minus one vector: there is no plane."""
    metric = Loss(torch.nn.MSELoss(), torch.rand(4, 1), torch.rand(4, 2))

    with pytest.raises(ValueError, match="span fewer than two dimensions"):
        loss_landscapes.random_plane(_linear(1, 2, [[1.0], [0.0]]), metric, steps=2)


def test_random_plane_keeps_small_genuine_directions_when_filter_norms_differ(monkeypatch):
    """Filter norms 1000 times apart leave the second direction ~0.2% of its norm after projection.

    That remainder is far above rounding error, so it is a genuine direction and must be kept.
    """
    metric = Loss(torch.nn.MSELoss(), torch.rand(4, 1), torch.rand(4, 2))

    def model():
        layer = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1.0], [0.001]]))
        return layer

    for seed in range(20):
        torch.manual_seed(seed)
        one, two = _capture_plane_directions(monkeypatch, model(), metric, "filter")
        assert cosine(one, two) == pytest.approx(0.0, abs=1e-5)
    assert np.isfinite(loss_landscapes.random_plane(model(), metric, steps=2)).all()


def test_random_plane_precision_ignores_unperturbed_tensors():
    """A float16 bias that ignore_1d leaves unperturbed must not loosen the float32 collapse bound."""

    class Float16Offset(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = _linear(1, 2, [[1.0], [0.001]])
            self.linear.bias = None
            self.offset = torch.nn.Parameter(torch.tensor([0.5, -1.0], dtype=torch.float16))

        def forward(self, x):
            return self.linear(x) + self.offset.float()

    metric = Loss(torch.nn.MSELoss(), torch.rand(4, 1), torch.rand(4, 2))

    for seed in range(20):
        torch.manual_seed(seed)
        assert np.isfinite(loss_landscapes.random_plane(Float16Offset(), metric, steps=2)).all()

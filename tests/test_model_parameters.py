"""Linear algebra over ModelParameters.

These are the operations the whole library is built on, and the ones whose repeated
breakage in 2019 produced the run of "fix broken normalization/step scaling" commits.
"""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from loss_landscapes.model_interface.model_parameters import (
    ModelParameters,
    orthogonal_to,
    rand_u_like,
)

from .conftest import parameters_like

scalars = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)


def assert_allclose(actual: ModelParameters, expected: ModelParameters, tol=1e-5):
    for a, b in zip(actual._get_parameters(), expected._get_parameters()):
        assert torch.allclose(a, b, atol=tol), f"{a} != {b}"


# --- vector space laws -----------------------------------------------------


def test_addition_and_subtraction_are_inverse(parameters, other_parameters):
    assert_allclose((parameters + other_parameters) - other_parameters, parameters)


def test_addition_is_commutative(parameters, other_parameters):
    assert_allclose(parameters + other_parameters, other_parameters + parameters)


@given(scalar=scalars)
@settings(max_examples=25, deadline=None)
def test_scalar_multiplication_distributes_over_addition(scalar):
    a = parameters_like((3, 4), (3,), seed=1)
    b = parameters_like((3, 4), (3,), seed=2)
    assert_allclose((a + b) * scalar, (a * scalar) + (b * scalar))


@given(scalar=scalars)
@settings(max_examples=25, deadline=None)
def test_multiplication_and_division_are_inverse(scalar):
    a = parameters_like((3, 4), (3,), seed=3)
    assert_allclose((a * scalar) / scalar, a)


def test_in_place_operations_match_constructive_ones(parameters, other_parameters):
    expected = parameters + other_parameters
    in_place = ModelParameters([p.clone() for p in parameters._get_parameters()])
    in_place.add_(other_parameters)
    assert_allclose(in_place, expected)


def test_numel_counts_individual_parameters():
    assert parameters_like((3, 4), (3,)).numel() == 15


def test_as_numpy_is_flat_and_complete():
    flat = parameters_like((3, 4), (3,)).as_numpy()
    assert flat.shape == (15,)


# --- norms -----------------------------------------------------------------


def test_model_norm_matches_flattened_l2(parameters):
    expected = float(torch.linalg.vector_norm(torch.from_numpy(parameters.as_numpy()), ord=2))
    assert parameters.model_norm(2) == pytest.approx(expected, rel=1e-5)


def test_layer_norm_matches_per_layer_l2(parameters):
    for index, layer in enumerate(parameters._get_parameters()):
        expected = float(torch.linalg.vector_norm(layer.flatten(), ord=2))
        assert parameters.layer_norm(index, 2) == pytest.approx(expected, rel=1e-5)


def test_filter_norm_matches_per_filter_l2(parameters):
    layer_index = 0
    layer = parameters[layer_index]
    for filter_index in range(len(layer)):
        expected = float(torch.linalg.vector_norm(layer[filter_index].flatten(), ord=2))
        assert parameters.filter_norm((layer_index, filter_index), 2) == pytest.approx(expected, rel=1e-5)


def test_dot_matches_flat_inner_product(parameters, other_parameters):
    expected = float(torch.from_numpy(parameters.as_numpy()) @ torch.from_numpy(other_parameters.as_numpy()))
    assert parameters.dot(other_parameters) == pytest.approx(expected, rel=1e-4)


# --- normalization ---------------------------------------------------------
#
# The contract of each `*_normalize_` is: rescale this vector so that its norms, at the
# stated granularity, equal those of the reference point.


def test_filter_normalize_gives_each_filter_the_reference_filter_norm(parameters):
    direction = rand_u_like(parameters)
    direction.filter_normalize_(parameters)

    for layer_index, layer in enumerate(parameters._get_parameters()):
        for filter_index in range(len(layer)):
            index = (layer_index, filter_index)
            assert direction.filter_norm(index) == pytest.approx(parameters.filter_norm(index), rel=1e-4)


def test_layer_normalize_gives_each_layer_the_reference_layer_norm(parameters):
    direction = rand_u_like(parameters)
    direction.layer_normalize_(parameters)

    for index in range(len(parameters)):
        assert direction.layer_norm(index) == pytest.approx(parameters.layer_norm(index), rel=1e-4)


def test_model_normalize_gives_the_reference_model_norm(parameters):
    direction = rand_u_like(parameters)
    direction.model_normalize_(parameters)
    assert direction.model_norm() == pytest.approx(parameters.model_norm(), rel=1e-4)


def test_model_normalize_is_correct_for_a_single_layer():
    """Match the reference norm when the vector consists of one tensor."""
    reference = parameters_like((3, 4))
    direction = rand_u_like(reference)
    direction.model_normalize_(reference)
    assert direction.model_norm() == pytest.approx(reference.model_norm(), rel=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("order", [2, 4])
def test_model_normalize_preserves_direction_and_matches_reference_norm(dtype, order):
    """Scale every component by two, independent of tensor shape and the norm order."""
    direction = ModelParameters([torch.tensor([-3.0], dtype=dtype), torch.tensor([[4.0]], dtype=dtype)])
    reference = ModelParameters([torch.tensor([8.0], dtype=dtype), torch.tensor([[6.0]], dtype=dtype)])

    direction.model_normalize_(reference, order=order)

    torch.testing.assert_close(direction[0], torch.tensor([-6.0], dtype=dtype))
    torch.testing.assert_close(direction[1], torch.tensor([[8.0]], dtype=dtype))
    flat = torch.cat([p.flatten() for p in direction])
    reference_flat = torch.cat([p.flatten() for p in reference])
    torch.testing.assert_close(
        torch.linalg.vector_norm(flat, ord=order), torch.linalg.vector_norm(reference_flat, ord=order)
    )
    torch.testing.assert_close(reference[0], torch.tensor([8.0], dtype=dtype))
    torch.testing.assert_close(reference[1], torch.tensor([[6.0]], dtype=dtype))


def test_model_normalize_is_independent_of_tensor_order():
    """Reordering parameter tensors must only reorder the normalised result."""
    forward = ModelParameters([torch.tensor([3.0]), torch.tensor([4.0]), torch.tensor([12.0])])
    reverse = ModelParameters([p.clone() for p in reversed(forward.parameters)])
    reference = ModelParameters([torch.tensor([26.0])])

    forward.model_normalize_(reference)
    reverse.model_normalize_(reference)

    expected = torch.tensor([6.0, 8.0, 24.0])
    torch.testing.assert_close(torch.cat(list(forward)), expected)
    torch.testing.assert_close(torch.cat(list(reversed(reverse.parameters))), expected)


def test_model_normalize_is_independent_of_tensor_partition():
    """Splitting one vector across tensors must not change model-wise normalisation."""
    whole = ModelParameters([torch.tensor([3.0, 4.0])])
    split = ModelParameters([torch.tensor([3.0]), torch.tensor([4.0])])
    reference = ModelParameters([torch.tensor([10.0])])

    whole.model_normalize_(reference)
    split.model_normalize_(reference)

    torch.testing.assert_close(whole[0], torch.tensor([6.0, 8.0]))
    torch.testing.assert_close(torch.cat(list(split)), whole[0])


def test_model_normalize_to_zero_reference_sets_every_tensor_to_zero():
    """A nonzero vector scales to zero when the reference norm is zero."""
    direction = ModelParameters([torch.tensor([3.0]), torch.tensor([4.0])])
    reference = ModelParameters([torch.tensor([0.0])])

    direction.model_normalize_(reference)

    for parameter in direction:
        torch.testing.assert_close(parameter, torch.zeros_like(parameter))


def test_model_normalize_rejects_zero_direction_without_mutating_reference():
    """Preserve the existing division-by-zero error for an undefined direction."""
    direction = ModelParameters([torch.tensor([0.0]), torch.tensor([0.0])])
    reference = ModelParameters([torch.tensor([10.0])])

    with pytest.raises(ZeroDivisionError):
        direction.model_normalize_(reference)

    torch.testing.assert_close(reference[0], torch.tensor([10.0]))


# --- orthogonality ---------------------------------------------------------


def test_orthogonal_to_produces_an_orthogonal_vector(parameters):
    direction = rand_u_like(parameters)
    other = orthogonal_to(direction)

    normaliser = direction.model_norm() * other.model_norm()
    assert other.dot(direction) / normaliser == pytest.approx(0.0, abs=1e-5)


def test_orthogonal_to_is_not_degenerate(parameters):
    """A zero vector would be trivially orthogonal; make sure we did not get one."""
    direction = rand_u_like(parameters)
    assert orthogonal_to(direction).model_norm() > 1e-6


def test_rand_u_like_preserves_shape_and_dtype(parameters):
    sampled = rand_u_like(parameters)
    assert len(sampled) == len(parameters)
    for new, old in zip(sampled._get_parameters(), parameters._get_parameters()):
        assert new.size() == old.size()
        assert new.dtype == old.dtype


def test_rand_u_like_is_uniform_on_unit_interval(parameters):
    sampled = rand_u_like(parameters).as_numpy()
    assert sampled.min() >= 0.0
    assert sampled.max() < 1.0


def test_matmul_is_explicitly_unimplemented(parameters, other_parameters):
    with pytest.raises(NotImplementedError):
        parameters @ other_parameters


def test_equality_is_structural(parameters):
    same = ModelParameters([p.clone() for p in parameters._get_parameters()])
    assert parameters == same
    assert not parameters == (parameters * 2.0)


def test_l2_norm_is_the_root_of_the_sum_of_squares(parameters):
    expected = math.sqrt(sum(float(p.pow(2).sum()) for p in parameters._get_parameters()))
    assert parameters.model_norm(2) == pytest.approx(expected, rel=1e-5)


# TODO: take absolute values in the `*_norm` methods, then drop this marker.
@pytest.mark.xfail(
    reason="The `*_norm` methods compute pow(sum(x**order), 1/order) without taking absolute "
    "values, so for odd orders negative parameters cancel out and the result is a signed sum "
    "rather than a norm. Even orders, including the default of 2, are unaffected.",
    strict=True,
)
def test_l1_norm_is_the_sum_of_absolute_values(parameters):
    expected = sum(float(p.abs().sum()) for p in parameters._get_parameters())
    assert parameters.model_norm(1) == pytest.approx(expected, rel=1e-5)

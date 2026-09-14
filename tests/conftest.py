"""Shared fixtures. Everything here is deliberately tiny and CPU-only."""

import numpy as np
import pytest
import torch

from loss_landscapes.model_interface.model_parameters import ModelParameters

SEED = 20190830


@pytest.fixture(autouse=True)
def deterministic():
    """Seed every test. The library samples random directions internally."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)


@pytest.fixture
def model():
    """A two-layer MLP: small enough to be fast, deep enough to exercise layer-wise code."""
    torch.manual_seed(SEED)
    return torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    )


@pytest.fixture
def other_model():
    """A second model of identical architecture but different parameters."""
    torch.manual_seed(SEED + 1)
    return torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    )


@pytest.fixture
def data():
    """Fixed inputs and targets for the supervised metrics."""
    torch.manual_seed(SEED)
    return torch.rand(8, 4), torch.rand(8, 2)


@pytest.fixture
def parameters(model):
    return ModelParameters([p.detach().clone() for p in model.parameters()])


@pytest.fixture
def other_parameters(other_model):
    return ModelParameters([p.detach().clone() for p in other_model.parameters()])


def parameters_like(*shapes, seed=SEED):
    """Build a ModelParameters from explicit shapes, for tests that need known structure."""
    generator = torch.Generator().manual_seed(seed)
    return ModelParameters([torch.rand(shape, generator=generator) for shape in shapes])

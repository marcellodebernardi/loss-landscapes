"""Shared fixtures. Everything here is deliberately tiny and CPU-only."""

import platform

import numpy as np
import pytest
import torch

import loss_landscapes
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


def pytest_addoption(parser):
    parser.addoption("--compatibility-check", action="store_true", help="Fail compatibility runs on ordinary skips")


def pytest_configure(config):
    if config.getoption("--compatibility-check"):
        config.pluginmanager.register(CompatibilityPolicy(), "compatibility-policy")


def pytest_report_header(config):
    if config.getoption("--compatibility-check"):
        return [
            f"Runtime: Python {platform.python_version()}, PyTorch {torch.__version__}, NumPy {np.__version__}",
            f"Platform: {platform.platform()} ({platform.machine()})",
            f"Package: {loss_landscapes.__file__}",
        ]


class CompatibilityPolicy:
    """Keep pytest's normal outcomes, but reject incomplete coverage from skips."""

    def __init__(self):
        self.skips = []

    def pytest_collectreport(self, report):
        if report.skipped:
            self.skips.append(report.nodeid)

    def pytest_runtest_logreport(self, report):
        if report.skipped and not getattr(report, "wasxfail", False):
            self.skips.append(report.nodeid)

    def pytest_sessionfinish(self, session):
        if self.skips and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter):
        if self.skips:
            terminalreporter.write_sep("=", "ordinary skips prevent compatibility success", red=True)
            for nodeid in self.skips:
                terminalreporter.write_line(nodeid)

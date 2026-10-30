"""Guards on the shape of the package: its public API, and which modules import."""

import importlib
import os
import pkgutil
import sys
import sysconfig
from importlib.metadata import metadata
from pathlib import Path

import pytest
import torch
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

import loss_landscapes

PUBLIC_API = [
    "point",
    "linear_interpolation",
    "random_line",
    "planar_interpolation",
    "random_plane",
    "ModelWrapper",
    "GeneralModelWrapper",
]

# Subpackages that are expected to import cleanly. `contrib` is excluded on purpose; see
# test_contrib_is_still_broken below.
IMPORTABLE_MODULES = [
    "loss_landscapes.main",
    "loss_landscapes.metrics",
    "loss_landscapes.metrics.metric",
    "loss_landscapes.metrics.sl_metrics",
    "loss_landscapes.metrics.rl_metrics",
    "loss_landscapes.model_interface.model_parameters",
    "loss_landscapes.model_interface.model_wrapper",
]


@pytest.mark.parametrize("name", PUBLIC_API)
def test_public_api_is_exported(name):
    assert hasattr(loss_landscapes, name)


@pytest.mark.parametrize("name", IMPORTABLE_MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_source_tree_contains_no_unexpected_top_level_modules():
    """The importable package has exactly the four subpackages we expect.

    Inspects the source tree rather than a built artifact: under an editable install,
    `loss_landscapes.__path__` points at `src/`.
    """
    expected = {"main", "metrics", "model_interface", "contrib"}
    found = {module.name for module in pkgutil.iter_modules(loss_landscapes.__path__)}

    assert found == expected


# TODO: repair or delete `contrib`, then drop this marker.
@pytest.mark.xfail(
    reason="contrib was orphaned by the 2019-07-17 refactor: it imports "
    "loss_landscapes.model_interface.model_interface, which no longer exists, and calls the "
    "removed get_parameter_tensor/set_parameter_tensor API. It only appears to work for users "
    "of the published wheel because that wheel still contains the deleted module.",
    strict=True,
    raises=ModuleNotFoundError,
)
@pytest.mark.parametrize("name", ["loss_landscapes.contrib.connecting_paths", "loss_landscapes.contrib.trajectories"])
def test_contrib_is_still_broken(name):
    importlib.import_module(name)


def test_version_is_available():
    from importlib.metadata import version

    assert version("loss_landscapes")


def test_declared_support_boundaries():
    info = metadata("loss_landscapes")
    python = SpecifierSet(info["Requires-Python"])
    assert all(f"3.{minor}.0" in python for minor in range(11, 16))
    assert "3.10.99" not in python and "3.16.0" not in python
    requirements = {r.name: r for r in map(Requirement, info.get_all("Requires-Dist", []))}
    assert "numpy" in requirements
    torch_versions = requirements["torch"].specifier
    assert "2.15.0" in torch_versions and "2.15.99" in torch_versions
    assert "2.14.99" not in torch_versions and "2.16.0" not in torch_versions


def test_compatibility_environment_identity():
    """Check the actual installed artifact and runtime when invoked through tox."""
    name = os.environ.get("TOX_ENV_NAME", "")
    if not name:
        return  # Ordinary editable development runs are supported too.
    minor = int(name.split("-")[0][3:]) if name.startswith("py3") else sys.version_info.minor
    if name == "sdist":
        minor = 12
    assert 11 <= minor <= 15
    assert sys.version_info[:2] == (3, minor)
    assert sys.version_info.releaselevel == "final"
    assert sys.implementation.name == "cpython"
    assert not sysconfig.get_config_var("Py_GIL_DISABLED")
    assert torch.__version__.split(".")[:2] == ["2", "15"]
    assert Path(loss_landscapes.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    if sys.platform == "linux":
        assert torch.version.cuda is None and torch.version.hip is None

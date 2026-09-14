"""Guards on the shape of the package: its public API, and which modules import."""

import importlib
import pkgutil

import pytest

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


@pytest.mark.xfail(
    reason="contrib was orphaned by the 2019-07-17 refactor: it imports "
    "loss_landscapes.model_interface.model_interface, which no longer exists, and calls the "
    "removed get_parameter_tensor/set_parameter_tensor API. It only appears to work for users "
    "of the published wheel because that wheel still contains the deleted module. "
    "See https://github.com/marcellodebernardi/loss-landscapes/issues/TBD",
    strict=True,
    raises=ModuleNotFoundError,
)
@pytest.mark.parametrize("name", ["loss_landscapes.contrib.connecting_paths", "loss_landscapes.contrib.trajectories"])
def test_contrib_is_still_broken(name):
    importlib.import_module(name)


def test_version_is_available():
    from importlib.metadata import version

    assert version("loss_landscapes")

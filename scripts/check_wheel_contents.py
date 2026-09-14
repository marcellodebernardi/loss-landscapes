"""Check that a built wheel contains exactly the modules it is supposed to.

The wheel published as 3.0.6 carried roughly sixty modules left over from the 1.x and 2.x
refactors, because it was built over a dirty `build/` directory. Nothing in the test suite
can catch that: tests import the installed package, which under an editable install is the
source tree. This inspects the artifact itself.

Usage:
    python scripts/check_wheel_contents.py dist/*.whl
"""

from __future__ import annotations

import argparse
import posixpath
import zipfile

PACKAGE = "loss_landscapes"

EXPECTED_MODULES = {
    "__init__.py",
    "contrib/__init__.py",
    "contrib/connecting_paths.py",
    "contrib/trajectories.py",
    "main.py",
    "metrics/__init__.py",
    "metrics/metric.py",
    "metrics/rl_metrics.py",
    "metrics/sl_metrics.py",
    "model_interface/__init__.py",
    "model_interface/model_parameters.py",
    "model_interface/model_wrapper.py",
}


def package_contents(wheel: str) -> set[str]:
    """List the wheel's payload paths, relative to the package root.

    Args:
        wheel: path to a wheel file.

    Returns:
        Paths inside the package directory, excluding `.dist-info` metadata.
    """
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]

    prefix = PACKAGE + "/"
    return {posixpath.relpath(n, prefix) for n in names if n.startswith(prefix)}


def main() -> int:
    """Compare each given wheel against the expected module list.

    Returns:
        0 if every wheel matches, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", help="wheel files to check")
    args = parser.parse_args()

    failed = False
    for wheel in args.wheels:
        found = package_contents(wheel)
        unexpected = sorted(found - EXPECTED_MODULES)
        missing = sorted(EXPECTED_MODULES - found)

        print(f"{wheel}: {len(found)} files under {PACKAGE}/")
        for name in unexpected:
            print(f"  unexpected: {name}")
        for name in missing:
            print(f"  missing:    {name}")

        if unexpected or missing:
            failed = True

    if failed:
        print("\nthe wheel does not contain the expected set of modules")
        return 1

    print("\nOK: wheel contents are exactly as expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

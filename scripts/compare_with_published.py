"""Verify that this repository still builds the dependency contract it published.

The packaging rewrite (setup.py -> pyproject.toml/hatchling) is intended to be a no-op
for consumers: same distribution name, same version, same `Requires-Python`, same
`Requires-Dist`. This script proves that rather than asserting it, by building the
current tree under the published version number and diffing the resulting core metadata
against the metadata of the published wheel.

Purely presentational metadata differences (the `Metadata-Version` bump, PEP 639 license
expressions, `Home-page` becoming `Project-URL`) are expected and ignored: they carry no
meaning for resolvers.

Usage:
    python scripts/compare_with_published.py [--version 3.0.6]
"""

from __future__ import annotations

import argparse
import email.parser
import glob
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = "loss-landscapes"

# Fields that a resolver acts on. Everything else is descriptive.
CONTRACT_FIELDS = ("Name", "Version", "Requires-Python")
CONTRACT_MULTI_FIELDS = ("Requires-Dist", "Provides-Extra")


def _normalise_name(name: str) -> str:
    """PyPI treats '_' and '-' in distribution names as equivalent."""
    return name.replace("_", "-").lower()


def fetch_published_metadata(version: str, workdir: str) -> email.message.Message:
    """Download the wheel PyPI holds for `version` and return its parsed METADATA.

    Args:
        version: released version to fetch.
        workdir: scratch directory the wheel is downloaded into.

    Returns:
        The parsed METADATA of the published wheel.
    """
    with urllib.request.urlopen(f"https://pypi.org/pypi/{PACKAGE}/{version}/json") as response:
        release = json.load(response)

    wheels = [f for f in release["urls"] if f["packagetype"] == "bdist_wheel"]
    if not wheels:
        sys.exit(f"error: {PACKAGE} {version} has no wheel published on PyPI")

    wheel_path = os.path.join(workdir, "published.whl")
    urllib.request.urlretrieve(wheels[0]["url"], wheel_path)
    return read_wheel_metadata(wheel_path)


def build_current_tree(version: str, workdir: str) -> email.message.Message:
    """Build the working tree as if it were the published version, into a scratch dir."""
    outdir = os.path.join(workdir, "dist")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", outdir],
        cwd=REPO_ROOT,
        env={**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION": version},
        check=True,
        stdout=subprocess.DEVNULL,
    )

    built = sorted(glob.glob(os.path.join(outdir, f"*{version}*.whl")))
    if not built:
        sys.exit("error: build produced no wheel")
    return read_wheel_metadata(built[-1])


def read_wheel_metadata(path: str) -> email.message.Message:
    """Parse the METADATA file out of the wheel at `path`.

    Args:
        path: path to a wheel.

    Returns:
        The parsed METADATA of that wheel.
    """
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        return email.parser.Parser().parsestr(archive.read(name).decode("utf-8"))


def compare(published: email.message.Message, built: email.message.Message) -> list[str]:
    """Diff the resolver-visible fields of two metadata blocks.

    Args:
        published: metadata of the release on PyPI.
        built: metadata of the wheel built from this tree.

    Returns:
        One human-readable line per contract field that differs; empty if none do.
    """
    problems = []

    for field in CONTRACT_FIELDS:
        old, new = published.get(field), built.get(field)
        if field == "Name":
            old, new = _normalise_name(old or ""), _normalise_name(new or "")
        if old != new:
            problems.append(f"{field}: published {old!r} -> built {new!r}")

    for field in CONTRACT_MULTI_FIELDS:
        old, new = sorted(published.get_all(field) or []), sorted(built.get_all(field) or [])
        if old != new:
            problems.append(f"{field}: published {old} -> built {new}")

    return problems


def main() -> int:
    """Run the comparison and report.

    Returns:
        0 if the contract is unchanged, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="3.0.6", help="published version to compare against")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as workdir:
        published = fetch_published_metadata(args.version, workdir)
        built = build_current_tree(args.version, workdir)
        problems = compare(published, built)

    print(f"comparing built metadata against published {PACKAGE} {args.version}\n")
    for field in CONTRACT_FIELDS:
        print(f"  {field:<16} {built.get(field)}")
    for field in CONTRACT_MULTI_FIELDS:
        for value in built.get_all(field) or []:
            print(f"  {field:<16} {value}")

    if problems:
        print("\nthe dependency contract changed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nOK: the dependency contract is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

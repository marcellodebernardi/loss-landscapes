"""Exercise compatibility session outcomes through pytest's public test harness."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ("def test_ok(): pass", 0),
        ("@pytest.mark.xfail(reason='known defect', strict=True)\ndef test_known(): assert False", 0),
        ("@pytest.mark.xfail(reason='known defect', strict=True)\ndef test_fixed(): pass", 1),
        ("def test_failure(): assert False", 1),
        ("pytest.skip('collection', allow_module_level=True)", 5),
        ("@pytest.fixture(autouse=True)\ndef fixture(): pytest.skip('setup')\ndef test_ok(): pass", 1),
        ("def test_skip(): pytest.skip('call')", 1),
        (
            "@pytest.fixture(autouse=True)\ndef fixture():\n    yield\n"
            "    pytest.skip('teardown')\ndef test_ok(): pass",
            1,
        ),
    ],
    ids=["pass", "xfail", "xpass", "failure", "collection-skip", "setup-skip", "call-skip", "teardown-skip"],
)
def test_compatibility_session_outcomes(pytester, body, status):
    """Check exit statuses and diagnostics for passes, known failures, errors and skips."""
    pytester.makeconftest(Path(__file__).with_name("conftest.py").read_text())
    pytester.makepyfile("import pytest\n" + body)
    result = pytester.runpytest("--compatibility-check", "-o", "addopts=")
    assert result.ret == status
    if "skip" in body:
        result.stdout.fnmatch_lines(["*ordinary skip*compatibility*"])
    if status == 0:
        result.stdout.fnmatch_lines(["*Runtime: Python*PyTorch*NumPy*", "*Package:*loss_landscapes*"])


def test_collection_skip_fails_with_passing_tests(pytester):
    """Reject collection skips even when another module contains a passing test."""
    pytester.makeconftest(Path(__file__).with_name("conftest.py").read_text())
    pytester.makepyfile(
        test_skipped="import pytest\npytest.skip('collection', allow_module_level=True)",
        test_passing="def test_ok(): pass",
    )
    result = pytester.runpytest("--compatibility-check", "-o", "addopts=")
    result.assert_outcomes(passed=1, skipped=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*ordinary skip*compatibility*"])


def test_ordinary_development_session_allows_skips(pytester):
    """Allow ordinary skips when compatibility verification is not requested."""
    pytester.makeconftest(Path(__file__).with_name("conftest.py").read_text())
    pytester.makepyfile("import pytest\ndef test_skip(): pytest.skip('optional development check')")
    result = pytester.runpytest("-o", "addopts=")
    assert result.ret == 0


@pytest.mark.parametrize("wrong_environment", ["interpreter", "import-origin"])
def test_environment_identity_rejects_mismatch(wrong_environment):
    """Reject an installed-package check using the wrong Python minor or checkout imports."""
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    minor = sys.version_info.minor
    if wrong_environment == "interpreter":
        minor = 11 if minor != 11 else 12
    else:
        env["PYTHONPATH"] = str(root / "src")
    env["TOX_ENV_NAME"] = f"py3{minor}-torch215"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(root / "tests/test_packaging.py"), "-k", "environment_identity", "-q"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "test_compatibility_environment_identity" in result.stdout
    assert "1 failed" in result.stdout

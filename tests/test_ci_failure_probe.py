"""Temporary acceptance probe; removed after checking independent CI failures."""

import os


def test_one_matrix_environment_fails():
    assert os.environ.get("TOX_ENV_NAME") != "py311-torch215", "controlled compatibility failure"

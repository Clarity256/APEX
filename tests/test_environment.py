"""Environment smoke tests."""

from __future__ import annotations


def test_core_dependencies_import() -> None:
    """Verify that the configured scientific stack imports in the project venv."""
    import cvxpy
    import gymnasium
    import numpy
    import pandas
    import scipy
    import skyfield
    import torch

    assert numpy.__version__
    assert scipy.__version__
    assert pandas.__version__
    assert torch.__version__
    assert gymnasium.__version__
    assert skyfield.__version__
    assert "ECOS" in cvxpy.installed_solvers()


def test_project_package_import() -> None:
    """Verify editable package installation."""
    import leo_alloc

    assert leo_alloc.__version__ == "0.1.0"

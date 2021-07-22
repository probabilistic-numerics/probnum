"""Tests for the maximum iterations stopping criterion."""

import pathlib

from pytest_cases import parametrize_with_cases

from probnum.linalg.solvers import ProbabilisticLinearSolverState, stopping_criteria

case_modules = (pathlib.Path(__file__).parent / "cases").stem
cases_stopping_criteria = case_modules + ".stopping_criteria"
cases_states = case_modules + ".states"


def test_maxiter_None():
    """Test whether if ``maxiter=None``, the maximum number of iterations is set to
    :math:`10n`, where :math:`n` is the dimension of the linear system."""
    pass  # TODO

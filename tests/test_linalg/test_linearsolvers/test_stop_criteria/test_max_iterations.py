"""Tests for the maximum iterations stopping criterion."""
import pytest

from probnum.linalg.linearsolvers.beliefs import LinearSystemBelief
from probnum.linalg.linearsolvers.stop_criteria import MaxIterations, StoppingCriterion
from probnum.problems import LinearSystem


@pytest.mark.parametrize("maxiter", [-1, 0, 1.0, 100], ids=lambda i: f"iter{i}")
def test_stop_if_iter_larger_or_equal_than_maxiter(
    stopcrit: StoppingCriterion,
    linsys_spd: LinearSystem,
    prior: LinearSystemBelief,
    solver_state_init,
    maxiter: int,
):
    """Test if stopping criterion returns true for iteration >= maxiter."""
    has_converged = MaxIterations(maxiter=maxiter)(
        problem=linsys_spd,
        belief=prior,
        solver_state=solver_state_init,
    )
    assert has_converged == (solver_state_init.iteration >= maxiter)

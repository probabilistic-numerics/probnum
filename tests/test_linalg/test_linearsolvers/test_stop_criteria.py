"""Tests for stopping criteria of probabilistic linear solvers."""

import numpy as np
import pytest

from probnum.linalg.linearsolvers import LinearSolverState
from probnum.linalg.linearsolvers.beliefs import LinearSystemBelief
from probnum.linalg.linearsolvers.stop_criteria import (
    MaxIterations,
    PosteriorContraction,
    Residual,
    StoppingCriterion,
)
from probnum.problems import LinearSystem

# pylint: disable="invalid-name"


def test_stop_crit_returns_bool(
    stopcrit: StoppingCriterion, linsys_spd: LinearSystem, prior: LinearSystemBelief
):
    """Test whether stopping criteria return a boolean value."""
    has_converged = stopcrit(
        problem=linsys_spd,
        belief=prior,
        solver_state=None,
    )
    assert isinstance(has_converged, (bool, np.bool_))


def test_solver_state_none(
    stopcrit: StoppingCriterion, linsys_spd: LinearSystem, prior: LinearSystemBelief
):
    """Test whether all stopping criteria can be computed without a solver state."""
    _ = stopcrit(
        problem=linsys_spd,
        belief=prior,
        solver_state=None,
    )


@pytest.mark.parametrize(
    "stopcrit",
    [
        Residual(),
        PosteriorContraction(),
    ],
    indirect=True,
)
def test_stops_if_true_solution(
    stopcrit: StoppingCriterion,
    linsys_spd: LinearSystem,
    belief_groundtruth: LinearSystemBelief,
):
    """Test if stopping criterion returns True for the exact solution."""
    assert stopcrit(
        problem=linsys_spd,
        belief=belief_groundtruth,
    )


class TestMaxIterations:
    """Tests for the maximum iterations stopping criterion."""

    @pytest.mark.parametrize("maxiter", [-1, 0, 1.0, 100], ids=lambda i: f"iter{i}")
    def test_stop_if_iter_larger_or_equal_than_maxiter(
        self,
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


class TestResidual:
    """Tests for the residual stopping criterion."""

    @pytest.mark.parametrize(
        "norm_ord", [np.inf, -np.inf, 0.5, 1, 2, 10], ids=lambda i: f"ord{i}"
    )
    def test_different_norms(
        self, linsys_spd: LinearSystem, prior: LinearSystemBelief, norm_ord: float
    ):
        """Test if stopping criterion can be computed for different norms."""
        Residual(norm_ord=norm_ord)(
            problem=linsys_spd,
            belief=prior,
        )


@pytest.mark.parametrize("stopcrit", [PosteriorContraction()], indirect=True)
class PosteriorContractionTestCase:
    """Test case for the posterior contraction stopping criterion."""

"""Test cases for policies of probabilistic linear solvers."""

import numpy as np
import pytest

import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import LinearSolverState
from probnum.linalg.linearsolvers.beliefs import LinearSystemBelief
from probnum.linalg.linearsolvers.policies import (
    ConjugateDirections,
    ExploreExploit,
    Policy,
    ThompsonSampling,
)

# pylint: disable="invalid-name"


def test_policy_returns_vector(policy, linsys, belief):
    """Test whether stochastic policies return a (column) vector of length n."""
    action, _ = policy(
        problem=linsys,
        belief=belief,
        solver_state=None,
    )
    assert (
        isinstance(action, np.ndarray),
        f"Action {action} returned by {policy.__class__.__name__} is "
        "not an np.ndarray.",
    )
    assert (
        (action.shape == (linsys.A.shape[1], 1)),
        f"Action returned by {policy.__class__.__name__} has shape" f" {action.shape}.",
    )


@pytest.mark.parametrize(["linsys", "belief"], [])
def test_is_deterministic_or_stochastic(policy, linsys, belief, solver_state):
    """Test whether evaluating the policy is deterministic or stochastic."""
    action1, _ = policy(problem=linsys, belief=belief, solver_state=solver_state)
    action2, _ = policy(problem=linsys, belief=belief, solver_state=solver_state)

    if policy.is_deterministic:
        assert (
            np.all(action1 == action2),
            "Policy returned two different actions for the same input.",
        )
    else:
        assert (
            np.all(action1 == action2),
            "Policy returned the same action for two subsequent evaluations.",
        )


def test_belief_equals_truth_solves_problem_in_one_step(self):
    """Test whether the solver converges in one step to the solution, if the model for
    the matrix, inverse and right hand side matches the truth."""
    Ainv = np.linalg.inv(self.linsys.A)
    x = self.rng.random((self.dim, 1))
    belief = LinearSystemBelief(
        x=rvs.Constant(x),
        A=rvs.Constant(self.linsys.A),
        Ainv=rvs.Constant(Ainv),
        b=rvs.Constant(self.linsys.b),
    )
    action, _ = self.thompson_sampling(problem=self.linsys, belief=belief)

    self.assertAllClose(
        self.linsys.solution,
        x + action,
    )


def test_directions_are_conjugate(self):
    """Test whether the actions given by the policy are A-conjugate."""
    # TODO: use ProbabilisticLinearSolver's solve_iter function to test this

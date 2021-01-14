"""Test cases for policies of probabilistic linear solvers."""

import numpy as np
import pytest

from probnum.linalg.linearsolvers.beliefs import LinearSystemBelief
from probnum.linalg.linearsolvers.policies import (
    ConjugateDirections,
    ExploreExploit,
    Policy,
    ThompsonSampling,
)
from probnum.problems import LinearSystem

# pylint: disable="invalid-name"


def test_policy_returns_vector(
    policy: Policy, linsys_spd: LinearSystem, prior: LinearSystemBelief
):
    """Test whether stochastic policies return a (column) vector of length n."""
    action, _ = policy(
        problem=linsys_spd,
        belief=prior,
        solver_state=None,
    )
    assert (
        isinstance(action, np.ndarray),
        f"Action {action} returned by {policy.__class__.__name__} is "
        "not an np.ndarray.",
    )
    assert (
        (action.shape == (linsys_spd.A.shape[1], 1)),
        f"Action returned by {policy.__class__.__name__} has shape" f" {action.shape}.",
    )


def test_is_deterministic_or_stochastic(
    policy: Policy, linsys_spd: LinearSystem, prior: LinearSystemBelief
):
    """Test whether evaluating the policy is deterministic or stochastic."""
    action1, _ = policy(problem=linsys_spd, belief=prior, solver_state=None)
    action2, _ = policy(problem=linsys_spd, belief=prior, solver_state=None)

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


@pytest.mark.parametrize(
    "policy",
    [
        ConjugateDirections(),
        ThompsonSampling(random_state=1),
        ExploreExploit(random_state=1),
    ],
    indirect=True,
)
def test_ground_truth_belief_solves_problem_in_one_step(
    policy: Policy, linsys_spd: LinearSystem, belief_groundtruth: LinearSystemBelief
):
    """Test whether the returned action is the step to the solution, if the model for
    the matrix, inverse and right hand side matches the truth."""
    action, _ = policy(problem=linsys_spd, belief=belief_groundtruth)

    np.testing.assert_allclose(
        linsys_spd.solution,
        belief_groundtruth.x.mean + action,
    )


def test_directions_are_conjugate():
    """Test whether the actions given by the ConjugateDirections policy are
    A-conjugate."""
    # TODO: use ProbabilisticLinearSolver's solve_iter function to test this

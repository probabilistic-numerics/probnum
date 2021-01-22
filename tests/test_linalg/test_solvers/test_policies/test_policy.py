"""Test cases for policies of probabilistic linear solvers."""

import numpy as np
import pytest

from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.linalg.solvers.policies import (
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
    action = policy(
        problem=linsys_spd,
        belief=prior,
        solver_state=None,
    )
    assert isinstance(
        action, np.ndarray
    ), f"Action {action} returned by {policy.__class__.__name__} is not an np.ndarray."
    assert action.shape == (linsys_spd.A.shape[1], 1), (
        f"Action returned by {policy.__class__.__name__} has shape" f" {action.shape}.",
    )


def test_is_deterministic_or_stochastic(
    policy: Policy, linsys_spd: LinearSystem, prior: LinearSystemBelief
):
    """Test whether evaluating the policy is deterministic or stochastic."""
    action1 = policy(problem=linsys_spd, belief=prior, solver_state=None)
    action2 = policy(problem=linsys_spd, belief=prior, solver_state=None)

    if policy.is_deterministic:
        assert np.all(
            action1 == action2
        ), "Policy returned two different actions for the same input."
    else:
        assert not np.all(
            action1 == action2
        ), "Policy returned the same action for two subsequent evaluations."


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
    action = policy(problem=linsys_spd, belief=belief_groundtruth)

    np.testing.assert_allclose(
        linsys_spd.solution,
        belief_groundtruth.x.mean + action,
    )


@pytest.mark.xfail(
    reason="The induced belief on x is not yet implemented for multiple rhs. "
    "github #302",
)
def test_multiple_rhs(
    policy: Policy,
    linsys_spd_multiple_rhs: LinearSystem,
    symm_belief_multiple_rhs: LinearSystemBelief,
):
    """Test whether the policy returns multiple actions for multiple right hand sides of
    the linear system."""
    actions = policy(problem=linsys_spd_multiple_rhs, belief=symm_belief_multiple_rhs)

    assert actions.shape == symm_belief_multiple_rhs.x.shape


@pytest.mark.parametrize(
    "policy",
    [ThompsonSampling(random_state=1), ExploreExploit(random_state=1)],
    indirect=True,
)
class TestStochasticPolicies:
    """Tests for stochastic policies."""

    def test_fixed_random_state(
        self,
        policy: Policy,
        linsys_spd: LinearSystem,
        prior: LinearSystemBelief,
        random_state: np.random.RandomState,
    ):
        """Test whether a fixed random state produces reproducible results."""
        action0 = type(policy)(random_state=random_state)(
            problem=linsys_spd, belief=prior
        )
        action1 = type(policy)(random_state=random_state)(
            problem=linsys_spd, belief=prior
        )
        np.testing.assert_allclose(
            action0,
            action1,
            rtol=10 ** 2 * np.finfo(float).eps,
            atol=10 ** 2 * np.finfo(float).eps,
        )

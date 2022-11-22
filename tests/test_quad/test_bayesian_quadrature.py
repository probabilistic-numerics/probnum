"""Basic tests for Bayesian quadrature method."""

import numpy as np
import pytest

from probnum import LambdaStoppingCriterion
from probnum.quad.integration_measures import LebesgueMeasure
from probnum.quad.solvers import BayesianQuadrature
from probnum.quad.solvers.policies import RandomPolicy, VanDerCorputPolicy
from probnum.quad.solvers.stopping_criteria import ImmediateStop
from probnum.randprocs.kernels import ExpQuad
from probnum.randvars import Normal


@pytest.fixture
def input_dim():
    return 3


@pytest.fixture
def data(input_dim):
    def fun(x):
        return 2 * np.ones(x.shape[0])

    nodes = np.ones([5, input_dim])
    fun_evals = fun(nodes)
    return nodes, fun_evals, fun


@pytest.fixture
def bq(input_dim):
    return BayesianQuadrature.from_problem(
        input_dim=input_dim,
        domain=(np.zeros(input_dim), np.ones(input_dim)),
    )


@pytest.fixture
def bq_no_policy(input_dim):
    return BayesianQuadrature.from_problem(
        input_dim=input_dim,
        domain=(np.zeros(input_dim), np.ones(input_dim)),
        policy=None,
    )


# Tests for correct assignments start here.


@pytest.mark.parametrize(
    "policy, policy_type", [("bmc", RandomPolicy), ("vdc", VanDerCorputPolicy)]
)
def test_bq_from_problem_policy_assignment(policy, policy_type):
    """Test if correct policy is assigned from string identifier."""
    bq = BayesianQuadrature.from_problem(input_dim=1, domain=(0, 1), policy=policy)
    assert isinstance(bq.policy, policy_type)


def test_bq_from_problem_defaults(bq_no_policy, bq):

    # default policy and stopping criterion
    assert isinstance(bq.policy, RandomPolicy)
    assert isinstance(bq.stopping_criterion, LambdaStoppingCriterion)

    # default stopping criterion if no policy is available
    assert bq_no_policy.policy is None
    assert isinstance(bq_no_policy.stopping_criterion, ImmediateStop)

    # default measure
    assert isinstance(bq_no_policy.measure, LebesgueMeasure)
    assert isinstance(bq.measure, LebesgueMeasure)

    # default kernel
    assert isinstance(bq_no_policy.kernel, ExpQuad)
    assert isinstance(bq.kernel, ExpQuad)


# Tests for input checks and exception raises start here.


def test_bq_from_problem_wrong_inputs(input_dim):

    # neither measure nor domain is provided
    with pytest.raises(ValueError):
        BayesianQuadrature.from_problem(input_dim=input_dim)


# Tests for integrate function start here.


def test_integrate_no_policy_wrong_input(bq_no_policy, data):
    nodes, fun_evals, fun = data

    # no nodes provided
    with pytest.raises(ValueError):
        bq_no_policy.integrate(fun=None, nodes=None, fun_evals=fun_evals)

    # fun is ignored if fun_evals are given
    with pytest.warns(Warning):
        bq_no_policy.integrate(fun=fun, nodes=nodes, fun_evals=fun_evals)


def test_integrate_wrong_input(bq, bq_no_policy, data):
    nodes, fun_evals, fun = data

    # no integrand provided
    with pytest.raises(ValueError):
        bq.integrate(fun=None, nodes=nodes, fun_evals=None)
    with pytest.raises(ValueError):
        bq_no_policy.integrate(fun=None, nodes=nodes, fun_evals=None)

    # wrong fun_evals shape
    with pytest.raises(ValueError):
        bq.integrate(fun=fun, nodes=nodes, fun_evals=fun_evals[:, None])
    with pytest.raises(ValueError):
        bq_no_policy.integrate(fun=None, nodes=nodes, fun_evals=fun_evals[:, None])

    # wrong nodes shape
    with pytest.raises(ValueError):
        bq.integrate(fun=fun, nodes=nodes[:, None], fun_evals=None)
    with pytest.raises(ValueError):
        bq_no_policy.integrate(fun=None, nodes=nodes[:, None], fun_evals=None)

    # number of points in nodes and fun_evals do not match
    wrong_nodes = np.vstack([nodes, np.ones([1, nodes.shape[1]])])
    with pytest.raises(ValueError):
        bq.integrate(fun=fun, nodes=wrong_nodes, fun_evals=fun_evals)
    with pytest.raises(ValueError):
        bq_no_policy.integrate(fun=None, nodes=wrong_nodes, fun_evals=fun_evals)


def test_integrate_max_evals_output(data, rng):
    nodes, fun_evals, fun = data
    input_dim = nodes.shape[1]
    max_evals = 10

    # no initial data
    bq = BayesianQuadrature.from_problem(
        input_dim=input_dim, domain=(0, 1), options=dict(max_evals=max_evals)
    )
    res, bq_state, info = bq.integrate(fun=fun, nodes=None, fun_evals=None, rng=rng)
    assert isinstance(res, Normal)
    assert isinstance(bq_state.integral_belief, Normal)
    assert isinstance(bq_state.scale_sq, float)
    assert len(bq_state.previous_integral_beliefs) == max_evals
    assert len(bq_state.kernel_means) == max_evals
    assert bq_state.nodes.shape == (max_evals, input_dim)
    assert bq_state.fun_evals.shape == (max_evals,)
    assert bq_state.gram.shape == (max_evals, max_evals)


@pytest.mark.parametrize("no_data", [True, False])
def test_integrate_max_evals_output(no_data, data, rng):
    nodes, fun_evals, fun = data

    nevals, input_dim = nodes.shape
    max_evals = 10
    assert max_evals > nevals  # make sure that some nodes are collected

    # if there is data, the number of updates is shorter
    num_updates = max_evals - nevals + 1
    if no_data:
        nodes, fun_evals, num_updates = None, None, max_evals

    # the total number of evals is the same, but the number of initial nodes differs
    bq = BayesianQuadrature.from_problem(
        input_dim=input_dim, domain=(0, 1), options=dict(max_evals=max_evals)
    )
    res, bq_state, info = bq.integrate(
        fun=fun, nodes=nodes, fun_evals=fun_evals, rng=rng
    )
    assert isinstance(res, Normal)
    assert isinstance(bq_state.integral_belief, Normal)
    assert isinstance(bq_state.scale_sq, float)
    assert len(bq_state.kernel_means) == max_evals
    assert len(bq_state.previous_integral_beliefs) == num_updates
    assert bq_state.nodes.shape == (max_evals, input_dim)
    assert bq_state.fun_evals.shape == (max_evals,)
    assert bq_state.gram.shape == (max_evals, max_evals)

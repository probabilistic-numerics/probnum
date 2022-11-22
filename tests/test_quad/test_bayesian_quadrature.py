"""Basic tests for Bayesian quadrature method."""

import numpy as np
import pytest

from probnum import LambdaStoppingCriterion
from probnum.quad.integration_measures import LebesgueMeasure
from probnum.quad.solvers import BayesianQuadrature
from probnum.quad.solvers.policies import RandomPolicy, VanDerCorputPolicy
from probnum.quad.solvers.stopping_criteria import (
    ImmediateStop,
    IntegralVarianceTolerance,
    MaxNevals,
    RelativeMeanChange,
)
from probnum.randprocs.kernels import ExpQuad


@pytest.fixture
def input_dim():
    return 3


@pytest.fixture
def data(input_dim):
    def fun(x):
        return 2 * np.ones(x.shape[0])

    nodes = np.ones([20, input_dim])
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


def test_bq_from_problem_wrong_inputs(input_dim):

    # neither measure nor domain is provided
    with pytest.raises(ValueError):
        BayesianQuadrature.from_problem(input_dim=input_dim)


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


@pytest.mark.parametrize(
    "max_evals, var_tol, rel_tol, t",
    [
        (None, None, None, LambdaStoppingCriterion),
        (1000, None, None, MaxNevals),
        (None, 1e-5, None, IntegralVarianceTolerance),
        (None, None, 1e-5, RelativeMeanChange),
        (None, 1e-5, 1e-5, LambdaStoppingCriterion),
        (1000, None, 1e-5, LambdaStoppingCriterion),
        (1000, 1e-5, None, LambdaStoppingCriterion),
        (1000, 1e-5, 1e-5, LambdaStoppingCriterion),
    ],
)
def test_bq_from_problem_stopping_condition_assignment(max_evals, var_tol, rel_tol, t):
    bq = BayesianQuadrature.from_problem(
        input_dim=2,
        domain=(0, 1),
        max_evals=max_evals,
        var_tol=var_tol,
        rel_tol=rel_tol,
    )
    assert isinstance(bq.stopping_criterion, t)


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


@pytest.mark.parametrize("rng", [np.random.default_rng(42), 42])
def test_integrate_runs_with_integer_rng(bq, data, rng):
    # rng is a generator
    nodes, fun_evals, fun = data
    bq.integrate(fun=fun, nodes=None, fun_evals=None, rng=rng)

    # rng is a seed
    nodes, fun_evals, fun = data
    bq.integrate(fun=fun, nodes=None, fun_evals=None, rng=rng)

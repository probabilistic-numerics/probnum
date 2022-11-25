"""Basic tests for Bayesian quadrature method."""

import numpy as np
import pytest

from probnum import LambdaStoppingCriterion
from probnum.quad.integration_measures import LebesgueMeasure
from probnum.quad.solvers import BayesianQuadrature
from probnum.quad.solvers.belief_updates import BQStandardBeliefUpdate
from probnum.quad.solvers.initial_designs import LatinDesign, MCDesign
from probnum.quad.solvers.policies import RandomPolicy, VanDerCorputPolicy
from probnum.quad.solvers.stopping_criteria import (
    ImmediateStop,
    IntegralVarianceTolerance,
    MaxNevals,
    RelativeMeanChange,
)
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
        domain=(0, 1),
    )


@pytest.fixture
def bq_no_policy(input_dim):
    return BayesianQuadrature.from_problem(
        input_dim=input_dim,
        domain=(0, 1),
        policy=None,
    )


@pytest.fixture
def bq_initial_design(input_dim):
    return BayesianQuadrature.from_problem(
        input_dim=input_dim,
        domain=(0, 1),
        policy="bmc",
        initial_design="mc",
        options=dict(num_initial_design_nodes=2),
    )


# =============================================
# Tests for '__init__' function start here.
# =============================================


def test_bayesian_quadrature_wrong_input(input_dim):
    """These exceptions are raised in the __init__ method."""
    measure = LebesgueMeasure(domain=(0, 1), input_dim=input_dim)

    # initial design is given but policy is not given
    with pytest.raises(ValueError):
        BayesianQuadrature(
            kernel=ExpQuad(input_shape=(input_dim,)),
            measure=measure,
            policy=None,
            belief_update=BQStandardBeliefUpdate(jitter=1e-6, scale_estimation=None),
            stopping_criterion=MaxNevals(max_nevals=10),
            initial_design=MCDesign(num_nodes=3, measure=measure),
        )


# =============================================
# Tests for 'from_problem' function start here.
# =============================================

# Tests for correct assignments start here.


@pytest.mark.parametrize(
    "policy, policy_type", [("bmc", RandomPolicy), ("vdc", VanDerCorputPolicy)]
)
def test_bq_from_problem_policy_assignment(policy, policy_type):
    """Test if correct policy is assigned from string identifier."""
    bq = BayesianQuadrature.from_problem(input_dim=1, domain=(0, 1), policy=policy)
    assert isinstance(bq.policy, policy_type)


@pytest.mark.parametrize(
    "design, design_type", [("mc", MCDesign), ("latin", LatinDesign)]
)
def test_bq_from_problem_initial_design_assignment(design, design_type):
    """Test if correct initial design is assigned from string identifier."""
    bq = BayesianQuadrature.from_problem(
        input_dim=1, domain=(0, 1), initial_design=design
    )
    assert isinstance(bq.initial_design, design_type)


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
def test_bq_from_problem_stopping_criterion_assignment(max_evals, var_tol, rel_tol, t):
    bq = BayesianQuadrature.from_problem(
        input_dim=2,
        domain=(0, 1),
        options=dict(max_evals=max_evals, var_tol=var_tol, rel_tol=rel_tol),
    )
    assert isinstance(bq.stopping_criterion, t)


def test_bq_from_problem_defaults(bq, bq_no_policy):

    # defaults if policy is available
    assert isinstance(bq.measure, LebesgueMeasure)
    assert isinstance(bq.kernel, ExpQuad)
    assert isinstance(bq.stopping_criterion, LambdaStoppingCriterion)
    assert isinstance(bq.policy, RandomPolicy)
    assert bq.initial_design is None

    # defaults if no policy is available
    assert isinstance(bq_no_policy.measure, LebesgueMeasure)
    assert isinstance(bq_no_policy.kernel, ExpQuad)
    assert isinstance(bq_no_policy.stopping_criterion, ImmediateStop)
    assert bq_no_policy.policy is None
    assert bq_no_policy.initial_design is None


# Tests for input checks and exception raises start here.


def test_bq_from_problem_wrong_inputs(input_dim):

    # neither measure nor domain is provided
    with pytest.raises(ValueError):
        BayesianQuadrature.from_problem(input_dim=input_dim)

    # unknown policy is provided
    with pytest.raises(NotImplementedError):
        BayesianQuadrature.from_problem(
            input_dim=input_dim, domain=(0, 1), policy="unknown_policy"
        )

    # unknown initial_design is provided
    with pytest.raises(NotImplementedError):
        BayesianQuadrature.from_problem(
            input_dim=input_dim,
            domain=(0, 1),
            policy="bmc",
            initial_design="unknown_design",
        )


# ==========================================
# Tests for 'integrate' function start here.
# ==========================================


@pytest.mark.parametrize("initial_design_provided", [True, False])
@pytest.mark.parametrize("nodes_provided", [True, False])
def test_integrate_output_shapes(initial_design_provided, nodes_provided, data, rng):
    # the test uses max_evals stopping condition in order to check the shapes
    # consistently.

    max_evals = 15
    num_design_nodes = 4

    # get data
    nodes, fun_evals, fun = data
    (num_nodes, input_dim) = nodes.shape

    params = dict(input_dim=input_dim, domain=(0, 1))
    options = dict(max_evals=max_evals)

    num_updates = max_evals
    # get correct shapes and values
    if nodes_provided:
        num_updates += -num_nodes + 1
    else:
        nodes, fun_evals = None, None

    if initial_design_provided:
        num_updates += -num_design_nodes + (1 - 1 * nodes_provided) * 1
        params["initial_design"] = "mc"
        options["num_initial_design_nodes"] = num_design_nodes

    assert num_updates > 1  # make sure that some nodes are collected

    bq = BayesianQuadrature.from_problem(**params, options=options)
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


# Tests for 'integrate' input checks and exception raises start here.


def test_integrate_wrong_input(bq, data, rng):
    """Exception tests shared by all bq methods."""
    # The combination of inputs below is important to trigger the correct exception.

    nodes, fun_evals, fun = data

    # no integrand provided
    with pytest.raises(ValueError):
        bq.integrate(fun=None, nodes=nodes, fun_evals=None, rng=rng)

    # wrong fun_evals shape
    with pytest.raises(ValueError):
        bq.integrate(fun=fun, nodes=nodes, fun_evals=fun_evals[:, None], rng=rng)

    # wrong nodes shape
    with pytest.raises(ValueError):
        bq.integrate(fun=fun, nodes=nodes[:, None], fun_evals=None, rng=rng)

    # number of points in nodes and fun_evals do not match
    wrong_nodes = np.vstack([nodes, np.ones([1, nodes.shape[1]])])
    with pytest.raises(ValueError):
        bq.integrate(fun=fun, nodes=wrong_nodes, fun_evals=fun_evals, rng=rng)

    # no rng provided but policy requires it
    with pytest.raises(ValueError):
        bq.integrate(fun=fun, nodes=nodes, fun_evals=fun_evals, rng=None)


def test_integrate_no_policy_wrong_input(bq_no_policy, data):
    """Exception tests specific to when no policy is given."""
    # The combination of inputs below is important to trigger the correct exception.

    nodes, fun_evals, fun = data

    # no nodes provided
    with pytest.raises(ValueError):
        bq_no_policy.integrate(fun=None, nodes=None, fun_evals=fun_evals)

    # fun is ignored if fun_evals are given
    with pytest.warns(Warning):
        bq_no_policy.integrate(fun=fun, nodes=nodes, fun_evals=fun_evals)


def test_integrate_initial_design_wrong_input(bq_initial_design, data):
    """Exception tests specific to when an initial design is given."""
    # The combination of inputs below is important to trigger the correct exception.

    nodes, fun_evals, fun = data

    # no fun provided but initial design requires it
    with pytest.raises(ValueError):
        bq_initial_design.integrate(fun=None, nodes=nodes, fun_evals=fun_evals)

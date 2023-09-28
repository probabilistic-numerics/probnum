from itertools import product
from typing import Callable

import numpy as np
import pytest
from pytest_cases import parametrize_with_cases

from probnum.problems import QuadratureProblem
from probnum.problems.zoo.quad import (
    bratley1992,
    genz_continuous,
    genz_cornerpeak,
    genz_discontinuous,
    genz_gaussian,
    genz_oscillatory,
    genz_productpeak,
    gfunction,
    morokoff_caflisch_1,
    morokoff_caflisch_2,
    roos_arnold,
)
from tests.test_problems.test_zoo.test_quad.test_quadproblems_uniform_cases import (
    GenzUniformCases,
    OtherIntegrandsUniformCases,
)


@pytest.mark.parametrize(
    "genz_problem",
    [
        genz_continuous,
        genz_cornerpeak,
        genz_discontinuous,
        genz_gaussian,
        genz_oscillatory,
        genz_productpeak,
    ],
)
def test_genz_uniform_param_checks(genz_problem):
    with pytest.raises(ValueError):
        genz_problem(2, a=np.ones(shape=(1,)))

    with pytest.raises(ValueError):
        genz_problem(3, u=np.ones(shape=(2, 1)))

    with pytest.raises(ValueError):
        genz_problem(3, u=np.full((3,), 1.1))

    with pytest.raises(ValueError):
        genz_problem(3, u=np.full((3,), -0.1))


@pytest.mark.parametrize(
    "quad_problem_constructor",
    [
        bratley1992,
        genz_continuous,
        genz_cornerpeak,
        genz_discontinuous,
        genz_gaussian,
        genz_oscillatory,
        genz_productpeak,
        gfunction,
        morokoff_caflisch_1,
        morokoff_caflisch_2,
        roos_arnold,
    ],
)
def test_integrand_eval_checks(
    quad_problem_constructor: Callable[..., QuadratureProblem]
):
    quad_problem = quad_problem_constructor(2)

    with pytest.raises(ValueError):
        quad_problem.fun(np.zeros((4, 3)))

    with pytest.raises(ValueError):
        quad_problem.fun(np.full((4, 2), -0.1))


dim_values = [1, 2]
quad_prob_constructor_values = [
    bratley1992,
    genz_continuous,
    genz_cornerpeak,
    genz_discontinuous,
    genz_gaussian,
    genz_oscillatory,
    genz_productpeak,
    gfunction,
    morokoff_caflisch_1,
    morokoff_caflisch_2,
    roos_arnold,
]
param_combinations = list(product(dim_values, quad_prob_constructor_values))


@pytest.mark.parametrize("dim, quad_prob_constructor", param_combinations)
def test_integrand_solution_float(
    dim: int, quad_prob_constructor: Callable[..., QuadratureProblem]
):
    quadprob = quad_prob_constructor(dim)
    if np.ndim(quadprob.solution) != 0:
        raise ValueError(f"The solution of {quadprob} is not a scalar.")


@parametrize_with_cases(
    "quadprob, rtol",
    cases=(GenzUniformCases, OtherIntegrandsUniformCases),
)
def test_quadprob_uniform_with_mc(quadprob, rtol):
    """Compare a Monte Carlo estimator against uniform measure with a very large number
    of samples to the true value of the integral.

    The former should be an approximation of the latter.
    """

    # Number of Monte Carlo samples for the test
    n = 10000000

    # generate some uniformly distributed points from [0, 1]
    rng = np.random.default_rng(0)
    x_uniform = quadprob.measure.sample(n, rng=rng)  # Monte Carlo samples x_1,...,x_n

    # Test that all Monte Carlo estimators approximate the true value of the integral
    # (integration against [0, 1])
    np.testing.assert_allclose(
        np.sum(quadprob.fun(x_uniform)) / n,
        quadprob.solution,
        rtol=rtol,
    )

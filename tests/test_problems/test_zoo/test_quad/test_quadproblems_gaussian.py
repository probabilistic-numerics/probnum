from itertools import product
from typing import Callable, Optional, Union

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
    sum_polynomials,
    uniform_to_gaussian_quadprob,
)
from probnum.quad.integration_measures import LebesgueMeasure
from probnum.quad.typing import DomainLike
from tests.test_problems.test_zoo.test_quad.test_quadproblems_gaussian_cases import (
    GenzStandardNormalCases,
    GenzVariedNormalCases,
    OtherIntegrandsGaussianCases,
)

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


@pytest.mark.parametrize("quad_prob_constructor", quad_prob_constructor_values)
@pytest.mark.parametrize("dim", dim_values)
def test_wrapping_all_test_functions_works(
    dim: int,
    quad_prob_constructor: Callable[[int], QuadratureProblem],
):
    """Integration test that wrapping all problems works."""
    quadprob = quad_prob_constructor(dim)
    gaussian_quadprob = uniform_to_gaussian_quadprob(quadprob)
    assert isinstance(gaussian_quadprob, QuadratureProblem)


@pytest.mark.parametrize(
    "domain, dim",
    [
        ((0.5, 1.0), 1),
        ((0.5, 1.0), 2),
        ((0.0, 1.5), 1),
        ((0.0, 1.5), 2),
    ],
)
def test_wrong_measure_bounds_scalar_raises(domain: DomainLike, dim: int):
    uniform_measure = LebesgueMeasure(domain=domain, input_dim=dim)
    quadprob = genz_continuous(dim)
    quadprob.measure = uniform_measure
    with pytest.raises(ValueError) as exc_info:
        uniform_to_gaussian_quadprob(quadprob)

    assert "[0,1]" in str(exc_info.value)


@pytest.mark.parametrize(
    "domain",
    [
        (np.array([0.0, 0.5]), np.array([1.0, 1.0])),
        (np.array([0.0, 0.0]), np.array([1.0, 1.5])),
    ],
)
def test_wrong_measure_bounds_array_raises(domain: DomainLike):
    lower_bd, _ = domain
    dim = len(lower_bd)
    uniform_measure = LebesgueMeasure(domain=domain, input_dim=dim)
    quadprob = genz_continuous(dim)
    quadprob.measure = uniform_measure
    with pytest.raises(ValueError) as exc_info:
        uniform_to_gaussian_quadprob(quadprob)

    assert "[0,1]" in str(exc_info.value)


def test_wrong_data_shapes_for_mean_raises():
    with pytest.raises(TypeError) as exc_info:
        uniform_to_gaussian_quadprob(
            genz_continuous(2), mean=np.array([[0.0, 0.5]]), std=np.array([0.0, 0.5])
        )

    assert "mean parameter" in str(exc_info.value)


def test_wrong_data_shapes_for_std_raises():
    with pytest.raises(TypeError) as exc_info:
        uniform_to_gaussian_quadprob(
            genz_continuous(2), mean=np.array([0.0, 0.5]), std=np.array([[0.0, 0.5]])
        )

    assert "std parameter" in str(exc_info.value)


def test_wrong_shaped_a_in_sum_polynomials_raises():
    a = np.array([1.0, 1.0])
    dim = 2
    with pytest.raises(ValueError) as exc_info:
        sum_polynomials(dim=dim, a=a)

    assert "Invalid shape" in str(exc_info.value)
    assert "parameter `a`" in str(exc_info.value)


def test_wrong_shaped_b_in_sum_polynomials_raises():
    b = np.array([1.0, 1.0])
    dim = 2
    with pytest.raises(ValueError) as exc_info:
        sum_polynomials(dim=dim, b=b)

    assert "Invalid shape" in str(exc_info.value)
    assert "parameter `b`" in str(exc_info.value)


def test_mismatch_dim_a_in_sum_polynomials_raises():
    a = np.array([[1.0, 1.0]])
    dim = 1
    with pytest.raises(ValueError) as exc_info:
        sum_polynomials(dim=dim, a=a)

    assert "parameter `a`" in str(exc_info.value)
    assert f"Expected {dim} columns" in str(exc_info.value)


def test_mismatch_dim_b_in_sum_polynomials_raises():
    b = np.array([[1.0, 1.0]])
    dim = 1
    with pytest.raises(ValueError) as exc_info:
        sum_polynomials(dim=dim, b=b)

    assert "parameter `b`" in str(exc_info.value)
    assert f"Expected {dim} columns" in str(exc_info.value)


def test_negative_values_for_b_in_sum_polynomials_raises():
    b = np.array([[-0.5, 1.0]])
    dim = 2
    with pytest.raises(ValueError) as exc_info:
        sum_polynomials(dim=dim, b=b)

    assert "negative" in str(exc_info.value)
    assert "parameters `b`" in str(exc_info.value)


@parametrize_with_cases(
    "quadprob, rtol",
    cases=(
        GenzStandardNormalCases,
        GenzVariedNormalCases,
        OtherIntegrandsGaussianCases,
    ),
)
def test_gaussian_quadprob(quadprob: QuadratureProblem, rtol: float):
    """Compare a Monte Carlo estimator against Gaussian measure with a very large number
    of samples to the true value of the integral.

    The former should be an approximation of the latter.
    """
    # Number of Monte Carlo samples for the test
    n = 100000

    # generate some normally distributed points from N(0, 1)
    rng = np.random.default_rng(0)
    x_gaussian = quadprob.measure.sample(n, rng=rng)  # Monte Carlo samples x_1,...,x_n

    # Test that all Monte Carlo estimators approximate the true value of the integral
    # (integration against Gaussian)
    np.testing.assert_allclose(
        np.sum(quadprob.fun(x_gaussian)) / n,
        quadprob.solution,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "dim, a, b, var",
    [
        (1, None, None, 1.0),
        (2, None, None, 1.0),
        (1, np.array([[0.5]]), np.array([[1]]), 0.5),
        (1, np.array([[0.5]]), np.array([[1]]), np.array([[0.5]])),
        (
            2,
            np.array([[0.5, 1.0], [2.0, 2.0]]),
            np.array([[1, 2], [2, 3]]),
            np.array([[1.0, 0], [0, 0.5]]),
        ),
    ],
)
def test_sum_polynomials(
    dim: int,
    a: Optional[np.ndarray],
    b: Optional[np.ndarray],
    var: Union[float, np.ndarray],
):
    """Compare a Monte Carlo estimator against Gaussian measure with a very large number
    of samples to the true value of the integral.

    The former should be an approximation of the latter.
    """

    # Number of Monte Carlo samples for the test
    n = 100000

    quadprob = sum_polynomials(dim, a, b, var)

    # generate some normally distributed points from N(0, 1)
    rng = np.random.default_rng(0)
    x_gaussian = quadprob.measure.sample(n, rng=rng)  # Monte Carlo samples x_1,...,x_n

    # Test that all Monte Carlo estimators approximate the true value of the integral
    # (integration against uniform)
    np.testing.assert_allclose(
        np.sum(quadprob.fun(x_gaussian)) / n,
        quadprob.solution,
        atol=1e-02,
    )

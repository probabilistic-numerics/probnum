import numpy as np
import pytest

import probnum.problems.zoo.quad.quadproblems_uniform


@pytest.mark.parametrize(
    "a, u, dim",
    [
        (5, 0.5, 1),
        (5.0, 1.0, 1),
        (2.0, 1.2, 1),
        (5, 0.5, 4),
        (5.0, 1.0, 4),
        (2.0, 1.2, 4),
    ],
)
def test_Genz_uniform(a, u, dim):
    """Compare a Monte Carlo estimator with a very large number of samples to the true
    value of the integral.

    The former should be an approximation of the latter.
    """

    # Number of Monte Carlo samples for the test
    n = 10000000

    # Define parameters a and u
    a_vec = np.repeat(a, dim)
    u_vec = np.repeat(u, dim)

    # generate some uniformly distributed points on [0,1]^d
    x_unif = np.random.uniform(
        low=0.0, high=1.0, size=(n, dim)
    )  # Monte Carlo samples x_1,...,x_n

    # Test that all Monte Carlo estimators approximate the true value of the integral (integration against uniform)
    np.testing.assert_allclose(
        np.sum(Genz_continuous(x_unif, a_vec, u_vec)) / n,
        integral_Genz_continuous(a_vec, u_vec),
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        np.sum(Genz_cornerpeak(x_unif, a_vec, u_vec)) / n,
        integral_Genz_cornerpeak(a_vec, u_vec),
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        np.sum(Genz_discontinuous(x_unif, a_vec, u_vec)) / n,
        integral_Genz_discontinuous(a_vec, u_vec),
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        np.sum(Genz_oscillatory(x_unif, a_vec, u_vec)) / n,
        integral_Genz_oscillatory(a_vec, u_vec),
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        np.sum(Genz_productpeak(x_unif, a_vec, u_vec)) / n,
        integral_Genz_productpeak(a_vec, u_vec),
        rtol=1e-03,
    )


@pytest.mark.parametrize("dim", [(1), (4), (10)])
def test_integrands_uniform(dim):
    """Compare a Monte Carlo estimator with a very large number of samples to the true
    value of the integral.

    The former should be an approximation of the latter.
    """

    # Number of Monte Carlo samples for the test
    n = 10000000

    # generate some uniformly distributed points on [0,1]^d
    x_unif = np.random.uniform(
        low=0.0, high=1.0, size=(n, dim)
    )  # Monte Carlo samples x_1,...,x_n

    # Test that all Monte Carlo estimators approximate the true value of the integral (integration against uniform)
    np.testing.assert_allclose(
        np.sum(Bratley1992(x_unif)) / n, integral_Bratley1992(dim), rtol=1e-03
    )
    np.testing.assert_allclose(
        np.sum(RoosArnold(x_unif)) / n, integral_RoosArnold(), rtol=1e-03
    )
    np.testing.assert_allclose(
        np.sum(Gfunction(x_unif)) / n, integral_Gfunction(), rtol=1e-03
    )
    np.testing.assert_allclose(
        np.sum(MorokoffCaflisch1(x_unif)) / n, integral_MorokoffCaflisch1(), rtol=1e-03
    )
    np.testing.assert_allclose(
        np.sum(MorokoffCaflisch2(x_unif)) / n, integral_MorokoffCaflisch1(), rtol=1e-03
    )

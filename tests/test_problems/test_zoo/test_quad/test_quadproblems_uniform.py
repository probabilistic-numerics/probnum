import numpy as np
import pytest

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
def test_integrand_eval_checks(quad_problem_constructor):
    quad_problem = quad_problem_constructor(2)

    with pytest.raises(ValueError):
        quad_problem.integrand(np.zeros((4, 3)))

    with pytest.raises(ValueError):
        quad_problem.integrand(np.full((4, 2), -0.1))


@pytest.mark.parametrize(
    "a, u, dim",
    [
        (None, None, 3),
        (5, 0.5, 1),
        (5.0, 1.0, 1),
        (2.0, 0.8, 1),
        (5, 0.5, 4),
        (5.0, 1.0, 4),
        (2.0, 0.8, 4),
    ],
)
def test_genz_uniform(a, u, dim):
    """Compare a Monte Carlo estimator with a very large number of samples to the true
    value of the integral.

    The former should be an approximation of the latter.
    """

    # Number of Monte Carlo samples for the test
    n = 10000000

    # Define parameters a and u
    a_vec = np.repeat(a, dim) if a is not None else None
    u_vec = np.repeat(u, dim) if u is not None else None

    quadprob_genz_continuous = genz_continuous(dim=dim, a=a_vec, u=u_vec)
    quadprob_genz_cornerpeak = genz_cornerpeak(dim=dim, a=a_vec, u=u_vec)
    quadprob_genz_discontinuous = genz_discontinuous(dim=dim, a=a_vec, u=u_vec)
    quadprob_genz_gaussian = genz_gaussian(dim=dim, a=a_vec, u=u_vec)
    quadprob_genz_oscillatory = genz_oscillatory(dim=dim, a=a_vec, u=u_vec)
    quadprob_genz_productpeak = genz_productpeak(dim=dim, a=a_vec, u=u_vec)

    # generate some uniformly distributed points on [0,1]^d
    np.random.seed(0)
    x_unif = np.random.uniform(
        low=0.0, high=1.0, size=(n, dim)
    )  # Monte Carlo samples x_1,...,x_n

    # Test that all Monte Carlo estimators approximate the true value of the integral
    # (integration against uniform)
    np.testing.assert_allclose(
        np.sum(quadprob_genz_continuous.integrand(x_unif)) / n,
        quadprob_genz_continuous.solution,
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        np.sum(quadprob_genz_cornerpeak.integrand(x_unif)) / n,
        quadprob_genz_cornerpeak.solution,
        rtol=3e-03,
    )
    np.testing.assert_allclose(
        np.sum(quadprob_genz_discontinuous.integrand(x_unif)) / n,
        quadprob_genz_discontinuous.solution,
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        np.sum(quadprob_genz_gaussian.integrand(x_unif)) / n,
        quadprob_genz_gaussian.solution,
        rtol=2e-03,
    )
    np.testing.assert_allclose(
        np.sum(quadprob_genz_oscillatory.integrand(x_unif)) / n,
        quadprob_genz_oscillatory.solution,
        rtol=3e-02,
    )
    np.testing.assert_allclose(
        np.sum(quadprob_genz_productpeak.integrand(x_unif)) / n,
        quadprob_genz_productpeak.solution,
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
    np.random.seed(0)
    x_unif = np.random.uniform(
        low=0.0, high=1.0, size=(n, dim)
    )  # Monte Carlo samples x_1,...,x_n

    # Set the integrands to test
    quadprob_bratley1992 = bratley1992(dim=dim)
    quadprob_roos_arnold = roos_arnold(dim=dim)
    quadprob_gfunction = gfunction(dim=dim)
    quadprob_morokoff_caflisch_1 = morokoff_caflisch_1(dim=dim)
    quadprob_morokoff_caflisch_2 = morokoff_caflisch_2(dim=dim)

    # Test that all Monte Carlo estimators approximate the true value of the integral
    # (integration against uniform)
    np.testing.assert_allclose(
        np.sum(quadprob_bratley1992.integrand(x_unif)) / n,
        quadprob_bratley1992.solution,
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        np.sum(quadprob_roos_arnold.integrand(x_unif)) / n,
        quadprob_roos_arnold.solution,
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        np.sum(quadprob_gfunction.integrand(x_unif)) / n,
        quadprob_gfunction.solution,
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        np.sum(quadprob_morokoff_caflisch_1.integrand(x_unif)) / n,
        quadprob_morokoff_caflisch_1.solution,
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        np.sum(quadprob_morokoff_caflisch_2.integrand(x_unif)) / n,
        quadprob_morokoff_caflisch_2.solution,
        rtol=1e-03,
    )

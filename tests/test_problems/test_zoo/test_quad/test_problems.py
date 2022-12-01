"""Tests for quadrature problems."""

import numpy as np
import pytest

from probnum.problems.zoo.quad import circulargaussian2d, hennig1d, hennig2d, sombrero2d
from probnum.quad.integration_measures import LebesgueMeasure


@pytest.fixture(params=[hennig1d, hennig2d, sombrero2d, circulargaussian2d])
def problem(request):
    """quad problems with default values"""
    return request.param()


@pytest.fixture(
    params=[
        (hennig2d, dict(c=np.array([[2, 0.1], [0.1, 1]]))),
        (circulargaussian2d, dict(m=1.0, v=1.5)),
        (sombrero2d, dict(w=2.0)),
    ],
    ids=["hennig2d", "circulargaussian2d", "sombrero2d"],
)
def problem_parametrized(request):
    """quad problems with custom parameters"""
    return request.param[0](**request.param[1])


def test_problem_solution_type(problem):
    assert isinstance(problem.solution, float)


def test_problem_parametrized_solution_type(problem_parametrized):
    # problems that do not have an analytic solution currently do not provide any.
    assert problem_parametrized.solution is None


@pytest.mark.parametrize("num_dat", [1, 5])
def test_problem_fun_shapes(problem, num_dat):
    input_dim = problem.input_dim
    res = problem.fun(np.ones([num_dat, input_dim]))
    assert res.shape == (num_dat,)


def test_problem_correct_solution_value(problem):
    s = problem.solution
    m = problem.measure
    if s is not None and isinstance(s, float) and problem.input_dim <= 2:
        x = m.sample(int(1e3), rng=np.random.default_rng(0))
        f = problem.fun(x)
        s_test = f.mean()
        c_test = f.std()

        # scale MC estimator with domain volume for unnormalized Lebesgue measure.
        if isinstance(m, LebesgueMeasure) and not m.normalized:
            volume = np.prod(m.domain[1] - m.domain[0])
            s_test *= volume
            c_test *= volume

        # Check if integral lies in a 95% confidence interval of the MC estimator.
        assert s_test - 2 * c_test < s < s_test + 2 * c_test


def test_problem_sombrero2d_raises():

    # frequency is negative
    with pytest.raises(ValueError):
        sombrero2d(w=-2.0)


def test_problem_hennig2d_raises():

    # wrong c shape
    with pytest.raises(ValueError):
        hennig2d(c=np.eye(3))

    # c not pos def
    with pytest.raises(ValueError):
        hennig2d(c=-np.eye(3))


def test_problem_circulargaussian2d_raises():

    # mean m negative
    with pytest.raises(ValueError):
        circulargaussian2d(m=-2.0)

    # mean v non-positive
    with pytest.raises(ValueError):
        circulargaussian2d(v=0.0)

"""import numpy as np import pytest from scipy.integrate._ivp import rk.

from probnum import diffeq
from probnum.diffeq.perturbedsolvers import perturbedstepsolution
from pn_ode_benchmarks import scipy_solver


@pytest.fixture
def initrv():
    return np.array([0.1])


@pytest.fixture
def ivp():
    return diffeq.logistic([0.0, 10], initrv)


@pytest.fixture
def testsolver45(ivp, initrv):
    testsolver = rk.RK45(ivp.rhs, ivp.t0, initrv, ivp.tmax)
    return scipy_solver.ScipyRungeKutta(testsolver, order=4)


@pytest.fixture
def steprule():
    return diffeq.ConstantSteps(0.1)


@pytest.fixture
def solutiontest(testsolver45, steprule):
    return testsolver45.solve(steprule)


@pytest.fixture
def solutionscipy(solutiontest):
    return solutiontest.scipy_solution


def test_t(solutionscipy, solutiontest):
    scipy_t = solutionscipy.ts
    probnum_t = solutiontest.t
    np.testing.assert_allclose(scipy_t, probnum_t, atol=1e-14, rtol=1e-14)



def test_y(solutionscipy, solutiontest):
    scipy_states = np.array(solutionscipy(solutionscipy.ts)).T
    probnum_states = np.array(solutiontest.y.mean)
    np.testing.assert_allclose(scipy_states, probnum_states, atol=1e-14, rtol=1e-14)


def test_call__(solutionscipy, solutiontest):
    scipy_call = solutionscipy(solutionscipy.ts)
    probnum_call = solutiontest(solutionscipy.ts).mean.reshape(scipy_call.shape)
    np.testing.assert_allclose(scipy_call, probnum_call, atol=1e-14, rtol=1e-14)


def test_len__(solutionscipy, solutiontest):
    scipy_len = len(solutionscipy.ts)
    probnum_len = len(solutiontest)
    np.testing.assert_allclose(scipy_len, probnum_len, atol=1e-14, rtol=1e-14)


def test_getitem__(solutionscipy, solutiontest):
    scipy_item = solutionscipy.interpolants[1](solutionscipy.ts[1])
    probnum_item = solutiontest[1]
    np.testing.assert_allclose(scipy_item, probnum_item, atol=1e-14, rtol=1e-14)


def test_sample(solutiontest):
    probnum_sample = solutiontest.sample(5)
    np.testing.assert_string_equal(probnum_sample, "Sampling not possible")
"""

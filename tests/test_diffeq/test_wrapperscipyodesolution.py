import pathlib

import numpy as np
import pytest
import pytest_cases
from scipy.integrate._ivp import rk

from probnum import diffeq
from probnum.diffeq import wrapperscipysolver

case_modules = [
    ".test_wrapperscipyodesolution_cases." + path.stem
    for path in (
        pathlib.Path(__file__).parent / "test_wrapperscipyodesolution_cases"
    ).glob("*_cases.py")
]


@pytest.fixture
def solutiontest():
    ivp = diffeq.logistic([0.0, 10], np.array([1]))
    scipysolver = rk.RK45(ivp.rhs, ivp.t0, np.array([1]), ivp.tmax)
    testsolver = wrapperscipysolver.WrapperScipyRungeKutta(scipysolver, order=4)
    return testsolver.solve(diffeq.ConstantSteps(0.5))


@pytest_cases.parametrize_with_cases("testsolution,scipysolution", cases=case_modules)
def test_locations(testsolution, scipysolution):
    scipy_t = scipysolution.ts
    probnum_t = testsolution.locations
    np.testing.assert_allclose(scipy_t, probnum_t, atol=1e-14, rtol=1e-14)


@pytest_cases.parametrize_with_cases("testsolution,scipysolution", cases=case_modules)
def test_states(testsolution, scipysolution):
    scipy_states = np.array(scipysolution(scipysolution.ts)).T
    probnum_states = np.array(testsolution.states.mean)
    np.testing.assert_allclose(scipy_states, probnum_states, atol=1e-14, rtol=1e-14)


@pytest_cases.parametrize_with_cases("testsolution,scipysolution", cases=case_modules)
def test_call__(testsolution, scipysolution):
    scipy_call = scipysolution(scipysolution.ts)
    probnum_call = testsolution(scipysolution.ts).mean.T
    np.testing.assert_allclose(scipy_call, probnum_call, atol=1e-14, rtol=1e-14)


@pytest_cases.parametrize_with_cases("testsolution,scipysolution", cases=case_modules)
def test_len__(testsolution, scipysolution):
    scipy_len = len(scipysolution.ts)
    probnum_len = len(testsolution)
    np.testing.assert_allclose(scipy_len, probnum_len, atol=1e-14, rtol=1e-14)


@pytest_cases.parametrize_with_cases("testsolution,scipysolution", cases=case_modules)
def test_getitem__(testsolution, scipysolution):
    scipy_item = scipysolution.interpolants[1](scipysolution.ts[1])
    probnum_item = testsolution[1]
    np.testing.assert_allclose(scipy_item, probnum_item, atol=1e-14, rtol=1e-14)


def test_sample(solutiontest):
    probnum_sample = solutiontest.sample(5)
    np.testing.assert_string_equal(probnum_sample, "Sampling not possible")

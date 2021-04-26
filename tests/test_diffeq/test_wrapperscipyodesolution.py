import pathlib

import numpy as np
import pytest
import pytest_cases
from scipy.integrate._ivp import rk

from probnum import diffeq
from probnum.diffeq import wrappedscipysolver

case_modules = [
    ".test_wrappedscipyodesolution_cases." + path.stem
    for path in (
        pathlib.Path(__file__).parent / "test_wrappedscipyodesolution_cases"
    ).glob("*_cases.py")
]


@pytest.fixture
def solutiontest():
    ivp = diffeq.logistic([0.0, 10], np.array([1]))
    scipysolver = rk.RK45(ivp.rhs, ivp.t0, np.array([1]), ivp.tmax)
    testsolver = wrappedscipysolver.WrappedScipyRungeKutta(scipysolver, order=4)
    return testsolver.solve(diffeq.ConstantSteps(0.5))


@pytest.fixture
@parametrize(params=["lorenz", "logistic"])
def ivp(requests):
    which_ode = requests.param
    choose_ode = {
        "lorenz": probnum.diffeq.lorenz(y0=(10, 10, 10)),
        "logistic": logistic(),
    }
    return choose_ode[which_ode]


@pytest.fixture
def testsolution_and_scipysolution(ivp, steprule):
    (...)
    return testsolution, scipysolution


all_odes = pytest.mark.parametrize(
    ["ivp", [probnum.diffeq.lorenz(y0=(10, 10, 10), t0=0.0, tmax=1.0), logistic(), ...]]
)
all_steprule = pytest.mark.parametrize(["steprule", [ConstantSteps(), AdaptiveSteps()]])


@pytest_cases.parametrize_with_cases("testsolution,scipysolution", cases=case_modules)
@all_odes
@all_steprules
def test_locations(testsolution_and_scipysolution):
    testsolution, scipysolution = testsolution_and_scipysolution
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

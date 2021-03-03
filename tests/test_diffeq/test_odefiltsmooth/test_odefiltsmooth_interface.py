import numpy as np
import pytest

from probnum.diffeq import ode
from probnum.diffeq.odefiltsmooth import probsolve_ivp
from probnum.random_variables import Constant


@pytest.fixture
def ivp():
    initrv = Constant(20.0 * np.ones(2))
    return ode.lotkavolterra([0.0, 0.5], initrv)


@pytest.mark.parametrize("method", ["ek0", "ek1"])
@pytest.mark.parametrize(
    "which_prior",
    [
        "ibm1",
        "ibm2",
        "ibm3",
        "ibm4",
        "ioup1",
        "ioup2",
        "ioup3",
        "ioup4",
        "matern32",
        "matern52",
        "matern72",
        "matern92",
    ],
)
@pytest.mark.parametrize("dense_output", ["True", "False"])
def test_adaptive_solver_successfull(ivp, method, which_prior, dense_output):
    f = ivp.rhs
    df = ivp.jacobian
    t0, tmax = ivp.timespan
    y0 = ivp.initrv.mean
    probsolve_ivp(
        f,
        t0,
        tmax,
        y0,
        df=df,
        atol=1e-1,
        rtol=1e-1,
        which_prior=which_prior,
        method=method,
        dense_output=dense_output,
    )

import numpy as np
import pytest

import probnum.problems as pnpr
import probnum.problems.zoo.diffeq as diffeqzoo

ODE_LIST = [
    diffeqzoo.vanderpol(),
    diffeqzoo.threebody(),
    diffeqzoo.rigidbody(),
    diffeqzoo.lotkavolterra(),
    diffeqzoo.logistic(),
    diffeqzoo.seir(),
    diffeqzoo.fitzhughnagumo(),
    diffeqzoo.lorenz(),
]

all_odes = pytest.mark.parametrize("ivp", ODE_LIST)


@all_odes
def test_isinstance(ivp):
    assert isinstance(ivp, pnpr.InitialValueProblem)


@all_odes
def test_eval(ivp):
    f0 = ivp.f(ivp.t0, ivp.y0)
    assert isinstance(f0, np.ndarray)
    if ivp.df is not None:
        df0 = ivp.df(ivp.t0, ivp.y0)
        assert isinstance(df0, np.ndarray)
    if ivp.ddf is not None:
        ddf0 = ivp.ddf(ivp.t0, ivp.y0)
        assert isinstance(ddf0, np.ndarray)


@all_odes
def test_df0(ivp):
    if ivp.df is not None:
        step = 1e-6

        time = ivp.t0 + 0.1 * np.random.rand()
        direction = step * (1.0 + 0.1 * np.random.rand(len(ivp.y0)))
        increment = step * direction
        point = ivp.y0 + 0.1 * np.random.rand(len(ivp.y0))

        fd_approx = (
            ivp.f(time, point + increment) - ivp.f(time, point - increment)
        ) / (2.0 * step)

        np.testing.assert_allclose(
            fd_approx, ivp.df(time, point) @ direction, rtol=1e-3, atol=1e-3
        )

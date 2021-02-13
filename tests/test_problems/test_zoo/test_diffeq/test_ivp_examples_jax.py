import pytest

import probnum.diffeq as pnd
import probnum.problems as pnprob
import probnum.problems.zoo.diffeq as diffeq_zoo


@pytest.mark.parametrize(
    "ivp_jax", [diffeq_zoo.threebody_jax(), diffeq_zoo.vanderpol_jax()]
)
@pytest.mark.parametrize("order", [1, 2, 3])
def test_compute_all_derivatives_terminates_successfully(ivp_jax, order):
    """Test asserts that the examples in diffeq-zoo are compatible with
    `compute_all_derivatives`, which happens if they are implemented in jax, and jax is
    available in the current environment."""

    ivp = pnd.compute_all_derivatives(ivp_jax, order=1)
    assert isinstance(ivp, pnprob.InitialValueProblem)


@pytest.mark.parametrize(
    "ivp_jax", [diffeq_zoo.threebody_jax(), diffeq_zoo.vanderpol_jax()]
)
def test_f(ivp_jax):
    ivp_jax.f(ivp_jax.t0, ivp_jax.y0)


@pytest.mark.parametrize(
    "ivp_jax", [diffeq_zoo.threebody_jax(), diffeq_zoo.vanderpol_jax()]
)
def test_df(ivp_jax):
    ivp_jax.df(ivp_jax.t0, ivp_jax.y0)


@pytest.mark.parametrize(
    "ivp_jax", [diffeq_zoo.threebody_jax(), diffeq_zoo.vanderpol_jax()]
)
def test_ddf(ivp_jax):
    ivp_jax.ddf(ivp_jax.t0, ivp_jax.y0)

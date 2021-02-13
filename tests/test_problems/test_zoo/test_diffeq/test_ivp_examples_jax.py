import pytest

import probnum.diffeq as pnd
import probnum.problems as pnprob
import probnum.problems.zoo.diffeq as diffeq_zoo


@pytest.mark.parametrize(
    "ivp_jax", [diffeq_zoo.threebody_jax(), diffeq_zoo.vanderpol_jax()]
)
def test_compute_all_derivatives_terminates_successfully(ivp_jax):
    """Test asserts that the examples in diffeq-zoo are compatible with
    `compute_all_derivatives`, which happens if they are implemented in jax, and jax is
    available in the current environment."""

    ivp = pnd.compute_all_derivatives(ivp_jax, order=1)
    assert isinstance(ivp, pnprob.InitialValueProblem)

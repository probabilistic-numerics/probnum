"""Tests for IntegratedOrnsteinUhlenbeckProcessTransitions."""


import numpy as np
import pytest

from probnum import randprocs, randvars
from tests.test_randprocs.test_markov.test_continuous import test_sde
from tests.test_randprocs.test_markov.test_integrator import test_integrator


@pytest.mark.parametrize("driftspeed", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("initarg", [0.0, 2.0])
@pytest.mark.parametrize("num_derivatives", [0, 1, 4])
@pytest.mark.parametrize("wiener_process_dimension", [1, 2, 3])
@pytest.mark.parametrize("use_initrv", [True, False])
@pytest.mark.parametrize("diffuse", [True, False])
def test_ioup_construction(
    driftspeed, initarg, num_derivatives, wiener_process_dimension, use_initrv, diffuse
):
    if use_initrv:
        d = (num_derivatives + 1) * wiener_process_dimension
        initrv = randvars.Normal(np.arange(d), np.diag(np.arange(1, d + 1)))
    else:
        initrv = None
    if use_initrv and diffuse:
        with pytest.warns(Warning):
            randprocs.markov.integrator.IntegratedOrnsteinUhlenbeckProcess(
                driftspeed=driftspeed,
                initarg=initarg,
                num_derivatives=num_derivatives,
                wiener_process_dimension=wiener_process_dimension,
                initrv=initrv,
                diffuse=diffuse,
            )

    else:
        ioup = randprocs.markov.integrator.IntegratedOrnsteinUhlenbeckProcess(
            driftspeed=driftspeed,
            initarg=initarg,
            num_derivatives=num_derivatives,
            wiener_process_dimension=wiener_process_dimension,
            initrv=initrv,
            diffuse=diffuse,
        )

        isinstance(
            ioup,
            randprocs.markov.integrator.IntegratedOrnsteinUhlenbeckProcess,
        )
        isinstance(ioup, randprocs.markov.MarkovProcess)
        isinstance(
            ioup.transition,
            randprocs.markov.integrator.IntegratedOrnsteinUhlenbeckTransition,
        )


class TestIntegratedOrnsteinUhlenbeckProcessTransition(
    test_sde.TestLTISDE, test_integrator.TestIntegratorTransition
):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        some_num_derivatives,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):
        self.some_num_derivatives = some_num_derivatives
        wiener_process_dimension = 1  # make tests compatible with some_normal_rv1, etc.
        self.transition = (
            randprocs.markov.integrator.IntegratedOrnsteinUhlenbeckTransition(
                num_derivatives=self.some_num_derivatives,
                wiener_process_dimension=wiener_process_dimension,
                driftspeed=1.2345,
                forward_implementation=forw_impl_string_linear_gauss,
                backward_implementation=backw_impl_string_linear_gauss,
            )
        )

        self.G = lambda t: self.transition.driftmat
        self.v = lambda t: self.transition.forcevec
        self.L = lambda t: self.transition.dispmat

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)

    @property
    def integrator(self):
        return self.transition

"""Tests for IOUPs."""


import pytest

from probnum import randprocs
from tests.test_randprocs.test_markov.test_continuous import test_sde
from tests.test_randprocs.test_markov.test_continuous.test_integrator import (
    test_integrator,
)


class TestIOUP(test_sde.TestLTISDE, test_integrator.TestIntegrator):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        some_ordint,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):
        self.some_ordint = some_ordint
        spatialdim = 1  # make tests compatible with some_normal_rv1, etc.
        self.transition = randprocs.markov.continuous.integrator.IOUP(
            ordint=self.some_ordint,
            spatialdim=spatialdim,
            driftspeed=1.2345,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        self.G = lambda t: self.transition.driftmat
        self.v = lambda t: self.transition.forcevec
        self.L = lambda t: self.transition.dispmat

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)

    @property
    def integrator(self):
        return self.transition

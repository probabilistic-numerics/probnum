import numpy as np
import pytest

from probnum import randprocs, randvars
from tests.test_randprocs.test_markov.test_continuous import test_linear_sde


class TestLTISDE(test_linear_sde.TestLinearSDE):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        test_ndim,
        spdmat1,
        spdmat2,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):

        self.G_const = spdmat1
        self.v_const = np.arange(test_ndim)
        self.L_const = spdmat2

        self.transition = randprocs.markov.continuous.LTISDE(
            drift_matrix=self.G_const,
            force_vector=self.v_const,
            dispersion_matrix=self.L_const,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        self.G = lambda t: spdmat1
        self.v = lambda t: np.arange(test_ndim)
        self.L = lambda t: spdmat2

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)
        self.l = lambda t, x: self.L(t)

    def test_discretise(self):
        out = self.transition.discretise(dt=0.1)
        assert isinstance(out, randprocs.markov.discrete.LTIGaussian)

    def test_discretise_no_force(self):
        """LTISDE.discretise() works if there is zero force (there is an "if" in the
        fct)."""
        new_trans = self.transition.duplicate(
            force_vector=0.0 * self.transition.force_vector
        )
        assert (
            np.linalg.norm(new_trans.force_vector_function(0.0)) == 0.0
        )  # side quest/test
        out = new_trans.discretise(dt=0.1)
        assert isinstance(out, randprocs.markov.discrete.LTIGaussian)

    def test_backward_rv(self, some_normal_rv1, some_normal_rv2):
        out, _ = self.transition.backward_rv(
            some_normal_rv1, some_normal_rv2, t=0.0, dt=0.1
        )
        assert isinstance(out, randvars.Normal)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        out, _ = self.transition.backward_realization(
            some_normal_rv1.mean, some_normal_rv2, t=0.0, dt=0.1
        )
        assert isinstance(out, randvars.Normal)

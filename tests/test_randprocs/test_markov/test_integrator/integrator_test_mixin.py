import numpy as np
import pytest

from probnum import randprocs, randvars
from probnum.problems.zoo import linalg as linalg_zoo


class IntegratorMixInTestMixIn:
    def test_uses_mixin(self):
        assert isinstance(self.transition, randprocs.markov.integrator.IntegratorMixIn)

    def test_num_derivatives(self):

        # Check for > 0, because we do not know how many the test cases generate.
        # If this property does not exist, however, this test will fail.
        assert self.transition.num_derivatives > 0

    def test_proj2coord(self):
        base = np.zeros(self.transition.num_derivatives + 1)
        base[0] = 1
        e_0_expected = np.kron(np.eye(1), base)
        e_0 = self.transition.proj2coord(coord=0)
        np.testing.assert_allclose(e_0, e_0_expected)

        base = np.zeros(self.transition.num_derivatives + 1)
        base[-1] = 1
        e_q_expected = np.kron(np.eye(1), base)
        e_q = self.transition.proj2coord(coord=self.transition.num_derivatives)
        np.testing.assert_allclose(e_q, e_q_expected)

    def test_precon(self):

        assert isinstance(
            self.transition.precon,
            randprocs.markov.integrator.NordsieckLikeCoordinates,
        )

    def test_state_ordering(self):
        assert self.transition.state_ordering in ["coordinate", "derivative"]

import numpy as np
import pytest

from probnum import randprocs, randvars
from probnum.problems.zoo import linalg as linalg_zoo
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

        self.G = lambda t: self.G_const
        self.v = lambda t: self.v_const
        self.L = lambda t: self.L_const

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)
        self.l = lambda t, x: self.L(t)

    def test_discretise(self):
        out = self.transition.discretise(dt=0.1)
        assert isinstance(out, randprocs.markov.discrete.LTIGaussian)

    def test_discretise_no_force(self):
        """LTISDE.discretise() works if there is zero force (there is an "if" in the
        fct)."""
        new_trans = self.transition.duplicate()
        new_trans.force_vector = np.zeros(len(new_trans.force_vector))

        # Sanity checks: if this does not work, the test is meaningless
        np.testing.assert_allclose(new_trans.force_vector_function(0.0), 0.0)
        np.testing.assert_allclose(new_trans.force_vector, 0.0)

        # Test discretisation
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

    def test_drift_matrix(self):
        # 1. Access works as expected.
        np.testing.assert_allclose(self.transition.drift_matrix, self.G_const)

        # 2. Wrong shape raises error
        with pytest.raises(ValueError):
            self.transition.drift_matrix = np.arange(len(self.G_const))

        # 3. Setting works as expected
        I_dxd = np.eye(len(self.G_const))
        self.transition.drift_matrix = I_dxd
        np.testing.assert_allclose(self.transition.drift_matrix, I_dxd)

        # 4. super() is updated correctly
        dummy_time = 0.1  # value does not matter.
        np.testing.assert_allclose(
            self.transition.drift_matrix_function(dummy_time), I_dxd
        )

    def test_force_vector(self):
        # 1. Access works as expected.
        np.testing.assert_allclose(self.transition.force_vector, self.v_const)

        # 2. Wrong shape raises error
        with pytest.raises(ValueError):
            self.transition.force_vector = np.arange(len(self.G_const) - 2)

        # 3. Setting works as expected
        v = 1 + 0.1 * np.random.rand(len(self.G_const))
        self.transition.force_vector = v
        np.testing.assert_allclose(self.transition.force_vector, v)

        # 4. super() is updated correctly
        dummy_time = 0.1  # value does not matter.
        np.testing.assert_allclose(self.transition.force_vector_function(dummy_time), v)

    def test_dispersion_matrix(self):
        # 1. Access works as expected.
        np.testing.assert_allclose(self.transition.dispersion_matrix, self.L_const)

        # 2. Wrong shape raises error
        with pytest.raises(ValueError):
            self.transition.dispersion_matrix = np.arange(len(self.L_const))

        # 3. Setting works as expected
        L = 1 + 0.1 * np.random.rand(*self.L_const.shape)
        self.transition.dispersion_matrix = L
        np.testing.assert_allclose(self.transition.dispersion_matrix, L)

        # 4. super() is updated correctly
        dummy_time = 0.1  # value does not matter.
        np.testing.assert_allclose(
            self.transition.dispersion_matrix_function(dummy_time), L
        )

    def test_duplicate_with_changed_coordinates(self):

        rng = np.random.default_rng(seed=2)
        spectrum1 = 1 + 0.25 * rng.uniform(size=len(self.G_const))
        spectrum2 = 1 + 0.25 * rng.uniform(size=len(self.G_const))
        P1 = np.diag(spectrum1)
        P2 = np.diag(spectrum2)
        P1inv = np.diag(1.0 / spectrum1)
        P2inv = np.diag(1.0 / spectrum2)

        # for now
        self.transition.force_vector = np.zeros(len(self.transition.force_vector))
        new_trans = self.transition.duplicate_with_changed_coordinates(
            outgoing=P1, incoming=P2
        )

        vector = rng.uniform(size=len(self.G_const))
        transformed_vector = P2inv @ vector

        assert np.linalg.norm(new_trans.force_vector) == 0.0
        disc1 = self.transition.discretise(0.2)
        disc2 = new_trans.discretise(0.2)

        print(P1 @ disc1.state_trans_mat @ P2, disc2.state_trans_mat)

        forw_classic, _ = self.transition.forward_realization(vector, t=0.1, dt=0.15)
        forw_transformed, _ = new_trans.forward_realization(
            transformed_vector, t=0.1, dt=0.15
        )

        raise RuntimeError("What is going on here?!")
        np.testing.assert_allclose(forw_classic.mean, P1inv @ forw_transformed.mean)

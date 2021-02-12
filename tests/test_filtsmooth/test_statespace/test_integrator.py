import unittest
import warnings

import numpy as np
import pytest

import probnum.random_variables as pnrv
from probnum.filtsmooth import statespace as pnfss
from probnum.problems.zoo.linalg import random_spd_matrix

from .test_sde import TestLTISDE


@pytest.fixture
def some_ordint(test_ndim):
    return test_ndim - 1


class TestIntegrator:
    """An integrator should be usable as is, but its tests are also useful for IBM,
    IOUP, etc."""

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self, some_ordint):
        self.some_ordint = some_ordint
        self.integrator = pnfss.Integrator(ordint=self.some_ordint, spatialdim=1)

    def test_proj2coord(self):
        base = np.zeros(self.some_ordint + 1)
        base[0] = 1
        e_0_expected = np.kron(np.eye(1), base)
        e_0 = self.integrator.proj2coord(coord=0)
        np.testing.assert_allclose(e_0, e_0_expected)

        base = np.zeros(self.some_ordint + 1)
        base[-1] = 1
        e_q_expected = np.kron(np.eye(1), base)
        e_q = self.integrator.proj2coord(coord=self.some_ordint)
        np.testing.assert_allclose(e_q, e_q_expected)

    def test_precon(self):

        assert isinstance(self.integrator.precon, pnfss.NordsieckLikeCoordinates)


class TestIBM(TestLTISDE, TestIntegrator):

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
        self.transition = pnfss.IBM(
            ordint=self.some_ordint,
            spatialdim=spatialdim,
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


class TestIOUP(TestLTISDE, TestIntegrator):

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
        self.transition = pnfss.IOUP(
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


class TestMatern(TestLTISDE, TestIntegrator):

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
        self.transition = pnfss.Matern(
            ordint=self.some_ordint,
            spatialdim=spatialdim,
            lengthscale=1.2345,
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


@pytest.fixture
def dt():
    return 0.1


@pytest.fixture
def ah_22_ibm(dt):
    return np.array(
        [
            [1.0, dt, dt ** 2 / 2.0],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def qh_22_ibm(dt):
    return np.array(
        [
            [dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
            [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
            [dt ** 3 / 6.0, dt ** 2 / 2.0, dt],
        ]
    )


@pytest.fixture
def spdmat3x3():
    return random_spd_matrix(3)


@pytest.fixture
def normal_rv3x3(spdmat3x3):

    return pnrv.Normal(
        mean=np.random.rand(3),
        cov=spdmat3x3,
        cov_cholesky=np.linalg.cholesky(spdmat3x3),
    )


class TestIBMValues:

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):
        spatialdim = 1  # make tests compatible with some_normal_rv1, etc.
        self.transition = pnfss.IBM(
            ordint=2,
            spatialdim=spatialdim,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

    def test_discretise_values(self, ah_22_ibm, qh_22_ibm, dt):
        discrete_model = self.transition.discretise(dt=dt)
        np.testing.assert_allclose(discrete_model.state_trans_mat, ah_22_ibm)
        np.testing.assert_allclose(discrete_model.proc_noise_cov_mat, qh_22_ibm)

    def test_forward_rv_values(self, normal_rv3x3, diffusion, ah_22_ibm, qh_22_ibm, dt):
        rv, _ = self.transition.forward_rv(
            normal_rv3x3, t=0.0, dt=dt, _diffusion=diffusion
        )
        np.testing.assert_allclose(ah_22_ibm @ normal_rv3x3.mean, rv[:3].mean)
        np.testing.assert_allclose(
            ah_22_ibm @ normal_rv3x3.cov @ ah_22_ibm.T + diffusion * qh_22_ibm,
            rv.cov,
        )

    def test_forward_realization_values(
        self, normal_rv3x3, diffusion, ah_22_ibm, qh_22_ibm, dt
    ):
        real = normal_rv3x3.sample()
        rv, _ = self.transition.forward_realization(
            real, t=0.0, dt=dt, _diffusion=diffusion
        )
        np.testing.assert_allclose(ah_22_ibm @ real, rv.mean)
        np.testing.assert_allclose(diffusion * qh_22_ibm, rv.cov)


#
# class TestIOUP(unittest.TestCase, NumpyAssertions):
#     def setUp(self):
#         driftspeed = 0.151231
#         self.ioup = pnfs.statespace.IOUP(2, 2, driftspeed)
#
#     def test_transition_rv(self):
#         mean, cov = np.ones(self.ioup.dimension), np.eye(self.ioup.dimension)
#         initrv = pnrv.Normal(mean, cov)
#         self.ioup.forward_rv(initrv, t=0.0, dt=STEP, _diffusion=DIFFCONST)
#
#     def test_transition_realization(self):
#         mean, cov = np.ones(self.ioup.dimension), np.eye(self.ioup.dimension)
#         real = pnrv.Normal(mean, cov).sample()
#         self.ioup.forward_realization(real, t=0.0, dt=STEP, _diffusion=DIFFCONST)
#
#     def test_asymptotically_ibm(self):
#         """For driftspeed==0, it coincides with the IBM prior."""
#         ioup_speed0 = pnfs.statespace.IOUP(2, 3, driftspeed=0.0)
#
#         ibm = pnfs.statespace.IBM(2, 3)
#         self.assertAllClose(ioup_speed0.driftmat, ibm.driftmat)
#         self.assertAllClose(ioup_speed0.forcevec, ibm.forcevec)
#         self.assertAllClose(ioup_speed0.dispmat, ibm.dispmat)
#
#         mean, cov = np.ones(ibm.dimension), np.eye(ibm.dimension)
#         rv = pnrv.Normal(mean, cov)
#         ibm_out, _ = ibm.forward_rv(rv, t=0.0, dt=STEP, _diffusion=1.2345)
#         ioup_out, _ = ioup_speed0.forward_rv(rv, t=0.0, dt=STEP, _diffusion=1.2345)
#         self.assertAllClose(ibm_out.mean, ioup_out.mean)
#         self.assertAllClose(ibm_out.cov, ioup_out.cov)
#
#         real = rv.sample()
#         ibm_out, _ = ibm.forward_realization(real, t=0.0, dt=STEP, _diffusion=DIFFCONST)
#         ioup_out, _ = ioup_speed0.forward_realization(
#             real, t=0.0, dt=STEP, _diffusion=DIFFCONST
#         )
#         self.assertAllClose(ibm_out.mean, ioup_out.mean)
#         self.assertAllClose(ibm_out.cov, ioup_out.cov)
#
#
# class TestMatern(unittest.TestCase, NumpyAssertions):
#     """Test whether coefficients for q=1, 2 match closed form.
#
#     ... and whether coefficients for q=0 are Ornstein Uhlenbeck.
#     """
#
#     def setUp(self):
#         lenscale = np.random.rand()
#         self.diffusion = np.random.rand()
#         self.mat0 = pnfs.statespace.Matern(0, 1, lenscale)
#         self.mat1 = pnfs.statespace.Matern(1, 1, lenscale)
#         self.mat2 = pnfs.statespace.Matern(2, 1, lenscale)
#
#     def test_n0(self):
#         """Closed form solution for ordint=0.
#
#         This is the OUP.
#         """
#         xi = np.sqrt(2 * (self.mat0.dimension - 0.5)) / self.mat0.lengthscale
#         self.assertAlmostEqual(self.mat0.driftmat[0, 0], -xi)
#
#     def test_n1(self):
#         """Closed form solution for ordint=1."""
#         xi = np.sqrt(2 * (self.mat1.dimension - 0.5)) / self.mat1.lengthscale
#         expected = np.array([-(xi ** 2), -2 * xi])
#         self.assertAllClose(self.mat1.driftmat[-1, :], expected)
#
#     def test_n2(self):
#         """Closed form solution for n=2."""
#         xi = np.sqrt(2 * (self.mat2.dimension - 0.5)) / self.mat2.lengthscale
#         expected = np.array([-(xi ** 3), -3 * xi ** 2, -3 * xi])
#         self.assertAllClose(self.mat2.driftmat[-1, :], expected)
#
#     def test_larger_shape(self):
#         mat2d = pnfs.statespace.Matern(2, 2, 1.0)
#         self.assertEqual(mat2d.dimension, 2 * (2 + 1))
#
#     def test_transition_rv(self):
#         mean, cov = np.ones(self.mat1.dimension), np.eye(self.mat1.dimension)
#         initrv = pnrv.Normal(mean, cov)
#         self.mat1.forward_rv(initrv, t=0.0, dt=STEP, _diffusion=self.diffusion)
#
#     def test_transition_real(self):
#         mean, cov = np.ones(self.mat1.dimension), np.eye(self.mat1.dimension)
#         real = pnrv.Normal(mean, cov).sample()
#         self.mat1.forward_realization(real, t=0.0, dt=STEP, _diffusion=self.diffusion)
#
#
# if __name__ == "__main__":
#     unittest.main()

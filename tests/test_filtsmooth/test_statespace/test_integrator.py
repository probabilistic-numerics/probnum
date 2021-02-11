import unittest

import numpy as np
import pytest

import probnum.random_variables as pnrv
from probnum.filtsmooth import statespace as pnfss

from .test_sde import TestLTISDE


class TestIntegrator:
    """An integrator should be usable as is, but its tests are also useful for IBM,
    IOUP, etc."""

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        test_ndim,
    ):
        self.some_ordint = 2
        self.integrator = pnfss.Integrator(
            ordint=self.some_ordint, spatialdim=test_ndim
        )

    def test_proj2coord(self, test_ndim):
        base = np.zeros(self.some_ordint + 1)
        base[0] = 1
        e_0_expected = np.kron(np.eye(test_ndim), base)
        e_0 = self.integrator.proj2coord(coord=0)
        np.testing.assert_allclose(e_0, e_0_expected)

        base = np.zeros(self.some_ordint + 1)
        base[-1] = 1
        e_q_expected = np.kron(np.eye(test_ndim), base)
        e_q = self.integrator.proj2coord(coord=self.some_ordint)
        np.testing.assert_allclose(e_q, e_q_expected)

    def test_precon(self):

        assert isinstance(self.integrator.precon, pnfss.NordsieckLikeCoordinates)


@pytest.fixture
def dt():
    return 0.1


@pytest.fixture
def ah_22_ibm(dt):
    return np.array(
        [
            [1.0, dt, dt ** 2 / 2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, dt, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, dt, dt ** 2 / 2.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def qh_22_ibm(dt):
    return np.array(
        [
            [dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0, 0.0, 0.0, 0.0],
            [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0, 0.0, 0.0, 0.0],
            [dt ** 3 / 6.0, dt ** 2 / 2.0, dt, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
            [0.0, 0.0, 0.0, dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
            [0.0, 0.0, 0.0, dt ** 3 / 6.0, dt ** 2 / 2.0, dt],
        ]
    )


class TestIBM(TestLTISDE):

    pass


#
# class TestIBM(unittest.TestCase, NumpyAssertions):
#     def setUp(self):
#         self.sde = pnfs.statespace.IBM(ordint=2, spatialdim=2)
#
#     def test_discretise(self):
#         discrete_model = self.sde.discretise(dt=STEP)
#         self.assertAllClose(discrete_model.state_trans_mat, AH_22_IBM, 1e-14)
#
#     def test_transition_rv(self):
#         mean, cov = np.ones(self.sde.dimension), np.eye(self.sde.dimension)
#         initrv = pnrv.Normal(mean, cov)
#         rv, _ = self.sde.forward_rv(rv=initrv, t=0.0, dt=STEP, _diffusion=DIFFCONST)
#         self.assertAllClose(AH_22_IBM @ initrv.mean, rv.mean, 1e-14)
#         self.assertAllClose(
#             AH_22_IBM @ initrv.cov @ AH_22_IBM.T + QH_22_IBM, rv.cov, 1e-14
#         )
#
#     def test_transition_realization(self):
#         mean, cov = np.ones(self.sde.dimension), np.eye(self.sde.dimension)
#         state = pnrv.Normal(mean, cov).sample()
#         rv, _ = self.sde.forward_realization(
#             real=state, t=0.0, dt=STEP, _diffusion=DIFFCONST
#         )
#         self.assertAllClose(AH_22_IBM @ state, rv.mean, 1e-14)
#         self.assertAllClose(QH_22_IBM, rv.cov, 1e-14)
#
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

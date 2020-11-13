import unittest

import numpy as np

import probnum.filtsmooth as pnfs
import probnum.random_variables as pnrv
from tests.testing import NumpyAssertions


class TestIntegrator(unittest.TestCase, NumpyAssertions):
    def setUp(self) -> None:
        self.q = 3
        self.d = 2
        self.integrator = pnfs.statespace.Integrator(ordint=self.q, spatialdim=self.d)

    def test_proj2deriv(self):
        with self.subTest():
            base = np.zeros(self.q + 1)
            base[0] = 1
            e_0_expected = np.kron(np.eye(self.d), base)
            e_0 = self.integrator.proj2deriv(coord=0)
            self.assertAllClose(e_0, e_0_expected, rtol=1e-15, atol=0)

        with self.subTest():
            base = np.zeros(self.q + 1)
            base[-1] = 1
            e_q_expected = np.kron(np.eye(self.d), base)
            e_q = self.integrator.proj2deriv(coord=self.q)
            self.assertAllClose(e_q, e_q_expected, rtol=1e-15, atol=0)


STEP = np.random.rand()
DIFFCONST = np.random.rand()

AH_22_IBM = np.array(
    [
        [1.0, STEP, STEP ** 2 / 2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, STEP, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, STEP, STEP ** 2 / 2.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, STEP],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)

QH_22_IBM = DIFFCONST ** 2 * np.array(
    [
        [STEP ** 5 / 20.0, STEP ** 4 / 8.0, STEP ** 3 / 6.0, 0.0, 0.0, 0.0],
        [STEP ** 4 / 8.0, STEP ** 3 / 3.0, STEP ** 2 / 2.0, 0.0, 0.0, 0.0],
        [STEP ** 3 / 6.0, STEP ** 2 / 2.0, STEP, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, STEP ** 5 / 20.0, STEP ** 4 / 8.0, STEP ** 3 / 6.0],
        [0.0, 0.0, 0.0, STEP ** 4 / 8.0, STEP ** 3 / 3.0, STEP ** 2 / 2.0],
        [0.0, 0.0, 0.0, STEP ** 3 / 6.0, STEP ** 2 / 2.0, STEP],
    ]
)


AH_21_PRE = np.array([[1, 1, 0.5], [0, 1, 1], [0, 0, 1]])

QH_21_PRE = (
    DIFFCONST ** 2
    * STEP
    * np.array([[1 / 20, 1 / 8, 1 / 6], [1 / 8, 1 / 3, 1 / 2], [1 / 6, 1 / 2, 1]])
)


class TestIBM(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        self.sde = pnfs.statespace.IBM(ordint=2, spatialdim=2, diffconst=DIFFCONST)

    def test_discretise(self):
        discrete_model = self.sde.discretise(step=STEP)
        self.assertAllClose(discrete_model.dynamat, AH_22_IBM, 1e-14)

    def test_transition_rv(self):
        mean, cov = np.ones(self.sde.dimension), np.eye(self.sde.dimension)
        initrv = pnrv.Normal(mean, cov)
        rv, _ = self.sde.transition_rv(rv=initrv, start=0.0, stop=STEP)
        self.assertAllClose(AH_22_IBM @ initrv.mean, rv.mean, 1e-14)
        self.assertAllClose(
            AH_22_IBM @ initrv.cov @ AH_22_IBM.T + QH_22_IBM, rv.cov, 1e-14
        )

    def test_transition_rv_preconditioned(self):
        """Check that if the flag is set, the result is different!"""
        mean, cov = np.ones(self.sde.dimension), np.eye(self.sde.dimension)
        initrv = pnrv.Normal(mean, cov)
        rv1, _ = self.sde.transition_rv(rv=initrv, start=0.0, stop=STEP)
        rv2, _ = self.sde.transition_rv(
            rv=initrv, start=0.0, stop=STEP, already_preconditioned=True
        )
        diff1 = np.abs(rv1.mean - rv2.mean)
        diff2 = np.abs(rv1.cov - rv2.cov)

        # Choose some 'sufficiently positive' constant
        # that worked in the present example
        self.assertGreater(np.linalg.norm(diff1), 1e-2)
        self.assertGreater(np.linalg.norm(diff2), 1e-2)

    def test_transition_realization(self):
        mean, cov = np.ones(self.sde.dimension), np.eye(self.sde.dimension)
        state = pnrv.Normal(mean, cov).sample()
        rv, _ = self.sde.transition_realization(real=state, start=0.0, stop=STEP)
        self.assertAllClose(AH_22_IBM @ state, rv.mean, 1e-14)
        self.assertAllClose(QH_22_IBM, rv.cov, 1e-14)

    def test_transition_realization_preconditioned(self):
        mean, cov = np.ones(self.sde.dimension), np.eye(self.sde.dimension)
        state = pnrv.Normal(mean, cov).sample()
        rv1, _ = self.sde.transition_realization(real=state, start=0.0, stop=STEP)
        rv2, _ = self.sde.transition_realization(
            real=state, start=0.0, stop=STEP, already_preconditioned=True
        )
        diff1 = np.abs(rv1.mean - rv2.mean)
        diff2 = np.abs(rv1.cov - rv2.cov)

        # Choose some 'sufficiently positive' constant
        # that worked in the present example
        self.assertGreater(np.linalg.norm(diff1), 1e-2)
        self.assertGreater(np.linalg.norm(diff2), 1e-2)


if __name__ == "__main__":
    unittest.main()

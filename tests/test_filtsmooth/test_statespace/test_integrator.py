import unittest
import probnum.filtsmooth as pnfs
import numpy as np
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


if __name__ == "__main__":
    unittest.main()

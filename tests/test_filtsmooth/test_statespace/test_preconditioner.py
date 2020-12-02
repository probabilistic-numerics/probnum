import unittest

import numpy as np

import probnum.filtsmooth as pnfs
from tests.testing import NumpyAssertions


class PreconditionerTestCase:
    def test_call(self):
        P = self.precon(0.5)
        self.assertEqual(P.ndim, 2)
        self.assertEqual(P.shape[0], (self.some_order + 1) * self.some_dim)
        self.assertEqual(P.shape[1], (self.some_order + 1) * self.some_dim)

    def test_inverse(self):
        P = self.precon(0.5)
        Pinv = self.precon.inverse(step=0.5)
        self.assertAllClose(P @ Pinv, np.eye(*P.shape), rtol=1e-15, atol=0.0)
        self.assertAllClose(Pinv @ P, np.eye(*P.shape), rtol=1e-15, atol=0.0)


class TestNordsieckLikeCoordinates(
    PreconditionerTestCase, unittest.TestCase, NumpyAssertions
):
    def setUp(self):
        self.some_order = 3
        self.some_dim = 1
        self.precon = pnfs.statespace.NordsieckLikeCoordinates.from_order(
            self.some_order, self.some_dim
        )


if __name__ == "__main__":
    unittest.main()

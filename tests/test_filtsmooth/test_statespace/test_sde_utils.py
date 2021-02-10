import numpy as np
import pytest

import probnum.random_variables as pnrv
from probnum.filtsmooth import statespace as pnfss


@pytest.fixture
def dt():
    return 0.1


@pytest.fixture
def Ah_closedform(dt):
    return np.array([[1.0, dt], [0.0, 1.0]])


@pytest.fixture
def Qh_closedform(dt):
    return np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])


@pytest.fixture
def F():
    return np.array([[0.0, 1.0], [0.0, 0.0]])


def L0():
    return np.array([0.0, 1.0])


def L1():
    return np.array([[0.0], [1.0]])


@pytest.mark.parametrize("L", [L0(), L1()])
def test_matrix_fraction_decomposition(F, L, dt, Ah_closedform, Qh_closedform):
    """Test MFD against a closed-form IBM solution."""
    Ah, Qh, _ = pnfss.matrix_fraction_decomposition(F, L, dt=dt)

    np.testing.assert_allclose(Ah, Ah_closedform)
    np.testing.assert_allclose(Qh, Qh_closedform)


# The below shall stay for a bit.
#
# class TestMatrixFractionDecomposition(unittest.TestCase):
#     """Test MFD against closed-form IBM solution."""
#
#     def setUp(self):
#         self.a = np.array([[0, 1], [0, 0]])
#         self.dc = 1.23451432151241
#         self.b = self.dc * np.array([[0], [1]])
#         self.h = 0.1
#
#     def test_ibm_qh_stack(self):
#         *_, stack = pnfss.sde.matrix_fraction_decomposition(self.a, self.b, self.h)
#
#         with self.subTest("top left"):
#             error = np.linalg.norm(stack[:2, :2] - self.a)
#             self.assertLess(error, 1e-15)
#
#         with self.subTest("top right"):
#             error = np.linalg.norm(stack[:2, 2:] - self.b @ self.b.T)
#             self.assertLess(error, 1e-15)
#
#         with self.subTest("bottom left"):
#             error = np.linalg.norm(stack[2:, 2:] + self.a.T)
#             self.assertLess(error, 1e-15)
#
#         with self.subTest("bottom right"):
#             error = np.linalg.norm(stack[2:, :2] - 0.0)
#             self.assertLess(error, 1e-15)
#
#     def test_ibm_ah(self):
#         Ah, *_ = pnfss.sde.matrix_fraction_decomposition(self.a, self.b, self.h)
#         expected = np.array([[1, self.h], [0, 1]])
#         error = np.linalg.norm(Ah - expected)
#         self.assertLess(error, 1e-15)
#
#     def test_ibm_qh(self):
#         _, Qh, _ = pnfss.sde.matrix_fraction_decomposition(self.a, self.b, self.h)
#         expected = self.dc ** 2 * np.array(
#             [[self.h ** 3 / 3, self.h ** 2 / 2], [self.h ** 2 / 2, self.h]]
#         )
#         error = np.linalg.norm(Qh - expected)
#         self.assertLess(error, 1e-15)
#
#     def test_type_error_captured(self):
#         good_A = np.array([[0, 1], [0, 0]])
#         good_B = np.array([[0], [1]])
#         good_h = 0.1
#         with self.subTest(culprit="F"):
#             with self.assertRaises(ValueError):
#                 pnfss.sde.matrix_fraction_decomposition(
#                     np.random.rand(2), good_B, good_h
#                 )
#
#         with self.subTest(culprit="L"):
#             with self.assertRaises(ValueError):
#                 pnfss.sde.matrix_fraction_decomposition(
#                     good_A, np.random.rand(2), good_h
#                 )
#
#         with self.subTest(culprit="h"):
#             with self.assertRaises(ValueError):
#                 pnfss.sde.matrix_fraction_decomposition(
#                     good_A, good_B, np.random.rand(2)
#                 )

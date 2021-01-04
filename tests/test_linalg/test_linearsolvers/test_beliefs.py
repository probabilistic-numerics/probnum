"""Test cases for linear system beliefs."""

import unittest

import numpy as np

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers.beliefs import (
    LinearSystemBelief,
    NoisyLinearSystemBelief,
    WeakMeanCorrespondenceBelief,
)
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix
from tests.testing import NumpyAssertions

# pylint: disable="invalid-name"


class LinearSystemBeliefTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for linear system beliefs."""

    def setUp(self) -> None:
        """Test resources for linear system beliefs."""
        self.rng = np.random.default_rng(42)
        self.linsys = LinearSystem.from_matrix(
            A=random_spd_matrix(dim=10), random_state=self.rng
        )

    def test_dimension_mismatch_raises_value_error(self):
        """Test whether mismatched components result in a ValueError."""
        m, n, nrhs = 5, 3, 2
        A = rvs.Normal(mean=np.ones((m, n)), cov=np.eye(m * n))
        Ainv = A
        x = rvs.Normal(mean=np.zeros((n, nrhs)), cov=np.eye(n * nrhs))
        b = rvs.Constant(np.ones((m, nrhs)))

        # A does not match b
        with self.assertRaises(ValueError):
            LinearSystemBelief(
                A=A, Ainv=Ainv, x=x, b=rvs.Constant(np.ones((m + 1, nrhs)))
            )

        # A does not match x
        with self.assertRaises(ValueError):
            LinearSystemBelief(
                A=A,
                Ainv=Ainv,
                x=rvs.Normal(mean=np.zeros((n + 1, nrhs)), cov=np.eye((n + 1) * nrhs)),
                b=b,
            )

        # x does not match b
        with self.assertRaises(ValueError):
            LinearSystemBelief(
                A=A,
                Ainv=Ainv,
                x=rvs.Normal(mean=np.zeros((n, nrhs + 1)), cov=np.eye(n * (nrhs + 1))),
                b=b,
            )

        # A does not match Ainv
        with self.assertRaises(ValueError):
            LinearSystemBelief(
                A=A,
                Ainv=rvs.Normal(mean=np.ones((m + 1, n)), cov=np.eye((m + 1) * n)),
                x=x,
                b=b,
            )

    def test_beliefs_are_two_dimensional(self):
        """Check whether all beliefs over quantities of interest are 2 dimensional."""
        m, n = 5, 3
        A = rvs.Normal(mean=np.ones((m, n)), cov=np.eye(m * n))
        Ainv = A
        x = rvs.Normal(mean=np.zeros(m), cov=np.eye(m))
        b = rvs.Constant(np.ones(m))
        belief = LinearSystemBelief(A=A, Ainv=Ainv, x=x, b=b)

        self.assertEqual(belief.A.ndim, 2)
        self.assertEqual(belief.Ainv.ndim, 2)
        self.assertEqual(belief.x.ndim, 2)
        self.assertEqual(belief.b.ndim, 2)

    def test_non_two_dimensional_raises_value_error(self):
        """Test whether specifying higher-dimensional random variables raise a
        ValueError."""
        A = rvs.Constant(np.eye(5))
        Ainv = rvs.Constant(np.eye(5))
        x = rvs.Constant(np.ones((5, 1)))
        b = rvs.Constant(np.ones((5, 1)))

        # A.ndim == 3
        with self.assertRaises(ValueError):
            LinearSystemBelief(A=A[:, None], Ainv=Ainv, x=x, b=b)

        # Ainv.ndim == 3
        with self.assertRaises(ValueError):
            LinearSystemBelief(A=A, Ainv=Ainv[:, None], x=x, b=b)

        # x.ndim == 3
        with self.assertRaises(ValueError):
            LinearSystemBelief(A=A, Ainv=Ainv, x=x[:, None], b=b)

        # b.ndim == 3
        with self.assertRaises(ValueError):
            LinearSystemBelief(A=A, Ainv=Ainv, x=x, b=b[:, None])

    def test_from_solution_array(self):
        """Test whether a linear system belief can be created from a solution estimate
        given as an array."""
        x0 = self.rng.normal(size=self.linsys.A.shape[1])
        LinearSystemBelief.from_solution(x0=x0, problem=self.linsys)

    def test_induced_solution_has_correct_distribution(self):
        """"""

    def test_from_solution_generates_consistent_inverse_belief(self):
        """"""

    def test_from_solution_creates_better_initialization(self):
        """"""

    def test_from_matrix_array(self):
        """Test whether a linear system belief can be created from a system matrix
        estimate given as an array."""
        A0 = self.rng.normal(size=self.linsys.A.shape)
        LinearSystemBelief.from_matrix(A0=A0, problem=self.linsys)

    def test_from_inverse_array(self):
        """Test whether a linear system belief can be created from an inverse estimate
        given as an array."""
        Ainv0 = self.rng.normal(size=self.linsys.A.shape)
        LinearSystemBelief.from_inverse(Ainv0=Ainv0, problem=self.linsys)


class WeakMeanCorrespondenceBeliefTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for the weak mean correspondence belief."""

    def test_from_matrix_satisfies_mean_correspondence(self):
        """"""

    def test_from_inverse_satisfies_mean_correspondence(self):
        """"""

    def test_means_correspond_weakly(self):
        """Test whether :math:`\mathbb{E}[A]^{-1}y = \mathbb{E}[H]y` for all actions
        :math:`y`."""

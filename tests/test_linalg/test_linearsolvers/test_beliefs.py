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
from tests.testing import NumpyAssertions

# pylint: disable="invalid-name"


class LinearSystemBeliefTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for linear system beliefs."""

    def setUp(self) -> None:
        """Test resources for linear system beliefs."""

    def test_dimension_mismatch_raises_value_error(self):
        """Test whether mismatched beliefs result in a ValueError."""
        A = rvs.Constant(np.eye(5))

        # A does not match b
        with self.assertRaises(ValueError):
            LinearSystemBelief(
                A=A,
                Ainv=A,
                x=rvs.Constant(np.zeros(A.shape[1])),
                b=rvs.Constant(np.zeros(A.shape[0] + 1)),
            )

        # A does not match x
        with self.assertRaises(ValueError):
            LinearSystemBelief(
                A=A,
                Ainv=A,
                x=rvs.Constant(np.zeros(A.shape[1] + 1)),
                b=rvs.Constant(np.zeros(A.shape[0])),
            )

        # x does not match b
        with self.assertRaises(ValueError):
            LinearSystemBelief(
                A=A,
                Ainv=A,
                x=rvs.Constant(np.zeros((A.shape[1], 3))),
                b=rvs.Constant(np.zeros((A.shape[0], 4))),
            )

    def test_beliefs_are_two_dimensional(self):
        """Test whether all beliefs are represented as 2D random variables."""
        belief = LinearSystemBelief(
            x=rvs.Normal(mean=np.ones(5), cov=np.eye(5)),
            A=rvs.Normal(mean=linops.Identity(5), cov=linops.Identity(25)),
            Ainv=rvs.Normal(mean=linops.Identity(5), cov=linops.Identity(25)),
            b=rvs.Constant(np.ones(5)),
        )

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
